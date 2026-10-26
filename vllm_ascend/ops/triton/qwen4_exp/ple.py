# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend PLE convolution kernels with compile-time token-map addressing.

Adapted from vLLM Qwen4Exp PLE kernels. Remove once upstream provides the
constexpr token-map branch and safe inactive map addresses on Ascend.
"""

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_reqs", "bs_iters"])
def _ple_conv_kernel(
    x_ptr,
    state_ptr,
    w_ptr,
    residual_ptr,
    outer_residual_ptr,
    state_idx_ptr,
    qsl_ptr,
    num_acc_ptr,
    has_init_ptr,
    token_idx_ptr,
    has_token_map: tl.constexpr,
    num_reqs,
    bs_iters,
    state_idx_stride,
    state_bs,
    state_ws,
    state_cs,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    STATE_LEN: tl.constexpr,
    DILATION: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    MODE: tl.constexpr,
    HAS_INIT: tl.constexpr,
    NULL_STATE_ID: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    t = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    # Mode-specific launches use logical row indices. Mixed batches map them
    # back to the original token order without gathering inputs or residuals.
    output_t = t.to(tl.int64)
    if has_token_map:
        output_t = tl.load(token_idx_ptr + t).to(tl.int64)

    if MODE == "decode":
        r = t
        q_start = t
        j = tl.full([], 0, tl.int32)
        slot_off = tl.full([], 0, tl.int32)
        in_range = True
    else:
        # Locate the request containing this token.
        lo = tl.full([], 1, tl.int32)
        hi = tl.full([], num_reqs + 1, tl.int32)
        for _ in range(bs_iters):
            mid = (lo + hi) // 2
            qmid = tl.load(qsl_ptr + mid, mask=mid <= num_reqs, other=0)
            pred = qmid <= t
            lo = tl.where(pred, mid + 1, lo)
            hi = tl.where(pred, hi, mid)
        r = tl.minimum(lo - 1, num_reqs - 1)
        q_start = tl.load(qsl_ptr + r)
        j = (t - q_start).to(tl.int32)
        total_real = tl.load(qsl_ptr + num_reqs)
        in_range = t < total_real
        if MODE == "spec":
            num_acc = tl.load(num_acc_ptr + r)
            # Roll back draft state to the last accepted token.
            slot_off = tl.minimum(tl.maximum(num_acc - 1, 0), SPEC_QUERY_LEN - 1).to(tl.int32)
        else:
            slot_off = tl.full([], 0, tl.int32)

    sid = tl.load(state_idx_ptr + r * state_idx_stride).to(tl.int64)
    state_ok = sid != NULL_STATE_ID
    sid_safe = tl.where(state_ok, sid, 0)
    if HAS_INIT:
        has_init = tl.load(has_init_ptr + r, mask=state_ok, other=0) != 0
    else:
        has_init = state_ok
    if MODE == "spec":
        # A new spec chunk can produce output from its own tokens without a
        # prior state; null slots in other modes denote inactive rows.
        read_state = state_ok
        out_ok = in_range
    else:
        read_state = state_ok & has_init
        out_ok = in_range & state_ok

    # state[sid, w, c] may be a strided view into a shared pool.
    base_state = state_ptr + sid_safe * state_bs
    acc = tl.zeros([BLOCK_C], tl.float32)
    for k in tl.static_range(0, KERNEL_SIZE):
        h = j + DILATION * k
        from_state = h <= STATE_LEN - 1
        state_tap = tl.load(
            base_state + (h + slot_off) * state_ws + c_offs * state_cs,
            mask=c_mask & read_state & from_state,
            other=0.0,
        )
        input_t = q_start + h - STATE_LEN
        if has_token_map:
            # Ascend scalar masked loads may read before selecting other.
            safe_input_t = tl.where(out_ok & (~from_state), input_t, 0)
            input_t = tl.load(
                token_idx_ptr + safe_input_t,
                mask=out_ok & (~from_state),
                other=input_t,
            )
        input_t = input_t.to(tl.int64)
        input_tap = tl.load(
            x_ptr + input_t * C + c_offs,
            mask=c_mask & out_ok & (~from_state),
            other=0.0,
        )
        tap = tl.where(from_state, state_tap, input_tap).to(tl.float32)
        weight = tl.load(
            w_ptr + c_offs * KERNEL_SIZE + k,
            mask=c_mask,
            other=0.0,
        ).to(tl.float32)
        acc += weight * tap

    # F.conv1d materializes its output dtype before SiLU.
    conv = acc.to(residual_ptr.dtype.element_ty).to(tl.float32)
    y = conv * tl.sigmoid(conv)
    conv_output = tl.where(out_ok, y, 0.0).to(residual_ptr.dtype.element_ty)
    residual = tl.load(
        residual_ptr + output_t * C + c_offs,
        mask=c_mask,
        other=0.0,
    )
    # Preserve the original eager operation boundaries: short convolution is
    # first accumulated into the BF16/FP16 PLE output, then the outer residual
    # is added to that rounded value.
    ple_output = (residual + conv_output).to(residual_ptr.dtype.element_ty)
    outer_residual = tl.load(
        outer_residual_ptr + output_t * C + c_offs,
        mask=c_mask,
        other=0.0,
    )
    ple_output = outer_residual.to(tl.float32) + ple_output.to(tl.float32)
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(
        residual_ptr + output_t * C + c_offs,
        ple_output,
        mask=c_mask,
    )

    if MODE == "decode":
        # Decode has one program per request and channel block, so no peer can
        # still be reading this state slice when it is updated.
        decode_input = tl.load(
            x_ptr + output_t * C + c_offs,
            mask=c_mask & state_ok,
            other=0.0,
        )
        for i in tl.static_range(0, STATE_LEN):
            if i < STATE_LEN - 1:
                next_state = tl.load(
                    base_state + (i + 1) * state_ws + c_offs * state_cs,
                    mask=c_mask & state_ok & has_init,
                    other=0.0,
                )
            else:
                next_state = decode_input
            tl.store(
                base_state + i * state_ws + c_offs * state_cs,
                next_state,
                mask=c_mask & state_ok,
            )


# Prefill and spec write back separately so every token reads the old state.
@triton.jit
def _ple_conv_writeback_kernel(
    x_ptr,
    state_ptr,
    state_idx_ptr,
    qsl_ptr,
    num_acc_ptr,
    has_init_ptr,
    token_idx_ptr,
    has_token_map: tl.constexpr,
    state_idx_stride,
    state_bs,
    state_ws,
    state_cs,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    STATE_LEN: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    MODE: tl.constexpr,
    HAS_INIT: tl.constexpr,
    NULL_STATE_ID: tl.constexpr,
):
    r = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    sid = tl.load(state_idx_ptr + r * state_idx_stride).to(tl.int64)
    state_ok = sid != NULL_STATE_ID
    if not state_ok:
        return

    q_start = tl.load(qsl_ptr + r)
    q_end = tl.load(qsl_ptr + r + 1)
    qlen = (q_end - q_start).to(tl.int32)
    if MODE == "spec":
        num_acc = tl.load(num_acc_ptr + r)
        # Use the same rollback window as the output kernel.
        slot_off = tl.minimum(tl.maximum(num_acc - 1, 0), SPEC_QUERY_LEN - 1).to(tl.int32)
        shift = 1
    else:
        slot_off = tl.full([], 0, tl.int32)
        shift = qlen
        if qlen <= 0:
            return

    has_init = tl.load(has_init_ptr + r) != 0 if HAS_INIT else True
    src_ok = state_ok if MODE == "spec" else state_ok & has_init

    WRITE_W: tl.constexpr = STATE_WIDTH if MODE == "spec" else STATE_LEN
    base_state = state_ptr + sid * state_bs
    for i in tl.static_range(0, WRITE_W):
        m = shift + i
        from_state = m <= STATE_LEN - 1
        # Short or graph-padded spec chunks preserve the unused state tail.
        do_write = i < STATE_LEN + qlen - 1 if MODE == "spec" else True
        state_value = tl.load(
            base_state + (slot_off + m) * state_ws + c_offs * state_cs,
            mask=c_mask & from_state & src_ok,
            other=0.0,
        )
        input_t = q_start + m - STATE_LEN
        if has_token_map:
            # Keep even inactive scalar map addresses inside the map.
            safe_input_t = tl.where((~from_state) & do_write, input_t, 0)
            input_t = tl.load(
                token_idx_ptr + safe_input_t,
                mask=(~from_state) & do_write,
                other=input_t,
            )
        input_t = input_t.to(tl.int64)
        input_value = tl.load(
            x_ptr + input_t * C + c_offs,
            mask=c_mask & (~from_state) & do_write,
            other=0.0,
        )
        value = tl.where(from_state, state_value, input_value)
        tl.store(
            base_state + i * state_ws + c_offs * state_cs,
            value,
            mask=c_mask & do_write,
        )
