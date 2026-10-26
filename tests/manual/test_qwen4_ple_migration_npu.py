# SPDX-License-Identifier: Apache-2.0
"""Single-NPU checks for migrated PLE kernels and pinned input staging."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import torch
import torch_npu  # noqa: F401
from vllm.models.qwen4_exp.common.ops import ple
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_ascend.models.qwen4_exp.input_adapter import Qwen4ExpInputContext, Qwen4ExpPLEInputAdapter
from vllm_ascend.ops.triton.qwen4_exp.ple import _ple_conv_kernel, _ple_conv_writeback_kernel
from vllm_ascend.patch.worker.patch_qwen4_ple import apply_patch


def check_kernel(mode, mapped, null_slot, dtype):
    lengths = [1, 1] if mode == "decode" else [2, 3]
    count = sum(lengths)
    channels, kernel_size, dilation = 37, 3, 2
    history_size = (kernel_size - 1) * dilation
    rows = [6, 0, 3, 2, 5][:count] if mapped else list(range(count))
    inputs = torch.randn(7 if mapped else count, channels).to(dtype)
    residual = torch.randn_like(inputs)
    outer = torch.randn_like(inputs)
    weights = torch.randn(channels, kernel_size).to(dtype)
    # Exercise noncontiguous state strides as used by shared state pools.
    state = torch.randn(2, history_size, channels).to(dtype).transpose(1, 2)
    expected = residual.clone()
    expected_state = state.clone()
    slots = [0, NULL_BLOCK_ID if null_slot else 1]
    initialized = [True, False]
    offset = 0
    for req, length in enumerate(lengths):
        selected = rows[offset : offset + length]
        slot = slots[req]
        if slot == NULL_BLOCK_ID:
            conv = torch.zeros(length, channels)
        else:
            history = state[slot].float() if initialized[req] else torch.zeros(channels, history_size)
            sequence = torch.cat([history, inputs[selected].float().T], dim=1)
            conv = torch.stack(
                [
                    sum(weights[:, k].float() * sequence[:, j + dilation * k] for k in range(kernel_size))
                    for j in range(length)
                ]
            )
            expected_state[slot] = sequence[:, -history_size:].to(dtype)
        activated = torch.nn.functional.silu(conv.to(dtype).float()).to(dtype)
        expected[selected] = (
            (residual[selected].float() + activated.float()).to(dtype).float() + outer[selected].float()
        ).to(dtype)
        offset += length
    device = torch.device("npu:0")
    x, y, z, w, cache = [t.to(device) for t in (inputs, residual, outer, weights, state)]
    qsl = torch.tensor([0, lengths[0], count], dtype=torch.int32, device=device)
    mapping = torch.tensor(rows, dtype=torch.int32, device=device) if mapped else None
    ple.ple_conv(
        inputs=x,
        residual=y,
        conv_state=cache,
        conv_weights=w,
        state_indices=torch.tensor(slots, dtype=torch.int32, device=device),
        outer_residual=z,
        mode=mode,
        dilation=dilation,
        query_start_loc=qsl,
        has_initial_states=torch.tensor(initialized, dtype=torch.bool, device=device),
        token_indices=mapping,
    )
    torch.npu.synchronize()
    tolerance = 0.04 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(y.cpu(), expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(cache.cpu(), expected_state, atol=0, rtol=0)


def check_staging():
    adapter = Qwen4ExpPLEInputAdapter(
        SimpleNamespace(ngram_size=3, eos_token_id=999), max_num_reqs=2, device=torch.device("npu:0")
    )
    context = Qwen4ExpInputContext(
        is_dummy=False,
        num_reqs=1,
        num_reqs_padded=2,
        query_start_loc=torch.tensor([0, 1, 1], dtype=torch.int32),
        num_computed_tokens=np.array([3]),
        token_ids=np.array([[7, 8, 9, 0]], dtype=np.int32),
    )
    pointers = None
    for i in range(32):
        source = torch.tensor([[i, i + 1], [999, 999]], dtype=torch.int32)
        values = {"ngram_context": source}
        adapter.prepare(values, context)
        # The non-pinned caller source can be reused after prepare returns.
        source.fill_(-1)
        torch.npu.synchronize()
        assert values["ngram_context"].cpu().tolist() == [[i, i + 1], [999, 999]]
        assert values["query_start_loc"].cpu().tolist() == [0, 1, 1]
        current = (values["ngram_context"].data_ptr(), values["query_start_loc"].data_ptr())
        assert pointers is None or current == pointers
        pointers = current


if __name__ == "__main__":
    torch.npu.set_device(0)
    torch.manual_seed(42)
    apply_patch()
    assert ple._ple_conv_kernel is _ple_conv_kernel
    assert ple._ple_conv_writeback_kernel is _ple_conv_writeback_kernel
    for case in product(["decode", "prefill"], [False, True], [False, True], [torch.float32, torch.bfloat16]):
        check_kernel(*case)
        print("KERNEL_PASS", case, flush=True)
    check_staging()
    print("PASS: 16 kernel cases and 32 staging iterations", flush=True)
