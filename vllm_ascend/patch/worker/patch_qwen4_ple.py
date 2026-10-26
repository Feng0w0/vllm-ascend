# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind Ascend-safe kernels before Qwen4Exp model loading and graph capture."""

from importlib import import_module


def apply_patch() -> None:
    target = "vllm.models.qwen4_exp.common.ops.ple"
    try:
        upstream_ple = import_module(target)
    except ModuleNotFoundError as error:
        # Other supported vLLM versions may not contain Qwen4Exp.
        if error.name == target or (error.name and target.startswith(error.name + ".")):
            return
        raise

    from vllm_ascend.ops.triton.qwen4_exp.ple import _ple_conv_kernel, _ple_conv_writeback_kernel

    upstream_ple._ple_conv_kernel = _ple_conv_kernel
    upstream_ple._ple_conv_writeback_kernel = _ple_conv_writeback_kernel


apply_patch()
