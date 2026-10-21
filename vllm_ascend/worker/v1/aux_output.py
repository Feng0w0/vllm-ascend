# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Bridge V1 batches to vLLM's shared routed-experts output connector.

The connector owns prefix-cache storage and request lifetimes. This adapter
only translates the V1 batch and schedules its output copy. Remove it when
the retained V1 runner is removed. Speculative decoding and PP are excluded
by platform validation until their acceptance and rank mapping are covered.
"""

from types import SimpleNamespace

import numpy as np
import torch
from vllm.v1.outputs import AsyncModelRunnerOutput


def prepare_aux_output(connector, input_batch, scheduler_output):
    """Snapshot routes with real, unpadded boundaries in V1 request order."""
    request_ids = list(input_batch.req_ids)
    lengths = [scheduler_output.num_scheduled_tokens[request_id] for request_id in request_ids]
    query_start_loc = np.zeros(len(request_ids) + 1, dtype=np.int64)
    np.cumsum(lengths, out=query_start_loc[1:])
    assert query_start_loc[-1] == scheduler_output.total_num_scheduled_tokens
    batch = SimpleNamespace(
        req_ids=request_ids,
        query_start_loc_np=query_start_loc,
        num_computed_tokens_np=input_batch.num_computed_tokens_cpu[: len(request_ids)].copy(),
    )
    return connector.prepare_output(batch)


class AuxOutputModelRunnerOutput(AsyncModelRunnerOutput):
    """Keep route snapshots alive until the corresponding token output arrives."""

    def __init__(self, output, pending, copy_stream):
        self.output = output
        self.pending = pending
        self.copy_done = torch.npu.Event()
        counts = np.zeros(len(pending.request_ids), dtype=np.int32)
        compute_stream = torch.npu.current_stream()
        with torch.npu.stream(copy_stream):
            if copy_stream != compute_stream:
                copy_stream.wait_stream(compute_stream)
            pending.enqueue_cpu_copy(counts, counts.copy())
            self.copy_done.record()

    def get_output(self):
        output = self.output.get_output() if isinstance(self.output, AsyncModelRunnerOutput) else self.output
        self.copy_done.synchronize()
        assert output.req_ids == self.pending.request_ids
        # With MTP disabled, each sampled row contains zero or one accepted
        # token. Incomplete prefill chunks must report zero sampled tokens.
        self.pending.num_sampled = np.asarray([len(row) for row in output.sampled_token_ids], dtype=np.int32)
        assert np.all(self.pending.num_sampled <= 1)
        output.aux_output_connector_output = self.pending.process_output()
        return output
