# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.distributed.aux_output_connector import worker
from vllm.distributed.aux_output_connector.connector import AuxOutputConnectorMetadata, PackedBlockHashes
from vllm.v1.outputs import AsyncModelRunnerOutput

from vllm_ascend.worker.v1.aux_output import AuxOutputModelRunnerOutput, prepare_aux_output


@pytest.fixture
def connector(monkeypatch):
    capturer = SimpleNamespace(shape_per_token=(1, 2), output_dtype_name="int32")
    capturer.snapshot_routing_data = lambda n: torch.arange(n * 2, dtype=torch.int32).reshape(n, 1, 2)
    monkeypatch.setattr(worker, "RoutedExpertsCapturer", lambda **kwargs: capturer)
    monkeypatch.setattr(worker, "bind_routed_experts_capturer", lambda *args: None)
    monkeypatch.setattr(worker, "get_tp_group", lambda: SimpleNamespace(is_first_rank=True))
    monkeypatch.setattr(worker, "resolve_kv_cache_block_sizes", lambda *args: (4, 4))
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16, max_num_seqs=4),
        aux_output_config=SimpleNamespace(max_bytes=4096),
        max_concurrent_batches=2,
    )
    result = worker.AuxOutputWorkerConnector(vllm_config=config, model=None, kv_cache_config=None)
    yield result
    result.close()


@pytest.fixture
def copy_stream(monkeypatch):
    event = Mock()
    stream = Mock()
    compute = object()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(Event=lambda: event, current_stream=lambda: compute, stream=lambda stream: nullcontext()),
    )
    stream.compute = compute
    stream.event = event
    return stream


def prepare(connector, ids, starts, lengths, hashes=None):
    connector.begin_step(
        AuxOutputConnectorMetadata(
            generation=0,
            requests={request: 0 for request in ids},
            block_hashes=hashes or {},
            finished_requests=(),
        )
    )
    batch = SimpleNamespace(req_ids=ids, num_computed_tokens_cpu=np.array(starts))
    scheduler = SimpleNamespace(num_scheduled_tokens=dict(zip(ids, lengths)), total_num_scheduled_tokens=sum(lengths))
    return prepare_aux_output(connector, batch, scheduler), batch


def complete(pending, ids, samples, copy_stream):
    output = SimpleNamespace(req_ids=ids, sampled_token_ids=samples)
    return AuxOutputModelRunnerOutput(output, pending, copy_stream).get_output().aux_output_connector_output


@pytest.mark.parametrize("same_stream", [False, True])
def test_unpadded_request_order_and_batch_snapshot(connector, copy_stream, monkeypatch, same_stream):
    if same_stream:
        monkeypatch.setattr(torch.npu, "current_stream", lambda: copy_stream)
    pending, batch = prepare(connector, ["b", "a"], [0, 0], [1, 2])
    batch.req_ids.reverse()
    batch.num_computed_tokens_cpu[:] = 99
    output = complete(pending, ["b", "a"], [[1], [2]], copy_stream)
    np.testing.assert_array_equal(output["b"].rows, [[[0, 1]]])
    np.testing.assert_array_equal(output["a"].rows, [[[2, 3]], [[4, 5]]])
    if same_stream:
        copy_stream.wait_stream.assert_not_called()
    else:
        copy_stream.wait_stream.assert_called_once_with(copy_stream.compute)
    copy_stream.event.synchronize.assert_called_once()


def test_chunked_prefill_materializes_all_routes(connector, copy_stream):
    first, _ = prepare(connector, ["a"], [0], [2])
    assert complete(first, ["a"], [[]], copy_stream) == {}
    second, _ = prepare(connector, ["a"], [2], [2], {"a": PackedBlockHashes(b"hash", 4)})
    output = complete(second, ["a"], [[7]], copy_stream)
    assert output["a"].token_start == 0
    np.testing.assert_array_equal(output["a"].rows[:, 0], [[0, 1], [2, 3], [0, 1], [2, 3]])


def test_prefix_hit_restores_original_routes(connector, copy_stream):
    hashes = {"a": PackedBlockHashes(b"hash", 4)}
    first, _ = prepare(connector, ["a"], [0], [4], hashes)
    complete(first, ["a"], [[7]], copy_stream)
    hit, _ = prepare(connector, ["hit"], [4], [1], {"hit": PackedBlockHashes(b"hash", 4)})
    output = complete(hit, ["hit"], [[8]], copy_stream)
    assert output["hit"].token_start == 0
    np.testing.assert_array_equal(output["hit"].rows[:, 0], [[0, 1], [2, 3], [4, 5], [6, 7], [0, 1]])


def test_async_token_output_is_consumed_before_route_commit(connector, copy_stream):
    pending, _ = prepare(connector, ["a"], [0], [1])

    class Tokens(AsyncModelRunnerOutput):
        def get_output(self):
            return SimpleNamespace(req_ids=["a"], sampled_token_ids=[[9]])

    result = AuxOutputModelRunnerOutput(Tokens(), pending, copy_stream).get_output()
    assert result.sampled_token_ids == [[9]]
    assert result.aux_output_connector_output["a"].rows.shape == (1, 1, 2)


def test_two_inflight_steps_keep_token_positions(connector, copy_stream):
    first, batch = prepare(connector, ["a"], [0], [1])
    second, _ = prepare(connector, ["a"], [1], [1])
    batch.num_computed_tokens_cpu[:] = 999
    one = complete(first, ["a"], [[1]], copy_stream)["a"]
    two = complete(second, ["a"], [[2]], copy_stream)["a"]
    assert (one.token_start, two.token_start) == (0, 1)
    assert one.rows.shape == two.rows.shape == (1, 1, 2)


def test_finished_request_waits_for_inflight_output(connector, copy_stream):
    pending, _ = prepare(connector, ["a"], [0], [1])
    connector.begin_step(
        AuxOutputConnectorMetadata(generation=0, requests={}, block_hashes={}, finished_requests=("a",))
    )
    assert "a" in connector._requests
    result = complete(pending, ["a"], [[1]], copy_stream)
    assert result["a"].token_start == 0
    assert "a" not in connector._requests
