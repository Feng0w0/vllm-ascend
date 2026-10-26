# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager
from queue import Queue
from threading import Thread
from types import SimpleNamespace

import numpy as np
import zmq
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc
from vllm.v1.serial_utils import MsgpackDecoder

from vllm_ascend.patch.platform import patch_engine_core, patch_engine_output


def test_copied_outputs_roundtrip_with_coordinator_and_shutdown(monkeypatch):
    calls = []

    @contextmanager
    def connect_socket(ctx, endpoint, socket_type, **kwargs):
        with ctx.socket(socket_type) as sock:
            sock.connect(endpoint)

            class OutputSocket:
                def send_multipart(self, frames, **options):
                    calls.append(options.get("copy", True))
                    return sock.send_multipart(frames, **options)

                def send(self, message):
                    return sock.send(message)

            yield OutputSocket()

    monkeypatch.setattr(patch_engine_output, "make_zmq_socket", connect_socket)
    queue = Queue()
    with zmq.Context() as ctx, ctx.socket(zmq.PULL) as client, ctx.socket(zmq.PULL) as coordinator:
        for sock in (client, coordinator):
            sock.setsockopt(zmq.RCVTIMEO, 10000)
            sock.bind("tcp://127.0.0.1:*")
        endpoint = client.getsockopt(zmq.LAST_ENDPOINT).decode()
        coord_endpoint = coordinator.getsockopt(zmq.LAST_ENDPOINT).decode()
        thread = Thread(
            target=patch_engine_output.process_output_sockets,
            args=(SimpleNamespace(output_queue=queue), [endpoint], coord_endpoint, 3),
            daemon=True,
        )
        thread.start()
        try:
            # Force auxiliary frames larger than the encoder's inline threshold.
            for i in range(8):
                route = np.full((128, 48, 10), i, dtype=np.int32)
                queue.put(
                    (
                        0,
                        EngineCoreOutputs(
                            outputs=[
                                EngineCoreOutput(request_id=str(i), new_token_ids=[i, i + 1], routed_experts=route)
                            ]
                        ),
                    )
                )
            queue.put((-1, EngineCoreOutputs(wave_complete=7)))
            decoder = MsgpackDecoder(EngineCoreOutputs)
            for i in range(8):
                result = decoder.decode(client.recv_multipart())
                assert result.engine_index == 3
                assert result.outputs[0].request_id == str(i)
                assert result.outputs[0].new_token_ids == [i, i + 1]
                np.testing.assert_array_equal(result.outputs[0].routed_experts, i)
            assert decoder.decode(coordinator.recv_multipart()).wave_complete == 7
            assert calls == [True] * 9
        finally:
            queue.put(EngineCoreProc.ENGINE_CORE_DEAD)
            thread.join(timeout=10)
        assert not thread.is_alive()
        assert client.recv() == EngineCoreProc.ENGINE_CORE_DEAD


def test_child_entry_installs_output_patch_before_engine_creation(monkeypatch):
    original = EngineCoreProc.process_output_sockets
    monkeypatch.setattr(EngineCoreProc, "process_output_sockets", original)
    monkeypatch.setattr(
        patch_engine_core,
        "init_ascend_config",
        lambda _: SimpleNamespace(
            scheduler_config=SimpleNamespace(profiling_chunk_config=SimpleNamespace(enabled=False))
        ),
    )
    monkeypatch.setattr(patch_engine_core, "_patch_dp_engine_core_proc", lambda *args: None)

    def start_engine(*args, **kwargs):
        assert EngineCoreProc.process_output_sockets is patch_engine_output.process_output_sockets
        assert DPEngineCoreProc.process_output_sockets is patch_engine_output.process_output_sockets
        return "started"

    monkeypatch.setattr(patch_engine_core, "_OriginalRunEngineCore", start_engine)
    assert patch_engine_core._run_engine_core_patch_func(vllm_config=object()) == "started"
