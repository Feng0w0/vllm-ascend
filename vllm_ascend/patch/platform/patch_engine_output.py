# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy EngineCore output frames before their backing storage can be reused.

This is the conservative transport used by the Qwen4Exp rollout experiments.
Keep the upstream coordinator and shutdown behavior. Remove when output buffer
ownership has been validated for zero-copy transport on the supported stack.
"""

from contextlib import ExitStack

import zmq
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.serial_utils import MsgpackEncoder


def process_output_sockets(self, output_paths: list[str], coord_output_path: str | None, engine_index: int):
    """Output socket IO thread."""
    # Msgpack serialization encoding.
    encoder = MsgpackEncoder()
    # We must set linger to ensure the ENGINE_CORE_DEAD
    # message is sent prior to closing the socket.
    with ExitStack() as stack, zmq.Context() as ctx:
        sockets = [
            stack.enter_context(make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000))
            for output_path in output_paths
        ]
        coord_socket = (
            stack.enter_context(make_zmq_socket(ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000))
            if coord_output_path is not None
            else None
        )

        while True:
            output = self.output_queue.get()
            if output == EngineCoreProc.ENGINE_CORE_DEAD:
                for socket in sockets:
                    socket.send(output)
                break
            assert not isinstance(output, bytes)
            client_index, outputs = output
            outputs.engine_index = engine_index

            if client_index == -1:
                # Don't reuse buffer for coordinator message
                # which will be very small.
                assert coord_socket is not None
                coord_socket.send_multipart(encoder.encode(outputs))
                continue

            # Keep the encoded header immutable and let ZeroMQ copy all
            # frames before returning. Do not reuse a mutable header while
            # output messages may still be in flight.
            buffers = encoder.encode(outputs)
            sockets[client_index].send_multipart(buffers, copy=True)


def apply_patch() -> None:
    # Called inside each EngineCore child, before output threads are created.
    # DP subclasses inherit this method; assignment is idempotent.
    EngineCoreProc.process_output_sockets = process_output_sockets
