"""
Utility functions that call RPC forward or backward on a single remote server
"""
import asyncio
from typing import Iterable, List, Optional, Tuple

import torch
from hivemind.p2p import StubBase
from hivemind.p2p.p2p_daemon_bindings.control import DEFAULT_MAX_MSG_SIZE, MAX_UNARY_PAYLOAD_SIZE
from hivemind.proto import runtime_pb2
from hivemind.utils.asyncio import aiter_with_timeout, iter_as_aiter
from hivemind.utils.streaming import split_for_streaming

from bloombee.client.config import ClientConfig
from bloombee.data_structures import ModuleUID, RPCInfo
from bloombee.utils.backward_protocol import BACKWARD_CHAIN_VERSION, BACKWARD_PROTOCOL_VERSION
from bloombee.utils.hivemind_compat import MSGPackSerializer, nested_flatten, nested_pack
from bloombee.utils.lossless_transport import (
    deserialize_tensor_stream,
    deserialize_torch_tensor,
    serialize_torch_tensor,
)


async def _forward_unary(
    uid: str,
    serialized_tensors: Iterable[runtime_pb2.Tensor],
    stub,
    config: ClientConfig,
    response_metadata=None,
    **kwargs,
) -> List[torch.Tensor]:
    outputs: runtime_pb2.ExpertResponse = await stub.rpc_forward(
        runtime_pb2.ExpertRequest(uid=uid, tensors=list(serialized_tensors), **kwargs),
        timeout=config.request_timeout,
    )
    if response_metadata is not None:
        response_metadata.update(MSGPackSerializer.loads(outputs.metadata) if outputs.metadata else {})
    return [deserialize_torch_tensor(t) for t in outputs.tensors]


async def _backward_unary(
    uid: str,
    serialized_tensors: Iterable[runtime_pb2.Tensor],
    stub,
    config: ClientConfig,
    **kwargs,
) -> List[torch.Tensor]:
    grad_inputs: runtime_pb2.ExpertResponse = await stub.rpc_backward(
        runtime_pb2.ExpertRequest(uid=uid, tensors=list(serialized_tensors), **kwargs),
        timeout=config.request_timeout,
    )
    _check_chain_response(grad_inputs)
    return [deserialize_torch_tensor(t) for t in grad_inputs.tensors]


async def _forward_stream(
    uid: str,
    serialized_tensors: Iterable[runtime_pb2.Tensor],
    stub,
    config: ClientConfig,
    response_metadata=None,
    **kwargs,
) -> List[torch.Tensor]:
    parts = (
        runtime_pb2.ExpertRequest(uid=uid, tensors=[part], **kwargs)
        for tensor in serialized_tensors
        for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE)
    )
    outputs = await asyncio.wait_for(stub.rpc_forward_stream(iter_as_aiter(parts)), config.connect_timeout)
    outputs = aiter_with_timeout(outputs, config.request_timeout)

    async def parts_with_metadata():
        async for msg in outputs:
            if response_metadata is not None:
                response_metadata.update(MSGPackSerializer.loads(msg.metadata) if msg.metadata else {})
            yield msg.tensors

    return await deserialize_tensor_stream(parts_with_metadata())


async def _backward_stream(uid, serialized_tensors, stub, config, *, collect=False, **kwargs):
    parts = [
        runtime_pb2.ExpertRequest(uid=uid, tensors=[part], **kwargs)
        for tensor in serialized_tensors
        for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE)
    ] or [runtime_pb2.ExpertRequest(uid=uid, **kwargs)]
    method = stub.rpc_collect_stream if collect else stub.rpc_backward_stream
    responses = await asyncio.wait_for(method(iter_as_aiter(parts)), config.connect_timeout)

    async def checked_parts():
        async for msg in aiter_with_timeout(responses, config.request_timeout):
            _check_chain_response(msg)
            yield msg.tensors

    return await deserialize_tensor_stream(checked_parts())


def _check_chain_response(response):
    metadata = MSGPackSerializer.loads(response.metadata) if response.metadata else {}
    if metadata.get(BACKWARD_CHAIN_VERSION) != BACKWARD_PROTOCOL_VERSION:
        raise RuntimeError("Worker did not acknowledge the full backward chain")


async def _serialize_tensors(inputs, compression):
    loop = asyncio.get_running_loop()
    return await asyncio.gather(
        *(loop.run_in_executor(None, serialize_torch_tensor, tensor, compression) for tensor in inputs)
    )


async def run_remote_backward_chain(uid, stub, tensors, *, config, metadata, compression, collect=False):
    """Send only gradients, or collect a result directly from the first worker."""
    inputs = tuple(tensor.detach().cpu().contiguous() for tensor in tensors)
    serialized = await _serialize_tensors(inputs, compression)
    size = sum(tensor.element_size() * tensor.nelement() for tensor in inputs)
    call = _backward_stream if collect or size > MAX_UNARY_PAYLOAD_SIZE // 2 else _backward_unary
    return await call(
        uid,
        serialized,
        stub,
        config,
        metadata=MSGPackSerializer.dumps(metadata),
        **({"collect": True} if collect else {}),
    )


async def run_remote_forward(
    uid: ModuleUID,
    stub: StubBase,
    rpc_info: RPCInfo,
    *inputs: torch.Tensor,
    config: ClientConfig,
    metadata: Optional[bytes] = None,
    response_metadata: Optional[dict] = None,
    **kwargs,
) -> Tuple[torch.Tensor, ...]:
    """
    Serializes input tensors and calls "rpc_forward" on a remote server.
    Mostly adapted from https://github.com/learning-at-home/hivemind/blob/7a7c93aefffc9494c39e7b170c07cb06d8c09c4c/hivemind/moe/client/expert.py#L198
    but without RemoteExpertWorker.run_coroutine() call that leads to deadlock here.
    """

    # Note: *inputs are flattened input tensors that follow the expert's info['input_schema']
    # detach to avoid pickling the computation graph
    assert len(kwargs) == len(rpc_info["keyword_names"]), f"Keyword args should be {rpc_info['keyword_names']}"
    kwargs = {key: kwargs[key] for key in rpc_info["keyword_names"]}

    # Note: we put keyword arguments in the same order as on a server to prevent f(a=1, b=2) != f(b=2, a=1) errors
    forward_inputs = tuple(nested_flatten((inputs, kwargs)))
    args_schema, _ = rpc_info["forward_schema"]
    compression = args_schema[0].compression
    inputs = tuple(tensor.detach().cpu().contiguous() for tensor in forward_inputs)
    # TODO: create more explicit way to check servers schema and client's structure
    assert len(inputs) >= len(args_schema) + 1, "Inputs and prompt tensors are necessary for a forward step"

    serialized_tensors = await _serialize_tensors(inputs, compression)

    # call RPC on remote server
    size = sum(t.element_size() * t.nelement() for t in inputs)
    forward_fn = _forward_stream if size > MAX_UNARY_PAYLOAD_SIZE // 2 else _forward_unary
    # Hotfix: we use "// 2" since hivemind==1.1.5 serializes bfloat16 tensors in float32, so they take 2x more space
    deserialized_outputs = await forward_fn(
        uid, serialized_tensors, stub, config, metadata=metadata, response_metadata=response_metadata, **kwargs
    )
    return nested_pack(deserialized_outputs, structure=rpc_info["outputs_schema"])
