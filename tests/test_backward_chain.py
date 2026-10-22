import asyncio
from types import SimpleNamespace

import pytest
import torch
from hivemind.proto import runtime_pb2
from hivemind.utils.tensor_descr import BatchTensorDescriptor

from bloombee.server import backward_chain as worker
from bloombee.utils.backward_protocol import (
    BACKWARD_CHAIN,
    BACKWARD_CHAIN_VERSION,
    MAX_BACKWARD_STAGES,
    validate_backward_chain,
)
from bloombee.utils.hivemind_compat import MSGPackSerializer, PeerID
from bloombee.utils.misc import DUMMY, is_dummy

PEER_A = "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG"
PEER_B = "QmZatFVTzzh66FNsd2aRHUNm45uvnwJEe8eKvPqagW6RQZ"


def route():
    return [dict(uid="model.1", peer_id=PEER_B, metadata={}), dict(uid="model.0", peer_id=PEER_A, metadata={})]


@pytest.mark.parametrize("failure", ["empty", "too_long", "tensor_count", "wrong_uid", "order", "prefix", "adapter"])
def test_rejects_invalid_chain(failure):
    stages, count, uid = route(), 5, "model.1"
    if failure == "empty":
        stages = []
    elif failure == "too_long":
        stages *= MAX_BACKWARD_STAGES
    elif failure == "tensor_count":
        count = 4
    elif failure == "wrong_uid":
        uid = "model.0"
    elif failure == "order":
        stages.reverse()
    elif failure == "prefix":
        stages[1]["uid"] = "other.0"
    elif failure == "adapter":
        stages[1]["metadata"]["active_adapter"] = "trainable"
    with pytest.raises(ValueError):
        validate_backward_chain(stages, uid, count)


@pytest.mark.parametrize("deep", [False, True])
def test_worker_gradient_goes_directly_to_previous_peer(monkeypatch, deep):
    stages = route()
    calls = []
    weights = {"model.0": 2.0, "model.1": 3.0}

    async def compute(inputs, grad, prompts, *, requested_backends, **kwargs):
        x = inputs.detach().requires_grad_(True)
        targets = [x]
        if not is_dummy(prompts):
            p = prompts.detach().requires_grad_(True)
            targets.append(p)
            y = x + p[0]
        else:
            y = x
        y = y * weights[requested_backends[0].uid]
        return list(torch.autograd.grad(y, targets, grad))

    handlers = {}
    for peer in [PEER_A, PEER_B]:
        handlers[peer] = SimpleNamespace(
            _p2p=SimpleNamespace(peer_id=PeerID.from_base58(peer)),
            _prioritizer=None,
            request_timeout=30,
            get_stub=lambda p2p, next_peer: str(next_peer),
        )
    backend = lambda uid: SimpleNamespace(uid=uid, optimizer=None, args_schema=[SimpleNamespace(compression=0)])

    async def direct(uid, stub, tensors, *, metadata, **kwargs):
        calls.append((str(handlers[PEER_B]._p2p.peer_id), stub, uid))
        return await worker.run_backward_chain(handlers[stub], uid, tensors, [backend(uid)], metadata)

    monkeypatch.setattr(worker, "run_rpc_backward", compute)
    monkeypatch.setattr(worker, "run_remote_backward_chain", direct)
    a = torch.randn(2, 3, 4)
    pa = torch.randn(1, 2, 3, 4) if deep else DUMMY
    pb = torch.randn(1, 2, 3, 4) if deep else DUMMY
    b = (a + pa[0] if deep else a) * 2
    metadata = {BACKWARD_CHAIN: stages, BACKWARD_CHAIN_VERSION: 1}
    grads = asyncio.run(
        worker.run_backward_chain(
            handlers[PEER_B],
            "model.1",
            [b, torch.ones_like(b), pb, a, pa],
            [backend("model.1")],
            metadata,
        )
    )
    torch.testing.assert_close(grads[0], torch.full_like(a, 6))
    assert calls == [(PEER_B, PEER_A, "model.0")]
    if deep:
        torch.testing.assert_close(grads[1][0], torch.full_like(a, 6))
        torch.testing.assert_close(grads[1][1], torch.full_like(a, 3))
    else:
        assert len(grads) == 1


def test_unacknowledged_worker_response_is_rejected():
    from bloombee.client.remote_forward_backward import _check_chain_response

    with pytest.raises(RuntimeError, match="acknowledge"):
        _check_chain_response(runtime_pb2.ExpertResponse())
    _check_chain_response(runtime_pb2.ExpertResponse(metadata=MSGPackSerializer.dumps({BACKWARD_CHAIN_VERSION: 1})))


@pytest.mark.parametrize("mode", ["unsupported", "success", "failed_chain"])
def test_client_negotiates_and_retains_fallback_inputs(monkeypatch, mode):
    from bloombee.client import sequential_autograd as client
    from bloombee.server.handler import TransformerConnectionHandler

    x = torch.randn(2, 3, 4)
    inputs = [x, x * 2]
    spans = [
        SimpleNamespace(peer_id=PeerID.from_base58(peer), start=i, end=i + 1) for i, peer in enumerate([PEER_A, PEER_B])
    ]
    info = {"forward_schema": ([BatchTensorDescriptor(3, 4, dtype=torch.float32, compression=0)], {})}
    info[BACKWARD_CHAIN_VERSION] = 0 if mode == "unsupported" else 1

    async def rpc_info(*args, **kwargs):
        return runtime_pb2.ExpertInfo(serialized_info=MSGPackSerializer.dumps(info))

    monkeypatch.setattr(TransformerConnectionHandler, "get_stub", lambda *args: SimpleNamespace(rpc_info=rpc_info))
    sent = []

    async def request(uid, stub, tensors, **kwargs):
        sent.append(uid)
        assert len(tensors) == 5
        if mode == "failed_chain":
            raise RuntimeError("Upstream worker disconnected after partial execution")
        return [torch.full_like(x, 6)]

    monkeypatch.setattr(client, "run_remote_backward_chain", request)
    legacy_calls = []

    async def legacy(uid, stub, info, inputs, grad, prompts, **kwargs):
        legacy_calls.append(uid)
        torch.testing.assert_close(inputs, x * (2 if uid == "model.1" else 1))
        return [grad * (3 if uid == "model.1" else 2)]

    monkeypatch.setattr(client, "run_remote_backward", legacy)
    manager = SimpleNamespace(
        config=SimpleNamespace(use_server_to_server_backward=True, active_adapter=None, request_timeout=5),
        rpc_info=info,
        block_uids=["model.0", "model.1"],
        state=SimpleNamespace(p2p=None),
        get_request_metadata=lambda *args, **kwargs: {},
        on_request_success=lambda peer: None,
    )
    result = asyncio.run(client.sequential_backward([torch.ones_like(x)], inputs, DUMMY, spans, manager))
    # The function copies inputs to CPU and consumes the shared route list.
    assert len(inputs) == 2 and len(spans) == 0
    torch.testing.assert_close(result[0][0], torch.full_like(x, 6))
    if mode == "unsupported":
        assert sent == [] and legacy_calls == ["model.1", "model.0"]
    elif mode == "failed_chain":
        assert sent == ["model.1"] and legacy_calls == ["model.1", "model.0"]
    else:
        assert sent == ["model.1"] and not legacy_calls


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("chained", [False, True])
@pytest.mark.parametrize("acknowledged", [False, True])
def test_backward_wire_protocol_and_legacy_streaming(monkeypatch, stream, chained, acknowledged):
    from bloombee.client import remote_forward_backward as wire
    from bloombee.client.config import ClientConfig
    from bloombee.utils.lossless_transport import serialize_torch_tensor

    expected = torch.randn(2, 3, 4)
    metadata = MSGPackSerializer.dumps({BACKWARD_CHAIN_VERSION: 1}) if acknowledged else b""
    response = runtime_pb2.ExpertResponse(tensors=[serialize_torch_tensor(expected, 0)], metadata=metadata)
    calls = []

    async def unary(request, **kwargs):
        calls.append("unary")
        return response

    async def streaming(requests, **kwargs):
        calls.append("stream")
        assert len([request async for request in requests]) >= 1

        async def parts():
            yield response

        return parts()

    stub = SimpleNamespace(rpc_backward=unary, rpc_backward_stream=streaming)
    config = ClientConfig(request_timeout=5, connect_timeout=5)
    monkeypatch.setattr(wire, "MAX_UNARY_PAYLOAD_SIZE", 1 if stream else 10**9)

    async def run():
        if chained:
            return await wire.run_remote_backward_chain(
                "model.1",
                stub,
                [expected],
                config=config,
                metadata={},
                compression=0,
            )
        fn = wire._backward_stream if stream else wire._backward_unary
        return await fn("model.1", [serialize_torch_tensor(expected, 0)], stub, config)

    if chained and not acknowledged:
        with pytest.raises(RuntimeError, match="acknowledge"):
            asyncio.run(run())
    else:
        actual = asyncio.run(run())
        torch.testing.assert_close(actual[0], expected)
    assert calls == ["stream" if stream else "unary"]
