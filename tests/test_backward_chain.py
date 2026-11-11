import asyncio
import time
import uuid
from types import SimpleNamespace

import pytest
import torch
from hivemind.proto import runtime_pb2
from hivemind.utils.tensor_descr import BatchTensorDescriptor

from bloombee.client import remote_forward_backward as wire
from bloombee.client.config import ClientConfig
from bloombee.client.sequential_autograd import sequential_backward
from bloombee.server import backward_chain as worker
from bloombee.server.handler import TransformerConnectionHandler
from bloombee.utils.backward_protocol import (
    BACKWARD_CACHE_ID,
    BACKWARD_CHAIN,
    BACKWARD_CHAIN_VERSION,
    BACKWARD_NAMESPACE,
    BACKWARD_PROTOCOL_VERSION,
    MAX_BACKWARD_STAGES,
    validate_backward_chain,
)
from bloombee.utils.hivemind_compat import MSGPackSerializer, PeerID
from bloombee.utils.lossless_transport import deserialize_tensor_stream
from bloombee.utils.misc import DUMMY, is_dummy


def peer(i):
    return PeerID(b"\x12\x20" + bytes([i]) * 32)


def route(n=2):
    return [
        {
            "uid": f"model.{i}",
            "peer_id": str(peer(i + 1)),
            BACKWARD_CACHE_ID: uuid.uuid4().hex,
            BACKWARD_NAMESPACE: f"backward-{i % 2}",
        }
        for i in reversed(range(n))
    ]


@pytest.mark.parametrize(
    "failure", ["empty", "too_long", "wrong_uid", "order", "prefix", "cache_id", "namespace", "position"]
)
def test_rejects_invalid_chain(failure):
    stages, uid, pos = route(), "model.1", 0
    if failure == "empty":
        stages = []
    elif failure == "too_long":
        stages *= MAX_BACKWARD_STAGES
    elif failure == "wrong_uid":
        uid = "model.0"
    elif failure == "order":
        stages.reverse()
    elif failure == "prefix":
        stages[1]["uid"] = "other.0"
    elif failure == "cache_id":
        stages[1][BACKWARD_CACHE_ID] = "invalid"
    elif failure == "namespace":
        stages[1][BACKWARD_NAMESPACE] = "arbitrary-rpc"
    elif failure == "position":
        pos = 100
    with pytest.raises(ValueError):
        validate_backward_chain(stages, uid, pos)


def make_service(index):
    uid = f"model.{index}"
    backend = SimpleNamespace(
        uid=uid, optimizer=None, args_schema=[BatchTensorDescriptor(2, 3, 4, dtype=torch.float32)], kwargs_schema={}
    )

    async def gather(requests, context):
        parts = [r async for r in requests]
        tensors = await deserialize_tensor_stream(_parts(parts))
        return parts[0].uid, tensors, MSGPackSerializer.loads(parts[0].metadata)

    handler = SimpleNamespace(
        _handler_index=index % 2,
        _handler_event_queues=[None, None],
        _p2p=SimpleNamespace(peer_id=peer(index + 1)),
        _prioritizer=None,
        request_timeout=2,
        module_backends={uid: backend},
        _gather_inputs=gather,
    )
    handler._serialize_grads = TransformerConnectionHandler._serialize_grads.__get__(handler)
    return worker.BackwardService(handler)


async def _parts(parts):
    for r in parts:
        yield r.tensors


@pytest.mark.parametrize("stages", [1, 2, 4])
@pytest.mark.parametrize("deep", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_default_cached_backward_direct_result_and_cleanup(monkeypatch, stages, deep, stream):
    async def run():
        client_peer = peer(99)
        records = list(reversed(route(stages)))
        services = {str(peer(i + 1)): make_service(i) for i in range(stages)}
        calls = []

        async def compute(inputs, grad, prompts, *, requested_backends, **kwargs):
            x = inputs.detach().requires_grad_(True)
            targets, y = [x], x
            if not is_dummy(prompts):
                p = prompts.detach().requires_grad_(True)
                targets.append(p)
                y = y + p[0]
            y = y * (int(requested_backends[0].uid.split(".")[-1]) + 2)
            return list(torch.autograd.grad(y, targets, grad))

        def get_stub(p2p, target, *, namespace):
            service = services[str(target)]
            assert namespace == service.namespace
            ctx = SimpleNamespace(remote_id=p2p.peer_id)

            async def backward(request, **kwargs):
                calls.append((str(p2p.peer_id), str(target), len(request.tensors)))
                return await service.rpc_backward(request, ctx)

            async def backward_stream(requests, **kwargs):
                parts = [r async for r in requests]
                tensors = await deserialize_tensor_stream(_parts(parts))
                calls.append((str(p2p.peer_id), str(target), len(tensors)))

                async def again():
                    for part in parts:
                        yield part

                return service.rpc_backward_stream(again(), ctx)

            async def collect(requests, **kwargs):
                return service.rpc_collect_stream(requests, ctx)

            async def release(request, **kwargs):
                return await service.rpc_release(request, ctx)

            return SimpleNamespace(
                rpc_backward=backward,
                rpc_backward_stream=backward_stream,
                rpc_collect_stream=collect,
                rpc_release=release,
            )

        monkeypatch.setattr(worker, "run_rpc_backward", compute)
        monkeypatch.setattr(worker.BackwardService, "get_stub", staticmethod(get_stub))
        monkeypatch.setattr(wire, "MAX_UNARY_PAYLOAD_SIZE", 1 if stream else 10**9)
        x = torch.randn(2, 3, 4)
        prompts = torch.randn(stages, 2, 3, 4) if deep else DUMMY
        spans = []
        for i, record in enumerate(records):
            p = prompts[i : i + 1] if deep else DUMMY
            services[record["peer_id"]].save(record["uid"], [x, p], record, client_peer)
            x = (x + p[0] if deep else x) * (i + 2)
            spans.append(SimpleNamespace(peer_id=peer(i + 1)))
        manager = SimpleNamespace(
            config=ClientConfig(request_timeout=2, connect_timeout=2),
            state=SimpleNamespace(p2p=SimpleNamespace(peer_id=client_peer)),
            rpc_info={"forward_schema": ([SimpleNamespace(compression=0)], {})},
            on_request_success=lambda p: None,
        )
        grads, gp = await sequential_backward([torch.ones_like(x)], records, prompts, spans, manager)
        total = 1
        for i in range(stages):
            total *= i + 2
        torch.testing.assert_close(grads[0], torch.full_like(x, total))
        if deep:
            for i in range(stages):
                expected = 1
                for j in range(i, stages):
                    expected *= j + 2
                torch.testing.assert_close(gp[i], torch.full_like(x, expected))
        else:
            assert gp is None
        assert len(calls) == stages
        assert calls[0] == (str(client_peer), str(peer(stages)), 1)
        for i, call in enumerate(calls[1:]):
            assert call == (str(peer(stages - i)), str(peer(stages - i - 1)), 2 if deep else 1)
        assert not records and not spans
        assert all(not s.entries and s.nbytes == 0 for s in services.values())
        assert not hasattr(ClientConfig(), "use_server_to_server_backward")

    asyncio.run(run())


def test_cache_clones_inputs_rejects_replay_and_expires():
    async def run():
        service = make_service(0)
        record = route(1)[0]
        x = torch.randn(2, 3, 4)
        original = x.clone()
        service.save("model.0", [x, DUMMY], record, peer(99))
        x.add_(5)
        torch.testing.assert_close(service.entries[record[BACKWARD_CACHE_ID]].inputs, original)
        with pytest.raises(ValueError, match="Duplicate"):
            service.save("model.0", [x, DUMMY], record, peer(99))
        service.entries[record[BACKWARD_CACHE_ID]].expires = time.monotonic() - 1
        service.prune()
        assert not service.entries and service.nbytes == 0

    asyncio.run(run())


def test_cache_capacity_and_optimizer_are_enforced():
    async def run():
        service = make_service(0)
        service.max_bytes = 1
        with pytest.raises(RuntimeError, match="full"):
            service.save("model.0", [torch.ones(2, 3, 4), DUMMY], route(1)[0], peer(99))
        service.handler.module_backends["model.0"].optimizer = object()
        with pytest.raises(ValueError, match="frozen"):
            service.save("model.0", [torch.ones(2, 3, 4), DUMMY], route(1)[0], peer(99))

    asyncio.run(run())


@pytest.mark.parametrize("version", [None, 1])
def test_old_protocol_fails_without_client_relay(version):
    response = runtime_pb2.ExpertResponse(metadata=MSGPackSerializer.dumps({BACKWARD_CHAIN_VERSION: version}))
    with pytest.raises(RuntimeError, match="acknowledge"):
        wire._check_chain_response(response)


def test_cache_ownership_and_missing_cache():
    async def run():
        service = make_service(0)
        stages = route(1)
        service.save("model.0", [torch.ones(2, 3, 4), DUMMY], stages[0], peer(99))
        meta = {
            BACKWARD_CHAIN: stages,
            BACKWARD_CHAIN_VERSION: BACKWARD_PROTOCOL_VERSION,
            "client_peer": str(peer(99)),
            "position": 0,
        }
        req = runtime_pb2.ExpertRequest(uid="model.0", metadata=MSGPackSerializer.dumps(meta))
        with pytest.raises(PermissionError):
            service._lookup(req, SimpleNamespace(remote_id=peer(98)))
        service.discard(stages[0][BACKWARD_CACHE_ID])
        with pytest.raises(RuntimeError, match="missing or expired"):
            service._lookup(req, SimpleNamespace(remote_id=peer(99)))

    asyncio.run(run())


def test_failed_chain_raises_and_releases_caches_without_relay(monkeypatch):
    async def run():
        records = list(reversed(route(2)))
        services = {r["peer_id"]: make_service(i) for i, r in enumerate(records)}
        client_peer = peer(99)
        calls = []
        for record in records:
            services[record["peer_id"]].save(record["uid"], [torch.ones(2, 3, 4), DUMMY], record, client_peer)

        async def compute(inputs, grad, prompts, **kwargs):
            raise RuntimeError("worker compute failed")

        def stub(p2p, target, *, namespace):
            service = services[str(target)]
            ctx = SimpleNamespace(remote_id=p2p.peer_id)

            async def backward(request, **kwargs):
                calls.append(str(target))
                return await service.rpc_backward(request, ctx)

            async def collect(requests, **kwargs):
                return service.rpc_collect_stream(requests, ctx)

            async def release(request, **kwargs):
                return await service.rpc_release(request, ctx)

            return SimpleNamespace(rpc_backward=backward, rpc_collect_stream=collect, rpc_release=release)

        monkeypatch.setattr(worker, "run_rpc_backward", compute)
        monkeypatch.setattr(worker.BackwardService, "get_stub", staticmethod(stub))
        monkeypatch.setattr(wire, "MAX_UNARY_PAYLOAD_SIZE", 10**9)
        manager = SimpleNamespace(
            config=ClientConfig(request_timeout=2, connect_timeout=2),
            state=SimpleNamespace(p2p=SimpleNamespace(peer_id=client_peer)),
            rpc_info={"forward_schema": ([SimpleNamespace(compression=0)], {})},
            on_request_success=lambda p: None,
        )
        spans = [SimpleNamespace(peer_id=peer(1)), SimpleNamespace(peer_id=peer(2))]
        with pytest.raises(RuntimeError, match="worker compute failed"):
            await sequential_backward([torch.ones(2, 3, 4)], records, DUMMY, spans, manager)
        assert calls == [str(peer(2))]
        assert all(not s.entries and s.nbytes == 0 for s in services.values())

    asyncio.run(run())
