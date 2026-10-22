"""Handler-affine forward caches and direct worker-to-worker backward RPCs."""

import asyncio
import os
import time
from dataclasses import dataclass
from typing import AsyncIterator

import torch
from async_timeout import timeout
from hivemind.p2p.p2p_daemon_bindings.control import DEFAULT_MAX_MSG_SIZE
from hivemind.proto import runtime_pb2
from hivemind.utils.streaming import split_for_streaming

from bloombee.client.config import ClientConfig
from bloombee.client.remote_forward_backward import run_remote_backward_chain
from bloombee.server.block_functions import run_rpc_backward
from bloombee.utils.backward_protocol import (
    BACKWARD_CACHE_ID,
    BACKWARD_CHAIN,
    BACKWARD_CHAIN_VERSION,
    BACKWARD_NAMESPACE,
    BACKWARD_PROTOCOL_VERSION,
    chain_response_metadata,
    validate_backward_chain,
)
from bloombee.utils.hivemind_compat import MSGPackSerializer, P2PContext, PeerID, ServicerBase, get_logger
from bloombee.utils.lossless_transport import deserialize_torch_tensor, serialize_torch_tensor

logger = get_logger(__name__)


@dataclass
class ForwardCacheEntry:
    uid: str
    owner: str
    inputs: torch.Tensor
    prompts: torch.Tensor
    metadata: dict
    expires: float
    result: asyncio.Future
    nbytes: int
    started: bool = False


class BackwardService(ServicerBase):
    """One uniquely addressed service per handler; caches never depend on RPC balancing."""

    def __init__(self, handler):
        self.handler = handler
        self.namespace = f"backward-{handler._handler_index}"
        self.entries = {}
        self.nbytes = 0
        total_bytes = int(os.getenv("BLOOMBEE_BACKWARD_CACHE_BYTES", str(512 * 1024**2)))
        self.max_bytes = total_bytes // len(handler._handler_event_queues)
        self.ttl = float(os.getenv("BLOOMBEE_BACKWARD_CACHE_TTL", "600"))
        if self.max_bytes <= 0 or self.ttl <= 0:
            raise ValueError("Backward cache capacity and TTL must be positive")
        self.cleanup_task = None

    async def start(self, p2p):
        await self.add_p2p_handlers(p2p, namespace=self.namespace)
        self.cleanup_task = asyncio.create_task(self._cleanup())

    async def close(self, p2p):
        if self.cleanup_task is not None:
            self.cleanup_task.cancel()
            await asyncio.gather(self.cleanup_task, return_exceptions=True)
        for key in list(self.entries):
            self.discard(key)
        await self.remove_p2p_handlers(p2p, namespace=self.namespace)

    async def _cleanup(self):
        while True:
            await asyncio.sleep(min(30, self.ttl / 2))
            self.prune()

    def prune(self):
        for key, entry in list(self.entries.items()):
            if entry.expires <= time.monotonic() and (not entry.started or entry.result.done()):
                self.discard(key)

    def discard(self, key):
        entry = self.entries.pop(key, None)
        if entry is not None:
            self.nbytes -= entry.nbytes
            if not entry.result.done():
                entry.result.cancel()

    def save(self, uid, tensors, metadata, owner):
        self.prune()
        key = metadata[BACKWARD_CACHE_ID]
        if key in self.entries:
            raise ValueError("Duplicate forward cache ID")
        backends = [self.handler.module_backends[u] for u in uid.split(" ")]
        if metadata.get("active_adapter") or any(b.optimizer is not None for b in backends):
            raise ValueError("Cached backward requires frozen backbones without worker optimizers or adapters")
        nbytes = sum(t.numel() * t.element_size() for t in tensors[:2])
        if self.nbytes + nbytes > self.max_bytes:
            raise RuntimeError(
                "Worker backward cache is full; reduce in-flight batches or increase BLOOMBEE_BACKWARD_CACHE_BYTES"
            )
        # run_rpc_forward adds deep prompts in place: preserve the original input.
        inputs, prompts = (t.detach().cpu().clone() for t in tensors[:2])
        self.entries[key] = ForwardCacheEntry(
            uid,
            str(owner),
            inputs,
            prompts,
            dict(metadata),
            time.monotonic() + self.ttl,
            asyncio.get_running_loop().create_future(),
            nbytes,
        )
        self.nbytes += nbytes
        logger.debug("[BACKWARD_CACHE_SAVE] uid=%s bytes=%d cached_bytes=%d", uid, nbytes, self.nbytes)
        return {**chain_response_metadata(), BACKWARD_CACHE_ID: key, BACKWARD_NAMESPACE: self.namespace}

    def _lookup(self, request, context, *, client_only=False):
        metadata = MSGPackSerializer.loads(request.metadata) if request.metadata else {}
        if metadata.get(BACKWARD_CHAIN_VERSION) != BACKWARD_PROTOCOL_VERSION:
            raise ValueError("Backward requires the cached worker protocol; upgrade all clients and workers")
        route = metadata[BACKWARD_CHAIN]
        pos = metadata.get("position", 0)
        validate_backward_chain(route, request.uid, pos)
        stage = route[pos]
        if stage["peer_id"] != str(self.handler._p2p.peer_id) or stage[BACKWARD_NAMESPACE] != self.namespace:
            raise ValueError("Backward reached the wrong forward cache owner")
        self.prune()
        entry = self.entries.get(stage[BACKWARD_CACHE_ID])
        if entry is None or entry.uid != request.uid:
            raise RuntimeError("Forward cache missing or expired; rerun forward before backward")
        client = metadata["client_peer"]
        expected = client if client_only or pos == 0 else route[pos - 1]["peer_id"]
        if entry.owner != client or str(context.remote_id) != expected:
            raise PermissionError("Backward request does not own this forward cache")
        entry.expires = time.monotonic() + self.ttl
        return metadata, stage[BACKWARD_CACHE_ID], entry

    async def _backward(self, request, context, tensors):
        metadata, key, entry = self._lookup(request, context)
        if entry.started:
            raise RuntimeError("Forward cache has already been consumed by backward")
        if len(tensors) not in (1, 2) or tensors[0].shape != entry.inputs.shape:
            raise ValueError("Backward accepts only the output gradient and optional downstream prompt gradients")
        entry.started = True
        route, pos = metadata[BACKWARD_CHAIN], metadata.get("position", 0)
        try:
            backends = tuple(self.handler.module_backends[u] for u in entry.uid.split(" "))
            grads = await run_rpc_backward(
                entry.inputs,
                tensors[0],
                entry.prompts,
                requested_backends=backends,
                prioritizer=self.handler._prioritizer,
                active_adapter="",
                points=entry.metadata.get("points", 0),
            )
            entry.inputs = entry.prompts = torch.empty(0)
            # Accumulate deep-prompt gradients in forward layer order.
            if len(tensors) == 2:
                if len(grads) != 2:
                    raise ValueError("Backward chain mixed prompt and non-prompt stages")
                grads[1] = torch.cat((grads[1], tensors[1]), dim=0)
            if pos + 1 == len(route):
                if not entry.result.done():
                    entry.result.set_result(grads)
                logger.info("[S2S_BACKWARD] result_ready uid=%s client=%s", entry.uid, entry.owner)
            else:
                next_stage = route[pos + 1]
                stub = self.get_stub(
                    self.handler._p2p,
                    PeerID.from_base58(next_stage["peer_id"]),
                    namespace=next_stage[BACKWARD_NAMESPACE],
                )
                logger.info(
                    "[S2S_BACKWARD] uid=%s -> peer=%s uid=%s gradient_bytes=%d activation_bytes=0",
                    entry.uid,
                    next_stage["peer_id"],
                    next_stage["uid"],
                    grads[0].numel() * grads[0].element_size(),
                )
                ack = await run_remote_backward_chain(
                    next_stage["uid"],
                    stub,
                    grads,
                    config=ClientConfig(
                        request_timeout=self.handler.request_timeout, connect_timeout=self.handler.request_timeout
                    ),
                    metadata={**metadata, "position": pos + 1},
                    compression=backends[0].args_schema[0].compression,
                )
                if ack:
                    raise RuntimeError("Worker returned gradients through the relay instead of acknowledging")
        finally:
            # The first stage retains its final result only until Client collects it.
            if pos + 1 < len(route) or not entry.result.done() or entry.result.cancelled():
                self.discard(key)

    async def rpc_backward(self, request: runtime_pb2.ExpertRequest, context: P2PContext) -> runtime_pb2.ExpertResponse:
        async with timeout(self.handler.request_timeout):
            await self._backward(request, context, [deserialize_torch_tensor(t) for t in request.tensors])
            return runtime_pb2.ExpertResponse(metadata=MSGPackSerializer.dumps(chain_response_metadata()))

    async def rpc_backward_stream(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> AsyncIterator[runtime_pb2.ExpertResponse]:
        async with timeout(self.handler.request_timeout):
            uid, tensors, metadata = await self.handler._gather_inputs(requests, context)
            await self._backward(
                runtime_pb2.ExpertRequest(uid=uid, metadata=MSGPackSerializer.dumps(metadata)), context, tensors
            )
            yield runtime_pb2.ExpertResponse(metadata=MSGPackSerializer.dumps(chain_response_metadata()))

    async def rpc_collect_stream(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> AsyncIterator[runtime_pb2.ExpertResponse]:
        async with timeout(self.handler.request_timeout):
            uid, tensors, metadata = await self.handler._gather_inputs(requests, context)
            if tensors:
                raise ValueError("Collect requests must not upload tensors")
            request = runtime_pb2.ExpertRequest(uid=uid, metadata=MSGPackSerializer.dumps(metadata))
            _, key, entry = self._lookup(request, context, client_only=True)
            if metadata["position"] != len(metadata[BACKWARD_CHAIN]) - 1:
                raise ValueError("Client must collect results directly from the first model stage")
            try:
                grads = await asyncio.shield(entry.result)
                backends = tuple(self.handler.module_backends[u] for u in entry.uid.split(" "))
                for grad in grads:
                    tensor = serialize_torch_tensor(grad, backends[0].args_schema[0].compression)
                    for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE):
                        yield runtime_pb2.ExpertResponse(
                            tensors=[part], metadata=MSGPackSerializer.dumps(chain_response_metadata())
                        )
                logger.info("[BACKWARD_RESULT_TO_CLIENT] uid=%s client=%s", entry.uid, entry.owner)
            finally:
                self.discard(key)

    async def rpc_release(self, request: runtime_pb2.ExpertRequest, context: P2PContext) -> runtime_pb2.ExpertResponse:
        metadata = MSGPackSerializer.loads(request.metadata)
        key = metadata[BACKWARD_CACHE_ID]
        entry = self.entries.get(key)
        if entry is not None:
            if entry.owner != str(context.remote_id):
                raise PermissionError("Only the forward client may release its cache")
            self.discard(key)
        return runtime_pb2.ExpertResponse(metadata=MSGPackSerializer.dumps(chain_response_metadata()))
