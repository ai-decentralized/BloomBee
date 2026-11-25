"""LoRA training RPCs on the worker's serial Runtime.

Protocol v2 (v1 requests are still served, statelessly):
  forward   [x, *A/B?]   run the whole span in one Runtime task. If the client will
                        backpropagate, keep each layer's input on the GPU under a cache id,
                        and (fast path, within budget) the layer's autograd graph too.
  backward  [dY]         gradients from the cached span: no input/adapter re-upload and,
                        where the graph was kept, no recomputation.
  stateless [x, dY, *A/B?]  forward + backward in one task, for cache misses (expired entry,
                        restarted or re-routed worker). Same result as the cached path.
  release   []           drop a cache entry early.
  collect   []           with a step token: return the adapter gradients summed over that
                        step's microbatches. Backward/stateless requests carrying the token
                        add their dA/dB there and return only dX, so adapter gradients cross
                        the network once per optimizer step instead of once per microbatch.
  upload    [*A/B]       store adapters without computing (the client's "seed" replica).
  probe     []           1 if this (client, span, version) adapter is cached (no exception on a miss).
  export    []           return the cached adapters of a (client, span, version) to a peer
                        worker; collect with `client` set does the same for gradient sums.
Data parallelism over replicas of a span: the client uploads each adapter version to one seed
replica; the others pull it from the seed (worker-to-worker, authorised by knowing the
unguessable version), and one leader replica sums the replicas' gradients before the client
downloads them, so client traffic does not grow with the number of replicas. Any pull failure
falls back to a direct client upload.
Adapters are cached per (client, span, adapter version): the client only uploads A/B when
they changed since that worker last saw them (normally once per optimizer step). Workers
never keep optimizer state or mutate adapters, so retries cannot double-apply an update.

All caches live in the Runtime thread of the main server process (where the GPU tensors
are); the forked connection handlers only pass ids. Every cache is bounded by bytes and
TTL; anything missing makes the client fall back to the stateless request.
"""

import os
import re
import time
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from typing import AsyncIterator

import torch
from async_timeout import timeout
from hivemind.p2p.p2p_daemon_bindings.control import DEFAULT_MAX_MSG_SIZE
from hivemind.proto import runtime_pb2
from hivemind.utils.asyncio import iter_as_aiter
from hivemind.utils.streaming import split_for_streaming

from bloombee.utils import lora_codec
from bloombee.utils.hivemind_compat import MSGPackSerializer, P2PContext, PeerID, ServicerBase, get_logger
from bloombee.utils.lora import LORA_NAMESPACE, LORA_PROTOCOL_VERSION, LoRAConfig, lora_context, projection_shapes
from bloombee.utils.lossless_transport import deserialize_tensor_stream, serialize_torch_tensor

logger = get_logger(__name__)

_WIRE_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
OPS = ("forward", "backward", "stateless", "release", "collect", "upload", "export", "probe")


def lora_enabled():
    return os.getenv("BLOOMBEE_ENABLE_LORA", "0") == "1"


def _env_flag(name, default):
    return os.getenv(name, "1" if default else "0") == "1"


class LoRACacheMiss(RuntimeError):
    """The worker no longer holds the requested adapter or activations; resend statelessly."""

    def __init__(self, what):
        super().__init__(f"LORA_CACHE_MISS:{what}")
        self.what = what

    def __reduce__(self):  # survives the MPFuture pickling between Runtime and handler
        return LoRACacheMiss, (self.what,)


# ---------------------------------------------------------------------------
# Runtime-side state (main process, Runtime thread only: no locking needed).
# ---------------------------------------------------------------------------

_BACKENDS = {}
_ADAPTERS = OrderedDict()  # (owner, span uid, version) -> (tensors on device, nbytes)
_ACTIVATIONS = OrderedDict()  # (owner, cache id) -> _Entry
_GRADIENTS = OrderedDict()  # (owner, span uid, step token) -> (summed adapter grads, expires)
_GRAPH_BYTES_PER_TOKEN = {}  # id(backend) -> measured autograd bytes per token
_usage = {"adapter_bytes": 0, "activation_bytes": 0}


@dataclass
class _Layer:
    x: torch.Tensor  # layer input (a leaf that requires grad if y is kept)
    params: tuple  # the adapter leaves used for this layer
    seed: int
    y: torch.Tensor = None  # graph-carrying output, or None (recompute in backward)
    graph_bytes: int = 0  # memory held by y's autograd graph


@dataclass
class _Entry:
    uid: str
    config: LoRAConfig
    training: bool
    input_dtype: torch.dtype
    expires: float
    layers: list = field(default_factory=list)
    nbytes: int = 0


def register_backend(backend):
    _BACKENDS[backend.name] = backend


def _device_total_bytes(device):
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).total_memory
    return 0


def _activation_budget(device):
    configured = os.getenv("BLOOMBEE_LORA_CACHE_BYTES")
    if configured is not None:
        return int(configured)
    # Default: a quarter of the GPU. Inputs alone are tiny; the rest holds autograd graphs.
    total = _device_total_bytes(device)
    return total // 4 if total else 2 * 1024**3


def _adapter_budget():
    return int(os.getenv("BLOOMBEE_LORA_ADAPTER_CACHE_BYTES", str(1024**3)))


def _ttl():
    return float(os.getenv("BLOOMBEE_LORA_CACHE_TTL", "300"))


def _allocated(device):
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device)
    return None


def _drop_activation(key):
    entry = _ACTIVATIONS.pop(key, None)
    if entry is not None:
        _usage["activation_bytes"] -= entry.nbytes
        entry.layers.clear()


def _downgrade(needed, entry=None):
    """Free `needed` accounted bytes by turning kept autograd graphs into inputs-only layers
    (they are recomputed in backward). Newest entries go first: their backward runs last.
    Cheaper than a cache miss, which makes the client redo the span statelessly."""
    freed = 0
    victims = [e for e in reversed(_ACTIVATIONS.values())]
    if entry is not None:
        victims.insert(0, entry)
    for victim in victims:
        for layer in reversed(victim.layers):
            if freed >= needed:
                return freed
            if layer is None or layer.y is None:
                continue
            layer.y = None
            victim.nbytes -= layer.graph_bytes
            if victim is not entry:
                _usage["activation_bytes"] -= layer.graph_bytes
            freed += layer.graph_bytes
            layer.graph_bytes = 0
    return freed


def _prune():
    now = time.monotonic()
    for key in [k for k, e in _ACTIVATIONS.items() if e.expires <= now]:
        _drop_activation(key)
    for key in [k for k, (_, expires) in _GRADIENTS.items() if expires <= now]:
        del _GRADIENTS[key]


def _finish_gradients(meta, gradients):
    """Return (dX, *dA/dB), or with a step token add dA/dB to the step's sum and return (dX,)."""
    token = meta.get("accumulate")
    if token is None:
        return gradients
    key = (meta["owner"], meta["uid"], token)
    current = _GRADIENTS.get(key)
    if current is None:
        _GRADIENTS[key] = ([g.clone() for g in gradients[1:]], time.monotonic() + _ttl())
    else:
        for total, g in zip(current[0], gradients[1:]):
            total.add_(g)
    return gradients[:1]


def _store_adapter(key, tensors):
    nbytes = sum(t.numel() * t.element_size() for t in tensors)
    old = _ADAPTERS.pop(key, None)
    if old is not None:
        _usage["adapter_bytes"] -= old[1]
    # One version per client span is enough: drop older versions of the same span.
    for stale in [k for k in _ADAPTERS if k[:2] == key[:2]]:
        _usage["adapter_bytes"] -= _ADAPTERS.pop(stale)[1]
    while _ADAPTERS and _usage["adapter_bytes"] + nbytes > _adapter_budget():
        _usage["adapter_bytes"] -= _ADAPTERS.popitem(last=False)[1][1]
    _ADAPTERS[key] = (tensors, nbytes)
    _usage["adapter_bytes"] += nbytes


def _validate_parameters(backend, config, parameters):
    shapes = projection_shapes(backend.config)
    for j, target in enumerate(config.target_modules):
        out_features, in_features = shapes[target]
        a, b = parameters[2 * j : 2 * j + 2]
        if a.shape != (config.r, in_features) or b.shape != (out_features, config.r):
            raise ValueError(f"Invalid LoRA A/B shapes for {target}")
        if a.dtype != torch.float32 or b.dtype != torch.float32:
            raise ValueError("LoRA optimizer/master parameters must be float32")
    if not all(torch.isfinite(p).all() for p in parameters):
        raise ValueError("LoRA parameters must be finite")


def _resolve_adapters(meta, backends, config, uploaded, device):
    width = 2 * len(config.target_modules)
    key = (meta["owner"], meta["uid"], meta["adapter_version"])
    if uploaded:
        if len(uploaded) != width * len(backends):
            raise ValueError("LoRA parameter count mismatch")
        tensors = tuple(p.to(device, non_blocking=True) for p in uploaded)
        for j, backend in enumerate(backends):
            _validate_parameters(backend, config, tensors[j * width : (j + 1) * width])
        if meta["adapter_version"] is not None:
            _store_adapter(key, tensors)
    else:
        cached = _ADAPTERS.get(key)
        if cached is None:
            raise LoRACacheMiss("adapter")
        _ADAPTERS.move_to_end(key)
        tensors = cached[0]
    return [tensors[j * width : (j + 1) * width] for j in range(len(backends))]


def _compute_dtype(backend, weights):
    if weights is not None:
        return weights["q_proj"].dtype
    block = backend.module.module_shards[0]
    if backend.config.model_type == "llama":
        return block.weight_home[0].val[0].dtype
    return next(block.parameters()).dtype


def _module_forward(backend, x):
    output = backend.module(x)
    return output[0] if isinstance(output, tuple) else output


def _layer_fn(backend, device):
    from bloombee.server.lora_fastpath import block_forward, block_weights

    if len(backend.module.module_shards) != 1:
        raise ValueError("LoRA does not yet support tensor-parallel blocks")
    backend._ensure_model_on_device()
    weights = block_weights(backend, device) if _env_flag("BLOOMBEE_LORA_FASTPATH", True) else None
    if weights is not None:
        return partial(block_forward, w=weights), True, _compute_dtype(backend, weights)
    return partial(_module_forward, backend), False, _compute_dtype(backend, None)


def _saved_tensor_ctx(backend, grad):
    # HF offload restores Parameter.data to its CPU home before autograd runs. Retain
    # separate handles to the staged storage for this block's backward.
    if grad and getattr(backend.module, "_cpu_offload", False):
        return torch.autograd.graph.saved_tensors_hooks(lambda t: t.detach(), lambda t: t)
    return nullcontext()


def _run_forward(backends, x, meta, layer_params, *, cache):
    """Forward through the span; optionally build a cache entry. Returns (output, entry)."""
    config = LoRAConfig(**meta["config"])
    device = x.device
    entry = None
    if cache:
        entry = _Entry(meta["uid"], config, meta["training"], x.dtype, time.monotonic() + _ttl())
    budget = _activation_budget(device)
    keep_graph_allowed = cache and _env_flag("BLOOMBEE_LORA_KEEP_GRAPH", True) and _allocated(device) is not None
    hidden = x
    for j, backend in enumerate(backends):
        layer, fast, dtype = _layer_fn(backend, device)
        hidden = hidden.to(dtype)
        seed = meta["seed"] + j
        tokens = hidden.shape[0] * hidden.shape[1]
        if entry is None:
            with torch.no_grad(), lora_context(config, layer_params[j], training=meta["training"], seed=seed):
                hidden = layer(hidden)
            continue
        input_bytes = hidden.numel() * hidden.element_size()
        estimate = _GRAPH_BYTES_PER_TOKEN.get(id(backend), 0) * tokens
        used = _usage["activation_bytes"] + entry.nbytes
        keep = keep_graph_allowed and fast and used + input_bytes + estimate <= budget
        if used + input_bytes > budget:
            _downgrade(used + input_bytes - budget, entry)
            used = _usage["activation_bytes"] + entry.nbytes
            if used + input_bytes > budget:
                raise LoRACacheMiss("budget")  # even inputs alone do not fit: client goes stateless
        if keep:
            leaf_x = hidden.detach().requires_grad_(True)
            leaf_p = tuple(p.detach().requires_grad_(True) for p in layer_params[j])
            before = _allocated(device)
            with torch.enable_grad(), lora_context(config, leaf_p, training=meta["training"], seed=seed):
                y = layer(leaf_x)
            graph_bytes = max(0, _allocated(device) - before)
            _GRAPH_BYTES_PER_TOKEN[id(backend)] = graph_bytes / max(tokens, 1)
            entry.layers.append(_Layer(leaf_x, leaf_p, seed, y, graph_bytes))
            entry.nbytes += input_bytes + graph_bytes
            hidden = y.detach()
        else:
            entry.layers.append(_Layer(hidden.detach(), tuple(layer_params[j]), seed))
            entry.nbytes += input_bytes
            with torch.no_grad(), lora_context(config, layer_params[j], training=meta["training"], seed=seed):
                hidden = layer(hidden)
    return hidden, entry


def _run_backward(backends, entry, grad):
    device = grad.device
    grads_per_layer = [None] * len(entry.layers)
    for j in reversed(range(len(entry.layers))):
        state = entry.layers[j]
        backend = backends[j]
        if state.y is not None:
            y, x, params = state.y, state.x, state.params
        else:
            layer, _, dtype = _layer_fn(backend, device)
            x = state.x.to(dtype).detach().requires_grad_(True)
            params = tuple(p.detach().requires_grad_(True) for p in state.params)
            with torch.enable_grad(), _saved_tensor_ctx(backend, True), lora_context(
                entry.config, params, training=entry.training, seed=state.seed
            ):
                y = layer(x)
        result = torch.autograd.grad(y, (x, *params), grad.to(y.dtype), allow_unused=False)
        grad = result[0]
        grads_per_layer[j] = tuple(g.detach() for g in result[1:])
        entry.layers[j] = None  # free this layer's graph and input as soon as possible
    return (grad.to(entry.input_dtype), *(g for layer in grads_per_layer for g in layer))


def _parse_sim_compute(value):
    """BLOOMBEE_SIM_COMPUTE="<fwd_s>,<bwd_s>[,<fixed_s>]": seconds per (layer x token) of forward and
    backward (backward includes the recompute of kept-input layers), plus a fixed cost per task."""
    if not value:
        return None
    parts = [float(x) for x in value.split(",")]
    if len(parts) not in (2, 3) or min(parts) < 0:
        raise ValueError("BLOOMBEE_SIM_COMPUTE must be '<fwd_s>,<bwd_s>[,<fixed_s>]' per layer-token")
    return parts[0], parts[1], parts[2] if len(parts) == 3 else 0.0


_SIM_COMPUTE = _parse_sim_compute(os.getenv("BLOOMBEE_SIM_COMPUTE", ""))
_TIMING_LOG = os.getenv("BLOOMBEE_LORA_TIMING_LOG")


def _simulate(backends, first, meta, tensors):
    """Benchmark-only worker: no model math, just the calibrated time the task would take on a
    dedicated GPU, with the same protocol state (adapters, cache entries, gradient sums) so that a
    swarm of many workers can be emulated on few GPUs. Outputs are zeros of the right shapes."""
    op, forward_s, backward_s, fixed_s = meta["op"], *_SIM_COMPUTE
    config = LoRAConfig(**meta["config"])
    key = (meta["owner"], meta.get("cache_id"))
    work = len(backends) * first.shape[0] * first.shape[1]
    if op == "release":
        _drop_activation(key)
        return (torch.empty(0),)
    if op == "collect":
        found = _GRADIENTS.pop((meta["owner"], meta["uid"], meta.get("accumulate")), None)
        if found is None:
            raise LoRACacheMiss("gradients")
        return tuple(found[0])
    if op in ("upload", "export", "probe"):
        return _adapter_op(op, meta, backends, first.device, tensors)
    if op == "backward":
        entry = _ACTIVATIONS.pop(key, None)
        if entry is None or entry.uid != meta["uid"]:
            raise LoRACacheMiss("activations")
        shapes, seconds = entry.shapes, fixed_s + work * backward_s
    else:
        layer_params = _resolve_adapters(meta, backends, config, tensors[1:] if op == "stateless" else tensors, first.device)
        shapes = [tuple(p.shape) for layer in layer_params for p in layer]
        seconds = fixed_s + work * (forward_s if op == "forward" else forward_s + backward_s)
        if op == "forward" and meta.get("cache_id") is not None:
            entry = _Entry(meta["uid"], config, meta["training"], first.dtype, time.monotonic() + _ttl())
            entry.shapes = shapes
            _ACTIVATIONS[key] = entry
    time.sleep(seconds)
    if op == "forward":
        return (torch.zeros_like(first),)
    grads = [torch.zeros(shape, device=first.device) for shape in shapes]
    return _finish_gradients(meta, (torch.zeros_like(first), *grads))


def run_lora_task(backend, first, meta, *tensors):
    """Executed by the Runtime. `first` is x (forward/stateless), dY (backward) or a dummy."""
    _prune()
    backends = [_BACKENDS[name] for name in meta["uid"].split(" ")]
    if _SIM_COMPUTE is not None:
        return _simulate(backends, first, meta, tensors)
    if _TIMING_LOG is None:
        return _execute(backends, first, meta, tensors)
    if first.is_cuda:
        torch.cuda.synchronize(first.device)
    start = time.perf_counter()
    result = _execute(backends, first, meta, tensors)
    if first.is_cuda:
        torch.cuda.synchronize(first.device)
    with open(_TIMING_LOG, "a") as log:  # op, layers, tokens, seconds
        log.write(f"{meta['op']} {len(backends)} {first.shape[0] * first.shape[1]} {time.perf_counter() - start:.6f}\n")
    return result


def run_lora_block(backend, inputs, grad_outputs, metadata, *parameters):
    """Stateless single-block forward (empty grad_outputs) or forward+backward; returns (y,) or (dX, *dA/dB)."""
    meta = dict(
        uid="local", owner="local", config=metadata["config"], training=metadata["training"],
        seed=metadata["seed"], adapter_version=None, cache_id=None,
        op="stateless" if grad_outputs.numel() else "forward",
    )  # fmt: skip
    rest = (grad_outputs, *parameters) if grad_outputs.numel() else parameters
    return _execute([backend], inputs, meta, rest)


def _adapter_op(op, meta, backends, device, tensors):
    if op == "upload":
        _resolve_adapters(meta, backends, LoRAConfig(**meta["config"]), tensors, device)
        return (torch.empty(0),)
    cached = _ADAPTERS.get((meta["owner"], meta["uid"], meta["adapter_version"]))
    if op == "probe":
        return (torch.tensor([cached is not None]),)
    if cached is None:
        raise LoRACacheMiss("adapter")
    return tuple(t.detach().cpu() for t in cached[0])


def _execute(backends, first, meta, tensors):
    op = meta["op"]
    key = (meta["owner"], meta.get("cache_id"))
    if op == "release":
        _drop_activation(key)
        return (torch.empty(0),)
    if op == "collect":
        found = _GRADIENTS.pop((meta["owner"], meta["uid"], meta.get("accumulate")), None)
        if found is None:
            raise LoRACacheMiss("gradients")
        return tuple(found[0])
    if op in ("upload", "export", "probe"):
        return _adapter_op(op, meta, backends, first.device, tensors)
    if op == "backward":
        entry = _ACTIVATIONS.pop(key, None)
        if entry is None or entry.uid != meta["uid"]:
            raise LoRACacheMiss("activations")
        _usage["activation_bytes"] -= entry.nbytes
        try:
            return _finish_gradients(meta, _run_backward(backends, entry, first))
        finally:
            entry.layers.clear()
    config = LoRAConfig(**meta["config"])
    if first.ndim != 3 or first.shape[-1] != backends[0].config.hidden_size or not first.is_floating_point():
        raise ValueError("LoRA inputs must be [batch, sequence, hidden_size]")
    if op == "forward":
        layer_params = _resolve_adapters(meta, backends, config, tensors, first.device)
        cache = meta.get("cache_id") is not None
        if cache and key in _ACTIVATIONS:
            raise ValueError("Duplicate LoRA cache id")
        output, entry = _run_forward(backends, first, meta, layer_params, cache=cache)
        if entry is not None:
            _ACTIVATIONS[key] = entry
            _usage["activation_bytes"] += entry.nbytes
        return (output.detach().to(first.dtype),)
    if op == "stateless":
        grad, uploaded = tensors[0], tensors[1:]
        if grad.shape != first.shape:
            raise ValueError("LoRA output gradient shape mismatch")
        layer_params = _resolve_adapters(meta, backends, config, uploaded, first.device)
        # Inputs only: compute is forward + per-layer recompute, without touching the cache budget.
        entry = _Entry(meta["uid"], config, meta["training"], first.dtype, 0.0)
        hidden = first
        for j, b in enumerate(backends):
            layer, _, dtype = _layer_fn(b, first.device)
            hidden = hidden.to(dtype)
            entry.layers.append(_Layer(hidden.detach(), tuple(layer_params[j]), meta["seed"] + j))
            with torch.no_grad(), lora_context(config, layer_params[j], training=meta["training"], seed=meta["seed"] + j):
                hidden = layer(hidden)
        return _finish_gradients(meta, _run_backward(backends, entry, grad))
    raise ValueError(f"Unknown LoRA op {op!r}")


def make_lora_pool(backend, pool_class, max_batch_size, device):
    register_backend(backend)
    return pool_class(
        partial(run_lora_task, backend), max_batch_size=max_batch_size, device=device, name=f"{backend.name}_lora"
    )


# ---------------------------------------------------------------------------
# Connection-handler side (forked processes): validation and dispatch only.
# ---------------------------------------------------------------------------


def _parse_v1(metadata, tensors, width, n_layers):
    """Translate a protocol v1 request (always stateless) into a v2 operation."""
    backward = metadata.get("backward")
    if not isinstance(backward, bool):
        raise ValueError("LoRA phase must be a boolean")
    if len(tensors) != 1 + width * n_layers + int(backward):
        raise ValueError("LoRA request tensor count mismatch")
    x, params = tensors[0], tensors[1 : 1 + width * n_layers]
    if backward:
        return "stateless", x, [tensors[-1], *params], None
    return "forward", x, list(params), None


def _miss_kind(error):
    if isinstance(error, LoRACacheMiss):
        return error.what
    found = re.search(r"LORA_CACHE_MISS:(\w+)", str(error))
    return found.group(1) if found else None


async def _peer_request(p2p, peer, uid, metadata, tensors, request_timeout):
    """One worker-to-worker rpc_lora_stream call; returns (tensors, response metadata)."""
    stub = LoRAService.get_stub(p2p, PeerID.from_base58(peer), namespace=LORA_NAMESPACE)
    packed = MSGPackSerializer.dumps(metadata)
    parts = []
    for tensor in tensors:
        wire = serialize_torch_tensor(tensor.detach().cpu().contiguous(), runtime_pb2.CompressionType.NONE)
        parts.extend(
            runtime_pb2.ExpertRequest(uid=uid, tensors=[part], metadata=packed)
            for part in split_for_streaming(wire, DEFAULT_MAX_MSG_SIZE)
        )
    parts = parts or [runtime_pb2.ExpertRequest(uid=uid, metadata=packed)]
    response_meta = {}

    async def stream():
        nonlocal response_meta
        async for response in await stub.rpc_lora_stream(iter_as_aiter(parts)):
            response_meta = MSGPackSerializer.loads(response.metadata)
            if response.tensors:
                yield response.tensors

    async with timeout(request_timeout):
        result = await deserialize_tensor_stream(stream())
    if response_meta.get("cache_miss"):
        raise LoRACacheMiss(response_meta["cache_miss"])
    return result, response_meta


class LoRAService(ServicerBase):
    def __init__(self, handler):
        self.handler = handler

    def _peer_metadata(self, metadata, op, owner, **extra):
        keep = ("version", "config", "training", "seed", "adapter_version", "accumulate")
        return {**{k: metadata[k] for k in keep if k in metadata}, "op": op, "client": owner, **extra}

    async def _run(self, backends, uid, first, meta, rest, metadata, context):
        """Submit to the Runtime; for data parallelism also pull adapters from a seed peer
        and reduce replica gradient sums (both optional, set by the client)."""
        pool, op = backends[0].lora_pool, meta["op"]
        p2p, request_timeout = self.handler._p2p, self.handler.request_timeout
        owner = meta["owner"]
        if op == "forward" and not rest and metadata.get("pull_from"):
            (have,) = await pool.submit_task(first, {**meta, "op": "probe"}, priority=2.0)
            if not bool(have.item()):
                began = time.perf_counter()
                try:
                    adapters, _ = await _peer_request(
                        p2p, metadata["pull_from"], uid,
                        self._peer_metadata(metadata, "export", owner), (), request_timeout,
                    )  # fmt: skip
                except Exception as error:
                    logger.warning(f"LoRA adapter pull from {metadata['pull_from']} failed: {error!r}")
                    raise LoRACacheMiss("pull") from None
                logger.info(f"[LORA_PEER] pulled adapters in {time.perf_counter() - began:.3f}s")
                return await pool.submit_task(first, meta, *adapters, priority=2.0)
        if op == "collect" and metadata.get("reduce_from"):
            totals = None
            try:
                totals = [t.float() for t in await pool.submit_task(first, meta, priority=2.0)]
            except Exception as error:
                if _miss_kind(error) != "gradients":
                    raise
            # One peer at a time: fetching all concurrently made the leader slower (its handler
            # process, not the link, is the bottleneck), measured with 7 peers.
            began = time.perf_counter()
            for peer in metadata["reduce_from"]:
                try:
                    other, _ = await _peer_request(
                        p2p, peer, uid, self._peer_metadata(metadata, "collect", owner), (), request_timeout
                    )
                except LoRACacheMiss:
                    continue  # that replica served no microbatch this step
                if totals is None:
                    totals = [t.float() for t in other]
                else:
                    for total, g in zip(totals, other):
                        total.add_(g.float())
            logger.info(
                f"[LORA_PEER] reduced gradients from {len(metadata['reduce_from'])} peers in "
                f"{time.perf_counter() - began:.3f}s"
            )
            if totals is None:
                raise LoRACacheMiss("gradients")
            return tuple(totals)
        return await pool.submit_task(first, meta, *rest, priority=2.0)

    async def rpc_lora_stream(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> AsyncIterator[runtime_pb2.ExpertResponse]:
        async with timeout(self.handler.request_timeout):
            uid, tensors, metadata = await self.handler._gather_inputs(requests, context)
            version = metadata.get("version")
            if version not in (1, LORA_PROTOCOL_VERSION):
                raise ValueError("Unsupported LoRA protocol")
            config = LoRAConfig(**metadata["config"])
            names = uid.split(" ")
            if not names or len(set(names)) != len(names):
                raise ValueError("Invalid LoRA layer chain")
            indices = [int(name.rsplit(".", 1)[1]) for name in names]
            if indices != list(range(indices[0], indices[0] + len(indices))):
                raise ValueError("LoRA layers must be a contiguous ordered span")
            backends = [self.handler.module_backends[name] for name in names]
            if any(getattr(b, "lora_pool", None) is None for b in backends):
                raise RuntimeError("Enable BLOOMBEE_ENABLE_LORA=1 on every worker")
            if not isinstance(metadata.get("training"), bool):
                raise ValueError("LoRA training flag must be a boolean")
            seed = metadata.get("seed")
            if not isinstance(seed, int) or not 0 <= seed < 2**62:
                raise ValueError("Invalid LoRA recompute seed")
            width = 2 * len(config.target_modules)
            meta = dict(
                uid=uid,
                owner=str(context.remote_id),
                config=config.to_dict(),
                training=metadata["training"],
                seed=seed,
                adapter_version=None,
                cache_id=None,
            )
            codec = "none"
            if version == 1:
                op, first, rest, _ = _parse_v1(metadata, tensors, width, len(names))
            else:
                op = metadata.get("op")
                if op not in OPS:
                    raise ValueError("Invalid LoRA op")
                cache_id = metadata.get("cache_id")
                if cache_id is not None and (not isinstance(cache_id, str) or len(cache_id) != 32):
                    raise ValueError("Invalid LoRA cache id")
                adapter_version = metadata.get("adapter_version")
                if adapter_version is not None and not isinstance(adapter_version, str):
                    raise ValueError("Invalid LoRA adapter version")
                accumulate = metadata.get("accumulate")
                if accumulate is not None and (not isinstance(accumulate, str) or len(accumulate) != 32):
                    raise ValueError("Invalid LoRA gradient accumulation token")
                meta.update(cache_id=cache_id, adapter_version=adapter_version, accumulate=accumulate)
                if op in ("backward", "release") and cache_id is None:
                    raise ValueError("LoRA backward/release need a cache id")
                if op == "collect" and accumulate is None:
                    raise ValueError("LoRA collect needs an accumulation token")
                codec = metadata.get("codec", "none")
                if codec not in lora_codec.CODECS:
                    raise ValueError("Unknown LoRA boundary codec")
                wire_dtype = _WIRE_DTYPES.get(metadata.get("wire_dtype", "float16"))
                if wire_dtype is None:
                    raise ValueError("Invalid LoRA wire dtype")
                if codec != "none" and op in ("forward", "backward", "stateless"):
                    k = lora_codec.wire_count(codec)
                    n_act = 2 if op == "stateless" else 1
                    if len(tensors) < k * n_act:
                        raise ValueError("LoRA request tensor count mismatch")
                    decoded = [lora_codec.decode(tensors[i * k : (i + 1) * k], codec, wire_dtype) for i in range(n_act)]
                    tensors = [*decoded, *tensors[k * n_act :]]
                adapter_wire = metadata.get("adapter_wire", "float32")
                if adapter_wire not in ("float32", "bfloat16"):
                    raise ValueError("Invalid LoRA adapter wire dtype")
                if adapter_wire != "float32" and op in ("forward", "stateless", "upload"):
                    n_act = {"forward": 1, "stateless": 2, "upload": 0}[op]
                    if any(t.dtype != torch.bfloat16 for t in tensors[n_act:]):
                        raise ValueError("Adapters must arrive in the announced wire dtype")
                    tensors = [*tensors[:n_act], *(t.float() for t in tensors[n_act:])]
                client = metadata.get("client")
                if client is not None:
                    # A peer worker acting for the client (export / reduce): the adapter version or
                    # gradient token, unguessable and known only to the client's workers, authorises it.
                    if op not in ("export", "collect", "probe") or not isinstance(client, str):
                        raise ValueError("Invalid LoRA peer request")
                    meta["owner"] = client
                if op in ("release", "collect", "export", "probe"):
                    if tensors:
                        raise ValueError("LoRA release/collect/export take no tensors")
                    first, rest = torch.empty(1, 1, 1), []
                elif op == "upload":
                    if len(tensors) != width * len(names):
                        raise ValueError("LoRA parameter count mismatch")
                    first, rest = torch.empty(1, 1, 1), tensors
                elif op == "backward":
                    if len(tensors) != 1:
                        raise ValueError("Cached LoRA backward accepts only the output gradient")
                    first, rest = tensors[0], []
                else:
                    first, rest = tensors[0], tensors[1:]
                    n_grad = int(op == "stateless")
                    if len(rest) not in (n_grad, n_grad + width * len(names)):
                        raise ValueError("LoRA request tensor count mismatch")
                    if n_grad and rest[0].shape != first.shape:
                        raise ValueError("LoRA output gradient shape mismatch")
            if first.ndim != 3 or min(first.shape) < 1:
                raise ValueError("LoRA requires nonempty [batch, sequence, hidden] tensors")
            meta["op"] = op
            response_meta = {"version": version}
            try:
                outputs = await self._run(backends, uid, first, meta, rest, metadata, context)
            except Exception as error:  # the exception class may not survive every pickling path
                what = _miss_kind(error)
                if what is None:
                    raise
                outputs, response_meta = (), {"version": version, "cache_miss": what}
            if op in ("release", "upload") and not response_meta.get("cache_miss"):
                outputs = ()
            if outputs and op in ("backward", "stateless", "collect") and metadata.get("adapter_wire") == "bfloat16":
                lead = 0 if op == "collect" else 1  # dX is not an adapter gradient
                outputs = (*outputs[:lead], *(g.to(torch.bfloat16) for g in outputs[lead:]))
            if outputs and codec != "none" and op in ("forward", "backward", "stateless"):
                outputs = (*lora_codec.encode(outputs[0], codec), *outputs[1:])
            packed = MSGPackSerializer.dumps(response_meta)
            if not outputs:
                yield runtime_pb2.ExpertResponse(metadata=packed)
            for tensor in outputs:
                wire = serialize_torch_tensor(tensor.detach().cpu().contiguous(), runtime_pb2.CompressionType.NONE)
                for part in split_for_streaming(wire, DEFAULT_MAX_MSG_SIZE):
                    yield runtime_pb2.ExpertResponse(tensors=[part], metadata=packed)
