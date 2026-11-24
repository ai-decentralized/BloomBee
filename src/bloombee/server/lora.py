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
from hivemind.utils.streaming import split_for_streaming

from bloombee.utils import lora_codec
from bloombee.utils.hivemind_compat import MSGPackSerializer, P2PContext, ServicerBase, get_logger
from bloombee.utils.lora import LORA_NAMESPACE, LORA_PROTOCOL_VERSION, LoRAConfig, lora_context, projection_shapes
from bloombee.utils.lossless_transport import serialize_torch_tensor

logger = get_logger(__name__)

_WIRE_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
OPS = ("forward", "backward", "stateless", "release", "collect")


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
            raise LoRACacheMiss("budget")  # client retries statelessly; never evict live entries
        if keep:
            leaf_x = hidden.detach().requires_grad_(True)
            leaf_p = tuple(p.detach().requires_grad_(True) for p in layer_params[j])
            before = _allocated(device)
            with torch.enable_grad(), lora_context(config, leaf_p, training=meta["training"], seed=seed):
                y = layer(leaf_x)
            graph_bytes = max(0, _allocated(device) - before)
            _GRAPH_BYTES_PER_TOKEN[id(backend)] = graph_bytes / max(tokens, 1)
            entry.layers.append(_Layer(leaf_x, leaf_p, seed, y))
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


def run_lora_task(backend, first, meta, *tensors):
    """Executed by the Runtime. `first` is x (forward/stateless), dY (backward) or a dummy."""
    _prune()
    return _execute([_BACKENDS[name] for name in meta["uid"].split(" ")], first, meta, tensors)


def run_lora_block(backend, inputs, grad_outputs, metadata, *parameters):
    """Stateless single-block forward (empty grad_outputs) or forward+backward; returns (y,) or (dX, *dA/dB)."""
    meta = dict(
        uid="local", owner="local", config=metadata["config"], training=metadata["training"],
        seed=metadata["seed"], adapter_version=None, cache_id=None,
        op="stateless" if grad_outputs.numel() else "forward",
    )  # fmt: skip
    rest = (grad_outputs, *parameters) if grad_outputs.numel() else parameters
    return _execute([backend], inputs, meta, rest)


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


class LoRAService(ServicerBase):
    def __init__(self, handler):
        self.handler = handler

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
                if op in ("release", "collect"):
                    if tensors:
                        raise ValueError("LoRA release/collect take no tensors")
                    first, rest = torch.empty(1, 1, 1), []
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
                outputs = await backends[0].lora_pool.submit_task(first, meta, *rest, priority=2.0)
            except LoRACacheMiss as miss:
                outputs, response_meta = (), {"version": version, "cache_miss": miss.what}
            except Exception as error:  # the exception class may not survive every pickling path
                if "LORA_CACHE_MISS:" not in str(error):
                    raise
                what = re.search(r"LORA_CACHE_MISS:(\w+)", str(error)).group(1)
                outputs, response_meta = (), {"version": version, "cache_miss": what}
            if op == "release" and not response_meta.get("cache_miss"):
                outputs = ()
            if outputs and codec != "none" and op in ("forward", "backward", "stateless"):
                outputs = (*lora_codec.encode(outputs[0], codec), *outputs[1:])
            packed = MSGPackSerializer.dumps(response_meta)
            if not outputs:
                yield runtime_pb2.ExpertResponse(metadata=packed)
            for tensor in outputs:
                wire = serialize_torch_tensor(tensor.detach().cpu().contiguous(), runtime_pb2.CompressionType.NONE)
                for part in split_for_streaming(wire, DEFAULT_MAX_MSG_SIZE):
                    yield runtime_pb2.ExpertResponse(tensors=[part], metadata=packed)
