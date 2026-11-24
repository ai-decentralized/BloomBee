"""Client-owned standard LoRA parameters, ordinary torch optimizers, remote gradients."""

import asyncio
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import torch
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.p2p.p2p_daemon_bindings.control import DEFAULT_MAX_MSG_SIZE
from hivemind.proto import runtime_pb2
from hivemind.utils.asyncio import aiter_with_timeout, iter_as_aiter
from hivemind.utils.streaming import split_for_streaming
from safetensors.torch import load_file, save_file
from torch import nn
from torch.autograd.function import once_differentiable

from bloombee.client.remote_sequential import RemoteSequential
from bloombee.server.lora import LoRAService
from bloombee.utils import lora_codec, sim_link
from bloombee.utils.hivemind_compat import MSGPackSerializer
from bloombee.utils.lora import LORA_NAMESPACE, LORA_PROTOCOL_VERSION, LoRAConfig, LoRAProjection, projection_shapes
from bloombee.utils.lossless_transport import deserialize_tensor_stream, serialize_torch_tensor


async def _call_lora(manager, span, inputs, metadata):
    """One LoRA RPC to a span's worker. Returns (tensors, response metadata)."""
    stub = LoRAService.get_stub(manager.state.p2p, span.peer_id, namespace=LORA_NAMESPACE)
    uid = " ".join(manager.block_uids[span.start : span.end])
    packed = MSGPackSerializer.dumps(metadata)
    parts = []
    for tensor in inputs:
        wire = serialize_torch_tensor(tensor.detach().cpu().contiguous(), runtime_pb2.CompressionType.NONE)
        parts.extend(
            runtime_pb2.ExpertRequest(uid=uid, tensors=[part], metadata=packed)
            for part in split_for_streaming(wire, DEFAULT_MAX_MSG_SIZE)
        )
    if not parts:
        parts.append(runtime_pb2.ExpertRequest(uid=uid, metadata=packed))
    await sim_link.account_send(sum(part.ByteSize() for part in parts))
    responses = await asyncio.wait_for(
        stub.rpc_lora_stream(iter_as_aiter(parts)), timeout=manager.config.connect_timeout
    )
    received = 0
    response_meta = {}

    async def checked():
        nonlocal received, response_meta
        async for response in aiter_with_timeout(responses, manager.config.request_timeout):
            response_meta = MSGPackSerializer.loads(response.metadata)
            if response_meta.get("version") != metadata["version"]:
                raise RuntimeError(f"Worker did not acknowledge LoRA protocol v{metadata['version']}")
            received += response.ByteSize()
            if response.tensors:
                yield response.tensors

    result = await deserialize_tensor_stream(checked())
    sim_link.account_receive(received)
    delay = sim_link.link_delay(received, "down")
    if delay:
        await asyncio.sleep(delay)
    return result, response_meta


class _SpanCall:
    """Everything one span needs for its forward and (cached or stateless) backward."""

    def __init__(self, sequence, span, metadata, parameters, needs_grad):
        self.manager = sequence.sequence_manager
        self.span = span
        self.metadata = metadata
        self.needs_grad = needs_grad
        self.adapters = sequence._adapter_acks
        self.key = (str(span.peer_id), span.start, span.end)
        # Optimizer steps and copy_() bump tensor version counters: a new adapter version
        # is uploaded exactly when some A/B changed since this worker last received it.
        self.version = f"{sequence._session}:{hash(tuple((id(p), p._version) for p in parameters)) & (2**63 - 1):x}"
        self.cache_id = uuid.uuid4().hex if needs_grad else None
        self.lock = sequence._upload_locks.setdefault(self.key, asyncio.Lock())
        self.codec = sequence.boundary_codec
        self.adapter_wire = sequence.adapter_wire
        # Inside train_step: workers sum dA/dB over the step and the client collects them once.
        self.accumulate = getattr(sequence._local, "accumulate", None)

    def _meta(self, op, **extra):
        meta = {
            **self.metadata,
            "op": op,
            "adapter_version": self.version,
            "cache_id": self.cache_id,
            "accumulate": self.accumulate,
            **extra,
        }
        if self.codec != "none":
            meta.update(codec=self.codec, wire_dtype=self.wire_dtype)
        if self.adapter_wire != "float32":
            meta["adapter_wire"] = self.adapter_wire
        return meta

    def _wire_params(self, parameters):
        if self.adapter_wire == "float32":
            return tuple(parameters)
        return tuple(p.detach().to(getattr(torch, self.adapter_wire)) for p in parameters)

    def _encode(self, tensor):
        self.wire_dtype = str(tensor.dtype).replace("torch.", "")
        return lora_codec.encode(tensor, self.codec)

    def _decode(self, tensors, dtype):
        k = lora_codec.wire_count(self.codec)
        if len(tensors) < k:
            raise RuntimeError("Invalid remote LoRA response")
        return [lora_codec.decode(tensors[:k], self.codec, dtype), *tensors[k:]]

    async def _forward(self, inputs, parameters, send):
        for _ in range(3):
            tensors = (*self._encode(inputs), *self._wire_params(parameters)) if send else tuple(self._encode(inputs))
            result, response = await _call_lora(self.manager, self.span, tensors, self._meta("forward"))
            miss = response.get("cache_miss")
            if miss is None:
                break
            if miss == "adapter" and not send:
                send = True
            elif miss == "budget" and self.cache_id is not None:
                self.cache_id = None  # worker has no room: backward will be stateless
            else:
                raise RuntimeError(f"Unexpected LoRA cache miss: {miss}")
        else:
            raise RuntimeError("LoRA forward kept missing worker caches")
        self.adapters[self.key] = self.version
        result = self._decode(result, inputs.dtype)
        if len(result) != 1 or result[0].shape != inputs.shape:
            raise RuntimeError("Invalid remote LoRA forward response")
        return result[0]

    async def forward(self, inputs, parameters):
        if self.adapters.get(self.key) == self.version:
            return await self._forward(inputs, parameters, send=False)
        async with self.lock:  # concurrent microbatches upload a changed adapter only once
            send = self.adapters.get(self.key) != self.version
            if send:
                return await self._forward(inputs, parameters, send=True)
        return await self._forward(inputs, parameters, send=False)

    async def backward(self, grad_outputs, inputs, parameters):
        gradients = None
        if self.cache_id is not None:
            gradients, response = await _call_lora(
                self.manager, self.span, tuple(self._encode(grad_outputs)), self._meta("backward")
            )
            if response.get("cache_miss"):
                gradients = None
        if gradients is None:  # expired / evicted / restarted worker: same result, recomputed
            tensors = (*self._encode(inputs), *self._encode(grad_outputs), *self._wire_params(parameters))
            gradients, _ = await _call_lora(self.manager, self.span, tensors, self._meta("stateless", cache_id=None))
            self.adapters[self.key] = self.version
        self.cache_id = None
        gradients = self._decode(gradients, inputs.dtype)
        expected = (inputs,) if self.accumulate else (inputs, *parameters)
        if len(gradients) != len(expected) or any(g.shape != p.shape for g, p in zip(gradients, expected)):
            raise RuntimeError("Invalid remote LoRA gradient response")
        if self.accumulate:
            return [gradients[0], *(None for _ in parameters)]
        return gradients


async def _collect_gradients(manager, span, metadata, parameters):
    gradients, response = await _call_lora(manager, span, (), {**metadata, "op": "collect"})
    if response.get("cache_miss") or len(gradients) != len(parameters):
        raise RuntimeError(f"Worker {span.peer_id} lost this step's adapter gradients; redo the step")
    if any(g.shape != p.shape for g, p in zip(gradients, parameters)):
        raise RuntimeError("Invalid collected LoRA gradients")
    return gradients


class _RemoteLoRAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, call, *parameters):
        ctx.call = call
        ctx.save_for_backward(inputs, *parameters)
        result = RemoteExpertWorker.run_coroutine(call.forward(inputs, parameters))
        return result.to(inputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_outputs):
        # save_for_backward enforces PyTorch's version check: do not optimizer.step
        # between this forward and backward (including multiple in-flight batches).
        inputs, *parameters = ctx.saved_tensors
        gradients = RemoteExpertWorker.run_coroutine(ctx.call.backward(grad_outputs, inputs, parameters))
        gradients = [None if g is None else g.to(p) for g, p in zip(gradients, (inputs, *parameters))]
        return gradients[0], None, *gradients[1:]


class LoRARemoteSequential(RemoteSequential):
    def __init__(self, original, lora_config):
        super().__init__(original.config, sequence_manager=original.sequence_manager)
        self.lora_config = lora_config
        self._session = uuid.uuid4().hex[:12]
        self._adapter_acks = {}  # (peer, start, end) -> adapter version that worker holds
        self._local = threading.local()  # per-microbatch route / seed / step token (train_step)
        self._upload_locks = {}
        self.boundary_codec = "none"  # see bloombee.utils.lora_codec; set via DistributedLoRA
        self.adapter_wire = "float32"  # "bfloat16" halves adapter/gradient traffic; set via DistributedLoRA
        shapes = projection_shapes(original.config)
        self.adapters = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        name: LoRAProjection(shapes[name][1], shapes[name][0], lora_config.r)
                        for name in lora_config.target_modules
                    }
                )
                for _ in range(len(original))
            ]
        )

    def _check_inputs(self, inputs, prompts, kwargs):
        if self.active_session is not None:
            raise ValueError("LoRA uses stateless full-sequence forward; KV inference sessions are not supported")
        if prompts is not None:
            raise ValueError("Combining LoRA with prompt tuning is not supported")
        mask = kwargs.get("tree_attention_mask")
        if mask is not None and (mask.ndim != 2 or not bool((mask == 1).all())):
            raise ValueError("LoRA currently requires unpadded causal sequences; use labels=-100 for ignored targets")
        for name in ("hypo_ids", "kv_cache_position_ids", "is_spec_decoding", "prefill_length"):
            if kwargs.get(name) is not None:
                raise ValueError(f"LoRA does not support {name}")
        if inputs.ndim != 3 or inputs.shape[-1] != self.config.hidden_size:
            raise ValueError("LoRA expects [batch, sequence, hidden_size]")

    def span_parameters(self, span):
        return tuple(
            p
            for layer in self.adapters[span.start : span.end]
            for name in self.lora_config.target_modules
            for p in (layer[name].lora_A, layer[name].lora_B)
        )

    def make_calls(self, route, needs_grad):
        calls = []
        seed_base = getattr(self._local, "seed", None)
        for index, span in enumerate(route):
            parameters = self.span_parameters(span)
            if seed_base is None:
                seed = int(torch.randint(0, 2**31, (), device="cpu"))
            else:
                seed = (seed_base + 1000003 * index) % 2**31
            metadata = dict(
                version=LORA_PROTOCOL_VERSION,
                config=self.lora_config.to_dict(),
                training=self.training,
                seed=seed,
            )
            calls.append((_SpanCall(self, span, metadata, parameters, needs_grad), parameters))
        return calls

    def balanced_routes(self, count, candidates=8):
        """Routes for `count` concurrent microbatches, spreading layers over replicas."""
        loads, routes = {}, []
        for _ in range(count):
            best, best_cost = None, None
            for _ in range(candidates):
                route = self.sequence_manager.make_sequence(mode="max_throughput")
                cost = max(loads.get(s.peer_id, 0) + s.end - s.start for s in route)
                if best is None or cost < best_cost:
                    best, best_cost = route, cost
            for span in best:
                loads[span.peer_id] = loads.get(span.peer_id, 0) + span.end - span.start
            routes.append(best)
        return routes

    @contextmanager
    def microbatch(self, *, route=None, seed=None, accumulate=None):
        names = ("route", "seed", "accumulate")
        previous = [getattr(self._local, name, None) for name in names]
        for name, value in zip(names, (route, seed, accumulate)):
            setattr(self._local, name, value)
        try:
            yield
        finally:
            for name, value in zip(names, previous):
                setattr(self._local, name, value)

    def collect_gradients(self, routes, token):
        """Fetch each used span's summed adapter gradients once and add them to .grad."""
        spans = {(str(s.peer_id), s.start, s.end): s for route in routes for s in route}
        metadata = dict(
            version=LORA_PROTOCOL_VERSION,
            config=self.lora_config.to_dict(),
            training=self.training,
            seed=0,
            accumulate=token,
        )
        if self.adapter_wire != "float32":
            metadata["adapter_wire"] = self.adapter_wire
        jobs = [(span, self.span_parameters(span)) for span in spans.values()]

        async def gather():
            return await asyncio.gather(
                *(_collect_gradients(self.sequence_manager, span, metadata, params) for span, params in jobs)
            )

        for (_, parameters), gradients in zip(jobs, RemoteExpertWorker.run_coroutine(gather())):
            for p, g in zip(parameters, gradients):
                g = g.to(p)
                if p.grad is None:
                    p.grad = g
                else:
                    p.grad.add_(g)

    def forward(self, inputs, prompts=None, **kwargs):
        self._check_inputs(inputs, prompts, kwargs)
        route = getattr(self._local, "route", None) or self.sequence_manager.make_sequence(mode="max_throughput")
        needs_grad = torch.is_grad_enabled() and (
            inputs.requires_grad or any(p.requires_grad for p in self.adapters.parameters())
        )
        device = inputs.device
        # Remote spans exchange CPU tensors. Keeping the autograd boundary on the CPU (the
        # copies happen for serialization anyway) runs each remote backward on the thread
        # that called backward(), not on the shared CUDA autograd thread: concurrent
        # microbatches (train_step) then pipeline their backward passes too.
        hidden = inputs.to("cpu")
        for call, parameters in self.make_calls(route, needs_grad):
            hidden = _RemoteLoRAFunction.apply(hidden, call, *parameters)
        return hidden.to(device)

    def inference_session(self, **kwargs):
        raise ValueError("Use DistributedLoRA.generate for full-prefix LoRA inference; remote KV is not adapter-aware")

    def __getitem__(self, index):
        raise ValueError("Slicing a LoRA sequence is not supported; keep the complete adapter/model mapping")


class DistributedLoRA(nn.Module):
    """Attach standard LoRA to a loaded BloomBee LLaMA/Qwen3 causal model.

    A/B and optimizer state live on the client; workers receive immutable copies
    per RPC and compute their gradients. This favors correctness over bandwidth:
    weights are sent again at backward; there is no worker-side optimizer step.
    """

    def __init__(self, model, config=None, *, boundary_compression="none", adapter_wire="float32"):
        """``boundary_compression`` ("none", "int8", "hadamard-int8") is an opt-in, lossy
        codec for activations and activation gradients on the network (see
        bloombee.utils.lora_codec). ``adapter_wire="bfloat16"`` is a second opt-in: adapters
        are uploaded and their gradients returned as bf16 (fp32 masters, optimizer state and
        gradient sums stay fp32), halving the adapter traffic. Both default to exact."""
        super().__init__()
        if adapter_wire not in ("float32", "bfloat16"):
            raise ValueError("adapter_wire must be 'float32' or 'bfloat16'")
        if boundary_compression not in lora_codec.CODECS:
            raise ValueError(f"boundary_compression must be one of {lora_codec.CODECS}")
        self.lora_config = config or LoRAConfig()
        if not isinstance(self.lora_config, LoRAConfig):
            raise TypeError("Use bloombee.LoRAConfig for DistributedLoRA")
        projection_shapes(model.config)
        if getattr(model.config, "tuning_mode", None):
            raise ValueError("Start from a model without prompt tuning or active adapters")
        if getattr(model.config, "active_adapter", None):
            raise ValueError("Combining existing adapters with training LoRA is not supported")
        modules = [(name, module) for name, module in model.named_modules() if isinstance(module, RemoteSequential)]
        if len(modules) != 1 or isinstance(modules[0][1], LoRARemoteSequential):
            raise ValueError("Expected one unmodified RemoteSequential")
        name, original = modules[0]
        if len(original) != model.config.num_hidden_layers:
            raise ValueError("Attach LoRA to the complete model layer sequence")
        model.requires_grad_(False)
        parent_path, _, attribute = name.rpartition(".")
        sequence = LoRARemoteSequential(original, self.lora_config)
        sequence.boundary_codec = boundary_compression
        sequence.adapter_wire = adapter_wire
        setattr(model.get_submodule(parent_path), attribute, sequence)
        self.model = model
        self._sequence_path = name

    @property
    def remote_layers(self):
        return self.model.get_submodule(self._sequence_path)

    @property
    def config(self):
        return self.model.config

    def forward(self, *args, **kwargs):
        if kwargs.get("past_key_values") is not None:
            raise ValueError("LoRA training/inference does not use remote KV caches")
        if kwargs.get("position_ids") is not None or kwargs.get("cache_position") is not None:
            raise ValueError("LoRA currently uses consecutive causal positions starting at zero")
        if kwargs.pop("use_cache", None):
            raise ValueError("LoRA currently uses stateless forward; set use_cache=False")
        return self.model(*args, **kwargs)

    def train_step(
        self, input_ids, labels=None, *, micro_batches=1, max_in_flight=None, accumulate_on_workers=True, loss_scale=1.0
    ):
        """Forward and backward of one batch, split into microbatches that pipeline across workers.

        Accumulates into .grad exactly what ``self(input_ids, labels=labels).loss.backward()``
        on the whole batch would (token-weighted mean over the microbatches), and returns that
        loss. Up to ``max_in_flight`` microbatches (default: all) are in flight at once, so
        while one span computes microbatch m the next span computes m-1, forward and backward.
        With replicated spans, microbatches are spread over replicas (data parallelism; the
        adapter gradients simply add up on the client). Call optimizer.step() afterwards.

        With ``accumulate_on_workers`` (default) each worker sums its adapter gradients over
        the step's microbatches and the client fetches them once, instead of downloading
        fp32 dA/dB after every microbatch. The sums are the same up to fp32 addition order.
        ``loss_scale`` multiplies the backpropagated loss (e.g. 1/gradient_accumulation_steps);
        the returned loss is unscaled.
        """
        labels = input_ids if labels is None else labels
        if micro_batches < 1 or micro_batches > input_ids.shape[0]:
            raise ValueError("micro_batches must be between 1 and the batch size")
        id_chunks, label_chunks = input_ids.chunk(micro_batches), labels.chunk(micro_batches)
        counts = [int((chunk[:, 1:] != -100).sum()) for chunk in label_chunks]
        total = sum(counts)
        if total == 0:
            raise ValueError("No target tokens in this batch")
        sequence = self.remote_layers
        seeds = torch.randint(0, 2**31, (len(id_chunks),), device="cpu").tolist()
        routes = sequence.balanced_routes(len(id_chunks))
        token = uuid.uuid4().hex if accumulate_on_workers else None

        def run(i):
            if counts[i] == 0:
                return 0.0
            with sequence.microbatch(route=routes[i], seed=seeds[i], accumulate=token):
                loss = self(input_ids=id_chunks[i], labels=label_chunks[i], use_cache=False).loss
                weight = counts[i] / total
                (loss * (weight * loss_scale)).backward()
                return float(loss.detach()) * weight

        if len(id_chunks) == 1:
            loss = run(0)
        else:
            workers = min(max_in_flight or len(id_chunks), len(id_chunks))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lora-microbatch") as pool:
                loss = sum(pool.map(run, range(len(id_chunks))))
        if token is not None:
            sequence.collect_gradients([r for r, c in zip(routes, counts) if c], token)
        return loss

    @torch.inference_mode()
    def generate(self, input_ids, *, max_new_tokens=20, eos_token_id=None):
        """Greedy full-prefix inference. Does not use or contaminate base-model KV caches."""
        if self.training:
            raise ValueError("Call eval() before LoRA generation")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        tokens = input_ids
        eos = getattr(self.model.generation_config, "eos_token_id", None) if eos_token_id is None else eos_token_id
        eos = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
        finished = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        for _ in range(max_new_tokens):
            next_token = self(tokens).logits[:, -1].argmax(-1)
            if eos:
                next_token = torch.where(finished, torch.full_like(next_token, eos[0]), next_token)
            tokens = torch.cat((tokens, next_token[:, None]), dim=1)
            if eos:
                finished |= torch.isin(next_token, torch.tensor(eos, device=tokens.device))
                if bool(finished.all()):
                    break
        return tokens

    def _adapter_state(self):
        state = {}
        for index, layer in enumerate(self.remote_layers.adapters):
            for name in self.lora_config.target_modules:
                group = "self_attn" if name in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
                for factor in ("A", "B"):
                    key = f"base_model.model.model.layers.{index}.{group}.{name}.lora_{factor}.weight"
                    state[key] = getattr(layer[name], f"lora_{factor}")
        return state

    def save_pretrained(self, directory):
        """PEFT-compatible adapter_config.json + adapter_model.safetensors; no base weights."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            **self.lora_config.to_dict(),
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "base_model_name_or_path": getattr(self.config, "_name_or_path", ""),
        }
        (path / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in self._adapter_state().items()},
            str(path / "adapter_model.safetensors"),
        )

    def _optimizer_parameter_names(self, optimizer):
        names = {id(p): name for name, p in self._adapter_state().items()}
        groups = [[names.get(id(p)) for p in group["params"]] for group in optimizer.param_groups]
        flattened = [name for group in groups for name in group]
        if None in flattened or len(flattened) != len(names) or set(flattened) != set(names.values()):
            raise ValueError("Optimizer must own every LoRA parameter exactly once and no base parameters")
        return groups

    def save_training_checkpoint(self, directory, optimizer, *, step, metadata=None):
        """Save at an optimizer boundary after zero_grad(set_to_none=True).

        Includes adapter weights, optimizer state, completed step count and the
        client's torch CPU/CUDA RNG states. Use a fresh directory per checkpoint.
        """
        groups = self._optimizer_parameter_names(optimizer)
        if any(p.grad is not None for p in self._adapter_state().values()):
            raise ValueError("Checkpoint requires cleared gradients at an optimizer boundary")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("step must be a nonnegative completed optimizer step count")
        state = dict(
            version=1,
            step=step,
            metadata=metadata or {},
            parameter_names=groups,
            lora_config=self.lora_config.to_dict(),
            optimizer_type=f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
            optimizer=optimizer.state_dict(),
            cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        )
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        self.save_pretrained(path)
        # Written last: an incomplete adapter-only directory cannot resume training.
        torch.save(state, path / "training_state.pt")

    def load_training_state(self, directory, optimizer):
        """Restore optimizer/RNG after from_pretrained and device placement.

        Call before the next batch is sampled. The adapter already loaded into
        this model must match the checkpoint, preventing mixed weights/state.
        """
        path = Path(directory)
        state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
        if state["lora_config"] != self.lora_config.to_dict():
            raise ValueError("Training checkpoint LoRA configuration does not match")
        if (
            state["version"] != 1
            or state["parameter_names"] != self._optimizer_parameter_names(optimizer)
            or state["optimizer_type"] != f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
        ):
            raise ValueError("Training checkpoint optimizer type/parameter order does not match")
        saved = load_file(str(path / "adapter_model.safetensors"))
        current = self._adapter_state()
        if set(saved) != set(current) or any(not torch.equal(p.detach().cpu(), saved[k]) for k, p in current.items()):
            raise ValueError("Load the checkpoint adapter before restoring its optimizer state")
        if any(p.grad is not None for p in current.values()):
            raise ValueError("Clear gradients before restoring training state")
        if state["cuda_rng"] and len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("Exact RNG resume requires the same number of visible CUDA devices")
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return {"step": state["step"], "metadata": state["metadata"]}

    @classmethod
    def from_pretrained(cls, model, directory, **kwargs):
        path = Path(directory)
        config = json.loads((path / "adapter_config.json").read_text())
        if (
            config.get("peft_type") != "LORA"
            or config.get("bias", "none") != "none"
            or config.get("fan_in_fan_out", False)
            or config.get("use_rslora", False)
            or config.get("use_dora", False)
            or config.get("rank_pattern")
            or config.get("alpha_pattern")
            or config.get("modules_to_save")
            or config.get("lora_bias", False)
        ):
            raise ValueError("Only standard bias-free alpha/r LoRA adapters are supported")
        result = cls(
            model, LoRAConfig(**{k: config[k] for k in ("r", "lora_alpha", "lora_dropout", "target_modules")}), **kwargs
        )
        expected, state = result._adapter_state(), load_file(str(path / "adapter_model.safetensors"))
        if set(expected) != set(state) or any(expected[k].shape != state[k].shape for k in expected):
            raise ValueError("LoRA checkpoint keys/shapes do not match this model")
        with torch.no_grad():
            for key, parameter in expected.items():
                parameter.copy_(state[key])
        return result
