"""Client-owned standard LoRA parameters, ordinary torch optimizers, remote gradients."""

import asyncio
import json
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
from bloombee.utils.hivemind_compat import MSGPackSerializer
from bloombee.utils.lora import LORA_NAMESPACE, LORA_PROTOCOL_VERSION, LoRAConfig, LoRAProjection, projection_shapes
from bloombee.utils.lossless_transport import deserialize_tensor_stream, serialize_torch_tensor


async def _call_lora(manager, span, inputs, metadata):
    stub = LoRAService.get_stub(manager.state.p2p, span.peer_id, namespace=LORA_NAMESPACE)
    uid = " ".join(manager.block_uids[span.start : span.end])
    parts = []
    for tensor in inputs:
        wire = serialize_torch_tensor(tensor.detach().cpu().contiguous(), runtime_pb2.CompressionType.NONE)
        parts.extend(
            runtime_pb2.ExpertRequest(uid=uid, tensors=[part], metadata=MSGPackSerializer.dumps(metadata))
            for part in split_for_streaming(wire, DEFAULT_MAX_MSG_SIZE)
        )
    responses = await asyncio.wait_for(
        stub.rpc_lora_stream(iter_as_aiter(parts)), timeout=manager.config.connect_timeout
    )

    async def checked():
        async for response in aiter_with_timeout(responses, manager.config.request_timeout):
            if MSGPackSerializer.loads(response.metadata).get("version") != LORA_PROTOCOL_VERSION:
                raise RuntimeError("Worker did not acknowledge LoRA protocol v1")
            yield response.tensors

    return await deserialize_tensor_stream(checked())


class _RemoteLoRAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, manager, span, metadata, *parameters):
        ctx.manager, ctx.span, ctx.lora_metadata = manager, span, metadata
        ctx.save_for_backward(inputs, *parameters)
        result = RemoteExpertWorker.run_coroutine(
            _call_lora(manager, span, (inputs, *parameters), {**metadata, "backward": False})
        )
        if len(result) != 1 or result[0].shape != inputs.shape:
            raise RuntimeError("Invalid remote LoRA forward response")
        return result[0].to(inputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_outputs):
        # save_for_backward enforces PyTorch's version check: do not optimizer.step
        # between this forward and backward (including multiple in-flight batches).
        inputs, *parameters = ctx.saved_tensors
        gradients = RemoteExpertWorker.run_coroutine(
            _call_lora(
                ctx.manager, ctx.span, (inputs, *parameters, grad_outputs), {**ctx.lora_metadata, "backward": True}
            )
        )
        expected = (inputs, *parameters)
        if len(gradients) != len(expected) or any(g.shape != p.shape for g, p in zip(gradients, expected)):
            raise RuntimeError("Invalid remote LoRA gradient response")
        gradients = [g.to(p) for g, p in zip(gradients, expected)]
        return gradients[0], None, None, None, *gradients[1:]


class LoRARemoteSequential(RemoteSequential):
    def __init__(self, original, lora_config):
        super().__init__(original.config, sequence_manager=original.sequence_manager)
        self.lora_config = lora_config
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

    def forward(self, inputs, prompts=None, **kwargs):
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
        route = self.sequence_manager.make_sequence(mode="max_throughput")
        for span in route:
            parameters = tuple(
                p
                for layer in self.adapters[span.start : span.end]
                for name in self.lora_config.target_modules
                for p in (layer[name].lora_A, layer[name].lora_B)
            )
            metadata = dict(
                version=LORA_PROTOCOL_VERSION,
                config=self.lora_config.to_dict(),
                training=self.training,
                seed=int(torch.randint(0, 2**31, (), device="cpu")),
            )
            inputs = _RemoteLoRAFunction.apply(inputs, self.sequence_manager, span, metadata, *parameters)
        return inputs

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

    def __init__(self, model, config=None):
        super().__init__()
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
    def from_pretrained(cls, model, directory):
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
        result = cls(model, LoRAConfig(**{k: config[k] for k in ("r", "lora_alpha", "lora_dropout", "target_modules")}))
        expected, state = result._adapter_state(), load_file(str(path / "adapter_model.safetensors"))
        if set(expected) != set(state) or any(expected[k].shape != state[k].shape for k in expected):
            raise ValueError("LoRA checkpoint keys/shapes do not match this model")
        with torch.no_grad():
            for key, parameter in expected.items():
                parameter.copy_(state[key])
        return result
