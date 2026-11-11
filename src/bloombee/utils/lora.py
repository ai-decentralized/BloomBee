"""Standard alpha/r LoRA, shared by client parameters and worker computations.

Adapters are request-local tensors. No worker parameter or optimizer is mutated.
The context also reaches FlexGen's *actual* functional projection operators.
"""

import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

LORA_PROTOCOL_VERSION = 1
LORA_NAMESPACE = "lora-v1"
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_ACTIVE_LORA = ContextVar("bloombee_request_lora", default=None)


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    target_modules: tuple = ("q_proj", "v_proj")

    def __post_init__(self):
        object.__setattr__(self, "target_modules", tuple(self.target_modules))
        if isinstance(self.r, bool) or not isinstance(self.r, int) or not 1 <= self.r <= 256:
            raise ValueError("LoRA rank must be an integer in [1, 256]")
        if not math.isfinite(self.lora_alpha) or self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be finite and positive")
        if not math.isfinite(self.lora_dropout) or not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        targets = self.target_modules
        if not targets or len(set(targets)) != len(targets) or any(t not in PROJECTIONS for t in targets):
            raise ValueError(f"Choose unique LoRA targets from {PROJECTIONS}")

    def to_dict(self):
        return asdict(self)


def projection_shapes(config):
    if config.model_type not in ("llama", "qwen3"):
        raise ValueError("Distributed LoRA currently supports LLaMA and Qwen3 dense blocks")
    h, i = config.hidden_size, config.intermediate_size
    d = getattr(config, "head_dim", None) or h // config.num_attention_heads
    q = config.num_attention_heads * d
    kv = getattr(config, "num_key_value_heads", config.num_attention_heads) * d
    return {
        "q_proj": (q, h),
        "k_proj": (kv, h),
        "v_proj": (kv, h),
        "o_proj": (h, q),
        "gate_proj": (i, h),
        "up_proj": (i, h),
        "down_proj": (h, i),
    }


class LoRAProjection(nn.Module):
    def __init__(self, in_features, out_features, rank, *, device=None):
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def _apply(self, fn, recurse=True):
        # Preserve values as well as dtype: converting to half and back to float
        # would irreversibly round the optimizer's master weights and gradients.
        def keep_fp32(tensor):
            converted = fn(tensor)
            return (
                converted
                if converted.dtype == torch.float32
                else tensor.to(device=converted.device, dtype=torch.float32)
            )

        return super()._apply(keep_fp32, recurse=recurse)


@contextmanager
def lora_context(config, tensors, *, training, seed):
    if len(tensors) != 2 * len(config.target_modules):
        raise ValueError("LoRA A/B tensor count does not match target_modules")
    pairs = {name: tensors[2 * j : 2 * j + 2] for j, name in enumerate(config.target_modules)}
    token = _ACTIVE_LORA.set((config, pairs, bool(training), int(seed)))
    try:
        yield
    finally:
        _ACTIVE_LORA.reset(token)


def lora_delta(inputs, projection):
    state = _ACTIVE_LORA.get()
    if state is None or projection not in state[1]:
        return None
    config, pairs, training, seed = state
    a, b = pairs[projection]
    hidden = inputs.to(dtype=a.dtype)
    if training and config.lora_dropout:
        # A request-local RNG: exact recompute without modifying global CPU/CUDA RNG.
        generator = torch.Generator(device=hidden.device)
        generator.manual_seed((seed + PROJECTIONS.index(projection) * 104729) % (2**63 - 1))
        keep = torch.rand(hidden.shape, device=hidden.device, generator=generator) >= config.lora_dropout
        hidden = hidden * keep / (1.0 - config.lora_dropout)
    return F.linear(F.linear(hidden, a), b) * (config.lora_alpha / config.r)


def lora_linear(inputs, weight, projection):
    output = F.linear(inputs, weight)
    delta = lora_delta(inputs, projection)
    return output if delta is None else output + delta.to(output.dtype)


def install_lora_hooks(module):
    """Install once on HF projections. An empty context leaves base inference unchanged."""
    for name, child in module.named_modules():
        target = name.rsplit(".", 1)[-1]
        if target in PROJECTIONS and isinstance(child, nn.Linear) and not getattr(child, "_bb_lora_hook", False):

            def add_delta(linear, args, output, projection=target):
                delta = lora_delta(args[0], projection)
                return output if delta is None else output + delta.to(output.dtype)

            child.register_forward_hook(add_delta)
            child._bb_lora_hook = True
