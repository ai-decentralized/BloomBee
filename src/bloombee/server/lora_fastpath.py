"""Training fast path for dense LLaMA / Qwen3 blocks under LoRA.

The serving code path is built for inference: FlexGen's LLaMA attention computes
eager fp32 B x H x S x S scores, repeat-interleaves GQA heads, writes KV caches
and rebuilds Task objects on each call. LoRA training only needs a pure function
of (hidden_states, frozen weights, request-local adapters). This module is that
function. It reads the very same frozen weight tensors (no copies), applies
adapters through bloombee.utils.lora, and uses causal scaled_dot_product_attention,
the default attention of HuggingFace models.

`block_weights` returns None whenever the block cannot use the fast path (weights
offloaded to CPU/disk, compressed, mixed devices, tensor parallel, unsupported
family); callers then fall back to the regular module forward.
"""

import torch
import torch.nn.functional as F
from transformers.activations import ACT2FN

from bloombee.utils.lora import lora_linear_group

_ROPE_CACHE = {}


def _plain(tensor_like, device):
    data = getattr(tensor_like, "data", None)
    if not isinstance(data, torch.Tensor) or data.device != device:
        return None
    device_type = getattr(getattr(tensor_like, "device", None), "device_type", None)
    if device_type is not None and device_type.name not in ("CUDA", "CPU", "MPS"):
        return None
    return data


def _llama_weights(block, device):
    homes = getattr(block, "weight_home", None)
    if homes is None or len(homes) != 2:
        return None
    try:
        attn, mlp = homes[0].val, homes[1].val
    except Exception:
        return None
    if attn is None or mlp is None or len(attn) != 6 or len(mlp) != 4:
        return None
    tensors = [_plain(t, device) for t in (*attn, *mlp)]
    if any(t is None for t in tensors):
        return None
    w_q, w_k, w_v, w_o, ln1, inv_freq, gate, down, up, ln2 = tensors
    config = block.llama_config
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return dict(
        family="llama",
        ln1=ln1,
        q_proj=w_q,
        k_proj=w_k,
        v_proj=w_v,
        o_proj=w_o,
        ln2=ln2,
        gate_proj=gate,
        up_proj=up,
        down_proj=down,
        inv_freq=inv_freq,
        q_norm=None,
        k_norm=None,
        rotary=None,
        head_dim=head_dim,
        num_heads=config.num_attention_heads,
        num_kv_heads=w_k.shape[0] // head_dim,
        eps=float(getattr(config, "rms_norm_eps", 1e-5)),
        act=ACT2FN[config.hidden_act],
    )


def _qwen3_weights(block, device):
    attn, mlp = getattr(block, "self_attn", None), getattr(block, "mlp", None)
    if attn is None or mlp is None or not hasattr(attn, "q_norm"):
        return None
    linears = dict(
        q_proj=attn.q_proj, k_proj=attn.k_proj, v_proj=attn.v_proj, o_proj=attn.o_proj,
        gate_proj=mlp.gate_proj, up_proj=mlp.up_proj, down_proj=mlp.down_proj,
    )  # fmt: skip
    if any(type(m) is not torch.nn.Linear or m.bias is not None for m in linears.values()):
        return None  # quantized or biased projections: use the module itself
    weights = {name: m.weight for name, m in linears.items()}
    norms = dict(
        ln1=block.input_layernorm.weight,
        ln2=block.post_attention_layernorm.weight,
        q_norm=attn.q_norm.weight,
        k_norm=attn.k_norm.weight,
    )
    if any(t.device != device for t in (*weights.values(), *norms.values())):
        return None
    config = attn.config
    head_dim = getattr(attn, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return dict(
        family="qwen3",
        **weights,
        **norms,
        inv_freq=None,
        rotary=block._rotary_emb,
        head_dim=head_dim,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        eps=float(config.rms_norm_eps),
        act=ACT2FN[config.hidden_act],
        sliding_window=getattr(block, "sliding_window", None) if getattr(config, "use_sliding_window", False) else None,
    )


def block_weights(backend, device):
    """Frozen weights of a backend's block for the fast path, or None to use the module forward."""
    cached = getattr(backend, "_lora_fastpath_weights", None)
    if cached is not None:
        return cached if cached is not False else None
    shards = getattr(backend.module, "module_shards", None)
    result = None
    if shards is not None and len(shards) == 1:
        block = shards[0]
        if backend.config.model_type == "llama":
            result = _llama_weights(block, device)
        elif backend.config.model_type == "qwen3":
            result = _qwen3_weights(block, device)
            if result is not None and result["sliding_window"]:
                result = None  # sliding-window attention is not implemented here
    # Weight placement of a running server is fixed: cache the decision.
    backend._lora_fastpath_weights = result if result is not None else False
    return result


def _rms(x, weight, eps):
    dtype = x.dtype
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return weight * h.to(dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _llama_cos_sin(inv_freq, seq_len, dtype):
    """Same tables (and rounding) as FlexGen's precompute_freqs_cis + apply_rotary_emb."""
    key = (inv_freq.data_ptr(), inv_freq.device, seq_len, dtype)
    cached = _ROPE_CACHE.get(key)
    if cached is None:
        t = torch.arange(seq_len, device=inv_freq.device)
        freqs = torch.outer(t, inv_freq).float()
        cis = torch.polar(torch.ones_like(freqs), freqs)
        cos = torch.cat((cis.real, cis.real), dim=-1).to(dtype)
        sin = torch.cat((cis.imag, cis.imag), dim=-1).to(dtype)
        if len(_ROPE_CACHE) > 64:
            _ROPE_CACHE.clear()
        cached = _ROPE_CACHE[key] = (cos[None, None], sin[None, None])  # [1, 1, S, D]
    return cached


def block_forward(x, w):
    """One decoder layer. x: [B, S, hidden] in the block dtype. Adapters come from lora_context."""
    bsz, seq, _ = x.shape
    nh, nkv, hd = w["num_heads"], w["num_kv_heads"], w["head_dim"]

    h = _rms(x, w["ln1"], w["eps"])
    q, k, v = lora_linear_group(h, ((w["q_proj"], "q_proj"), (w["k_proj"], "k_proj"), (w["v_proj"], "v_proj")))
    q = q.view(bsz, seq, nh, hd)
    k = k.view(bsz, seq, nkv, hd)
    v = v.view(bsz, seq, nkv, hd)
    if w["q_norm"] is not None:
        q = _rms(q, w["q_norm"], w["eps"])
        k = _rms(k, w["k_norm"], w["eps"])
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [B, heads, S, D]
    if w["family"] == "llama":
        cos, sin = _llama_cos_sin(w["inv_freq"], seq, q.dtype)
    else:
        positions = torch.arange(seq, device=x.device).unsqueeze(0)
        cos, sin = w["rotary"](x, positions)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    if nkv != nh:
        # GQA: replicate K/V heads (B*S*H elements, cheap next to attention). SDPA's
        # enable_gqa would avoid the copy but forces the math kernel on pre-Ampere GPUs.
        k = k[:, :, None].expand(bsz, nkv, nh // nkv, seq, hd).reshape(bsz, nh, seq, hd)
        v = v[:, :, None].expand(bsz, nkv, nh // nkv, seq, hd).reshape(bsz, nh, seq, hd)
    attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=hd**-0.5)
    attn = attn.transpose(1, 2).reshape(bsz, seq, nh * hd)
    (o,) = lora_linear_group(attn, ((w["o_proj"], "o_proj"),))
    x = x + o

    h = _rms(x, w["ln2"], w["eps"])
    gate, up = lora_linear_group(h, ((w["gate_proj"], "gate_proj"), (w["up_proj"], "up_proj")))
    (down,) = lora_linear_group(w["act"](gate) * up, ((w["down_proj"], "down_proj"),))
    return x + down
