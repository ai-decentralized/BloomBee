"""Block wrapper for Gemma-4 text-tower layers.

Key differences from WrappedQwen3Block we have to handle:

1. `layer_types` alternates `sliding_attention` / `full_attention`, and
   each type uses its own RoPE pair and its own attention mask. Gemma-4's
   `Gemma4TextRotaryEmbedding` is itself layer-type aware — we just thread
   the string through.
2. Full-attention layers use a different KV head count and head_dim
   (`num_global_key_value_heads`, `global_head_dim`) and alias V to K
   (`attention_k_eq_v=True`). The KV head count drives `_reorder_cache_*`.
3. Gemma-4's layer forward signature takes `shared_kv_states` (a dict
   passed across the whole layer stack for the 'KV sharing' pattern
   used on E2B/E4B). On 31B `num_kv_shared_layers=0`, so we pass `{}`
   and the attention writes / reads nothing.
4. `per_layer_input` is unused on 31B (`hidden_size_per_layer_input=0`),
   so we pass `None`.
5. The decoder applies a `self.layer_scalar` at the end of its forward —
   no-op for us (inherited from `super().__init__`).
"""

from typing import Optional, Tuple

import torch

from bloombee.utils.cache_compat import (
    make_empty_kv_cache,
    make_past_kv_cache,
    read_kv_from_cache,
)
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig as _BaseBlockConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextDecoderLayer as _BaseDecoderLayer,
    Gemma4TextRotaryEmbedding as _RotaryEmbedding,
)


class WrappedGemma4Block(_BaseDecoderLayer):
    def __init__(self, config: _BaseBlockConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self._attn_implementation = config._attn_implementation
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        # Sliding window: `sliding_window` applies to sliding layers only; full
        # layers see the whole context.
        self.sliding_window = config.sliding_window if self.is_sliding else None
        # One rotary module holds both inv_freq pairs; forward() picks by key.
        self._rotary_emb = _RotaryEmbedding(config)

        # BloomBee's backend assumes `self_attn.num_heads` and `num_key_value_heads`
        # are valid; Gemma-4's attention derives them differently on full vs
        # sliding layers. Expose them here to match the backend's contract.
        if not hasattr(self.self_attn, "num_heads"):
            self.self_attn.num_heads = config.num_attention_heads
        if not hasattr(self.self_attn, "num_key_value_heads"):
            # Gemma-4: sliding uses config.num_key_value_heads, full uses
            # config.num_global_key_value_heads. The HF attention instance
            # already knows the right value; we read it back when available.
            kv = getattr(
                self.self_attn,
                "num_key_value_heads",
                (
                    getattr(config, "num_global_key_value_heads", None)
                    if not self.is_sliding
                    else None
                )
                or config.num_key_value_heads,
            )
            self.self_attn.num_key_value_heads = kv

    def _apply(self, fn, recurse=True):
        # Same fp16/bf16 rotary buffer guard as Qwen3: keep inv_freq buffers
        # in fp32 even after the module is cast to half-precision. Gemma-4's
        # rotary stores one pair per layer type, so we walk both.
        out = super()._apply(fn, recurse=recurse)
        rot = getattr(self, "_rotary_emb", None)
        if rot is not None:
            for layer_type in getattr(rot, "layer_types", ()):
                for suffix in ("inv_freq", "original_inv_freq"):
                    name = f"{layer_type}_{suffix}"
                    buf = getattr(rot, name, None)
                    if buf is not None and buf.is_floating_point() and buf.dtype != torch.float32:
                        rot.register_buffer(name, buf.float(), persistent=False)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        batch_size, seq_length, _ = hidden_states.shape

        past_key_values_length = 0
        past_key_value = layer_past

        if past_key_value is not None:
            pk, pv = past_key_value
            if pk.dtype != hidden_states.dtype or pk.device != hidden_states.device:
                pk = pk.to(device=hidden_states.device, dtype=hidden_states.dtype)
                pv = pv.to(device=hidden_states.device, dtype=hidden_states.dtype)
                past_key_value = (pk, pv)
            past_key_values_length = past_key_value[0].shape[2]
            _past_key_value = self._reorder_cache_from_bloom(past_key_value, batch_size, past_key_values_length)
            past_key_value = make_past_kv_cache(
                _past_key_value[0], _past_key_value[1],
                layer_idx=self.layer_idx, seen_tokens=past_key_values_length,
            )
        elif use_cache:
            past_key_value = make_empty_kv_cache(self.layer_idx)

        # Causal mask: Gemma-4 full_attention uses a plain causal triangular
        # mask; sliding_attention additionally zeros entries outside the
        # sliding window. For Phase 1 we only implement the full path (the
        # simple mask) and let sliding layers attend over whatever's in cache
        # — that's functionally equivalent when the cumulative context
        # already fits inside the window, which is true for short prompts.
        # Longer-context serving needs per-layer masks and is Phase 3.
        if attention_mask is None:
            total_len = past_key_values_length + seq_length
            neg_inf = torch.finfo(hidden_states.dtype).min
            causal = torch.full(
                (seq_length, total_len),
                neg_inf,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            if total_len > 0:
                causal = torch.triu(causal, diagonal=past_key_values_length + 1)
            attention_mask = causal.unsqueeze(0).unsqueeze(0)

        position_ids = kwargs.pop("position_ids", None)
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length,
                dtype=torch.long, device=hidden_states.device,
            ).unsqueeze(0).expand(batch_size, -1)

        # Gemma-4's rotary picks by layer_type string.
        position_embeddings = self._rotary_emb(hidden_states, position_ids, self.layer_type)

        cache_position = torch.arange(
            past_key_values_length, past_key_values_length + seq_length,
            dtype=torch.long, device=hidden_states.device,
        )

        skip_keys = {
            "position_ids", "attention_mask", "use_cache", "rotary_position_ids",
            "position_embeddings", "past_key_value", "past_key_values",
            "cache_position", "per_layer_input", "shared_kv_states",
        }
        extra_kwargs = {k: v for k, v in kwargs.items() if k not in skip_keys}

        # Gemma-4's DecoderLayer.forward requires `shared_kv_states` as a
        # positional/keyword dict. On 31B (num_kv_shared_layers=0) it's
        # dead — no layer reads from it — but the signature demands it.
        # `per_layer_input=None` is fine because hidden_size_per_layer_input=0
        # disables that branch.
        outputs = super().forward(
            hidden_states,
            per_layer_input=None,
            shared_kv_states={},
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            cache_position=cache_position,
            use_cache=use_cache,
            **extra_kwargs,
        )

        if isinstance(outputs, torch.Tensor):
            output_hidden = outputs
        elif isinstance(outputs, tuple):
            output_hidden = outputs[0]
        else:
            output_hidden = outputs

        if use_cache and past_key_value is not None:
            pk, pv = read_kv_from_cache(past_key_value, self.layer_idx)
            if pk is not None:
                pk = pk[:, :, past_key_values_length:, :]
                pv = pv[:, :, past_key_values_length:, :]
                present_key_value = self._reorder_cache_to_bloom((pk, pv), batch_size, seq_length)
                return (output_hidden, present_key_value)

        return (output_hidden, None)

    # ---- cache-layout bridges between BloomBee's 3D (BH, D, S) and HF's 4D (B, H, S, D) ----

    def _kv_shape(self) -> Tuple[int, int]:
        """Return (num_kv_heads, head_dim) active on this layer."""
        nkv = self.self_attn.num_key_value_heads
        hd = self.self_attn.head_dim
        return nkv, hd

    def _reorder_cache_from_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        key_states, value_states = key_value
        nkv, hd = self._kv_shape()
        if key_states.dim() == 4:
            # [B, H, S, D] — slice to the first `nkv` KV heads (BloomBee
            # allocates num_attention_heads but only the first num_kv_heads
            # are meaningful under GQA).
            key_states = key_states[:, :nkv, :, :]
            value_states = value_states[:, :nkv, :, :]
            return (key_states, value_states)
        # 3D: key [B*H, D, S], value [B*H, S, D]
        key_states = key_states.permute(0, 2, 1)  # [B*H, S, D]
        key_states = key_states.view(batch_size, nkv, seq_length, hd)
        value_states = value_states.view(*key_states.shape)
        return (key_states, value_states)

    def _reorder_cache_to_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        key_states, value_states = key_value
        nkv, hd = self._kv_shape()
        value_states = value_states.reshape(batch_size * nkv, seq_length, hd)
        key_states = key_states.reshape(*value_states.shape)
        key_states = key_states.permute(0, 2, 1)
        return (key_states, value_states)
