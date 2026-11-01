from typing import Optional, Tuple

import torch
from transformers.models.smollm3 import SmolLM3Config
from transformers.models.smollm3.modeling_smollm3 import SmolLM3DecoderLayer, SmolLM3RotaryEmbedding

from bloombee.utils.cache_compat import make_empty_kv_cache, make_past_kv_cache, read_kv_from_cache


class WrappedSmolLM3Block(SmolLM3DecoderLayer):
    """SmolLM3 decoder layer adapted to BloomBee's block interface.

    SmolLM3 is Llama-like with GQA. Two differences matter here:
    - NoPE layers (``config.no_rope_layers[i] == 0``) skip rotary embedding.
      HF's ``SmolLM3Attention`` handles this via ``self_attn.use_rope``, so we
      always compute and pass ``position_embeddings``.
    - Optional sliding-window layers (``use_sliding_window=True``). Eager/SDPA
      attention ignores the ``sliding_window`` kwarg, so the window is enforced
      in the additive mask built here. Released checkpoints have it disabled.
    """

    def __init__(self, config: SmolLM3Config, layer_idx: int):
        super().__init__(config, layer_idx)

        self._attn_implementation = config._attn_implementation
        self.layer_idx = layer_idx
        self.sliding_window = self.self_attn.sliding_window
        self._rotary_emb = SmolLM3RotaryEmbedding(config)

        # BloomBee's backend.py accesses self_attn.num_heads — add it for compatibility
        if not hasattr(self.self_attn, "num_heads"):
            self.self_attn.num_heads = config.num_attention_heads
        if not hasattr(self.self_attn, "num_key_value_heads"):
            self.self_attn.num_key_value_heads = config.num_key_value_heads

    def _apply(self, fn, recurse=True):
        # Keep rotary inv_freq buffers in fp32 across .to(dtype=fp16/bf16) calls
        # (same reason as WrappedQwen3Block: fp16 inv_freq corrupts positions).
        out = super()._apply(fn, recurse=recurse)
        rot = getattr(self, "_rotary_emb", None)
        if rot is not None:
            for name in ("inv_freq", "original_inv_freq"):
                buf = getattr(rot, name, None)
                if buf is not None and buf.is_floating_point() and buf.dtype != torch.float32:
                    rot.register_buffer(name, buf.float(), persistent=False)
        return out

    def _sliding_window_mask(self, seq_length: int, past_length: int, like: torch.Tensor) -> torch.Tensor:
        """Additive [1, 1, S, K] mask hiding keys older than ``sliding_window - 1`` positions."""
        total_len = past_length + seq_length
        q_pos = torch.arange(past_length, total_len, device=like.device)
        k_pos = torch.arange(total_len, device=like.device)
        too_old = k_pos.unsqueeze(0) < (q_pos.unsqueeze(1) - (self.sliding_window - 1))
        mask = torch.zeros(seq_length, total_len, dtype=like.dtype, device=like.device)
        mask = mask.masked_fill(too_old, torch.finfo(like.dtype).min)
        return mask.unsqueeze(0).unsqueeze(0)

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
            # Cache may differ in dtype/device from hidden_states (fp16 cache, pipeline parallelism).
            pk, pv = past_key_value
            if pk.dtype != hidden_states.dtype or pk.device != hidden_states.device:
                pk = pk.to(device=hidden_states.device, dtype=hidden_states.dtype)
                pv = pv.to(device=hidden_states.device, dtype=hidden_states.dtype)
                past_key_value = (pk, pv)
            past_key_values_length = past_key_value[0].shape[2]
            _past_key_value = self._reorder_cache_from_bloom(past_key_value, batch_size, past_key_values_length)
            past_key_value = make_past_kv_cache(
                _past_key_value[0],
                _past_key_value[1],
                layer_idx=self.layer_idx,
                seen_tokens=past_key_values_length,
            )
        elif use_cache:
            past_key_value = make_empty_kv_cache(self.layer_idx)

        # tf 5.x eager/SDPA attention only masks what it is given, and BloomBee wraps a
        # bare DecoderLayer (no SmolLM3Model to build the mask), so build it here.
        if attention_mask is None:
            total_len = past_key_values_length + seq_length
            causal = torch.full(
                (seq_length, total_len),
                torch.finfo(hidden_states.dtype).min,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            if total_len > 0:
                causal = torch.triu(causal, diagonal=past_key_values_length + 1)
            attention_mask = causal.unsqueeze(0).unsqueeze(0)
        elif attention_mask.dim() == 3:
            # BloomBee's backend builds [B, S, K]; HF attention expects [B, 1, S, K].
            attention_mask = attention_mask.unsqueeze(1)

        if self.sliding_window is not None and self.sliding_window > 0:
            # Both masks are additive (0 or dtype-min), so elementwise min = union of masked keys.
            sliding = self._sliding_window_mask(seq_length, past_key_values_length, attention_mask)
            attention_mask = torch.minimum(attention_mask, sliding)

        position_ids = kwargs.pop("position_ids", None)
        if position_ids is None:
            position_ids = (
                torch.arange(
                    past_key_values_length,
                    seq_length + past_key_values_length,
                    dtype=torch.long,
                    device=hidden_states.device,
                )
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        position_embeddings = self._rotary_emb(hidden_states, position_ids)

        # tf 5.x attention needs cache_position to know where to write new KV into the cache.
        cache_position = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_length,
            dtype=torch.long,
            device=hidden_states.device,
        )

        skip_keys = {
            "position_ids",
            "attention_mask",
            "use_cache",
            "rotary_position_ids",
            "position_embeddings",
            "past_key_value",
            "past_key_values",
            "cache_position",
        }
        extra_kwargs = {k: v for k, v in kwargs.items() if k not in skip_keys}

        outputs = super().forward(
            hidden_states,
            *args,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
            **extra_kwargs,
        )
        output_hidden = outputs[0] if isinstance(outputs, tuple) else outputs

        if use_cache and past_key_value is not None:
            pk, pv = read_kv_from_cache(past_key_value, self.layer_idx)
            if pk is not None:
                # Return only NEW tokens (BloomBee manages the cumulative cache externally)
                pk = pk[:, :, past_key_values_length:, :]
                pv = pv[:, :, past_key_values_length:, :]
                return (output_hidden, self._reorder_cache_to_bloom((pk, pv), batch_size, seq_length))

        return (output_hidden, None)

    def _reorder_cache_from_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        key_states, value_states = key_value
        if key_states.dim() == 4:
            # select_cache() returns [B, H, S, D] allocated for num_attention_heads;
            # only the first num_key_value_heads heads are valid (GQA).
            nkv = self.self_attn.num_key_value_heads
            return (key_states[:, :nkv, :, :], value_states[:, :nkv, :, :])
        # 3D case: key is [B*H, D, S], value is [B*H, S, D]
        key_states = key_states.permute(0, 2, 1)
        key_states = key_states.view(
            batch_size, self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        value_states = value_states.view(*key_states.shape)
        return (key_states, value_states)

    def _reorder_cache_to_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        key_states, value_states = key_value
        # reshape (not view): tensors from DynamicCache may be non-contiguous
        value_states = value_states.reshape(
            batch_size * self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        key_states = key_states.reshape(*value_states.shape)
        key_states = key_states.permute(0, 2, 1)
        return (key_states, value_states)
