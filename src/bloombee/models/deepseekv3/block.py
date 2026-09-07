from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.models.deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3DecoderLayer,
    DeepseekV3RotaryEmbedding,
)

from bloombee.utils.cache_compat import make_past_kv_cache, make_empty_kv_cache, read_kv_from_cache


class WrappedDeepseekV3Block(DeepseekV3DecoderLayer):
    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        # DeepseekV3MoE's expert dispatch reads config._experts_implementation at
        # forward time and falls back to a *different, buggy* default kernel path
        # when it's left as None (the transformers.integrations.moe warning calls
        # this the "standalone module" case, which is exactly what a bare
        # DecoderLayer is here -- BloomBee never builds the full DeepseekV3Model
        # that would normally set this). Left unset, MoE layers emit inf/nan on
        # single-token forward passes, i.e. every decode step after prefill.
        if getattr(config, "_experts_implementation", None) is None:
            config._experts_implementation = "eager"

        super().__init__(config, layer_idx)

        self._attn_implementation = config._attn_implementation
        self.layer_idx = layer_idx
        self._rotary_emb = DeepseekV3RotaryEmbedding(config)
        self._compressed_cache = hasattr(self.self_attn, "expand_kv")
        self._cache_key_dim = config.kv_lora_rank if self._compressed_cache else self.self_attn.qk_head_dim
        self._cache_value_dim = config.qk_rope_head_dim if self._compressed_cache else self.self_attn.v_head_dim
        self._cache_heads = 1 if self._compressed_cache else config.num_attention_heads
        self._cache_width = max(self._cache_key_dim, self._cache_value_dim)

        # BloomBee's backend accesses self_attn.num_heads / num_key_value_heads
        if not hasattr(self.self_attn, "num_heads"):
            self.self_attn.num_heads = config.num_attention_heads
        if not hasattr(self.self_attn, "num_key_value_heads"):
            self.self_attn.num_key_value_heads = config.num_key_value_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs
    ):
        batch_size, seq_length, _ = hidden_states.shape

        seq_length_with_past = seq_length
        past_key_values_length = 0

        past_key_value = layer_past

        if past_key_value is not None:
            pk, pv = past_key_value
            if pk.dtype != hidden_states.dtype or pk.device != hidden_states.device:
                pk = pk.to(device=hidden_states.device, dtype=hidden_states.dtype)
                pv = pv.to(device=hidden_states.device, dtype=hidden_states.dtype)
                past_key_value = (pk, pv)
            past_key_values_length = past_key_value[0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
            _past_key_value = self._reorder_cache_from_bloom(past_key_value, batch_size, past_key_values_length)
            past_key_value = make_past_kv_cache(
                _past_key_value[0], _past_key_value[1],
                layer_idx=self.layer_idx, seen_tokens=past_key_values_length,
            )
        elif use_cache:
            past_key_value = make_empty_kv_cache(self.layer_idx)

        # DeepSeek-V3's eager attention (same pattern as Qwen3's) does not add an
        # implicit causal mask when attention_mask=None -- it only adds a mask if
        # one is given. BloomBee wraps a bare DecoderLayer (no DeepseekV3Model in
        # front to build the mask), so build the additive causal mask here.
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
        elif attention_mask.dim() == 3:
            # BloomBee's backend builds the mask as [B, S, K]; DeepSeek-V3's
            # attention expects 4D [B, 1, S, K] so it broadcasts over the heads dim.
            attention_mask = attention_mask.unsqueeze(1)

        position_ids = kwargs.pop("position_ids", None)
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length,
                dtype=torch.long, device=hidden_states.device,
            ).unsqueeze(0).expand(batch_size, -1)

        position_embeddings = self._rotary_emb(hidden_states, position_ids)

        cache_position = torch.arange(
            past_key_values_length, past_key_values_length + seq_length,
            dtype=torch.long, device=hidden_states.device,
        )

        skip_keys = {'position_ids', 'attention_mask', 'use_cache', 'rotary_position_ids',
                     'position_embeddings', 'past_key_value', 'past_key_values', 'cache_position'}
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

    def _reorder_cache_from_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        """Convert BloomBee's stored KV back into DeepSeek-V3's native shapes.

        Recent Transformers stores single-head compressed KV and rotary keys;
        older versions store expanded multi-head K/V. Both use a shared padded
        width in BloomBee. Restore the native head count and remove padding.
        """
        key_states, value_states = key_value
        if key_states.dim() == 4:
            return (
                key_states[:, :self._cache_heads, :, :self._cache_key_dim],
                value_states[:, :self._cache_heads, :, :self._cache_value_dim],
            )

        # 3D case: key is [B*H, D, S], value is [B*H, S, D].
        key_states = key_states.permute(0, 2, 1)
        key_states = key_states.reshape(batch_size, self._cache_heads, seq_length, self._cache_width)
        value_states = value_states.reshape(batch_size, self._cache_heads, seq_length, self._cache_width)
        return (key_states[..., :self._cache_key_dim], value_states[..., :self._cache_value_dim])

    def _reorder_cache_to_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        """Pad both native cache tensors to the shared BloomBee storage width."""
        key_states, value_states = key_value
        key_states = F.pad(key_states, [0, self._cache_width - key_states.shape[-1]])
        value_states = F.pad(value_states, [0, self._cache_width - value_states.shape[-1]])
        value_states = value_states.reshape(batch_size * self._cache_heads, seq_length, self._cache_width)
        key_states = key_states.reshape(batch_size * self._cache_heads, seq_length, self._cache_width)
        key_states = key_states.permute(0, 2, 1)
        return (key_states, value_states)
