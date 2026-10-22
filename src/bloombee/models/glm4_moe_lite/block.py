from typing import Optional, Tuple

import torch

from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
    Glm4MoeLiteDecoderLayer,
    Glm4MoeLiteRotaryEmbedding,
)

from bloombee.utils.cache_compat import (
    make_empty_kv_cache,
    make_past_kv_cache,
    read_kv_from_cache,
)


class WrappedGlm4MoeLiteBlock(Glm4MoeLiteDecoderLayer):
    """
    BloomBee wrapper for GLM-4.7-Flash decoder layers.

    GLM uses MLA compressed cache:
        kv_nope: [B, 1, S, kv_lora_rank]
        k_rot:   [B, 1, S, qk_rope_head_dim]
    """

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

        self.layer_idx = layer_idx
        self._rotary_emb = Glm4MoeLiteRotaryEmbedding(config=config)

        if not hasattr(self.self_attn, "num_heads"):
            self.self_attn.num_heads = config.num_attention_heads

        if not hasattr(self.self_attn, "num_key_value_heads"):
            self.self_attn.num_key_value_heads = config.num_key_value_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        batch_size, seq_length, _ = hidden_states.shape

        past_length = 0
        past_key_values = None

        if layer_past is not None:
            kv_nope, k_rot = layer_past

            if kv_nope.device != hidden_states.device or kv_nope.dtype != hidden_states.dtype:
                kv_nope = kv_nope.to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )

            if k_rot.device != hidden_states.device or k_rot.dtype != hidden_states.dtype:
                k_rot = k_rot.to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )

            past_length = kv_nope.shape[2]

            past_key_values = make_past_kv_cache(
                kv_nope,
                k_rot,
                layer_idx=self.layer_idx,
                seen_tokens=past_length,
            )

        elif use_cache:
            past_key_values = make_empty_kv_cache(self.layer_idx)

        position_ids = kwargs.pop("position_ids", None)

        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                past_length + seq_length,
                dtype=torch.long,
                device=hidden_states.device,
            ).unsqueeze(0).expand(batch_size, -1)

        position_embeddings = self._rotary_emb(
            hidden_states,
            position_ids=position_ids,
        )

        # BloomBee produces [B, Q, K] additive masks.
        # HF attention expects them broadcastable across attention heads.
        if attention_mask is not None and attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        skip_keys = {
            "position_ids",
            "attention_mask",
            "use_cache",
            "past_key_value",
            "past_key_values",
            "position_embeddings",
            "rotary_position_ids",
            "cache_position",
        }

        extra_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in skip_keys
        }

        output_hidden = super().forward(
            hidden_states,
            *args,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **extra_kwargs,
        )

        if use_cache and past_key_values is not None:
            kv_nope, k_rot = read_kv_from_cache(
                past_key_values,
                self.layer_idx,
            )

            if kv_nope is not None:
                # BloomBee stores cumulative cache externally, so return
                # only the cache corresponding to the new tokens.
                kv_nope = kv_nope[:, :, past_length:, :]
                k_rot = k_rot[:, :, past_length:, :]

                return output_hidden, (kv_nope, k_rot)

        return output_hidden, None
