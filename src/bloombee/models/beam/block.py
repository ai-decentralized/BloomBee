from typing import Optional, Tuple

import torch
from transformers.models.beam.configuration_beam import BeamConfig
from transformers.models.beam.modeling_beam import (
    BeamDecoderLayer,
    BeamRotaryEmbedding,
    compute_attention_scaling,
)

from bloombee.utils.cache_compat import (
    make_empty_kv_cache,
    make_past_kv_cache,
    read_kv_from_cache,
)


KVPast = Tuple[torch.Tensor, torch.Tensor]


def _build_layer_type_mask(
    *,
    layer_type: str,
    sliding_window: Optional[int],
    query_length: int,
    past_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    total_len = past_length + query_length
    neg_inf = torch.finfo(dtype).min

    if total_len == 0:
        return torch.zeros(
            (1, 1, query_length, 0),
            dtype=dtype,
            device=device,
        )

    q_pos = torch.arange(
        past_length,
        past_length + query_length,
        device=device,
    )
    k_pos = torch.arange(total_len, device=device)

    mask = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)

    if (
        layer_type == "sliding_attention"
        and sliding_window is not None
        and sliding_window > 0
    ):
        too_old = k_pos.unsqueeze(0) < (
            q_pos.unsqueeze(1) - (sliding_window - 1)
        )
        mask = mask | too_old

    additive = torch.where(
        mask,
        torch.tensor(neg_inf, dtype=dtype, device=device),
        torch.tensor(0.0, dtype=dtype, device=device),
    )

    return additive.unsqueeze(0).unsqueeze(0)


class WrappedBeamBlock(BeamDecoderLayer):
    """BloomBee-compatible Beam decoder layer."""

    def __init__(self, config: BeamConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = (
            config.sliding_window if self.is_sliding else None
        )

        self._rotary_emb = BeamRotaryEmbedding(config=config)

        if not hasattr(self.self_attn, "num_heads"):
            self.self_attn.num_heads = config.num_attention_heads

        if not hasattr(self.self_attn, "num_key_value_heads"):
            self.self_attn.num_key_value_heads = config.num_key_value_heads

    def _kv_from_bloom(
        self,
        key_value: KVPast,
        batch_size: int,
        seq_length: int,
    ) -> KVPast:
        key_states, value_states = key_value

        if key_states.dim() == 4:
            return key_states, value_states

        num_kv_heads = self.self_attn.num_key_value_heads
        head_dim = self.self_attn.head_dim

        key_states = key_states.permute(0, 2, 1).reshape(
            batch_size,
            num_kv_heads,
            seq_length,
            head_dim,
        )

        value_states = value_states.reshape(
            batch_size,
            num_kv_heads,
            seq_length,
            head_dim,
        )

        return key_states, value_states

    def _kv_to_bloom(
        self,
        key_value: KVPast,
        batch_size: int,
        seq_length: int,
    ) -> KVPast:
        key_states, value_states = key_value

        num_kv_heads = self.self_attn.num_key_value_heads
        head_dim = self.self_attn.head_dim

        key_states = key_states.reshape(
            batch_size * num_kv_heads,
            seq_length,
            head_dim,
        ).permute(0, 2, 1)

        value_states = value_states.reshape(
            batch_size * num_kv_heads,
            seq_length,
            head_dim,
        )

        return key_states, value_states

    def _apply(self, fn, recurse=True):
        # Beam RoPE inverse frequencies must stay fp32.
        rot = getattr(self, "_rotary_emb", None)
        rotary_buffers = {}

        if rot is not None:
            for layer_type in getattr(rot, "layer_types", ()):
                for suffix in ("inv_freq", "original_inv_freq"):
                    name = f"{layer_type}_{suffix}"
                    value = getattr(rot, name, None)

                    if (
                        value is not None
                        and torch.is_tensor(value)
                        and value.is_floating_point()
                    ):
                        rotary_buffers[name] = value

        out = super()._apply(fn, recurse=recurse)

        if rot is not None:
            for name, value in rotary_buffers.items():
                target = getattr(rot, name)
                setattr(
                    rot,
                    name,
                    value.to(
                        device=target.device,
                        dtype=torch.float32,
                    ),
                )

        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[KVPast] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        batch_size, seq_length, _ = hidden_states.shape

        past_length = 0
        past_key_values = None

        if layer_past is not None:
            pk, pv = layer_past

            if (
                pk.dtype != hidden_states.dtype
                or pk.device != hidden_states.device
            ):
                pk = pk.to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                pv = pv.to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )

            # BloomBee K shape: [B*Hkv, D, S]
            past_length = pk.shape[2]

            hf_k, hf_v = self._kv_from_bloom(
                (pk, pv),
                batch_size,
                past_length,
            )

            past_key_values = make_past_kv_cache(
                hf_k,
                hf_v,
                layer_idx=self.layer_idx,
                seen_tokens=past_length,
            )

        elif use_cache:
            past_key_values = make_empty_kv_cache(self.layer_idx)

        if attention_mask is None:
            attention_mask = _build_layer_type_mask(
                layer_type=self.layer_type,
                sliding_window=self.sliding_window,
                query_length=seq_length,
                past_length=past_length,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        else:
            if attention_mask.dtype == torch.bool:
                attention_mask = torch.zeros_like(
                    attention_mask,
                    dtype=hidden_states.dtype,
                ).masked_fill(
                    ~attention_mask,
                    torch.finfo(hidden_states.dtype).min,
                )

            attention_mask = attention_mask.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)

            if self.is_sliding:
                sliding_mask = _build_layer_type_mask(
                    layer_type=self.layer_type,
                    sliding_window=self.sliding_window,
                    query_length=seq_length,
                    past_length=past_length,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                attention_mask = torch.minimum(
                    attention_mask,
                    sliding_mask,
                )

        position_ids = kwargs.pop("position_ids", None)

        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                past_length + seq_length,
                dtype=torch.long,
                device=hidden_states.device,
            ).unsqueeze(0).expand(batch_size, -1)

        # Beam:
        #   sliding layers -> RoPE
        #   full layers    -> NoPE + position-dependent query scaling
        if self.is_sliding:
            position_embeddings = self._rotary_emb(
                hidden_states,
                position_ids,
                self.layer_type,
            )
            attention_scaling = None
        else:
            position_embeddings = None
            attention_scaling = compute_attention_scaling(position_ids)

        skip_keys = {
            "position_ids",
            "attention_mask",
            "use_cache",
            "past_key_value",
            "past_key_values",
            "position_embeddings",
            "attention_scaling",
            "cache_position",
            "rotary_position_ids",
        }

        extra_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in skip_keys
        }

        output_hidden = super().forward(
            hidden_states,
            *args,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            attention_scaling=attention_scaling,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **extra_kwargs,
        )

        if use_cache and past_key_values is not None:
            pk, pv = read_kv_from_cache(
                past_key_values,
                self.layer_idx,
            )

            if pk is not None:
                # BloomBee's outer cache appends only newly produced K/V.
                pk = pk[:, :, past_length:, :]
                pv = pv[:, :, past_length:, :]

                present = self._kv_to_bloom(
                    (pk, pv),
                    batch_size,
                    seq_length,
                )

                return output_hidden, present

        return output_hidden, None
