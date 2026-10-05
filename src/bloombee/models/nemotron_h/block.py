from typing import Optional, Tuple

import torch
from transformers import DynamicCache
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHBlock

from bloombee.utils.cache_compat import (
    make_empty_kv_cache,
    make_past_kv_cache,
    read_kv_from_cache,
)


MambaPast = Tuple[torch.Tensor, torch.Tensor]
KVPast = Tuple[torch.Tensor, torch.Tensor]


class WrappedNemotronHBlock(NemotronHBlock):
    """
    BloomBee-compatible wrapper around a single Nemotron-H block.

    Supported state formats:

    linear_attention:
        (conv_state, recurrent_state)

    full_attention:
        BloomBee KV format:
        key   [B*Hkv, D, S]
        value [B*Hkv, S, D]

    moe / mlp:
        no persistent cache
    """

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)

        self.layer_idx = layer_idx
        self.block_type = config.layers_block_type[layer_idx]

        # BloomBee backend expects these attributes on attention modules.
        if self.block_type == "full_attention":
            if not hasattr(self.mixer, "num_heads"):
                self.mixer.num_heads = config.num_attention_heads

            if not hasattr(self.mixer, "num_key_value_heads"):
                self.mixer.num_key_value_heads = config.num_key_value_heads

    # ------------------------------------------------------------------
    # MAMBA / LINEAR ATTENTION
    # ------------------------------------------------------------------

    def _make_mamba_cache(
        self,
        layer_past: Optional[MambaPast],
    ) -> DynamicCache:
        cache = DynamicCache(config=self.config)

        if layer_past is None:
            return cache

        conv_state, recurrent_state = layer_past
        cache_layer = cache.layers[self.layer_idx]

        cache_layer.update_conv_state(
            conv_state,
            state_idx=0,
            conv_kernel_size=self.config.conv_kernel,
        )

        cache_layer.update_recurrent_state(
            recurrent_state,
            state_idx=0,
        )

        return cache

    # ------------------------------------------------------------------
    # FULL ATTENTION KV CONVERSION
    # ------------------------------------------------------------------

    def _kv_from_bloom(
        self,
        key_value: KVPast,
        batch_size: int,
        seq_length: int,
    ) -> KVPast:
        """
        BloomBee:
            K [B*Hkv, D, S]
            V [B*Hkv, S, D]

        HF:
            K [B, Hkv, S, D]
            V [B, Hkv, S, D]
        """

        key_states, value_states = key_value

        # Already in HF layout.
        if key_states.dim() == 4:
            return key_states, value_states

        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim

        key_states = key_states.permute(0, 2, 1)
        key_states = key_states.reshape(
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

        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim

        value_states = value_states.reshape(
            batch_size * num_kv_heads,
            seq_length,
            head_dim,
        )

        key_states = key_states.reshape(
            batch_size * num_kv_heads,
            seq_length,
            head_dim,
        )

        key_states = key_states.permute(0, 2, 1)

        return key_states, value_states

    # ------------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past=None,
        use_cache: bool = False,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        batch_size, seq_length, _ = hidden_states.shape

        # ==============================================================
        # MAMBA
        # ==============================================================

        if self.block_type == "linear_attention":
            # Optimized causal_conv1d requires conv_state dtype to match
            # the current Mamba input dtype. Keep the persistent BloomBee
            # cache FP32, but present conv_state to the HF kernel in the
            # compute dtype (BF16 for our deployment). Recurrent SSM state
            # remains FP32.
            if layer_past is not None:
                conv_state, recurrent_state = layer_past

                if (
                    conv_state.dtype != hidden_states.dtype
                    or conv_state.device != hidden_states.device
                ):
                    conv_state = conv_state.to(
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )

                if recurrent_state.device != hidden_states.device:
                    recurrent_state = recurrent_state.to(
                        device=hidden_states.device
                    )

                layer_past = (conv_state, recurrent_state)

            cache = self._make_mamba_cache(layer_past)

            # BloomBee normally builds an attention-score mask shaped
            # [batch, query_length, key_length]. Nemotron-H's Mamba mixer
            # does not consume a causal attention matrix; causality is
            # inherent in the recurrent state. It only accepts a 2-D
            # token/padding mask shaped [batch, sequence_length].
            #
            # Recover token validity from BloomBee's mask when one is
            # present. Only the current sequence belongs to this Mamba
            # invocation; cached history is already encoded in the
            # recurrent state.
            mamba_attention_mask = attention_mask

            if mamba_attention_mask is not None:
                if mamba_attention_mask.ndim >= 3:
                    current_mask = mamba_attention_mask[
                        ..., -seq_length:
                    ]

                    if current_mask.dtype == torch.bool:
                        mamba_attention_mask = current_mask.any(dim=-2)
                    else:
                        # BloomBee additive masks use 0 for allowed
                        # positions and negative values for masked ones.
                        mamba_attention_mask = (
                            current_mask >= 0
                        ).any(dim=-2)

                elif mamba_attention_mask.ndim == 2:
                    # Decode masks may still contain cached-history
                    # positions. Mamba only needs the current tokens.
                    mamba_attention_mask = mamba_attention_mask[
                        :, -seq_length:
                    ]

            output = super().forward(
                hidden_states,
                past_key_values=cache,
                attention_mask=mamba_attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
            )

            # Nemotron-H's GradientCheckpointingLayer/HF block returns
            # hidden_states directly for linear_attention. BloomBee,
            # however, uses the uniform (output, cache) interface.
            #
            # The Mamba cache is explicitly maintained by this wrapper,
            # so expose the updated conv/recurrent state as the second
            # return value.
            if not use_cache:
                return output, None

            cache_layer = cache.layers[self.layer_idx]

            present_state = (
                cache_layer.conv_states[0],
                cache_layer.recurrent_states[0],
            )

            return output, present_state

        # ==============================================================
        # FULL ATTENTION
        # ==============================================================

        if self.block_type == "full_attention":
            past_length = 0

            if layer_past is not None:
                pk, pv = layer_past

                if pk.dtype != hidden_states.dtype or pk.device != hidden_states.device:
                    pk = pk.to(
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                    pv = pv.to(
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )

                # BloomBee key layout is [B*Hkv, D, S].
                if pk.dim() == 3:
                    past_length = pk.shape[2]
                else:
                    past_length = pk.shape[2]

                hf_k, hf_v = self._kv_from_bloom(
                    (pk, pv),
                    batch_size,
                    past_length,
                )

                cache = make_past_kv_cache(
                    hf_k,
                    hf_v,
                    layer_idx=self.layer_idx,
                    seen_tokens=past_length,
                )

            elif use_cache:
                cache = make_empty_kv_cache(self.layer_idx)

            else:
                cache = None

            if position_ids is None:
                position_ids = torch.arange(
                    past_length,
                    past_length + seq_length,
                    device=hidden_states.device,
                    dtype=torch.long,
                ).unsqueeze(0).expand(batch_size, -1)

            cache_position = torch.arange(
                past_length,
                past_length + seq_length,
                device=hidden_states.device,
                dtype=torch.long,
            )

            output = super().forward(
                hidden_states,
                past_key_values=cache,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=use_cache,
            )

            if use_cache and cache is not None:
                pk, pv = read_kv_from_cache(
                    cache,
                    self.layer_idx,
                )

                if pk is not None:
                    # BloomBee stores only the newly generated part.
                    pk = pk[:, :, past_length:, :]
                    pv = pv[:, :, past_length:, :]

                    present = self._kv_to_bloom(
                        (pk, pv),
                        batch_size,
                        seq_length,
                    )

                    return output, present

            return output, None

        # ==============================================================
        # STATELESS BLOCKS
        # ==============================================================

        if self.block_type in ("moe", "mlp"):
            output = super().forward(
                hidden_states,
                past_key_values=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )

            return output, None

        raise ValueError(
            f"Unknown Nemotron-H block type: {self.block_type}"
        )
