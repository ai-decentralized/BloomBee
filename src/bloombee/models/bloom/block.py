"""
Bloom intermediate layer
Based on https://github.com/huggingface/transformers/commit/ca2a55e9dfb245527b5e1c954fec6ffbb7aef07b
See commit history for authorship.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bloom.modeling_bloom import (
    BloomAttention,
    BloomBlock,
    BloomConfig,
    BloomMLP,
    build_alibi_tensor,
    dropout_add,
)

from bloombee.utils.misc import is_dummy


class OptimizedBloomAttention(BloomAttention):
    """BloomAttention replacement that speaks BloomBee's tuple-based KV cache.

    TF 5.x's upstream ``BloomAttention.forward`` expects ``layer_past`` to be a
    ``transformers.cache_utils.Cache`` instance and calls ``layer_past.update()``
    to append. BloomBee's backend hands blocks ``(k, v)`` tensor tuples directly
    (see ``server/backend.py``), so the upstream path crashes. This subclass
    keeps the fused-qkv/linear weights from the parent but overrides ``forward``
    to do the tuple-cache concat inline, and returns ``(output, present)`` where
    ``present`` is a ``(k, v)`` tuple suitable for BloomBee's cache manager.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        batch_size, q_length, _ = hidden_states.shape
        fused_qkv = self.query_key_value(hidden_states)
        query_layer, key_layer, value_layer = self._reshape(fused_qkv)

        if layer_past is not None:
            past_key, past_value = layer_past
            # BloomBee backend stores cache as ``key=[B*H, D, S]``, ``value=[B*H, S, D]``;
            # reshape/permute back to ``[B, H, S, D]`` so we can concat along seq axis.
            if past_key.dim() == 3 and past_key.shape[-2] == self.head_dim:
                past_key = past_key.view(batch_size, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)
                past_value = past_value.view(batch_size, self.num_heads, -1, self.head_dim)
            # standard HF [B, H, S, D] cache format
            key_layer = torch.cat([past_key, key_layer], dim=2)
            value_layer = torch.cat([past_value, value_layer], dim=2)

        present = None

        # reshape qkv for further computations
        query_layer = query_layer.reshape(batch_size * self.num_heads, -1, self.head_dim)
        key_layer_bhd = key_layer.reshape(batch_size * self.num_heads, -1, self.head_dim).transpose(-1, -2)
        value_layer_bhd = value_layer.reshape(batch_size * self.num_heads, -1, self.head_dim)

        if use_cache:
            # BloomBee's memory_cache_manager expects kv in 3D layout:
            #   key=[B*H, D, S]  (i.e., head_dim before seq), value=[B*H, S, D].
            # We hand it ``present`` in exactly that shape so _write_kvs' assertion passes.
            present = (key_layer_bhd, value_layer_bhd)

        attention_scores = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer_bhd,
            beta=self.beta,
            alpha=self.inv_norm_factor,
        )

        attn_weights = attention_scores.view(batch_size, self.num_heads, q_length, -1)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_layer.dtype)
        attention_probs = self.attention_dropout(attention_probs)
        attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, -1)

        context_layer = torch.bmm(attention_probs_reshaped, value_layer_bhd)
        context_layer = self._merge_heads(context_layer)

        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)
        return output_tensor, present


class WrappedBloomBlock(BloomBlock):
    """BloomBlock wired into BloomBee's distributed backend.

    Two responsibilities:
      1. Build the alibi tensor and causal attention mask that BloomBee's
         backend doesn't provide, so the block can be called with just
         ``(hidden_states, layer_past=..., use_cache=...)``.
      2. Swap the upstream ``BloomAttention`` for ``OptimizedBloomAttention``
         so the block speaks tuple-KV instead of the TF 5.x ``Cache`` API.
    """

    def __init__(self, config: BloomConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.self_attention = OptimizedBloomAttention(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        alibi: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        batch_size, seq_length = hidden_states.shape[:2]

        # BloomBee cache_manager may hand us dummy tensors on the first decode step;
        # treat those as "no past" so upstream concat shortcuts don't blow up.
        if layer_past is not None and is_dummy(layer_past[0]):
            layer_past = None
        if layer_past is None:
            past_length = 0
        else:
            # Accept both cache layouts:
            #   - BloomBee backend tuples  [B, H, D, S] / [B, H, S, D]  (3D legacy or 4D modern)
            #   - Our own OptimizedBloomAttention output [B, H, S, D]
            head_dim = self.self_attention.head_dim
            pk = layer_past[0]
            if pk.dim() == 4:
                past_length = pk.shape[2] if pk.shape[-1] == head_dim else pk.shape[-1]
            else:
                past_length = pk.shape[-1] if pk.shape[-2] == head_dim else pk.shape[-2]
        seq_length_with_past = seq_length + past_length

        padding_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        if alibi is None:
            alibi = build_alibi_tensor(padding_mask, num_heads=self.num_heads, dtype=hidden_states.dtype)

        # Build [B, 1, q, kv] additive mask with -inf on future positions.
        causal_mask = torch.zeros(
            batch_size, 1, seq_length, seq_length_with_past, device=hidden_states.device, dtype=hidden_states.dtype
        )
        future = torch.triu(
            torch.ones(seq_length, seq_length_with_past, device=hidden_states.device, dtype=torch.bool),
            diagonal=past_length + 1,
        )
        causal_mask.masked_fill_(future, float("-inf"))

        # Call the upstream BloomBlock body with our OptimizedBloomAttention slot,
        # but we cannot delegate to super() because it passes layer_past positionally
        # expecting a Cache. Do the layer-norm + attention + MLP inline instead.
        layernorm_output = self.input_layernorm(hidden_states)
        residual = layernorm_output if self.apply_residual_connection_post_layernorm else hidden_states

        attn_output, present = self.self_attention(
            layernorm_output,
            residual,
            alibi=alibi,
            attention_mask=causal_mask,
            layer_past=layer_past,
            use_cache=use_cache,
        )

        layernorm_output = self.post_attention_layernorm(attn_output)
        residual = layernorm_output if self.apply_residual_connection_post_layernorm else attn_output
        output = self.mlp(layernorm_output, residual)

        if use_cache:
            return output, present
        return (output,)
