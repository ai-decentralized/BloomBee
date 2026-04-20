from typing import Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

from bloombee.utils.cache_compat import make_past_kv_cache, make_empty_kv_cache, read_kv_from_cache

try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer as _BaseDecoderLayer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding as _RotaryEmbedding
    from transformers.models.qwen3 import Qwen3Config as _BaseBlockConfig
except ImportError:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer as _BaseDecoderLayer
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding as _RotaryEmbedding
    from transformers import Qwen2Config as _BaseBlockConfig


class WrappedQwen3Block(_BaseDecoderLayer):
    def __init__(self, config: _BaseBlockConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self._attn_implementation = config._attn_implementation
        self.sliding_window = getattr(config, "sliding_window", None)
        self.layer_idx = layer_idx
        self._rotary_emb = _RotaryEmbedding(config)

        # BloomBee's backend.py accesses self_attn.num_heads — add it for compatibility
        if not hasattr(self.self_attn, "num_heads"):
            self.self_attn.num_heads = config.num_attention_heads
        if not hasattr(self.self_attn, "num_key_value_heads"):
            self.self_attn.num_key_value_heads = config.num_key_value_heads

    def _apply(self, fn, recurse=True):
        # Keep rotary inv_freq buffers in fp32 across .to(dtype=fp16/bf16) calls.
        # HF's full model achieves this via _keep_in_fp32_modules, but BloomBee
        # loads a bare block and calls .to(dtype=fp16). Without this override the
        # buffer rounds to fp16 and rotary positions accumulate quantization
        # error, corrupting generation (symptom: repeated tokens in groups).
        out = super()._apply(fn, recurse=recurse)
        rot = getattr(self, "_rotary_emb", None)
        if rot is not None:
            for name in ("inv_freq", "original_inv_freq"):
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
        **kwargs
    ):
        batch_size, seq_length, _ = hidden_states.shape

        seq_length_with_past = seq_length
        past_key_values_length = 0

        past_key_value = layer_past

        if past_key_value is not None:
            # Fix dtype and device mismatch (analogous to Falcon fixes #7 and pipeline-parallel fix):
            # Cache may be float16 while hidden_states is bfloat16, and may be on cuda:0
            # while this block lives on a different GPU (pipeline parallelism).
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

        # tf 5.x eager/SDPA attention does NOT add an implicit causal mask when
        # attention_mask=None — eager_attention_forward only masks what you pass in.
        # BloomBee wraps a bare DecoderLayer (no Qwen3Model in front to build the mask),
        # so build the additive causal mask here.
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
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=hidden_states.device
            ).unsqueeze(0).expand(batch_size, -1)

        position_embeddings = self._rotary_emb(hidden_states, position_ids)

        # tf 5.x attention needs cache_position to know where to write new KV into the cache.
        # Without it, DynamicCache is not updated and read-back returns only stale past KV.
        cache_position = torch.arange(
            past_key_values_length, past_key_values_length + seq_length,
            dtype=torch.long, device=hidden_states.device,
        )

        # Filter kwargs that conflict with our explicit args
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
            **extra_kwargs
        )

        # Extract hidden_states from outputs (may be tensor or tuple)
        if isinstance(outputs, torch.Tensor):
            output_hidden = outputs
        elif isinstance(outputs, tuple):
            output_hidden = outputs[0]
        else:
            output_hidden = outputs

        if use_cache and past_key_value is not None:
            pk, pv = read_kv_from_cache(past_key_value, self.layer_idx)
            if pk is not None:
                # Extract only NEW tokens (BloomBee manages cumulative cache externally)
                pk = pk[:, :, past_key_values_length:, :]
                pv = pv[:, :, past_key_values_length:, :]
                present_key_value = self._reorder_cache_to_bloom((pk, pv), batch_size, seq_length)
                return (output_hidden, present_key_value)

        return (output_hidden, None)

    def _reorder_cache_from_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        key_states, value_states = key_value
        if key_states.dim() == 4:
            # select_cache() returns [B, H, S, D] where H = num_attention_heads.
            # The cache is allocated for num_attention_heads but only the first num_key_value_heads
            # heads are valid (GQA). Slice to only valid KV heads.
            nkv = self.self_attn.num_key_value_heads
            key_states = key_states[:, :nkv, :, :]
            value_states = value_states[:, :nkv, :, :]
            return (key_states, value_states)
        # 3D case: key is [B*H, D, S], value is [B*H, S, D]
        key_states = key_states.permute(0, 2, 1)  # [B*H, D, S] -> [B*H, S, D]
        key_states = key_states.view(
            batch_size, self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        value_states = value_states.view(*key_states.shape)
        return (key_states, value_states)

    def _reorder_cache_to_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        key_states, value_states = key_value
        # Use reshape (not view) since tensors from DynamicCache may be non-contiguous
        value_states = value_states.reshape(
            batch_size * self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        key_states = key_states.reshape(*value_states.shape)
        key_states = key_states.permute(0, 2, 1)
        return (key_states, value_states)
