"""GPT-OSS decoder block and checkpoint loading."""

import torch
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer, GptOssRotaryEmbedding

from bloombee.utils.cache_compat import make_empty_kv_cache, make_past_kv_cache, read_kv_from_cache


class WrappedGptOssBlock(GptOssDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self._rotary_emb = GptOssRotaryEmbedding(config)
        self.self_attn.num_heads = config.num_attention_heads
        self.self_attn.num_key_value_heads = config.num_key_value_heads

    def _apply(self, fn, recurse=True):
        # Bare layers do not inherit the full model's precision exclusions.
        rotary_buffers = dict(self._rotary_emb.named_buffers())
        result = super()._apply(fn, recurse=recurse)
        self.input_layernorm.float()
        self.post_attention_layernorm.float()
        for name, value in rotary_buffers.items():
            if value.is_floating_point():
                target = getattr(self._rotary_emb, name)
                self._rotary_emb.register_buffer(
                    name, value.to(device=target.device, dtype=torch.float32), persistent=False
                )
        return result

    def forward(self, hidden_states, *args, attention_mask=None, layer_past=None, use_cache=False, **kwargs):
        batch, length, _ = hidden_states.shape
        heads = self.self_attn.num_key_value_heads
        dim = self.self_attn.head_dim
        past_length = 0
        cache = make_empty_kv_cache(self.layer_idx) if use_cache else None
        if layer_past is not None:
            key, value = (tensor.to(hidden_states) for tensor in layer_past)
            past_length = key.shape[2]
            if key.ndim == 3:
                key = key.transpose(1, 2).reshape(batch, heads, past_length, dim)
                value = value.reshape(batch, heads, past_length, dim)
            else:
                key, value = key[:, :heads], value[:, :heads]
            cache = make_past_kv_cache(key, value, self.layer_idx, past_length)

        positions = kwargs.pop("position_ids", None)
        if positions is None:
            positions = torch.arange(past_length, past_length + length, device=hidden_states.device)[None]
        total_length = past_length + length
        key_positions = torch.arange(total_length, device=hidden_states.device)
        if attention_mask is None:
            query_positions = torch.arange(past_length, total_length, device=hidden_states.device)
            allowed = key_positions[None, :] <= query_positions[:, None]
            attention_mask = torch.zeros(
                (1, 1, length, total_length), device=hidden_states.device, dtype=hidden_states.dtype
            ).masked_fill(~allowed, float("-inf"))
        else:
            if attention_mask.dtype == torch.bool:
                attention_mask = torch.zeros_like(attention_mask, dtype=hidden_states.dtype).masked_fill(
                    ~attention_mask, float("-inf")
                )
            attention_mask = attention_mask.to(hidden_states)
            if attention_mask.ndim == 3:
                attention_mask = attention_mask.unsqueeze(1)
        if self.self_attn.sliding_window is not None:
            # Position IDs retain tree depth for speculative attention masks.
            all_positions = torch.cat(
                (
                    torch.arange(past_length, device=hidden_states.device)[None].expand(positions.shape[0], -1),
                    positions,
                ),
                dim=-1,
            )
            too_old = all_positions[:, None, :] <= positions[:, :, None] - self.self_attn.sliding_window
            attention_mask = attention_mask.masked_fill(too_old[:, None], float("-inf"))
        for name in (
            "rotary_position_ids",
            "past_key_value",
            "past_key_values",
            "position_embeddings",
            "cache_position",
        ):
            kwargs.pop(name, None)
        output = super().forward(
            hidden_states,
            *args,
            attention_mask=attention_mask,
            position_ids=positions,
            past_key_values=cache,
            use_cache=use_cache,
            position_embeddings=self._rotary_emb(hidden_states, positions),
            **kwargs,
        )
        present = None
        if use_cache:
            key, value = read_kv_from_cache(cache, self.layer_idx)
            key = key[:, :, -length:].reshape(batch * heads, length, dim).transpose(1, 2)
            value = value[:, :, -length:].reshape(batch * heads, length, dim)
            present = (key, value)
        return output, present

    def load_checkpoint_state(self, state_dict, dtype):
        """Decode official MXFP4 expert tensors one layer at a time on CPU."""
        from transformers.integrations.mxfp4 import convert_moe_packed_tensors

        for projection in ("gate_up_proj", "down_proj"):
            prefix = f"mlp.experts.{projection}"
            if prefix + "_blocks" in state_dict:
                state_dict[prefix] = convert_moe_packed_tensors(
                    state_dict.pop(prefix + "_blocks"),
                    state_dict.pop(prefix + "_scales"),
                    dtype=dtype,
                    rows_per_chunk=65536,
                )
        # Check all learned parameters: strict=False must not leave random experts.
        expected = set(dict(self.named_parameters()))
        missing = expected - state_dict.keys()
        unexpected = state_dict.keys() - self.state_dict().keys()
        if missing or unexpected:
            raise ValueError(f"Invalid GPT-OSS checkpoint: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
        self.to(dtype=dtype)
        self.load_state_dict(state_dict, strict=False)
        return self
