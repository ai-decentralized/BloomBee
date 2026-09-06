"""GPT-OSS inference adapters for remotely hosted decoder layers."""

import copy
import os

import torch
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
    GptOssDecoderLayer,
    GptOssForCausalLM,
    GptOssModel,
    GptOssPreTrainedModel,
    GptOssRotaryEmbedding,
)

from bloombee.client.config import ClientConfig
from bloombee.client.from_pretrained import FromPretrainedMixin
from bloombee.client.lm_head import LMHead, LMHeadConfig
from bloombee.client.ptune import PTuneConfig, PTuneMixin
from bloombee.client.remote_generation import RemoteGenerationMixin
from bloombee.client.remote_sequential import RemoteSequential
from bloombee.models.qwen3.model import DistributedQwen3ForCausalLM, DistributedQwen3Model
from bloombee.utils.auto_config import register_model_classes
from bloombee.utils.cache_compat import make_empty_kv_cache, make_past_kv_cache, read_kv_from_cache

__all__ = [
    "DistributedGptOssConfig",
    "DistributedGptOssForCausalLM",
    "DistributedGptOssModel",
    "WrappedGptOssBlock",
]


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


class DistributedGptOssConfig(GptOssConfig, ClientConfig, PTuneConfig, LMHeadConfig):
    block_class = WrappedGptOssBlock
    attn_class = GptOssAttention
    block_prefix = "model.layers"

    @property
    def num_key_value_groups(self):
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_pretrained(cls, model_name_or_path, *args, dht_prefix=None, **kwargs):
        if dht_prefix is None and model_name_or_path is not None and not os.path.isdir(model_name_or_path):
            dht_prefix = str(model_name_or_path).replace(".", "-")
        result = super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        quantization = getattr(config, "quantization_config", None)
        if isinstance(quantization, dict) and quantization.get("quant_method") == "mxfp4":
            # Remote clients contain no expert modules; workers decode each block.
            config.quantization_config = dict(quantization, dequantize=True)
        return result


class _GptOssClientLoader(FromPretrainedMixin):
    @classmethod
    def from_pretrained(cls, model_name_or_path, *args, config=None, **kwargs):
        if config is None:
            config, kwargs = cls.config_class.from_pretrained(model_name_or_path, return_unused_kwargs=True, **kwargs)
        config = copy.deepcopy(config)
        quantization = getattr(config, "quantization_config", None)
        if isinstance(quantization, dict) and quantization.get("quant_method") == "mxfp4":
            # Embeddings, final norm and LM head are unquantized in official weights.
            # Do not initialize an expert quantizer for a client with no experts.
            del config.quantization_config
        return super().from_pretrained(model_name_or_path, *args, config=config, **kwargs)


class DistributedGptOssModel(_GptOssClientLoader, PTuneMixin, GptOssModel):
    config_class = DistributedGptOssConfig
    _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = [r"^model\.layers\."]

    def __init__(self, config, *, dht=None):
        count = config.num_hidden_layers
        config.num_hidden_layers = 0
        try:
            super().__init__(config)
        finally:
            config.num_hidden_layers = count
        with torch.device("cpu"):
            self.layers = RemoteSequential(config, dht=dht)
        self.requires_grad_(False)
        self.init_prompts(config)

    def forward(self, *args, output_router_logits=False, **kwargs):
        if output_router_logits:
            raise ValueError("Remote GPT-OSS inference does not return router logits")
        result = DistributedQwen3Model.forward(self, *args, **kwargs)
        return MoeModelOutputWithPast(**result)

    word_embeddings = DistributedQwen3Model.word_embeddings
    word_embeddings_layernorm = DistributedQwen3Model.word_embeddings_layernorm
    h = DistributedQwen3Model.h
    ln_f = DistributedQwen3Model.ln_f


class DistributedGptOssForCausalLM(_GptOssClientLoader, RemoteGenerationMixin, GptOssForCausalLM):
    config_class = DistributedGptOssConfig
    _keys_to_ignore_on_load_missing = DistributedGptOssModel._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedGptOssModel._keys_to_ignore_on_load_unexpected
    _supports_cache_class = True

    def __init__(self, config):
        GptOssPreTrainedModel.__init__(self, config)
        self.model = DistributedGptOssModel(config)
        self.lm_head = LMHead(config)
        self.vocab_size = config.vocab_size
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()

    prepare_inputs_for_generation = DistributedQwen3ForCausalLM.prepare_inputs_for_generation

    @property
    def transformer(self):
        return self.model


register_model_classes(
    config=DistributedGptOssConfig,
    model=DistributedGptOssModel,
    model_for_causal_lm=DistributedGptOssForCausalLM,
)
