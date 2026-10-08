from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
    Glm4MoeLiteForCausalLM,
    Glm4MoeLiteModel,
    Glm4MoeLitePreTrainedModel,
    Glm4MoeLiteRMSNorm,
)

from bloombee.client.from_pretrained import FromPretrainedMixin
from bloombee.client.lm_head import LMHead
from bloombee.client.ptune import PTuneMixin
from bloombee.client.remote_generation import (
    RemoteGenerationMixin,
    RemotePastKeyValues,
)
from bloombee.client.remote_sequential import RemoteSequential
from bloombee.models.glm4_moe_lite.config import DistributedGlm4MoeLiteConfig
from bloombee.utils.auto_config import DefaultRevisionMixin
from bloombee.utils.hivemind_compat import DHT


class DistributedGlm4MoeLiteModel(
    DefaultRevisionMixin,
    FromPretrainedMixin,
    PTuneMixin,
    Glm4MoeLiteModel,
):
    """GLM-4 MoE Lite with decoder layers hosted by the BloomBee swarm."""

    _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = [r"^model\.layers\."]

    config_class = DistributedGlm4MoeLiteConfig

    def __init__(
        self,
        config: DistributedGlm4MoeLiteConfig,
        *,
        dht: Optional[DHT] = None,
    ):
        # Avoid Glm4MoeLiteModel.__init__, which would instantiate all 47
        # decoder layers locally.
        Glm4MoeLitePreTrainedModel.__init__(self, config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )

        self.norm = Glm4MoeLiteRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        with torch.device("cpu"):
            self.layers = RemoteSequential(config, dht=dht)

        self.requires_grad_(False)
        self.init_prompts(config)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[RemotePastKeyValues] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You must specify input_ids or inputs_embeds"
            )

        assert (
            attention_mask is None or (attention_mask == 1).all()
        ), f"Custom attention masks are not supported, {attention_mask=}"

        if cache_position is not None:
            assert position_ids is not None
            assert torch.all(torch.eq(cache_position, position_ids)).item()

        assert (
            position_ids is None
            or (position_ids[:, 1:] - position_ids[:, :-1] == 1).all()
        ), f"Non-consecutive position_ids are not supported, {position_ids=}"

        assert use_cache is None or use_cache
        assert not output_attentions
        assert not output_hidden_states
        assert return_dict is None or return_dict

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_prompts = (
            self.config.tuning_mode
            and "ptune" in self.config.tuning_mode
            and self.h.position == 0
        )

        if use_prompts:
            batch_size = inputs_embeds.shape[0]
            prompts, intermediate_prompts = self.get_prompt(batch_size)
            inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
        else:
            prompts = intermediate_prompts = None

        hidden_states = inputs_embeds
        output_shape = input_shape + (hidden_states.size(-1),)

        if past_key_values is None:
            past_key_values = RemotePastKeyValues()

        past_key_values.update_seen(hidden_states.size(1))

        hidden_states = self.layers(
            hidden_states,
            prompts=intermediate_prompts,
            hypo_ids=past_key_values.hypo_ids,
        )

        if use_prompts:
            hidden_states = hidden_states[:, self.pre_seq_len :]

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @property
    def word_embeddings(self):
        return self.embed_tokens

    @property
    def word_embeddings_layernorm(self):
        return nn.Identity()

    @property
    def h(self):
        return self.layers

    @property
    def ln_f(self):
        return self.norm


class DistributedGlm4MoeLiteForCausalLM(
    FromPretrainedMixin,
    RemoteGenerationMixin,
    Glm4MoeLiteForCausalLM,
):
    _keys_to_ignore_on_load_missing = (
        DistributedGlm4MoeLiteModel._keys_to_ignore_on_load_missing
    )
    _keys_to_ignore_on_load_unexpected = (
        DistributedGlm4MoeLiteModel._keys_to_ignore_on_load_unexpected
    )

    config_class = DistributedGlm4MoeLiteConfig
    _supports_cache_class = True

    def __init__(self, config: DistributedGlm4MoeLiteConfig):
        Glm4MoeLitePreTrainedModel.__init__(self, config)

        self.model = DistributedGlm4MoeLiteModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = LMHead(config)

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def mark_tied_weights_as_initialized(self, loading_info):
        return

    def tie_weights(self, missing_keys=None, recompute_mapping=True):
        if getattr(self.config, "tie_word_embeddings", False):
            embed = self.get_input_embeddings()
            if embed is not None and getattr(embed, "weight", None) is not None:
                self.lm_head.weight = embed.weight

    @property
    def transformer(self):
        return self.model
