from typing import Optional

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import MoeModelOutputWithPast
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

        past_key_values.advance_seen(hidden_states.size(1))

        hidden_states = self.layers(
            hidden_states,
            prompts=intermediate_prompts,
            hypo_ids=past_key_values.hypo_ids,
        )

        if use_prompts:
            hidden_states = hidden_states[:, self.pre_seq_len :]

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
            router_logits=None,
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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=True,
        **kwargs,
    ):
        """
        Prepare incremental generation inputs for BloomBee's remote GLM cache.

        RemotePastKeyValues tracks how many tokens have already traversed the
        distributed decoder. During decode, send only tokens that have not
        previously been processed and keep position/cache positions aligned.
        """
        if past_key_values is not None:
            past_length = past_key_values.get_seq_length()

            # Keep only tokens that have not already traversed the remote model.
            if input_ids.shape[1] > past_length:
                input_ids = input_ids[:, past_length:]
            else:
                input_ids = input_ids[:, -1:]
        else:
            past_length = 0

        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

            # On cached decode only positions for the new tokens are needed.
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        else:
            position_ids = torch.arange(
                past_length,
                past_length + input_ids.shape[1],
                dtype=torch.long,
                device=input_ids.device,
            ).unsqueeze(0).expand(input_ids.shape[0], -1)

        # GLM's forward path explicitly checks that cache_position and
        # position_ids describe the same positions.
        cache_position = position_ids

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )

        return model_inputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ) -> dict:
        # BloomBee's remote cache tracks already-processed tokens. During
        # decode, trim input_ids so Transformers does not resend the full
        # prefill sequence to the remote swarm.
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = getattr(past_key_values, "_seen_tokens", None)

                if past_length is None:
                    past_length = cache_length

                if hasattr(past_key_values, "get_max_length"):
                    max_cache_length = past_key_values.get_max_length()
                elif hasattr(past_key_values, "get_max_cache_shape"):
                    max_cache_length = past_key_values.get_max_cache_shape()
                else:
                    max_cache_length = None
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[
                    :, -(attention_mask.shape[1] - past_length):
                ]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )

        return model_inputs

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
