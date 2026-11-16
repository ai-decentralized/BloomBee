from typing import Optional

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.nemotron_h import (
    NemotronHForCausalLM,
    NemotronHModel,
    NemotronHPreTrainedModel,
)
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHRMSNorm

from bloombee.client.from_pretrained import FromPretrainedMixin
from bloombee.client.lm_head import LMHead
from bloombee.client.ptune import PTuneMixin
from bloombee.client.remote_generation import RemoteGenerationMixin, RemotePastKeyValues
from bloombee.client.remote_sequential import RemoteSequential
from bloombee.models.nemotron_h.config import DistributedNemotronHConfig
from bloombee.utils.auto_config import DefaultRevisionMixin
from bloombee.utils.hivemind_compat import DHT, get_logger

logger = get_logger(__name__)


class DistributedNemotronHModel(
    DefaultRevisionMixin,
    FromPretrainedMixin,
    PTuneMixin,
    NemotronHModel,
):
    """
    NemotronHModel with all Nemotron-H blocks hosted by the BloomBee swarm.

    Unlike several Transformer architectures, Nemotron-H cannot temporarily set
    config.num_hidden_layers = 0 because that value is derived from
    layers_block_type. Therefore we intentionally bypass NemotronHModel.__init__
    and initialize only the client-side modules here.
    """

    _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = [r"^model\.layers\."]

    config_class = DistributedNemotronHConfig

    def __init__(
        self,
        config: DistributedNemotronHConfig,
        *,
        dht: Optional[DHT] = None,
    ):
        # Do NOT call NemotronHModel.__init__ here: it would instantiate all
        # 52 NemotronHBlock objects locally.
        NemotronHPreTrainedModel.__init__(self, config)

        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.norm_f = NemotronHRMSNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

        # Force CPU context because from_pretrained may initialize the model
        # while a CUDA device context is active.
        with torch.device("cpu"):
            self.layers = RemoteSequential(config, dht=dht)

        self.requires_grad_(False)
        self.init_prompts(config)

        # Initialize the local embedding/norm modules.
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
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds"
            )

        # BloomBee constructs the architecture-specific causal/recurrent masks
        # on the server side.
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

        assert use_cache is None or use_cache, f"{use_cache=} is not supported"
        assert not output_attentions, f"{output_attentions=} is not supported"
        assert not output_hidden_states, f"{output_hidden_states=} is not supported"
        assert return_dict is None or return_dict, f"{return_dict=} is not supported"

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

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

        hidden_states = self.norm_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
            router_logits=None,
        )

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    @property
    def word_embeddings(self) -> nn.Embedding:
        """Compatibility with RemoteGenerationMixin."""
        return self.embeddings

    @property
    def word_embeddings_layernorm(self) -> nn.Module:
        """Compatibility with RemoteGenerationMixin tests."""
        return nn.Identity()

    @property
    def h(self) -> RemoteSequential:
        """Compatibility with RemoteGenerationMixin."""
        return self.layers

    @property
    def ln_f(self) -> nn.Module:
        """Compatibility with BloomBee client model conventions."""
        return self.norm_f


class DistributedNemotronHForCausalLM(
    FromPretrainedMixin,
    RemoteGenerationMixin,
    NemotronHForCausalLM,
):
    _keys_to_ignore_on_load_missing = (
        DistributedNemotronHModel._keys_to_ignore_on_load_missing
    )
    _keys_to_ignore_on_load_unexpected = (
        DistributedNemotronHModel._keys_to_ignore_on_load_unexpected
    )

    config_class = DistributedNemotronHConfig
    _supports_cache_class = True

    _checkpoint_conversion_mapping = {
        r"^backbone\.embeddings\.": "model.embeddings.",
        r"^backbone\.norm_f\.": "model.norm_f.",
    }

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # NVIDIA's Nemotron-3.5 checkpoint stores the client-side embedding
        # and final norm under `backbone.*`, while BloomBee's distributed
        # CausalLM owns them under `model.*`.
        key_mapping = kwargs.pop("key_mapping", None)
        if key_mapping is None:
            key_mapping = dict(cls._checkpoint_conversion_mapping)
        else:
            key_mapping = {
                **cls._checkpoint_conversion_mapping,
                **key_mapping,
            }

        return super().from_pretrained(
            *args,
            key_mapping=key_mapping,
            **kwargs,
        )

    def __init__(self, config: DistributedNemotronHConfig):
        # Avoid NemotronHForCausalLM.__init__, because it would construct a
        # complete local NemotronHModel including all transformer/Mamba blocks.
        NemotronHPreTrainedModel.__init__(self, config)

        self.model = DistributedNemotronHModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = LMHead(config)

        self.post_init()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ) -> dict:
        # BloomBee's remote cache tracks already-processed tokens. Trim
        # input_ids during decode so Transformers 5.x does not repeatedly
        # resend the full prefill sequence to the remote swarm.
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
    def transformer(self) -> DistributedNemotronHModel:
        """Compatibility with RemoteGenerationMixin."""
        return self.model
