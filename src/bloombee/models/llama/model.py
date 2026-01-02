from typing import Optional

import hivemind
import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.utils.logging import get_logger
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama import LlamaForCausalLM, LlamaForSequenceClassification, LlamaModel, LlamaPreTrainedModel

from bloombee.client.from_pretrained import FromPretrainedMixin
from bloombee.client.lm_head import LMHead
from bloombee.client.ptune import PTuneMixin
from bloombee.client.remote_generation import RemoteGenerationMixin, RemotePastKeyValues
from bloombee.client.remote_sequential import RemoteSequential
from bloombee.models.llama.config import DistributedLlamaConfig

logger = get_logger(__name__)


class DistributedLlamaModel(FromPretrainedMixin, PTuneMixin, LlamaModel):
    # DistributedLlamaModel 类，继承自 FromPretrainedMixin、PTuneMixin 和 LlamaModel
    """LlamaModel, but all transformer layers are hosted by the swarm"""

    _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = [r"^model\.layers\."]

    config_class = DistributedLlamaConfig

    def __init__(self, config: DistributedLlamaConfig, *, dht: Optional[hivemind.DHT] = None):
        n_layer, config.num_hidden_layers = config.num_hidden_layers, 0  # Prevent initialization
        super().__init__(config)
        assert len(self.layers) == 0
        config.num_hidden_layers = n_layer

        self.layers = RemoteSequential(config, dht=dht) # create RemoteSequential instance to manage distributed layers

        self.requires_grad_(False)  # Forbid accumulate grads for embeddings and layernorm
        self.init_prompts(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[RemotePastKeyValues] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> BaseModelOutputWithPast:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # The causal mask will be added on the server-side
        if cache_position is not None:
            assert position_ids is not None and torch.all(torch.eq(cache_position, position_ids)).item()
        assert (
            position_ids is None or (position_ids[:, 1:] - position_ids[:, :-1] == 1).all()
        ), f"Non-consecutive position_ids are not supported, {position_ids=}"
        assert use_cache is None or use_cache, f"{use_cache=} is not supported"
        assert not output_attentions, f"{output_attentions=} is not supported"
        assert not output_hidden_states, f"{output_hidden_states=} is not supported"
        assert return_dict is None or return_dict, f"{return_dict=} is not supported"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_prompts = self.config.tuning_mode and "ptune" in self.config.tuning_mode and self.layers.position == 0
        if use_prompts:
            batch_size = inputs_embeds.shape[0]
            prompts, intermediate_prompts = self.get_prompt(batch_size)
            inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
        else:
            prompts = intermediate_prompts = None

        hidden_states = inputs_embeds
        # print('model.py llama model inputs_embeds, ', inputs_embeds)  # Temporarily commented for cleaner debug output
        output_shape = input_shape + (hidden_states.size(-1),)
        
        # logger.info(f"hidden_states: {hidden_states}")

        hidden_states = self.layers(
            hidden_states,
            prompts=intermediate_prompts,
            hypo_ids=past_key_values.hypo_ids if past_key_values is not None else None,
            tree_attention_mask=attention_mask,
            kv_cache_position_ids=past_key_values.kv_cache_position_ids if past_key_values is not None else None,
            draft_tokens = input_ids,
            is_spec_decoding = past_key_values.is_spec_decoding if past_key_values is not None else None,
        )

        if past_key_values is None:
            past_key_values = RemotePastKeyValues()
        past_key_values.update_seen(hidden_states.size(1))

        # Remove prefix
        if use_prompts:
            hidden_states = hidden_states[:, self.pre_seq_len :]
            # Recalculate output_shape after removing prefix tokens
            output_shape = (hidden_states.size(0), hidden_states.size(1), hidden_states.size(2))

        # Add last hidden state
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

    @property
    def word_embeddings(self) -> nn.Embedding:  # For compatibility with RemoteGenerationMixin
        return self.embed_tokens

    @property
    def word_embeddings_layernorm(self) -> nn.Module:  # For compatibility with RemoteGenerationMixin
        return nn.Identity()

    @property
    def h(self) -> RemoteSequential:  # For compatibility with RemoteGenerationMixin
        return self.layers

    @property
    def ln_f(self) -> nn.Module:  # For compatibility with RemoteGenerationMixin
        return self.norm


class DistributedLlamaForCausalLM(FromPretrainedMixin, RemoteGenerationMixin, LlamaForCausalLM):
    _keys_to_ignore_on_load_missing = DistributedLlamaModel._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedLlamaModel._keys_to_ignore_on_load_unexpected
    _supports_cache_class = True

    config_class = DistributedLlamaConfig

    def __init__(self, config: DistributedLlamaConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = DistributedLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = LMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ) -> dict:
        """
        Prepare inputs for generation, handling incremental token generation properly.
        This method is crucial for correct embedding updates during generation.
        """
        # print(f"🔧 prepare_inputs_for_generation called:")
        # print(f"   input_ids.shape: {input_ids.shape if input_ids is not None else None}")
        # print(f"   past_key_values type: {type(past_key_values)}")
        
        # if past_key_values is not None:
        #     print(f"   past_key_values._seen_tokens: {past_key_values._seen_tokens}")
        
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values._seen_tokens
                max_cache_length = past_key_values.get_max_length()
                # print(f"   Cache case: cache_length={cache_length}, past_length={past_length}")
            else:
                cache_length = past_length = past_key_values[0][0].shape[2] if hasattr(past_key_values[0][0], 'shape') else 0
                max_cache_length = None
                # print(f"   Non-Cache case: past_length={past_length}")

            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
                # print(f"   Attention mask case: new input_ids.shape={input_ids.shape}")
            elif past_length < input_ids.shape[1]:
                original_shape = input_ids.shape
                input_ids = input_ids[:, past_length:]
                # print(f"   Past length case: {original_shape} -> {input_ids.shape}, kept tokens: {input_ids}")
            else:
                logger.debug(f"No truncation needed: past_length={past_length}, input_ids.shape[1]={input_ids.shape[1]}")

            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
            logger.debug("Using inputs_embeds for first generation step")
        else:
            model_inputs = {"input_ids": input_ids}
            # logger.debug(f"Using input_ids: {input_ids}")

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        # print(f"   Final model_inputs keys: {list(model_inputs.keys())}")
        return model_inputs

    def get_output_embeddings(self):
        return self.lm_head

    @property
    def transformer(self) -> DistributedLlamaModel:  # For compatibility with RemoteGenerationMixin
        return self.model


class DistributedLlamaForSequenceClassification(FromPretrainedMixin, LlamaForSequenceClassification):
    _keys_to_ignore_on_load_missing = DistributedLlamaModel._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedLlamaModel._keys_to_ignore_on_load_unexpected

    config_class = DistributedLlamaConfig

    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels

        self.model = DistributedLlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def transformer(self) -> DistributedLlamaModel:  # For compatibility with RemoteGenerationMixin
        return self.model
