from typing import Optional

import torch
import torch.nn as nn
from hivemind import DHT
from hivemind.utils.logging import get_logger
from transformers.modeling_outputs import BaseModelOutputWithPast
try:
    from transformers.models.qwen3 import (
        Qwen3ForCausalLM as _BaseCausalLM,
        Qwen3ForSequenceClassification as _BaseSeqCls,
        Qwen3Model as _BaseModel,
        Qwen3PreTrainedModel as _BasePreTrained,
    )
except ImportError:
    from transformers.models.qwen2 import (
        Qwen2ForCausalLM as _BaseCausalLM,
        Qwen2ForSequenceClassification as _BaseSeqCls,
        Qwen2Model as _BaseModel,
        Qwen2PreTrainedModel as _BasePreTrained,
    )

from bloombee.client.from_pretrained import FromPretrainedMixin
from bloombee.client.lm_head import LMHead
from bloombee.client.ptune import PTuneMixin
from bloombee.client.remote_generation import RemoteGenerationMixin, RemotePastKeyValues
from bloombee.client.remote_sequential import RemoteSequential
from bloombee.models.qwen3.config import DistributedQwen3Config
from bloombee.utils.auto_config import DefaultRevisionMixin

logger = get_logger(__name__)


class DistributedQwen3Model(DefaultRevisionMixin, FromPretrainedMixin, PTuneMixin, _BaseModel):
    """Qwen3Model (backed by Qwen2 classes), but all transformer layers are hosted by the swarm"""

    _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = [r"^model\.layers\."]

    config_class = DistributedQwen3Config

    def __init__(self, config: DistributedQwen3Config, *, dht: Optional[DHT] = None):
        n_layer, config.num_hidden_layers = config.num_hidden_layers, 0  # Prevent initialization
        super().__init__(config)
        assert len(self.layers) == 0
        config.num_hidden_layers = n_layer

        # Force CPU context: transformers from_pretrained wraps __init__ in torch.device('cuda'),
        # which hijacks torch.empty() in hivemind's DHT/MPFuture to create CUDA tensors.
        # share_memory_() only works on CPU tensors, so we must reset device context here.
        with torch.device('cpu'):
            self.layers = RemoteSequential(config, dht=dht)

        self.requires_grad_(False)  # Forbid accumulate grads for embeddings and layernorm
        self.init_prompts(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[RemotePastKeyValues] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
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
        assert (
            attention_mask is None or (attention_mask == 1).all()
        ), f"Custom attention masks are not supported, {attention_mask=}"
        if cache_position is not None:
            assert position_ids is not None and torch.all(torch.eq(cache_position, position_ids)).item()
        assert (
            position_ids is None or (position_ids[:, 1:] - position_ids[:, :-1] == 1).all()
        ), f"Non-consecutive position_ids are not supported, {position_ids=}"
        assert head_mask is None, f"Custom head masks are not supported, {head_mask=}"
        assert use_cache is None or use_cache, f"{use_cache=} is not supported"
        assert not output_attentions, f"{output_attentions=} is not supported"
        assert not output_hidden_states, f"{output_hidden_states=} is not supported"
        assert return_dict is None or return_dict, f"{return_dict=} is not supported"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_prompts = self.config.tuning_mode and "ptune" in self.config.tuning_mode and self.h.position == 0
        if use_prompts:
            batch_size = inputs_embeds.shape[0]
            prompts, intermediate_prompts = self.get_prompt(batch_size)
            inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
        else:
            prompts = intermediate_prompts = None

        hidden_states = inputs_embeds
        output_shape = input_shape + (hidden_states.size(-1),)

        if not isinstance(past_key_values, RemotePastKeyValues):
            past_key_values = RemotePastKeyValues()
        past_key_values.update_seen(hidden_states.size(1))

        hidden_states = self.layers(
            hidden_states,
            prompts=intermediate_prompts,
            hypo_ids=past_key_values.hypo_ids,
        )

        # Remove prefix
        if use_prompts:
            hidden_states = hidden_states[:, self.pre_seq_len :]

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
    def word_embeddings_layernorm(self) -> nn.Module:  # For compatibility with RemoteGenerationMixin in tests
        return nn.Identity()

    @property
    def h(self) -> RemoteSequential:  # For compatibility with RemoteGenerationMixin
        return self.layers

    @property
    def ln_f(self) -> nn.Module:  # For compatibility with RemoteGenerationMixin in tests
        return self.norm


class DistributedQwen3ForCausalLM(FromPretrainedMixin, RemoteGenerationMixin, _BaseCausalLM):
    _keys_to_ignore_on_load_missing = DistributedQwen3Model._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedQwen3Model._keys_to_ignore_on_load_unexpected
    _supports_cache_class = True
    config_class = DistributedQwen3Config

    def __init__(self, config: DistributedQwen3Config):
        _BasePreTrained.__init__(self, config)
        self.model = DistributedQwen3Model(config)
        self.lm_head = LMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def mark_tied_weights_as_initialized(self, loading_info):
        """Override TF 5.x's tied-weight bookkeeping.

        Qwen3 has tie_word_embeddings=True, so the upstream method iterates
        ``all_tied_weights_keys`` (which includes ``lm_head.weight``) and calls
        ``self.get_parameter("lm_head.weight")``. BloomBee's LMHead holds
        ``self.weight = None`` until it gets bound to ``embed_tokens.weight``
        during state-dict loading, so ``get_parameter`` raises AttributeError.

        BloomBee manages the tying manually, so we just skip the bookkeeping.
        """
        return

    def tie_weights(self, missing_keys=None, recompute_mapping=True):
        """Bind LMHead.weight to embed_tokens.weight when tie_word_embeddings=True.

        The upstream TF 5.x implementation also walks ``all_tied_weights_keys``
        and calls ``get_parameter`` on each, which fails on our LMHead stub.
        Do the minimum BloomBee needs: hand embed_tokens.weight to LMHead.
        """
        if getattr(self.config, "tie_word_embeddings", False):
            embed = self.get_input_embeddings()
            if embed is not None and getattr(embed, "weight", None) is not None:
                self.lm_head.weight = embed.weight

    def get_output_embeddings(self):
        return self.lm_head

    @property
    def transformer(self) -> DistributedQwen3Model:  # For compatibility with RemoteGenerationMixin
        return self.model


class DistributedQwen3ForSequenceClassification(FromPretrainedMixin, _BaseSeqCls):
    _keys_to_ignore_on_load_missing = DistributedQwen3Model._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedQwen3Model._keys_to_ignore_on_load_unexpected

    config_class = DistributedQwen3Config

    def __init__(self, config: DistributedQwen3Config):
        _BasePreTrained.__init__(self, config)
        self.num_labels = config.num_labels

        self.model = DistributedQwen3Model(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def transformer(self) -> DistributedQwen3Model:  # For compatibility with RemoteGenerationMixin
        return self.model
