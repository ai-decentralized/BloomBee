"""Distributed Gemma-4 (text-only) classes for BloomBee.

Targets the text tower of the multimodal `Gemma4ForConditionalGeneration`
checkpoints. The checkpoint ships vision-tower weights in the same
state_dict under `model.vision_*` / `model.embed_vision.*` / `model.audio_tower.*`
and the text tower under `model.language_model.*` — we have to ignore the
vision/audio keys and remap `model.language_model.X` onto our
`Gemma4ForCausalLM`-shaped `model.X` slots at load time.

HF has no `Gemma4ForSequenceClassification`, so we don't ship one.
"""

from typing import Optional

import torch
import torch.nn as nn
from hivemind import DHT
from hivemind.utils.logging import get_logger
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForCausalLM as _BaseCausalLM,
    Gemma4PreTrainedModel as _BasePreTrained,
    Gemma4TextModel as _BaseTextModel,
)

from bloombee.client.from_pretrained import FromPretrainedMixin
from bloombee.client.lm_head import LMHead
from bloombee.client.ptune import PTuneMixin
from bloombee.client.remote_generation import RemoteGenerationMixin, RemotePastKeyValues
from bloombee.client.remote_sequential import RemoteSequential
from bloombee.models.gemma4.config import DistributedGemma4Config
from bloombee.utils.auto_config import DefaultRevisionMixin

logger = get_logger(__name__)


# Patterns for state-dict keys that live in the published Gemma-4
# multimodal checkpoint but don't belong to the text tower we serve.
# Applied via `_keys_to_ignore_on_load_unexpected` so the loader doesn't
# crash when it encounters them.
_VISION_IGNORE_KEYS = [
    r"^model\.vision_.*",
    r"^model\.embed_vision\..*",
    r"^model\.audio_.*",
    r"^model\.embed_audio\..*",
    r"^model\.vision_tower\..*",
    r"^vision_.*",
    r"^embed_vision\..*",
]


# Rename map for the published multimodal checkpoint.
#
# The 31B-it release is saved by `Gemma4ForConditionalGeneration`, which
# wraps `Gemma4TextModel` at `self.model.language_model`. So every text
# weight in the state_dict begins with `model.language_model.` (e.g.
# `model.language_model.embed_tokens.weight`, `model.language_model.norm.weight`,
# `model.language_model.layers.0.self_attn.q_proj.weight`, ...).
#
# Our adapter inherits from `Gemma4ForCausalLM`, whose text tower lives
# at `self.model` — so we want `model.X` not `model.language_model.X`.
# The per-layer weights are loaded by the server (see
# `server/from_pretrained.py`, which now reads from
# `config.block_prefix = "model.language_model.layers"`). The client
# still needs to load `embed_tokens` + `norm`, so we strip the
# `language_model.` segment on the way in.
#
# Passed to HF's `from_pretrained(..., key_mapping=...)`, which applies
# the rename as a regex replace across every key in the checkpoint.
_GEMMA4_KEY_MAPPING = {
    # Strip `language_model.` out of the `model.` namespace.
    # HF's key_mapping treats this as a regex where the source pattern is
    # searched and the target replaces it in-place.
    r"^model\.language_model\.": "model.",
    # `Gemma4ForConditionalGeneration` has `lm_head` directly under root,
    # just like us, so nothing to remap for the head.
}


class DistributedGemma4Model(DefaultRevisionMixin, FromPretrainedMixin, PTuneMixin, _BaseTextModel):
    """Gemma4TextModel with remote transformer layers."""

    _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
    # After `key_mapping` flattens `model.language_model.*` to `model.*`,
    # per-layer keys look like `model.layers.<i>.*` — ignored here so the
    # embed/norm-only client doesn't complain about them.
    _keys_to_ignore_on_load_unexpected = [r"^model\.layers\."] + _VISION_IGNORE_KEYS

    config_class = DistributedGemma4Config

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # Checkpoint is saved by the multimodal Gemma4ForConditionalGeneration
        # under `model.language_model.*`. Flatten onto our text-only namespace
        # (`model.*`) before HF matches it against our parameter names.
        kwargs.setdefault("key_mapping", dict(_GEMMA4_KEY_MAPPING))
        return super().from_pretrained(*args, **kwargs)

    def __init__(self, config: DistributedGemma4Config, *, dht: Optional[DHT] = None):
        # Prevent the base class from instantiating the 60 full layers;
        # BloomBee serves them remotely.
        n_layer, config.num_hidden_layers = config.num_hidden_layers, 0
        super().__init__(config)
        assert len(self.layers) == 0
        config.num_hidden_layers = n_layer

        # See qwen3/model.py — TF 5.x wraps from_pretrained's __init__ in a
        # cuda device context, which breaks hivemind's CPU tensor setup.
        with torch.device("cpu"):
            self.layers = RemoteSequential(config, dht=dht)

        self.requires_grad_(False)
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

        if not isinstance(past_key_values, RemotePastKeyValues):
            past_key_values = RemotePastKeyValues()
        past_key_values.update_seen(hidden_states.size(1))

        hidden_states = self.layers(
            hidden_states,
            prompts=intermediate_prompts,
            hypo_ids=past_key_values.hypo_ids,
        )

        if use_prompts:
            hidden_states = hidden_states[:, self.pre_seq_len:]

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

    @property
    def word_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    @property
    def word_embeddings_layernorm(self) -> nn.Module:
        return nn.Identity()

    @property
    def h(self) -> RemoteSequential:
        return self.layers

    @property
    def ln_f(self) -> nn.Module:
        return self.norm


class DistributedGemma4ForCausalLM(FromPretrainedMixin, RemoteGenerationMixin, _BaseCausalLM):
    _keys_to_ignore_on_load_missing = DistributedGemma4Model._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedGemma4Model._keys_to_ignore_on_load_unexpected
    _supports_cache_class = True
    config_class = DistributedGemma4Config

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # Same remap as the text-only class, since the top-level checkpoint
        # still nests the text weights at `model.language_model.*`.
        kwargs.setdefault("key_mapping", dict(_GEMMA4_KEY_MAPPING))
        return super().from_pretrained(*args, **kwargs)

    def __init__(self, config: DistributedGemma4Config):
        _BasePreTrained.__init__(self, config)
        self.model = DistributedGemma4Model(config)
        self.lm_head = LMHead(config)
        self.post_init()

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ) -> dict:
        # Same rationale as Qwen3 / Llama override: TF 5.x's default
        # prepare_inputs_for_generation does prefill-shaped work every step.
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

            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
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
            if past_key_values:
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

    def mark_tied_weights_as_initialized(self, loading_info):
        """No-op: BloomBee manages tying manually below. Upstream TF 5.x would
        walk `_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}`
        and call `get_parameter("lm_head.weight")`, which fails on our
        LMHead stub where `self.weight = None` until bound to embed_tokens."""
        return

    def tie_weights(self, missing_keys=None, recompute_mapping=True):
        """Bind LMHead.weight to embed_tokens.weight. Gemma-4 sets
        tie_word_embeddings=True."""
        if getattr(self.config, "tie_word_embeddings", False):
            embed = self.get_input_embeddings()
            if embed is not None and getattr(embed, "weight", None) is not None:
                self.lm_head.weight = embed.weight

    def get_output_embeddings(self):
        return self.lm_head

    @property
    def transformer(self) -> DistributedGemma4Model:
        return self.model


# Gemma-4 has no `Gemma4ForSequenceClassification` upstream, so we don't
# expose a distributed variant. Registration in __init__.py passes None.
DistributedGemma4ForSequenceClassification = None
