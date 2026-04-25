"""Gemma-4 adapter — text tower only.

We target the `Gemma4ForConditionalGeneration` checkpoints (e.g. the
31B-it release), but load only the text sub-config (`Gemma4TextConfig`)
and ignore vision-tower weights at load time.

Gemma-4 differs from Qwen3 in three non-trivial ways we have to honor:
 1. `layer_types` alternates sliding_attention + full_attention.
 2. Full-attention layers use `global_head_dim` + fewer KV heads than
    sliding layers, and have `attention_k_eq_v=True` (V aliases K).
 3. Two RoPE variants live inside one `Gemma4TextRotaryEmbedding`;
    layers pick theirs by string key.
"""

from transformers import AutoConfig

from bloombee.models.gemma4.block import WrappedGemma4Block
from bloombee.models.gemma4.config import DistributedGemma4Config
from bloombee.models.gemma4.model import (
    DistributedGemma4ForCausalLM,
    DistributedGemma4ForSequenceClassification,
    DistributedGemma4Model,
)
from bloombee.utils.auto_config import register_model_classes

# `gemma4` is TF-5.x-native; register only if upstream didn't already.
try:
    AutoConfig.register("gemma4", DistributedGemma4Config)
except ValueError:
    pass

register_model_classes(
    config=DistributedGemma4Config,
    model=DistributedGemma4Model,
    model_for_causal_lm=DistributedGemma4ForCausalLM,
    model_for_sequence_classification=DistributedGemma4ForSequenceClassification,
)
