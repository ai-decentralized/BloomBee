from transformers import AutoConfig

from bloombee.models.smollm3.block import WrappedSmolLM3Block
from bloombee.models.smollm3.config import DistributedSmolLM3Config
from bloombee.models.smollm3.model import (
    DistributedSmolLM3ForCausalLM,
    DistributedSmolLM3ForSequenceClassification,
    DistributedSmolLM3Model,
)
from bloombee.utils.auto_config import register_model_classes

# Register "smollm3" model_type with HuggingFace's AutoConfig.
# Skip if already registered (transformers has native smollm3 support).
try:
    AutoConfig.register("smollm3", DistributedSmolLM3Config)
except ValueError:
    pass  # Already known to transformers natively

register_model_classes(
    config=DistributedSmolLM3Config,
    model=DistributedSmolLM3Model,
    model_for_causal_lm=DistributedSmolLM3ForCausalLM,
    model_for_sequence_classification=DistributedSmolLM3ForSequenceClassification,
)
