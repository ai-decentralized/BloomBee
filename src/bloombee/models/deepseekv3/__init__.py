from bloombee.models.deepseekv3.block import WrappedDeepseekV3Block
from bloombee.models.deepseekv3.config import DistributedDeepseekV3Config
from bloombee.models.deepseekv3.model import (
    DistributedDeepseekV3ForCausalLM,
    DistributedDeepseekV3ForSequenceClassification,
    DistributedDeepseekV3Model,
)
from bloombee.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedDeepseekV3Config,
    model=DistributedDeepseekV3Model,
    model_for_causal_lm=DistributedDeepseekV3ForCausalLM,
    model_for_sequence_classification=DistributedDeepseekV3ForSequenceClassification,
)