from bloombee.models.glm4_moe_lite.block import WrappedGlm4MoeLiteBlock
from bloombee.models.glm4_moe_lite.config import DistributedGlm4MoeLiteConfig
from bloombee.models.glm4_moe_lite.model import (
    DistributedGlm4MoeLiteForCausalLM,
    DistributedGlm4MoeLiteModel,
)
from bloombee.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedGlm4MoeLiteConfig,
    model=DistributedGlm4MoeLiteModel,
    model_for_causal_lm=DistributedGlm4MoeLiteForCausalLM,
)
