"""GPT-OSS inference adapters for remotely hosted decoder layers."""

from bloombee.models.gpt_oss.block import WrappedGptOssBlock
from bloombee.models.gpt_oss.config import DistributedGptOssConfig
from bloombee.models.gpt_oss.model import DistributedGptOssForCausalLM, DistributedGptOssModel
from bloombee.utils.auto_config import register_model_classes

__all__ = [
    "DistributedGptOssConfig",
    "DistributedGptOssForCausalLM",
    "DistributedGptOssModel",
    "WrappedGptOssBlock",
]

register_model_classes(
    config=DistributedGptOssConfig,
    model=DistributedGptOssModel,
    model_for_causal_lm=DistributedGptOssForCausalLM,
)
