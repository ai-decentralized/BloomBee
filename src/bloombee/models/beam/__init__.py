from bloombee.models.beam.block import WrappedBeamBlock
from bloombee.models.beam.config import DistributedBeamConfig
from bloombee.models.beam.model import (
    DistributedBeamForCausalLM,
    DistributedBeamModel,
)
from bloombee.utils.auto_config import register_model_classes


register_model_classes(
    config=DistributedBeamConfig,
    model=DistributedBeamModel,
    model_for_causal_lm=DistributedBeamForCausalLM,
)
