from bloombee.models.nemotron_h.block import WrappedNemotronHBlock
from bloombee.models.nemotron_h.config import DistributedNemotronHConfig
from bloombee.models.nemotron_h.model import (
    DistributedNemotronHForCausalLM,
    DistributedNemotronHModel,
)
from bloombee.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedNemotronHConfig,
    model=DistributedNemotronHModel,
    model_for_causal_lm=DistributedNemotronHForCausalLM,
)
