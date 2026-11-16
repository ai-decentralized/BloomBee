import os
from typing import Optional, Union

from transformers.models.beam.configuration_beam import BeamConfig
from transformers.models.beam.modeling_beam import BeamAttention

from bloombee.client.config import ClientConfig
from bloombee.client.lm_head import LMHeadConfig
from bloombee.client.ptune import PTuneConfig
from bloombee.models.beam.block import WrappedBeamBlock
from bloombee.utils.hivemind_compat import get_logger


logger = get_logger(__name__)


class DistributedBeamConfig(
    BeamConfig,
    ClientConfig,
    PTuneConfig,
    LMHeadConfig,
):
    block_class = WrappedBeamBlock
    attn_class = BeamAttention
    block_prefix = "model.layers"

    @property
    def num_key_value_groups(self):
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, os.PathLike, None],
        *args,
        dht_prefix: Optional[str] = None,
        **kwargs,
    ):
        loading_from_repo = (
            model_name_or_path is not None
            and not os.path.isdir(model_name_or_path)
        )

        if loading_from_repo and dht_prefix is None:
            dht_prefix = str(model_name_or_path).replace(".", "-")
            logger.info(f"Using DHT prefix: {dht_prefix}")

        return super().from_pretrained(
            model_name_or_path,
            *args,
            dht_prefix=dht_prefix,
            **kwargs,
        )
