import os
from typing import Optional, Union

from transformers.models.glm4_moe_lite.configuration_glm4_moe_lite import (
    Glm4MoeLiteConfig,
)
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
    Glm4MoeLiteAttention,
)

from bloombee.client.config import ClientConfig
from bloombee.client.lm_head import LMHeadConfig
from bloombee.client.ptune import PTuneConfig
from bloombee.models.glm4_moe_lite.block import WrappedGlm4MoeLiteBlock
from bloombee.utils.hivemind_compat import get_logger


logger = get_logger(__name__)


class DistributedGlm4MoeLiteConfig(
    Glm4MoeLiteConfig,
    ClientConfig,
    PTuneConfig,
    LMHeadConfig,
):
    model_type = "glm4_moe_lite"

    block_class = WrappedGlm4MoeLiteBlock
    attn_class = Glm4MoeLiteAttention
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
