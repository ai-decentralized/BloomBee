import os
from typing import Optional, Union

from hivemind import get_logger
from transformers.models.bloom import BloomConfig
from transformers.models.bloom.modeling_bloom import BloomAttention

from bloombee.client.config import ClientConfig
from bloombee.client.lm_head import LMHeadConfig
from bloombee.client.ptune import PTuneConfig
from bloombee.models.bloom.block import WrappedBloomBlock

logger = get_logger(__name__)


class DistributedBloomConfig(BloomConfig, ClientConfig, PTuneConfig, LMHeadConfig):
    block_class = WrappedBloomBlock
    attn_class = BloomAttention
    block_prefix = "h"

    num_key_value_groups = 1

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, os.PathLike, None], *args, dht_prefix: Optional[str] = None, **kwargs
    ):
        logger.info("Make sure you follow the BLOOM terms of use: https://bit.ly/bloom-license")

        loading_from_repo = model_name_or_path is not None and not os.path.isdir(model_name_or_path)
        if loading_from_repo and dht_prefix is None:
            # We need "-bloombee" for backward compatibility with BloomBee < 1.2.0
            dht_prefix = str(model_name_or_path) + "-bloombee"
            dht_prefix = dht_prefix.replace(".", "-")
            logger.info(f"Using DHT prefix: {dht_prefix}")
        return super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
