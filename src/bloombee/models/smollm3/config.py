import os
from typing import Optional, Union

from transformers.models.smollm3 import SmolLM3Config
from transformers.models.smollm3.modeling_smollm3 import SmolLM3Attention

from bloombee.client.config import ClientConfig
from bloombee.client.lm_head import LMHeadConfig
from bloombee.client.ptune import PTuneConfig
from bloombee.models.smollm3.block import WrappedSmolLM3Block
from bloombee.utils.hivemind_compat import get_logger

logger = get_logger(__name__)


class DistributedSmolLM3Config(SmolLM3Config, ClientConfig, PTuneConfig, LMHeadConfig):
    model_type = "smollm3"

    block_class = WrappedSmolLM3Block
    attn_class = SmolLM3Attention
    block_prefix = "model.layers"

    # Cache descriptors are allocated at num_attention_heads; the block slices the
    # first num_key_value_heads (same convention as Qwen3).
    num_key_value_groups = 1

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, os.PathLike, None], *args, dht_prefix: Optional[str] = None, **kwargs
    ):
        loading_from_repo = model_name_or_path is not None and not os.path.isdir(model_name_or_path)
        if loading_from_repo and dht_prefix is None:
            dht_prefix = str(model_name_or_path)
            dht_prefix = dht_prefix.replace(".", "-")
            logger.info(f"Using DHT prefix: {dht_prefix}")
        result = super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        if config.pad_token_id is None:
            config.pad_token_id = 0
        return result
