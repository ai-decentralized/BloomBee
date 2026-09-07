import os
from typing import Optional, Union

from transformers.models.deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention

from bloombee.client.config import ClientConfig
from bloombee.client.lm_head import LMHeadConfig
from bloombee.client.ptune import PTuneConfig
from bloombee.models.deepseekv3.block import WrappedDeepseekV3Block
from bloombee.utils.hivemind_compat import get_logger

logger = get_logger(__name__)


class DistributedDeepseekV3Config(DeepseekV3Config, ClientConfig, PTuneConfig, LMHeadConfig):
    block_class = WrappedDeepseekV3Block
    attn_class = DeepseekV3Attention
    block_prefix = "model.layers"

    @property
    def num_key_value_groups(self):
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def cache_head_dim(self):
        # Compressed MLA caches latent KV and rotary keys with unequal widths.
        if hasattr(DeepseekV3Attention, "expand_kv"):
            return max(self.kv_lora_rank, self.qk_rope_head_dim)
        # DeepSeek-V3's MLA gives keys width qk_head_dim (qk_nope_head_dim +
        # qk_rope_head_dim) and values width v_head_dim, which usually differ.
        # `config.head_dim` is already claimed by upstream transformers for the
        # rotary embedding's frequency dimension (qk_rope_head_dim), so it can't
        # double as the KV cache width. server/backend.py's
        # _head_dim_for_this_block() reads this attribute to size the shared
        # per-block cache tensor at qk_head_dim (the wider of the two); the
        # block wrapper zero-pads/truncates values to match on write/read.
        return self.qk_head_dim

    @property
    def cache_num_key_value_heads(self):
        return 1 if hasattr(DeepseekV3Attention, "expand_kv") else self.num_attention_heads

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
