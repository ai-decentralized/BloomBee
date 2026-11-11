"""Distributed GPT-OSS configuration."""

import os

from transformers.models.gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention

from bloombee.client.config import ClientConfig
from bloombee.client.lm_head import LMHeadConfig
from bloombee.client.ptune import PTuneConfig
from bloombee.models.gpt_oss.block import WrappedGptOssBlock


class DistributedGptOssConfig(GptOssConfig, ClientConfig, PTuneConfig, LMHeadConfig):
    block_class = WrappedGptOssBlock
    attn_class = GptOssAttention
    block_prefix = "model.layers"

    @property
    def num_key_value_groups(self):
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_pretrained(cls, model_name_or_path, *args, dht_prefix=None, **kwargs):
        if dht_prefix is None and model_name_or_path is not None and not os.path.isdir(model_name_or_path):
            dht_prefix = str(model_name_or_path).replace(".", "-")
        result = super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        quantization = getattr(config, "quantization_config", None)
        if isinstance(quantization, dict) and quantization.get("quant_method") == "mxfp4":
            # Remote clients contain no expert modules; workers decode each block.
            config.quantization_config = dict(quantization, dequantize=True)
        return result
