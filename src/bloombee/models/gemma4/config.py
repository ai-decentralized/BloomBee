import os
from typing import Optional, Union

from transformers import AutoConfig
from transformers.models.gemma4.configuration_gemma4 import (
    Gemma4Config as _Gemma4MultimodalConfig,
    Gemma4TextConfig as _Gemma4TextConfig,
)
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention as _BaseAttention

from bloombee.client.config import ClientConfig
from bloombee.client.lm_head import LMHeadConfig
from bloombee.client.ptune import PTuneConfig
from bloombee.models.gemma4.block import WrappedGemma4Block
from bloombee.utils.hivemind_compat import get_logger

logger = get_logger(__name__)


class DistributedGemma4Config(_Gemma4TextConfig, ClientConfig, PTuneConfig, LMHeadConfig):
    """Text-only Gemma-4 config for the distributed swarm.

    We inherit from `Gemma4TextConfig` (the text sub-config) so the forward
    path in the block wrapper can use all Gemma-4 text-tower attributes
    natively, but we register under `model_type="gemma4"` so BloomBee's
    dispatcher resolves on the checkpoint's TOP-LEVEL model_type (the
    published checkpoints are the multimodal `Gemma4ForConditionalGeneration`
    bundles; their top-level config.json has `model_type=gemma4`).
    """

    # BloomBee's dispatcher keys on this string; must match what sits at
    # the top of a gemma-4-*.json. Do NOT leave this as "gemma4_text".
    model_type = "gemma4"

    block_class = WrappedGemma4Block
    attn_class = _BaseAttention
    block_prefix = "model.layers"

    num_key_value_groups = 1

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, os.PathLike, None],
        *args,
        dht_prefix: Optional[str] = None,
        **kwargs,
    ):
        loading_from_repo = model_name_or_path is not None and not os.path.isdir(model_name_or_path)
        if loading_from_repo and dht_prefix is None:
            dht_prefix = str(model_name_or_path).replace(".", "-")
            logger.info(f"Using DHT prefix: {dht_prefix}")

        # The published Gemma-4 checkpoints carry a multimodal top-level
        # config; `Gemma4TextConfig.from_pretrained` already knows how to
        # reach into `text_config` and pull out just the text sub-dict,
        # so we can delegate to the parent and it does the right thing.
        result = super().from_pretrained(
            model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs
        )
        config = result[0] if isinstance(result, tuple) else result

        # The parent load may have re-stamped `model_type` to `gemma4_text`
        # (because we inherit from Gemma4TextConfig). Force it back to the
        # top-level key the dispatcher expects.
        config.model_type = cls.model_type

        if config.pad_token_id is None:
            config.pad_token_id = 0
        return result
