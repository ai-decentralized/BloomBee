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

        # Why we can't just call `super().from_pretrained(...)`:
        # We set `model_type = "gemma4"` on this class so BloomBee's
        # dispatcher matches the top-level gemma-4 checkpoint. But that
        # causes HF's loader to parse the TOP-LEVEL config dict (which
        # has model_type="gemma4" but no `num_hidden_layers` — those
        # live under `text_config`). HF then silently falls back to the
        # `Gemma4TextConfig` defaults (30 layers, etc.), which is wrong.
        #
        # Fix: call `Gemma4TextConfig.from_pretrained` explicitly (it
        # knows how to reach into `text_config`), then rebuild our
        # subclass from that loaded instance's dict.
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        parent_result = _Gemma4TextConfig.from_pretrained(
            model_name_or_path, *args, return_unused_kwargs=True, **kwargs,
        )
        parent_cfg, unused = parent_result if isinstance(parent_result, tuple) else (parent_result, {})

        # Rebuild as our subclass so BloomBee-specific fields are populated
        # (ClientConfig / PTuneConfig / LMHeadConfig defaults). Pass
        # parent_cfg's fields directly via to_dict + our dht_prefix.
        src = parent_cfg.to_dict()
        src["model_type"] = cls.model_type  # "gemma4", matches dispatcher
        src["dht_prefix"] = dht_prefix
        config = cls.from_dict(src, **unused)

        if config.pad_token_id is None:
            config.pad_token_id = 0

        if return_unused_kwargs:
            return config, unused
        return config
