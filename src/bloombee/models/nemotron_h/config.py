import os
from typing import Optional, Union

from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHAttention
from bloombee.models.nemotron_h.block import WrappedNemotronHBlock

from bloombee.client.config import ClientConfig
from bloombee.client.lm_head import LMHeadConfig
from bloombee.client.ptune import PTuneConfig
from bloombee.utils.hivemind_compat import get_logger


logger = get_logger(__name__)


class DistributedNemotronHConfig(
    NemotronHConfig,
    ClientConfig,
    PTuneConfig,
    LMHeadConfig,
):
    """
    BloomBee configuration for NVIDIA Nemotron-H models.

    Nemotron-H is a heterogeneous architecture whose layers may be
    linear-attention (Mamba-2), full-attention, MoE, or MLP blocks.
    """

    # Temporary. This will become WrappedNemotronHBlock once the
    # BloomBee-compatible hybrid cache wrapper is implemented.
    block_class = WrappedNemotronHBlock

    attn_class = NemotronHAttention
    block_prefix = "backbone.layers"

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
