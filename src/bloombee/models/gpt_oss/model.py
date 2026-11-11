"""GPT-OSS clients for remotely hosted decoder layers."""

import copy

import torch
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM, GptOssModel, GptOssPreTrainedModel

from bloombee.client.from_pretrained import FromPretrainedMixin
from bloombee.client.lm_head import LMHead
from bloombee.client.ptune import PTuneMixin
from bloombee.client.remote_generation import RemoteGenerationMixin
from bloombee.client.remote_sequential import RemoteSequential
from bloombee.models.gpt_oss.config import DistributedGptOssConfig
from bloombee.models.qwen3.model import DistributedQwen3ForCausalLM, DistributedQwen3Model


class _GptOssClientLoader(FromPretrainedMixin):
    @classmethod
    def from_pretrained(cls, model_name_or_path, *args, config=None, **kwargs):
        if config is None:
            config, kwargs = cls.config_class.from_pretrained(model_name_or_path, return_unused_kwargs=True, **kwargs)
        config = copy.deepcopy(config)
        quantization = getattr(config, "quantization_config", None)
        if isinstance(quantization, dict) and quantization.get("quant_method") == "mxfp4":
            # Embeddings, final norm and LM head are unquantized in official weights.
            # Do not initialize an expert quantizer for a client with no experts.
            del config.quantization_config
        return super().from_pretrained(model_name_or_path, *args, config=config, **kwargs)


class DistributedGptOssModel(_GptOssClientLoader, PTuneMixin, GptOssModel):
    config_class = DistributedGptOssConfig
    _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = [r"^model\.layers\."]

    def __init__(self, config, *, dht=None):
        count = config.num_hidden_layers
        config.num_hidden_layers = 0
        try:
            super().__init__(config)
        finally:
            config.num_hidden_layers = count
        with torch.device("cpu"):
            self.layers = RemoteSequential(config, dht=dht)
        self.requires_grad_(False)
        self.init_prompts(config)

    def forward(self, *args, output_router_logits=False, **kwargs):
        if output_router_logits:
            raise ValueError("Remote GPT-OSS inference does not return router logits")
        result = DistributedQwen3Model.forward(self, *args, **kwargs)
        return MoeModelOutputWithPast(**result)

    word_embeddings = DistributedQwen3Model.word_embeddings
    word_embeddings_layernorm = DistributedQwen3Model.word_embeddings_layernorm
    h = DistributedQwen3Model.h
    ln_f = DistributedQwen3Model.ln_f


class DistributedGptOssForCausalLM(_GptOssClientLoader, RemoteGenerationMixin, GptOssForCausalLM):
    config_class = DistributedGptOssConfig
    _keys_to_ignore_on_load_missing = DistributedGptOssModel._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedGptOssModel._keys_to_ignore_on_load_unexpected
    _supports_cache_class = True

    def __init__(self, config):
        GptOssPreTrainedModel.__init__(self, config)
        self.model = DistributedGptOssModel(config)
        self.lm_head = LMHead(config)
        self.vocab_size = config.vocab_size
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()

    prepare_inputs_for_generation = DistributedQwen3ForCausalLM.prepare_inputs_for_generation

    @property
    def transformer(self):
        return self.model
