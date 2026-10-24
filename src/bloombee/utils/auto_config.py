import os
from dataclasses import dataclass
from typing import Optional, Type, Union

from transformers import AutoConfig, PretrainedConfig, PreTrainedModel

from bloombee.utils.hivemind_compat import get_logger
from bloombee.utils.hf_auth import always_needs_auth

logger = get_logger(__name__)


@dataclass
class _ModelClasses:
    config: Type[PretrainedConfig]
    model: Optional[Type[PreTrainedModel]] = None
    model_for_causal_lm: Optional[Type[PreTrainedModel]] = None
    model_for_speculative: Optional[Type[PreTrainedModel]] = None
    model_for_sequence_classification: Optional[Type[PreTrainedModel]] = None


_CLASS_MAPPING = {}  # Populated by bloombee.models.* subpackages with register_model_classes()


def register_model_classes(*, config: Type[PretrainedConfig], **kwargs):
    assert issubclass(config, PretrainedConfig)
    assert config.model_type not in _CLASS_MAPPING, f"Model type {config.model_type} is already registered"

    _CLASS_MAPPING[config.model_type] = _ModelClasses(config=config, **kwargs)


class _AutoDistributedBase:
    _mapping_field = None  # Should be defined in child classes

    @classmethod
    def from_pretrained(cls, model_name_or_path: Union[str, os.PathLike, None], *args, **kwargs) -> PretrainedConfig:
        if (
            always_needs_auth(model_name_or_path)
            and kwargs.get("token") is None
            and kwargs.get("use_auth_token") is None
        ):
            kwargs["use_auth_token"] = True

        config = AutoConfig.from_pretrained(model_name_or_path, *args, **kwargs)
        if config.model_type not in _CLASS_MAPPING:
            raise ValueError(f"BloomBee does not support model type {config.model_type}")

        proper_cls = getattr(_CLASS_MAPPING[config.model_type], cls._mapping_field)
        if proper_cls is None:
            raise ValueError(f"BloomBee does not have {cls.__name__} for model type {config.model_type}")

        return proper_cls.from_pretrained(model_name_or_path, *args, **kwargs)


class DefaultRevisionMixin:
    """Optional default revision overrides for HuggingFace ids.

    Falcon used to be pinned to old in-library `.bin` commits because TII
    briefly reverted `main`. Those pins now break Transformers 5 / recent
    torch, which refuse `torch.load` on the `.bin` shards. Hub `main` ships
    safetensors, so we no longer override the revision. Callers can still
    pass an explicit `revision=`.
    """

    DEFAULT_REVISIONS: dict[str, str] = {}

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, os.PathLike, None], *args, revision: Optional[str] = None, **kwargs
    ):
        if revision is None and model_name_or_path in cls.DEFAULT_REVISIONS:
            revision = cls.DEFAULT_REVISIONS[model_name_or_path]
            logger.info(f"Loading {model_name_or_path}, revision {revision}")
        return super().from_pretrained(model_name_or_path, *args, revision=revision, **kwargs)


class AutoDistributedConfig(DefaultRevisionMixin, _AutoDistributedBase):
    _mapping_field = "config"


class AutoDistributedModel(DefaultRevisionMixin, _AutoDistributedBase):
    _mapping_field = "model"


class AutoDistributedModelForCausalLM(DefaultRevisionMixin, _AutoDistributedBase):
    _mapping_field = "model_for_causal_lm"


class AutoDistributedSpeculativeModel(DefaultRevisionMixin, _AutoDistributedBase):
    _mapping_field = "model_for_speculative"


class AutoDistributedModelForSequenceClassification(DefaultRevisionMixin, _AutoDistributedBase):
    _mapping_field = "model_for_sequence_classification"
