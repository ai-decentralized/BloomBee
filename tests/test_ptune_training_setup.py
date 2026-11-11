"""Regression tests for prompt-tuning (training) setup. CPU-only, no swarm needed.

1. get_prompt() must work when the model was constructed on the "meta" device (transformers>=5 does this).
2. After loading, only the prompt embeddings (and the classification head) may require grad.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from bloombee.client.from_pretrained import _freeze_all_but_prompts
from bloombee.client.ptune import PTuneMixin


class _TinyPTuneModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.init_prompts(config)


def _config(tuning_mode):
    return SimpleNamespace(tuning_mode=tuning_mode, pre_seq_len=3, hidden_size=4, num_hidden_layers=2)


@pytest.mark.parametrize("tuning_mode", ["ptune", "deep_ptune"])
def test_get_prompt_after_meta_device_construction(tuning_mode):
    with torch.device("meta"):
        model = _TinyPTuneModel(_config(tuning_mode))
    # what from_pretrained() does afterwards: weights get real storage on cpu
    model.word_embeddings.to_empty(device="cpu")

    prompts, intermediate = model.get_prompt(batch_size=2)

    assert prompts.shape == (2, 3, 4)
    if tuning_mode == "deep_ptune":
        assert intermediate.shape == (2, 2, 3, 4)  # [layers, batch, prefix, hidden]


@pytest.mark.parametrize("tuning_mode", ["ptune", "deep_ptune"])
def test_only_prompts_are_trainable(tuning_mode):
    model = _TinyPTuneModel(_config(tuning_mode))
    model.score = nn.Linear(4, 2)  # a classification head
    assert all(p.requires_grad for p in model.parameters())  # simulates weights loaded as trainable

    _freeze_all_but_prompts(model)

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    expected = {"prompt_embeddings.weight", "score.weight", "score.bias"}
    if tuning_mode == "deep_ptune":
        expected.add("intermediate_prompt_embeddings.weight")
    assert trainable == expected
