"""Prompt indices must survive HF meta construction and parameter materialization."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bloombee.client.ptune import PTuneMixin


class PromptModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(16, config.hidden_size, dtype=torch.float16)
        self.init_prompts(config)


@pytest.mark.parametrize("mode", ["ptune", "deep_ptune"])
def test_prompt_forward_and_backward_after_meta_materialization(mode):
    config = SimpleNamespace(tuning_mode=mode, pre_seq_len=3, hidden_size=4, num_hidden_layers=2)
    with torch.device("meta"):
        model = PromptModel(config)
    model.to_empty(device="cpu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    prompts, intermediate = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 3, 4)
    assert prompts.dtype == torch.float16
    loss = prompts.float().sum()
    if mode == "deep_ptune":
        assert intermediate.shape == (2, 2, 3, 4)
        loss = loss + intermediate.float().sum()
    loss.backward()
    torch.testing.assert_close(
        model.prompt_embeddings.weight.grad, torch.full_like(model.prompt_embeddings.weight, 2.0)
    )
    if mode == "deep_ptune":
        torch.testing.assert_close(
            model.intermediate_prompt_embeddings.weight.grad,
            torch.full_like(model.intermediate_prompt_embeddings.weight, 2.0),
        )
