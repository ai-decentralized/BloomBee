"""Guard tied-head fallback without hiding corrupt untied checkpoints."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bloombee.models.llama.block import WrappedLlamaBlock
from bloombee.models.llama.flex_llama import FLEX_LlamaRMSNorm, SimpleLMHead


def make_block(tmp_path, tied, explicit_head=False):
    embedding = np.arange(32, dtype=np.float16).reshape(8, 4) / 32
    head = embedding + 1
    for name, array in [("norm.weight", np.ones(4, dtype=np.float16)), ("embed_tokens.weight", embedding)]:
        with (tmp_path / name).open("wb") as handle:
            np.save(handle, array)
    if explicit_head:
        with (tmp_path / "lm_head.weight").open("wb") as handle:
            np.save(handle, head)
    block = WrappedLlamaBlock.__new__(WrappedLlamaBlock)
    torch.nn.Module.__init__(block)
    block.config = SimpleNamespace(tie_word_embeddings=tied)
    block.expanded_apth = str(tmp_path)
    block.input_layernorm = FLEX_LlamaRMSNorm(4)
    block.lm_head = SimpleLMHead(4, 8)
    return block, embedding, head


def test_tied_checkpoint_uses_shared_embedding(tmp_path):
    block, embedding, _ = make_block(tmp_path, tied=True)
    block.load_lm_head()
    torch.testing.assert_close(block.lm_head.weight, torch.tensor(embedding, dtype=torch.float32))
    x = torch.randn(1, 2, 4)
    torch.testing.assert_close(block.lm_head(x), x @ torch.tensor(embedding, dtype=torch.float32).T)


def test_missing_untied_head_still_fails(tmp_path):
    block, _, _ = make_block(tmp_path, tied=False)
    with pytest.raises(FileNotFoundError, match="lm_head.weight"):
        block.load_lm_head()


def test_explicit_head_takes_precedence(tmp_path):
    block, _, head = make_block(tmp_path, tied=True, explicit_head=True)
    block.load_lm_head()
    torch.testing.assert_close(block.lm_head.weight, torch.tensor(head, dtype=torch.float32))
