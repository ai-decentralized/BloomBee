import pytest
import torch

from bloombee.server.block_utils import resolve_block_dtype
from bloombee.server.from_pretrained import load_pretrained_block
from bloombee.utils.auto_config import AutoDistributedConfig
from test_utils import MODEL_NAME


@pytest.mark.forked
@pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float16, "auto"])
def test_block_dtype(torch_dtype):
    config = AutoDistributedConfig.from_pretrained(MODEL_NAME)
    block = load_pretrained_block(MODEL_NAME, 0, config=config, torch_dtype=torch_dtype)
    expected_dtype = resolve_block_dtype(config, torch_dtype)
    assert all(param.dtype == expected_dtype for param in block.parameters())
