"""FlexGen task metadata must follow the actual batch and prompt shape."""

from types import SimpleNamespace

import pytest
from torch import nn

from bloombee.flexgen_utils.task import Task
from bloombee.models.llama.block import OptimizedLlamaDecoderLayer, get_test_inputs


def make_block(pad_token_id=0):
    block = OptimizedLlamaDecoderLayer.__new__(OptimizedLlamaDecoderLayer)
    nn.Module.__init__(block)
    block.llama_config = SimpleNamespace(pad_token_id=pad_token_id)
    block._test_inputs_cache = {}
    block._cached_task = None
    return block


@pytest.mark.parametrize("pad_token_id", [None, 7])
def test_task_inputs_match_shape_without_loading_tokenizer(pad_token_id, monkeypatch):
    block = make_block(pad_token_id)
    monkeypatch.setattr(block, "_get_tokenizer", lambda: pytest.fail("Task metadata must not load a tokenizer"))
    inputs = block._get_cached_test_inputs(5, 3)
    assert inputs == tuple([pad_token_id or 0] * 5 for _ in range(3))
    assert inputs[0] is not inputs[1]
    assert block._get_cached_test_inputs(5, 3) is inputs
    assert len(block._get_cached_test_inputs(5, 2)) == 2
    assert len(block._get_cached_test_inputs(6, 3)[0]) == 6


@pytest.mark.parametrize(
    "generation,prompt,batch,rebuild",
    [
        (8, 5, 3, False),
        (9, 5, 3, True),
        (8, 6, 3, True),
        (8, 5, 2, True),
    ],
)
def test_task_rebuilds_when_batch_or_lengths_change(generation, prompt, batch, rebuild):
    block = make_block()
    assert block._should_rebuild_task(generation, prompt, batch)
    block._cached_task = Task(
        inputs=block._get_cached_test_inputs(5, 3),
        prompt_len=5,
        gen_len=8,
        cut_gen_len=8,
        do_sample=False,
        temperature=1.0,
        stop=None,
    )
    assert block._should_rebuild_task(generation, prompt, batch) is rebuild


def test_placeholder_helper_preserves_prompt_length():
    assert get_test_inputs(4, 2, SimpleNamespace(pad_token_id=3)) == ([3] * 4, [3] * 4)
