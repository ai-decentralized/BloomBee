from transformers import GenerationConfig

from bloombee.models.llama.eagle_drafter import _default_draft_token_budget
from bloombee.models.llama.speculative_model import _merge_generation_config_kwargs


def test_eagle_legacy_budget_matches_chain_draft_depth():
    assert _default_draft_token_budget(beam_width=1, max_depth=5) == 5


def test_eagle_legacy_budget_matches_scalar_tree_plan():
    assert _default_draft_token_budget(beam_width=3, max_depth=4) == 12


def test_eagle_legacy_budget_matches_sequoia_width_plan():
    assert _default_draft_token_budget(beam_width=[2, 3, 1], max_depth=3) == 14


def test_speculative_generation_honors_do_sample_kwarg():
    cfg = GenerationConfig(do_sample=True, temperature=0.6, pad_token_id=0)
    kwargs = {"do_sample": False, "temperature": 1.0, "pad_token_id": 2}

    merged = _merge_generation_config_kwargs(cfg, kwargs)

    assert merged.do_sample is False
    assert merged.temperature == 1.0
    assert merged.pad_token_id == 2
    assert cfg.do_sample is True
    assert kwargs == {}
