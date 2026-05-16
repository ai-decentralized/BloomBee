from transformers import GenerationConfig

from bloombee.models.llama.eagle_drafter import (
    _default_draft_token_budget,
    _eagle2_max_candidate_depth,
)
from bloombee.models.llama.speculative_model import (
    _eos_token_ids,
    _merge_generation_config_kwargs,
)


def test_eagle_legacy_budget_matches_chain_draft_depth():
    assert _default_draft_token_budget(beam_width=1, max_depth=5) == 5


def test_eagle_legacy_budget_matches_scalar_tree_plan():
    assert _default_draft_token_budget(beam_width=3, max_depth=4) == 12


def test_eagle_legacy_budget_matches_sequoia_width_plan():
    assert _default_draft_token_budget(beam_width=[2, 3, 1], max_depth=3) == 14


def test_eagle_paper_budget_uses_official_depth_convention(monkeypatch):
    monkeypatch.delenv("BLOOMBEE_EAGLE_DEPTH", raising=False)

    assert _eagle2_max_candidate_depth(4, explicit_tree_budget=True) == 6
    assert _eagle2_max_candidate_depth(5, explicit_tree_budget=True) == 6


def test_eagle_legacy_depth_remains_path_length(monkeypatch):
    monkeypatch.delenv("BLOOMBEE_EAGLE_DEPTH", raising=False)

    assert _eagle2_max_candidate_depth(5, explicit_tree_budget=False) == 5


def test_eagle_depth_env_controls_paper_budget_depth(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_EAGLE_DEPTH", "7")

    assert _eagle2_max_candidate_depth(4, explicit_tree_budget=True) == 8


def test_speculative_generation_honors_do_sample_kwarg():
    cfg = GenerationConfig(do_sample=True, temperature=0.6, pad_token_id=0)
    kwargs = {"do_sample": False, "temperature": 1.0, "pad_token_id": 2}

    merged = _merge_generation_config_kwargs(cfg, kwargs)

    assert merged.do_sample is False
    assert merged.temperature == 1.0
    assert merged.pad_token_id == 2
    assert cfg.do_sample is True
    assert kwargs == {}


def test_speculative_generation_reads_generation_config_eos_ids():
    assert _eos_token_ids(GenerationConfig(eos_token_id=None)) == ()
    assert _eos_token_ids(GenerationConfig(eos_token_id=2)) == (2,)
    assert _eos_token_ids(GenerationConfig(eos_token_id=[2, 32000])) == (2, 32000)
