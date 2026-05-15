from bloombee.models.llama.eagle_drafter import _default_draft_token_budget


def test_eagle_legacy_budget_matches_chain_draft_depth():
    assert _default_draft_token_budget(beam_width=1, max_depth=5) == 5


def test_eagle_legacy_budget_matches_scalar_tree_plan():
    assert _default_draft_token_budget(beam_width=3, max_depth=4) == 12


def test_eagle_legacy_budget_matches_sequoia_width_plan():
    assert _default_draft_token_budget(beam_width=[2, 3, 1], max_depth=3) == 14
