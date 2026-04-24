"""Unit tests for the fast-generate eligibility predicate.

Exercises the pure-Python branching in ``RemoteGenerationMixin._fast_generate_eligible``
without needing a remote swarm or GPU. The test uses a minimal stub that
inherits the mixin but stubs out the transformer config attrs the predicate
reads.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
import torch

from bloombee.client.remote_generation import RemoteGenerationMixin


class _Stub(RemoteGenerationMixin):
    """Just enough of the mixin's surface to drive eligibility."""

    def __init__(self, tuning_mode=None):
        self._stub_transformer = MagicMock()
        self._stub_transformer.config = MagicMock(tuning_mode=tuning_mode, pre_seq_len=0)

    @property
    def transformer(self):
        return self._stub_transformer


@pytest.fixture(autouse=True)
def _enable_fast_generate(monkeypatch):
    """Ensure the env flag is ON for every test in this module."""
    monkeypatch.setenv("BLOOMBEE_FAST_GENERATE", "1")
    # Reimport-safe: the mixin reads the flag at module-import time, so we
    # poke the constant directly.
    import bloombee.client.remote_generation as rg
    monkeypatch.setattr(rg, "_FAST_GENERATE_ENABLED", True, raising=True)


def _basic_inputs():
    return torch.arange(8, dtype=torch.long).view(1, 8)


def test_greedy_max_new_tokens_is_eligible():
    m = _Stub()
    assert m._fast_generate_eligible(_basic_inputs(), (), {"max_new_tokens": 32}, None)


def test_greedy_max_length_is_eligible():
    m = _Stub()
    assert m._fast_generate_eligible(_basic_inputs(), (), {"max_length": 40}, None)


def test_do_sample_is_ineligible():
    m = _Stub()
    assert not m._fast_generate_eligible(
        _basic_inputs(), (), {"max_new_tokens": 32, "do_sample": True}, None
    )


def test_num_beams_over_one_is_ineligible():
    m = _Stub()
    assert not m._fast_generate_eligible(
        _basic_inputs(), (), {"max_new_tokens": 32, "num_beams": 4}, None
    )


def test_attention_mask_is_ineligible():
    m = _Stub()
    mask = torch.ones(1, 8)
    assert not m._fast_generate_eligible(
        _basic_inputs(), (), {"max_new_tokens": 32, "attention_mask": mask}, None
    )


def test_logits_processor_is_ineligible():
    m = _Stub()
    lp = [MagicMock()]
    assert not m._fast_generate_eligible(
        _basic_inputs(), (), {"max_new_tokens": 32, "logits_processor": lp}, None
    )


def test_stopping_criteria_is_ineligible():
    m = _Stub()
    sc = [MagicMock()]
    assert not m._fast_generate_eligible(
        _basic_inputs(), (), {"max_new_tokens": 32, "stopping_criteria": sc}, None
    )


def test_return_dict_is_ineligible():
    m = _Stub()
    assert not m._fast_generate_eligible(
        _basic_inputs(), (), {"max_new_tokens": 32, "return_dict_in_generate": True}, None
    )


def test_custom_generation_config_is_ineligible():
    m = _Stub()
    gc = MagicMock()
    assert not m._fast_generate_eligible(
        _basic_inputs(), (), {"max_new_tokens": 32, "generation_config": gc}, None
    )


def test_unknown_kwarg_is_ineligible():
    m = _Stub()
    assert not m._fast_generate_eligible(
        _basic_inputs(), (), {"max_new_tokens": 32, "some_future_feature": True}, None
    )


def test_missing_length_cap_is_ineligible():
    m = _Stub()
    assert not m._fast_generate_eligible(_basic_inputs(), (), {}, None)


def test_ptune_is_ineligible():
    m = _Stub(tuning_mode="ptune")
    assert not m._fast_generate_eligible(_basic_inputs(), (), {"max_new_tokens": 32}, None)


def test_env_off_disables_fast_path(monkeypatch):
    import bloombee.client.remote_generation as rg
    monkeypatch.setattr(rg, "_FAST_GENERATE_ENABLED", False, raising=True)
    m = _Stub()
    assert not m._fast_generate_eligible(_basic_inputs(), (), {"max_new_tokens": 32}, None)


def test_positional_extra_args_are_ineligible():
    m = _Stub()
    assert not m._fast_generate_eligible(_basic_inputs(), ("extra",), {"max_new_tokens": 32}, None)


def test_inputs_not_tensor_is_ineligible():
    m = _Stub()
    assert not m._fast_generate_eligible(None, (), {"max_new_tokens": 32}, None)
    assert not m._fast_generate_eligible([1, 2, 3], (), {"max_new_tokens": 32}, None)


def test_inputs_wrong_ndim_is_ineligible():
    m = _Stub()
    flat = torch.arange(8, dtype=torch.long)
    assert not m._fast_generate_eligible(flat, (), {"max_new_tokens": 32}, None)
