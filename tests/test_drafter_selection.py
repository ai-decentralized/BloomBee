"""Family-aware drafter selection — registry lookup + env override."""

from __future__ import annotations

import os

import pytest

from bloombee.models.llama.spec_decoding_drafter import (
    _DEFAULT_DRAFTER,
    select_drafter_for_target,
)


@pytest.mark.parametrize(
    "model_type,name,expected_prefix",
    [
        ("llama", "huggyllama/llama-7b", "JackFram/llama-160m"),
        ("llama", "meta-llama/Llama-3.2-8B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"),
        ("llama", "meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-3.2-1B-Instruct"),
        ("qwen3", "Qwen/Qwen3-14B", "Qwen/Qwen3-0.6B"),
        ("qwen3_moe", "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-0.6B"),
        ("bloom", "bigscience/bloom-7b1", "bigscience/bloom-560m"),
        ("falcon", "tiiuae/falcon-7b", "tiiuae/Falcon3-1B-Instruct"),
        ("mixtral", "mistralai/Mixtral-8x7B-v0.1", "mistralai/Mistral-7B-v0.3"),
    ],
)
def test_registry_picks_expected_default(model_type, name, expected_prefix):
    drafter_id, source = select_drafter_for_target(
        target_model_type=model_type, target_name_or_path=name,
    )
    assert drafter_id == expected_prefix, (
        f"{model_type=} {name=} picked {drafter_id!r}, expected {expected_prefix!r}"
    )
    assert source.startswith("registry:")


def test_unknown_model_type_falls_back_to_default():
    drafter_id, source = select_drafter_for_target(
        target_model_type="unknown_family_42", target_name_or_path="foo/bar",
    )
    assert drafter_id == _DEFAULT_DRAFTER
    assert source == "default"


def test_env_override_wins_over_registry(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_DRAFTER", "myorg/my-custom-drafter")
    drafter_id, source = select_drafter_for_target(
        target_model_type="llama", target_name_or_path="huggyllama/llama-7b",
    )
    assert drafter_id == "myorg/my-custom-drafter"
    assert source == "env"


def test_llama3_disambiguation_by_path():
    drafter_id_l2, src_l2 = select_drafter_for_target("llama", "huggyllama/llama-7b")
    drafter_id_l3, src_l3 = select_drafter_for_target("llama", "meta-llama/Llama-3.2-70B")
    assert drafter_id_l2.startswith("JackFram/llama")
    assert src_l2 == "registry:llama"
    assert drafter_id_l3.startswith("meta-llama/Llama-3")
    assert src_l3 == "registry:llama3"
