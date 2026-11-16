import torch

from bloombee.client.remote_generation import RemotePastKeyValues
from bloombee.models.nemotron_h.model import DistributedNemotronHForCausalLM


def _model():
    # prepare_inputs_for_generation does not use initialized model weights.
    return object.__new__(DistributedNemotronHForCausalLM)


def test_nemotron_declares_cache_class_support():
    assert DistributedNemotronHForCausalLM._supports_cache_class is True


def test_nemotron_prepare_inputs_prefill():
    model = _model()

    input_ids = torch.tensor([[10, 11, 12]])
    attention_mask = torch.ones_like(input_ids)

    prepared = model.prepare_inputs_for_generation(
        input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )

    assert torch.equal(prepared["input_ids"], input_ids)
    assert torch.equal(
        prepared["position_ids"],
        torch.tensor([[0, 1, 2]]),
    )
    assert prepared["past_key_values"] is None
    assert prepared["use_cache"] is True


def test_nemotron_prepare_inputs_decode_trims_seen_tokens():
    model = _model()

    cache = RemotePastKeyValues()
    cache.advance_seen(3)

    input_ids = torch.tensor([[10, 11, 12, 13]])
    attention_mask = torch.ones_like(input_ids)

    prepared = model.prepare_inputs_for_generation(
        input_ids,
        past_key_values=cache,
        attention_mask=attention_mask,
        use_cache=True,
    )

    assert prepared["input_ids"].shape == (1, 1)
    assert prepared["input_ids"].item() == 13

    assert prepared["position_ids"].shape == (1, 1)
    assert prepared["position_ids"].item() == 3

    assert prepared["past_key_values"] is cache
