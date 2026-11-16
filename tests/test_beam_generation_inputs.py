import torch

from bloombee.client.remote_generation import RemotePastKeyValues
from bloombee.models.beam.model import DistributedBeamForCausalLM


def _model():
    return object.__new__(DistributedBeamForCausalLM)


def test_beam_declares_cache_support():
    assert DistributedBeamForCausalLM._supports_cache_class is True


def test_beam_prepare_prefill():
    model = _model()

    ids = torch.tensor([[10, 11, 12]])
    mask = torch.ones_like(ids)

    out = model.prepare_inputs_for_generation(
        ids,
        attention_mask=mask,
        use_cache=True,
    )

    assert torch.equal(out["input_ids"], ids)
    assert torch.equal(
        out["position_ids"],
        torch.tensor([[0, 1, 2]]),
    )
    assert out["past_key_values"] is None


def test_beam_prepare_decode_trims_seen_tokens():
    model = _model()

    cache = RemotePastKeyValues()
    cache.advance_seen(3)

    ids = torch.tensor([[10, 11, 12, 13]])
    mask = torch.ones_like(ids)

    out = model.prepare_inputs_for_generation(
        ids,
        past_key_values=cache,
        attention_mask=mask,
        use_cache=True,
    )

    assert out["input_ids"].shape == (1, 1)
    assert out["input_ids"].item() == 13
    assert out["position_ids"].shape == (1, 1)
    assert out["position_ids"].item() == 3
    assert out["past_key_values"] is cache


def test_beam_prefill_preserves_inputs_embeds():
    model = _model()

    ids = torch.tensor([[10, 11]])
    mask = torch.ones_like(ids)
    embeds = torch.randn(1, 2, 8)

    out = model.prepare_inputs_for_generation(
        ids,
        attention_mask=mask,
        inputs_embeds=embeds,
        use_cache=True,
    )

    assert out["inputs_embeds"] is embeds
    assert "input_ids" not in out
