"""Check remote cache injection through the normal Transformers generation path."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

import bloombee.models.mixtral.model as mixtral_model
from bloombee.client.remote_generation import RemoteGenerationMixin, RemotePastKeyValues, _skipped_tokens
from bloombee.models.mixtral.config import DistributedMixtralConfig


@pytest.mark.parametrize("batch_size", [1, 2])
def test_mixtral_generate_uses_remote_cache_and_resumes(monkeypatch, local_remote_layers, batch_size):
    monkeypatch.setattr(mixtral_model, "RemoteSequential", local_remote_layers)
    config = DistributedMixtralConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=2,
        num_experts_per_tok=1,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    model = mixtral_model.DistributedMixtralForCausalLM(config).eval()
    monkeypatch.setattr(model, "_fast_generate_eligible", lambda *args: False)
    session = SimpleNamespace(position=0, output_ids=None)
    forwarded = []

    def record_forward(module, args, kwargs):
        cache = kwargs["past_key_values"]
        assert isinstance(cache, RemotePastKeyValues)
        assert cache.get_seq_length() == session.position
        forwarded.append(kwargs["input_ids"].clone())

    hook = model.model.register_forward_pre_hook(record_forward, with_kwargs=True)
    skipped = _skipped_tokens.set(0)
    generation_kwargs = dict(do_sample=False, session=session, suppress_tokens=[config.pad_token_id])
    try:
        prompt = torch.tensor([[3, 4, 5]]).expand(batch_size, -1)
        result = model.generate(prompt, max_new_tokens=3, **generation_kwargs)
        assert result.shape == (batch_size, 6)
        torch.testing.assert_close(torch.cat(forwarded, dim=1), result[:, :-1])
        assert session.position == 5

        previous = session.output_ids.clone()
        forwarded.clear()
        result = model.generate(None, max_new_tokens=2, **generation_kwargs)
        assert result.shape == (batch_size, 2)
        torch.testing.assert_close(torch.cat(forwarded, dim=1), torch.cat([previous[:, -1:], result[:, :-1]], 1))
        assert session.position == session.output_ids.shape[1] - 1
    finally:
        hook.remove()
        _skipped_tokens.reset(skipped)


class _CaptureGeneration:
    def generate(self, inputs, *args, **kwargs):
        self.captured_kwargs = kwargs
        return inputs


class _CacheProbe(RemoteGenerationMixin, _CaptureGeneration):
    def use_session(self, session):
        return contextlib.nullcontext(session)

    def _fast_generate_eligible(self, *args):
        return False


@pytest.mark.parametrize("support", [None, True, False])
def test_generate_cache_injection_respects_legacy_support_flag(support):
    model = _CacheProbe()
    if support is not None:
        model._supports_cache_class = support
    session = SimpleNamespace(position=7, output_ids=None)
    model.generate(torch.tensor([[3, 4]]), session=session)
    cache = model.captured_kwargs.get("past_key_values")
    if support is False:
        assert cache is None
    else:
        assert isinstance(cache, RemotePastKeyValues)
        assert cache.get_seq_length() == 7


def test_generate_preserves_explicit_cache():
    model = _CacheProbe()
    cache = RemotePastKeyValues()
    cache.update_seen(7)
    session = SimpleNamespace(position=7, output_ids=None)
    model.generate(torch.tensor([[3, 4]]), session=session, past_key_values=cache)
    assert model.captured_kwargs["past_key_values"] is cache
