"""Exercise Falcon generation preparation with real tensors and client code."""

from types import SimpleNamespace

import pytest
import torch
from transformers.cache_utils import DynamicCache

import bloombee.models.falcon.model as falcon_model
from bloombee.client.remote_generation import RemotePastKeyValues, _skipped_tokens
from bloombee.models.falcon.config import DistributedFalconConfig
from bloombee.models.falcon.model import DistributedFalconForCausalLM


def _prepare_model(alibi=False):
    model = DistributedFalconForCausalLM.__new__(DistributedFalconForCausalLM)
    torch.nn.Module.__init__(model)
    model.transformer = SimpleNamespace(use_alibi=alibi)
    return model


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("alibi", [False, True])
@pytest.mark.parametrize("suffix_length", [1, 3, 16, 25])
def test_falcon_pretrimmed_prompt_keeps_every_uncached_token(batch, alibi, suffix_length):
    prefix = 17
    cache = RemotePastKeyValues()
    cache.update_seen(torch.full((batch,), prefix, dtype=torch.long))
    suffix = torch.arange(batch * suffix_length).view(batch, suffix_length)
    mask = torch.ones(batch, prefix + suffix_length, dtype=torch.long)
    prepared = _prepare_model(alibi).prepare_inputs_for_generation(
        suffix, past_key_values=cache, attention_mask=mask, use_cache=True
    )
    torch.testing.assert_close(prepared["input_ids"], suffix)
    assert prepared["past_key_values"] is cache
    assert prepared["use_cache"] is True
    torch.testing.assert_close(prepared["attention_mask"], mask)
    if alibi:
        assert prepared["position_ids"] is None
    else:
        expected = torch.arange(prefix, prefix + suffix_length).expand(batch, -1)
        torch.testing.assert_close(prepared["position_ids"], expected)


@pytest.mark.parametrize("batch", [1, 2])
def test_falcon_full_history_discards_only_cached_tokens(batch):
    prefix = 17
    cache = RemotePastKeyValues()
    cache.update_seen(torch.full((batch,), prefix, dtype=torch.long))
    inputs = torch.arange(batch * 33).view(batch, 33)
    prepared = _prepare_model().prepare_inputs_for_generation(
        inputs, past_key_values=cache, attention_mask=torch.ones_like(inputs)
    )
    torch.testing.assert_close(prepared["input_ids"], inputs[:, prefix:])
    torch.testing.assert_close(prepared["position_ids"], torch.arange(prefix, 33).expand(batch, -1))


@pytest.mark.parametrize("cache_kind", ["legacy", "dynamic"])
def test_falcon_pretrimmed_prompt_supports_other_cache_types(cache_kind):
    keys = torch.zeros(2, 1, 17, 4)
    if cache_kind == "legacy":
        cache = ((keys, keys.clone()),)
    else:
        cache = DynamicCache()
        cache.update(keys, keys.clone(), 0)
    suffix = torch.arange(6).view(2, 3)
    prepared = _prepare_model().prepare_inputs_for_generation(
        suffix, past_key_values=cache, attention_mask=torch.ones(2, 20, dtype=torch.long)
    )
    torch.testing.assert_close(prepared["input_ids"], suffix)
    torch.testing.assert_close(prepared["position_ids"], torch.arange(17, 20).expand(2, -1))


@pytest.mark.parametrize("with_mask", [False, True])
@pytest.mark.parametrize("with_empty_cache", [False, True])
def test_falcon_first_prefill_keeps_full_prompt(with_mask, with_empty_cache):
    inputs = torch.arange(10).view(1, 10)
    cache = RemotePastKeyValues() if with_empty_cache else None
    prepared = _prepare_model().prepare_inputs_for_generation(
        inputs, past_key_values=cache, attention_mask=torch.ones_like(inputs) if with_mask else None
    )
    torch.testing.assert_close(prepared["input_ids"], inputs)


def test_falcon_pretrimmed_prompt_without_mask_is_preserved():
    cache = RemotePastKeyValues()
    cache.update_seen(17)
    suffix = torch.tensor([[11, 12, 13]])
    prepared = _prepare_model().prepare_inputs_for_generation(suffix, past_key_values=cache)
    torch.testing.assert_close(prepared["input_ids"], suffix)
    assert prepared["position_ids"] is None


def test_falcon_explicit_position_ids_match_the_uncached_suffix():
    cache = RemotePastKeyValues()
    cache.update_seen(17)
    suffix = torch.tensor([[11, 12, 13]])
    prepared = _prepare_model().prepare_inputs_for_generation(
        suffix,
        past_key_values=cache,
        attention_mask=torch.ones(1, 20, dtype=torch.long),
        position_ids=torch.arange(20).view(1, -1),
    )
    torch.testing.assert_close(prepared["input_ids"], suffix)
    torch.testing.assert_close(prepared["position_ids"], torch.tensor([[17, 18, 19]]))


@pytest.fixture
def local_falcon(monkeypatch, local_remote_layers):
    monkeypatch.setattr(falcon_model, "RemoteSequential", local_remote_layers)
    monkeypatch.setattr(
        falcon_model, "LMHead", lambda config: torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    )
    config = DistributedFalconConfig(
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        vocab_size=64,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    model = DistributedFalconForCausalLM(config).eval()
    token = _skipped_tokens.set(0)
    try:
        yield model
    finally:
        _skipped_tokens.reset(token)


@pytest.mark.parametrize("chunk_size", [None, 4])
def test_falcon_generate_prefill_and_ordinary_session_continuation(local_falcon, chunk_size):
    model = local_falcon
    session = SimpleNamespace(position=0, output_ids=None)
    forwarded = []
    hook = model.transformer.register_forward_pre_hook(lambda module, args: forwarded.append(args[0].clone()))
    kwargs = {"max_new_tokens": 2, "do_sample": False, "session": session, "suppress_tokens": [0]}
    if chunk_size is not None:
        kwargs["prefill_chunk_size"] = chunk_size
    try:
        first_prompt = torch.arange(1, 12).view(1, -1)
        model.generate(first_prompt, **kwargs)
        torch.testing.assert_close(torch.cat(forwarded, dim=1), session.output_ids[:, :-1])
        assert session.position == session.output_ids.shape[1] - 1

        # Exercise ordinary multi-turn generation and generate(None) after
        # either a regular or chunked initial prompt.
        kwargs.pop("prefill_chunk_size", None)
        last_token = session.output_ids[:, -1:].clone()
        forwarded.clear()
        new_prompt = torch.tensor([[21, 22, 23, 24, 25, 26]])
        model.generate(new_prompt, **kwargs)
        expected = torch.cat([last_token, new_prompt, session.output_ids[:, -2:-1]], dim=1)
        torch.testing.assert_close(torch.cat(forwarded, dim=1), expected)
        assert session.position == session.output_ids.shape[1] - 1

        last_token = session.output_ids[:, -1:].clone()
        forwarded.clear()
        model.generate(None, **kwargs)
        expected = torch.cat([last_token, session.output_ids[:, -2:-1]], dim=1)
        torch.testing.assert_close(torch.cat(forwarded, dim=1), expected)
        assert session.position == session.output_ids.shape[1] - 1
    finally:
        hook.remove()
