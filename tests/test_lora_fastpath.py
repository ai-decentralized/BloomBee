"""LoRA training fast path, fused shared-input adapters, and the cached v2 worker protocol."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from bloombee.server import lora as worker
from bloombee.utils.lora import PROJECTIONS, LoRAConfig, lora_context, lora_linear, lora_linear_group, projection_shapes


def _adapters(config, shapes, scale=0.05, generator=None):
    return tuple(
        t
        for name in config.target_modules
        for t in (
            torch.randn(config.r, shapes[name][1], generator=generator) * scale,
            torch.randn(shapes[name][0], config.r, generator=generator) * scale,
        )
    )


def test_grouped_projections_match_separate_lora_linear():
    torch.manual_seed(3)
    config = LoRAConfig(r=4, lora_alpha=8, target_modules=("q_proj", "v_proj"))
    x = torch.randn(2, 5, 16)
    weights = {n: torch.randn(8, 16) for n in ("q_proj", "k_proj", "v_proj")}
    params = tuple(t.requires_grad_() for t in (torch.randn(4, 16), torch.randn(8, 4), torch.randn(4, 16), torch.randn(8, 4)))
    g = [torch.randn(2, 5, 8) for _ in range(3)]
    with lora_context(config, params, training=True, seed=1):
        grouped = lora_linear_group(x, [(weights[n], n) for n in ("q_proj", "k_proj", "v_proj")])
        grads_grouped = torch.autograd.grad((grouped[0], grouped[2]), params, (g[0], g[2]))
        separate = [lora_linear(x, weights[n], n) for n in ("q_proj", "k_proj", "v_proj")]
        grads_separate = torch.autograd.grad((separate[0], separate[2]), params, (g[0], g[2]))
    for a, b in zip(grouped + list(grads_grouped), separate + list(grads_separate)):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    # No adapters active: plain projections.
    plain = lora_linear_group(x, [(weights["k_proj"], "k_proj")])
    torch.testing.assert_close(plain[0], F.linear(x, weights["k_proj"]))


def _qwen_backend(device="cpu"):
    from test_standard_lora import make_qwen_backend

    backend, block = make_qwen_backend(device)
    backend.name = f"qwen.{id(backend)}"
    return backend, block


def _run(backend, x, grad, params, config, monkeypatch, fast):
    monkeypatch.setenv("BLOOMBEE_LORA_FASTPATH", "1" if fast else "0")
    backend._lora_fastpath_weights = None
    meta = dict(config=config.to_dict(), training=True, seed=5)
    return worker.run_lora_block(backend, x, grad, meta, *params)


def test_qwen_fastpath_matches_module_forward_and_gradients(monkeypatch):
    torch.manual_seed(21)
    backend, _ = _qwen_backend()
    config = LoRAConfig(r=2, lora_alpha=4, target_modules=PROJECTIONS)
    params = _adapters(config, projection_shapes(backend.config))
    x, g = torch.randn(2, 7, 32), torch.randn(2, 7, 32)
    from bloombee.server.lora_fastpath import block_weights

    assert block_weights(backend, x.device) is not None
    for grad in (torch.empty(0), g):
        fast = _run(backend, x, grad, params, config, monkeypatch, True)
        slow = _run(backend, x, grad, params, config, monkeypatch, False)
        assert len(fast) == len(slow)
        for a, b in zip(fast, slow):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def test_llama_fastpath_matches_flexgen_attention_and_mlp():
    from bloombee.flexgen_utils.pytorch_backend import TorchDevice, TorchTensor
    from bloombee.server.lora_fastpath import block_forward
    from transformers.activations import ACT2FN

    torch.manual_seed(7)
    device = TorchDevice("cpu")
    batch, seq, n_head, n_kv, hd, inter = 2, 6, 8, 2, 16, 96
    h = n_head * hd
    w = dict(
        q_proj=torch.randn(h, h) * 0.05, k_proj=torch.randn(n_kv * hd, h) * 0.05,
        v_proj=torch.randn(n_kv * hd, h) * 0.05, o_proj=torch.randn(h, h) * 0.05,
        gate_proj=torch.randn(inter, h) * 0.05, up_proj=torch.randn(inter, h) * 0.05,
        down_proj=torch.randn(h, inter) * 0.05,
        ln1=torch.rand(h) + 0.5, ln2=torch.rand(h) + 0.5,
    )  # fmt: skip
    inv_freq = 1.0 / (10000 ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    x = torch.randn(batch, seq, h) * 0.5
    causal = torch.zeros(batch, 1, seq, seq)
    causal.masked_fill_(torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1), float("-inf"))
    cfg = SimpleNamespace(hidden_act="silu", rms_norm_eps=1e-5)
    config = LoRAConfig(r=2, lora_alpha=4, target_modules=PROJECTIONS)
    shapes = dict(
        q_proj=(h, h), k_proj=(n_kv * hd, h), v_proj=(n_kv * hd, h), o_proj=(h, h),
        gate_proj=(inter, h), up_proj=(inter, h), down_proj=(h, inter),
    )  # fmt: skip
    params = _adapters(config, shapes)

    def tt(t):
        return TorchTensor.create_from_torch(t, device)

    with lora_context(config, params, training=False, seed=0):
        attn, _, _ = device.mha_llama(
            tt(x.clone()), tt(causal), tt(w["q_proj"]), tt(w["k_proj"]), tt(w["v_proj"]), tt(w["o_proj"]),
            n_head, (False, False), False, None, tt(w["ln1"]), tt(inv_freq), None,
        )  # fmt: skip
        reference = device.mlp_llama(
            attn, tt(w["gate_proj"]), tt(w["down_proj"]), tt(w["up_proj"]), (False,), cfg, tt(w["ln2"])
        ).data
        weights = dict(
            family="llama", inv_freq=inv_freq, q_norm=None, k_norm=None, rotary=None, head_dim=hd,
            num_heads=n_head, num_kv_heads=n_kv, eps=1e-5, act=ACT2FN["silu"], **w,
        )  # fmt: skip
        fast = block_forward(x, weights)
    torch.testing.assert_close(fast, reference, rtol=2e-4, atol=2e-4)


@pytest.fixture
def fresh_caches(monkeypatch):
    monkeypatch.setattr(worker, "_ADAPTERS", type(worker._ADAPTERS)())
    monkeypatch.setattr(worker, "_ACTIVATIONS", type(worker._ACTIVATIONS)())
    monkeypatch.setattr(worker, "_usage", {"adapter_bytes": 0, "activation_bytes": 0})
    monkeypatch.setattr(worker, "_BACKENDS", {})


def _span(n=2):
    backends = []
    for _ in range(n):
        backend, _ = _qwen_backend()
        worker.register_backend(backend)
        backends.append(backend)
    return backends


def _meta(backends, config, op, **extra):
    meta = dict(
        uid=" ".join(b.name for b in backends), owner="peer", config=config.to_dict(), training=True,
        seed=11, adapter_version="v1", cache_id=None, op=op,
    )  # fmt: skip
    return {**meta, **extra}


@pytest.mark.parametrize("keep_graph", [False, True])
def test_cached_backward_matches_stateless(fresh_caches, monkeypatch, keep_graph):
    torch.manual_seed(4)
    if keep_graph:  # CPU has no allocator statistics; pretend so the graph path runs
        monkeypatch.setattr(worker, "_allocated", lambda device: 0)
        monkeypatch.setenv("BLOOMBEE_LORA_CACHE_BYTES", str(1 << 30))
    backends = _span()
    config = LoRAConfig(r=2, lora_alpha=4, lora_dropout=0.1, target_modules=PROJECTIONS)
    per_layer = [_adapters(config, projection_shapes(b.config)) for b in backends]
    params = tuple(p for layer in per_layer for p in layer)
    x, g = torch.randn(2, 6, 32), torch.randn(2, 6, 32)

    meta = _meta(backends, config, "forward", cache_id="a" * 32)
    (y,) = worker.run_lora_task(None, x, meta, *params)
    entry = worker._ACTIVATIONS[("peer", "a" * 32)]
    assert all((layer.y is not None) == keep_graph for layer in entry.layers)
    cached = worker.run_lora_task(None, g, _meta(backends, config, "backward", cache_id="a" * 32))
    assert ("peer", "a" * 32) not in worker._ACTIVATIONS and worker._usage["activation_bytes"] == 0
    stateless = worker.run_lora_task(None, x, _meta(backends, config, "stateless"), g, *params)
    assert len(cached) == len(stateless) == 1 + len(params)
    for a, b in zip(cached, stateless):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    # Forward without a cache id leaves nothing behind and gives the same output.
    (y2,) = worker.run_lora_task(None, x, _meta(backends, config, "forward"))  # adapter from cache
    torch.testing.assert_close(y, y2, rtol=0, atol=0)
    assert not worker._ACTIVATIONS


def test_cache_misses_are_reported(fresh_caches, monkeypatch):
    backends = _span(1)
    config = LoRAConfig(r=2, target_modules=("q_proj",))
    params = _adapters(config, projection_shapes(backends[0].config))
    x = torch.randn(1, 4, 32)
    with pytest.raises(worker.LoRACacheMiss, match="adapter"):
        worker.run_lora_task(None, x, _meta(backends, config, "forward", adapter_version="unknown"))
    with pytest.raises(worker.LoRACacheMiss, match="activations"):
        worker.run_lora_task(None, x, _meta(backends, config, "backward", cache_id="b" * 32))
    monkeypatch.setenv("BLOOMBEE_LORA_CACHE_BYTES", "16")
    with pytest.raises(worker.LoRACacheMiss, match="budget"):
        worker.run_lora_task(None, x, _meta(backends, config, "forward", cache_id="c" * 32), *params)
    assert not worker._ACTIVATIONS
    # A new adapter version for the same span replaces the old one.
    worker.run_lora_task(None, x, _meta(backends, config, "forward", adapter_version="v2"), *params)
    assert [k[2] for k in worker._ADAPTERS] == ["v2"]
    miss = worker.LoRACacheMiss("adapter")
    import pickle

    assert pickle.loads(pickle.dumps(miss)).what == "adapter"


def test_worker_accumulates_step_gradients_and_collects_once(fresh_caches, monkeypatch):
    monkeypatch.setattr(worker, "_GRADIENTS", type(worker._GRADIENTS)())
    torch.manual_seed(9)
    backends = _span()
    config = LoRAConfig(r=2, lora_alpha=4, target_modules=("q_proj", "down_proj"))
    params = tuple(p for b in backends for p in _adapters(config, projection_shapes(b.config)))
    xs = [torch.randn(1, 5, 32) for _ in range(3)]
    gs = [torch.randn(1, 5, 32) for _ in range(3)]
    token = "f" * 32
    expected = None
    for i, (x, g) in enumerate(zip(xs, gs)):
        plain = worker.run_lora_task(None, x, _meta(backends, config, "stateless"), g, *params)
        expected = list(plain[1:]) if expected is None else [e + p for e, p in zip(expected, plain[1:])]
        cache_id = f"{i:032x}"
        worker.run_lora_task(None, x, _meta(backends, config, "forward", cache_id=cache_id), *params)
        out = worker.run_lora_task(None, g, _meta(backends, config, "backward", cache_id=cache_id, accumulate=token))
        assert len(out) == 1
        torch.testing.assert_close(out[0], plain[0], rtol=1e-5, atol=1e-6)
    collected = worker.run_lora_task(None, torch.empty(1, 1, 1), _meta(backends, config, "collect", accumulate=token))
    assert len(collected) == len(params)
    for a, b in zip(collected, expected):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    with pytest.raises(worker.LoRACacheMiss, match="gradients"):
        worker.run_lora_task(None, torch.empty(1, 1, 1), _meta(backends, config, "collect", accumulate=token))


@pytest.mark.parametrize("codec", ["int8", "hadamard-int8"])
def test_boundary_codecs_roundtrip_with_outliers(codec):
    from bloombee.utils import lora_codec

    torch.manual_seed(0)
    x = torch.randn(2, 16, 256).half()
    x[..., 7] *= 200  # an outlier channel, as in real LLM hidden states
    parts = lora_codec.encode(x, codec)
    assert parts[0].dtype == torch.int8 and len(parts) == lora_codec.wire_count(codec)
    y = lora_codec.decode(parts, codec, torch.float16)
    assert y.shape == x.shape and y.dtype == torch.float16
    error = (y.float() - x.float()).norm() / x.float().norm()
    assert error < (0.01 if codec == "hadamard-int8" else 0.05)
    # Deterministic: a stateless retry sees the very same decoded values.
    assert torch.equal(lora_codec.decode(lora_codec.encode(x, codec), codec, torch.float16), y)
    small = x[..., :7].float()
    small_error = (y[..., :7].float() - small).norm() / small.norm()
    if codec == "hadamard-int8":
        assert small_error < 0.25  # rotation spreads the outlier: ordinary channels survive
    assert lora_codec.decode(lora_codec.encode(x, "none"), "none", torch.float16) is x
