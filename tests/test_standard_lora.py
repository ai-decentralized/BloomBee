"""Standard LoRA math, independent PEFT parity, parameter ownership and recompute."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bloombee.server.lora import run_lora_block
from bloombee.utils.lora import PROJECTIONS, LoRAConfig, LoRAProjection, install_lora_hooks, lora_context, lora_linear


@pytest.mark.parametrize("target", PROJECTIONS)
def test_standard_linear_matches_peft_forward_and_all_gradients(target):
    from peft.tuners.lora.layer import Linear

    torch.manual_seed(11)
    base = nn.Linear(12, 16, bias=False).requires_grad_(False)
    reference = Linear(base, "default", r=3, lora_alpha=6, lora_dropout=0.0)
    adapter = LoRAProjection(12, 16, 3)
    with torch.no_grad():
        adapter.lora_B.normal_(0, 0.03)
        reference.lora_A["default"].weight.copy_(adapter.lora_A)
        reference.lora_B["default"].weight.copy_(adapter.lora_B)
    x = torch.randn(2, 5, 12, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    g = torch.randn(2, 5, 16)
    with lora_context(
        LoRAConfig(r=3, lora_alpha=6, target_modules=(target,)),
        (adapter.lora_A, adapter.lora_B),
        training=True,
        seed=42,
    ):
        actual = lora_linear(x, base.weight, target)
        actual_grads = torch.autograd.grad(actual, (x, adapter.lora_A, adapter.lora_B), g)
    expected = reference(ref_x)
    ref_grads = torch.autograd.grad(
        expected, (ref_x, reference.lora_A["default"].weight, reference.lora_B["default"].weight), g
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for a, b in zip(actual_grads, ref_grads):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert base.weight.grad is None
    torch.testing.assert_close(lora_linear(x, base.weight, target), base(x), atol=0, rtol=0)


def test_dropout_recompute_is_exact_and_rng_is_unchanged():
    torch.manual_seed(7)
    config = LoRAConfig(r=2, lora_dropout=0.3, target_modules=("q_proj",))
    a, b = torch.randn(2, 8), torch.randn(8, 2)
    x, w = torch.randn(2, 6, 8), torch.randn(8, 8)
    state = torch.random.get_rng_state().clone()
    outputs = []
    for _ in range(2):
        with lora_context(config, (a, b), training=True, seed=41):
            outputs.append(lora_linear(x, w, "q_proj"))
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    assert torch.equal(state, torch.random.get_rng_state())
    with pytest.raises(RuntimeError), lora_context(config, (a, b), training=True, seed=41):
        raise RuntimeError("test cleanup")
    torch.testing.assert_close(lora_linear(x, w, "q_proj"), torch.nn.functional.linear(x, w))


def make_qwen_backend(device="cpu", policy=None):
    from bloombee.models.qwen3.block import WrappedQwen3Block
    from bloombee.models.qwen3.config import DistributedQwen3Config
    from bloombee.utils.convert_block import QuantType, convert_block

    config = DistributedQwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    block = WrappedQwen3Block(config, layer_idx=0).eval().requires_grad_(False)
    wrapper = convert_block(
        block, 0, config, [torch.device(device)], torch.device(device), QuantType.NONE, policy=policy
    )
    install_lora_hooks(block)
    return SimpleNamespace(config=config, module=wrapper, _ensure_model_on_device=lambda: None), block


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA weight offload test")
def test_qwen_lora_offload_gradients_match_resident_weights_and_restore_homes():
    torch.manual_seed(51)
    resident, _ = make_qwen_backend("cuda:0")
    torch.manual_seed(51)
    offloaded, block = make_qwen_backend("cuda:0", SimpleNamespace(w_cpu_percent=50, w_gpu_percent=50))
    homes = {name: p.data for name, p in block.named_parameters() if p.device.type == "cpu"}
    assert homes and len(homes) < len(list(block.parameters()))
    config = LoRAConfig(r=2, lora_dropout=0.2, target_modules=PROJECTIONS)
    from bloombee.utils.lora import projection_shapes

    shapes = projection_shapes(offloaded.config)
    params = tuple(
        t
        for name in PROJECTIONS
        for t in (torch.randn(2, shapes[name][1], device="cuda"), torch.randn(shapes[name][0], 2, device="cuda"))
    )
    x, g = torch.randn(2, 5, 32, device="cuda"), torch.randn(2, 5, 32, device="cuda")
    metadata = dict(config=config.to_dict(), training=True, seed=87)
    for gradient in (torch.empty(0, device="cuda"), g):
        actual = run_lora_block(offloaded, x, gradient, metadata, *params)
        expected = run_lora_block(resident, x, gradient, metadata, *params)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for name, parameter in block.named_parameters():
            assert parameter.grad is None
            if name in homes:
                assert parameter.device.type == "cpu" and parameter.data_ptr() == homes[name].data_ptr()


def test_qwen_block_gradients_update_only_adapters_and_accumulate():
    torch.manual_seed(13)
    backend, block = make_qwen_backend()
    before = {name: p.detach().clone() for name, p in block.named_parameters()}
    config = LoRAConfig(r=2, lora_alpha=4, target_modules=PROJECTIONS)
    from bloombee.utils.lora import projection_shapes

    shapes = projection_shapes(backend.config)
    adapters = nn.ModuleDict({t: LoRAProjection(shapes[t][1], shapes[t][0], 2) for t in PROJECTIONS})
    params = tuple(p for t in PROJECTIONS for p in (adapters[t].lora_A, adapters[t].lora_B))
    optimizer = torch.optim.AdamW(params, lr=0.01)
    x = torch.randn(2, 5, 32)
    g = torch.randn_like(x) / x.numel()
    meta = {"config": config.to_dict(), "training": True, "seed": 42}
    baseline = run_lora_block(backend, x, torch.empty(0), meta, *params)[0]
    first = run_lora_block(backend, x, g, meta, *params)
    assert first[0].norm() > 0
    for j in range(0, len(params), 2):
        assert torch.count_nonzero(first[j + 1]) == 0  # Standard zero-B initialization.
        assert first[j + 2].norm() > 0
    # Optimizer ownership is local; two microbatches explicitly accumulate.
    for _ in range(2):
        for p, grad in zip(params, first[1:]):
            p.grad = grad.clone() if p.grad is None else p.grad + grad
    optimizer.step()
    optimizer.zero_grad()
    after = run_lora_block(backend, x, torch.empty(0), meta, *params)[0]
    assert not torch.equal(baseline, after)
    second = run_lora_block(backend, x, g, meta, *params)
    assert any(second[j + 1].norm() > 0 for j in range(0, len(params), 2))
    for name, p in block.named_parameters():
        assert p.grad is None and not p.requires_grad
        assert torch.equal(before[name], p)
    with pytest.raises(ValueError, match="shapes"):
        run_lora_block(backend, x, g, meta, *[torch.empty(1) for _ in params])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"r": 0},
        {"r": True},
        {"lora_alpha": float("nan")},
        {"lora_dropout": 1},
        {"target_modules": ["bad"]},
        {"target_modules": ["q_proj", "q_proj"]},
    ],
)
def test_invalid_lora_config(kwargs):
    with pytest.raises(ValueError):
        LoRAConfig(**kwargs)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA master precision test")
        ),
    ],
)
def test_adapter_master_weights_survive_model_half(device):
    layer = LoRAProjection(8, 16, 2).to(device)
    with torch.no_grad():
        layer.lora_B.normal_()
    for p in layer.parameters():
        p.grad = torch.randn_like(p)
    before = [(p.detach().clone(), p.grad.clone()) for p in layer.parameters()]
    layer.half().bfloat16().double()
    for p, (value, grad) in zip(layer.parameters(), before):
        assert p.dtype == p.grad.dtype == torch.float32
        torch.testing.assert_close(p, value, rtol=0, atol=0)
        torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)


def test_fp32_client_fp16_worker_casts_forward_and_gradients():
    backend, block = make_qwen_backend()
    block.half()
    cfg = LoRAConfig(r=2, target_modules=("q_proj",))
    projection = LoRAProjection(32, 32, 2)
    x = torch.randn(1, 4, 32)
    g = torch.randn_like(x)
    metadata = dict(config=cfg.to_dict(), training=True, seed=3)
    output = run_lora_block(backend, x, torch.empty(0), metadata, projection.lora_A, projection.lora_B)[0]
    gradients = run_lora_block(backend, x, g, metadata, projection.lora_A, projection.lora_B)
    assert output.dtype == gradients[0].dtype == torch.float32
    assert torch.isfinite(output).all() and all(torch.isfinite(t).all() for t in gradients)


def test_remote_autograd_runs_for_frozen_inputs_and_detects_early_update(monkeypatch):
    from bloombee.client import lora as client

    async def local_call(manager, span, tensors, metadata):
        backward = metadata["backward"]
        with torch.set_grad_enabled(backward):
            x, a, b = [t.detach().requires_grad_(backward) for t in tensors[:3]]
            y = x + torch.nn.functional.linear(torch.nn.functional.linear(x, a), b)
            return list(torch.autograd.grad(y, (x, a, b), tensors[3])) if backward else [y.detach()]

    monkeypatch.setattr(client, "_call_lora", local_call)
    x = torch.randn(1, 3, 4)  # Frozen embedding, no gradient anchor necessary.
    a, b = nn.Parameter(torch.randn(2, 4)), nn.Parameter(torch.randn(4, 2))
    y = client._RemoteLoRAFunction.apply(x, None, None, {}, a, b)
    assert y.requires_grad
    y.square().mean().backward()
    assert a.grad.norm() > 0 and b.grad.norm() > 0
    stale = client._RemoteLoRAFunction.apply(x, None, None, {}, a, b)
    with torch.no_grad():
        b.add_(0.01)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        stale.sum().backward()


def checkpoint_model():
    # Only serialization is local here; the CLI integration test uses real RPC.
    from bloombee.client.lora import DistributedLoRA

    model = DistributedLoRA.__new__(DistributedLoRA)
    nn.Module.__init__(model)
    model.lora_config = LoRAConfig(r=2, lora_dropout=0.3, target_modules=("q_proj",))
    model.model = nn.Module()
    model.model.config = SimpleNamespace(_name_or_path="test")
    model.model.layers = nn.Module()
    model.model.layers.adapters = nn.ModuleList([nn.ModuleDict({"q_proj": LoRAProjection(4, 4, 2)})])
    model._sequence_path = "layers"
    return model


def test_training_checkpoint_resumes_adamw_sampling_and_dropout_exactly(tmp_path):
    torch.manual_seed(37)
    model = checkpoint_model()
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=0.02)

    def step(current, opt):
        x = torch.randn(2, 5, 4)
        seed = int(torch.randint(2**31, ()))
        p = tuple(current.parameters())
        with lora_context(current.lora_config, p, training=True, seed=seed):
            loss = lora_linear(x, torch.eye(4), "q_proj").square().mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss.detach().clone()

    step(model, optimizer)
    path = tmp_path / "checkpoint"
    model.save_training_checkpoint(path, optimizer, step=1, metadata={"data": "same"})
    expected_loss = step(model, optimizer)
    expected_rng = torch.get_rng_state().clone()
    restored = checkpoint_model()  # Construction intentionally consumes RNG.
    from safetensors.torch import load_file

    with torch.no_grad():
        for key, parameter in restored._adapter_state().items():
            parameter.copy_(load_file(str(path / "adapter_model.safetensors"))[key])
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=123.0)
    assert restored.load_training_state(path, restored_opt) == {"step": 1, "metadata": {"data": "same"}}
    torch.testing.assert_close(step(restored, restored_opt), expected_loss, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for actual, expected in zip(restored.parameters(), model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert restored_opt.param_groups[0]["lr"] == 0.02
    for actual, expected in zip(restored_opt.state.values(), optimizer.state.values()):
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_training_checkpoint_rejects_wrong_optimizer_pending_grads_and_mixed_weights(tmp_path):
    model = checkpoint_model()
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params)
    with pytest.raises(ValueError, match="every LoRA parameter"):
        model.save_training_checkpoint(tmp_path / "incomplete", torch.optim.AdamW(params[:1]), step=0)
    params[0].grad = torch.zeros_like(params[0])
    with pytest.raises(ValueError, match="cleared gradients"):
        model.save_training_checkpoint(tmp_path / "pending", optimizer, step=0)
    optimizer.zero_grad(set_to_none=True)
    path = tmp_path / "valid"
    model.save_training_checkpoint(path, optimizer, step=0)
    with pytest.raises(FileExistsError):
        model.save_training_checkpoint(path, optimizer, step=0)
    with pytest.raises(ValueError, match="parameter order"):
        model.load_training_state(path, torch.optim.AdamW(params[::-1]))
    with pytest.raises(ValueError, match="optimizer type"):
        model.load_training_state(path, torch.optim.SGD(params, lr=0.1))
    config = model.lora_config
    model.lora_config = LoRAConfig(r=2, lora_alpha=32, lora_dropout=0.3, target_modules=("q_proj",))
    with pytest.raises(ValueError, match="LoRA configuration"):
        model.load_training_state(path, optimizer)
    model.lora_config = config
    with torch.no_grad():
        params[0].add_(1)
    with pytest.raises(ValueError, match="checkpoint adapter"):
        model.load_training_state(path, optimizer)
