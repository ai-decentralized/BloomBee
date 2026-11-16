import torch
from transformers import BeamConfig

from bloombee.models.beam.block import WrappedBeamBlock


def config():
    cfg = BeamConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        sliding_window=4,
        num_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        mlp_layer_types=[
            "dense",
            "dense",
            "dense",
            "sparse",
            "sparse",
            "sparse",
        ],
        rope_parameters={
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
            "full_attention": None,
        },
    )
    cfg._attn_implementation = "eager"
    return cfg


def test_dense_sliding_prefill():
    cfg = config()
    block = WrappedBeamBlock(cfg, 0).eval()

    x = torch.randn(1, 3, cfg.hidden_size)

    with torch.no_grad():
        y, cache = block(x, use_cache=True)

    assert y.shape == x.shape
    assert cache is not None


def test_sparse_sliding_prefill_decode():
    cfg = config()
    block = WrappedBeamBlock(cfg, 3).eval()

    x = torch.randn(1, 3, cfg.hidden_size)

    with torch.no_grad():
        y, cache = block(x, use_cache=True)
        z, new_cache = block(
            torch.randn(1, 1, cfg.hidden_size),
            layer_past=cache,
            use_cache=True,
        )

    assert y.shape == x.shape
    assert z.shape == (1, 1, cfg.hidden_size)
    assert new_cache is not None


def test_sparse_full_attention_prefill_decode():
    cfg = config()
    block = WrappedBeamBlock(cfg, 5).eval()

    x = torch.randn(1, 3, cfg.hidden_size)

    with torch.no_grad():
        y, cache = block(x, use_cache=True)
        z, new_cache = block(
            torch.randn(1, 1, cfg.hidden_size),
            layer_past=cache,
            use_cache=True,
        )

    assert y.shape == x.shape
    assert z.shape == (1, 1, cfg.hidden_size)
    assert new_cache is not None


def test_rotary_inv_freq_stays_fp32():
    cfg = config()
    block = WrappedBeamBlock(cfg, 3).to(torch.bfloat16)

    assert (
        block._rotary_emb.sliding_attention_inv_freq.dtype
        == torch.float32
    )
