import torch

from bloombee.server.from_pretrained import (
    _remap_nemotron_h_expert_state_dict,
)


def test_nemotron_experts_are_packed_in_index_order():
    state = {
        "mixer.experts.0.up_proj.weight": torch.full((2, 3), 10.0),
        "mixer.experts.0.down_proj.weight": torch.full((3, 2), 20.0),
        "mixer.experts.1.up_proj.weight": torch.full((2, 3), 11.0),
        "mixer.experts.1.down_proj.weight": torch.full((3, 2), 21.0),
        "mixer.gate.weight": torch.randn(2, 3),
    }

    result = _remap_nemotron_h_expert_state_dict(state)

    assert result["mixer.experts.up_proj"].shape == (2, 2, 3)
    assert result["mixer.experts.down_proj"].shape == (2, 3, 2)

    assert torch.equal(
        result["mixer.experts.up_proj"][0],
        torch.full((2, 3), 10.0),
    )
    assert torch.equal(
        result["mixer.experts.up_proj"][1],
        torch.full((2, 3), 11.0),
    )

    assert "mixer.gate.weight" in result


def test_nemotron_remap_noop_for_dense_state():
    weight = torch.randn(3, 3)
    state = {"mixer.in_proj.weight": weight}

    result = _remap_nemotron_h_expert_state_dict(state)

    assert result["mixer.in_proj.weight"] is weight
