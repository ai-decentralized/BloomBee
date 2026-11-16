import torch

from bloombee.server.from_pretrained import (
    _remap_deepseekv3_expert_state_dict,
)


def test_beam_style_experts_pack_gate_up_down():
    # Small Beam-shaped synthetic example:
    # 4 experts, hidden=8, moe_intermediate=4.
    state = {}

    for i in range(4):
        state[f"mlp.experts.{i}.gate_proj.weight"] = torch.full(
            (4, 8), float(10 + i)
        )
        state[f"mlp.experts.{i}.up_proj.weight"] = torch.full(
            (4, 8), float(20 + i)
        )
        state[f"mlp.experts.{i}.down_proj.weight"] = torch.full(
            (8, 4), float(30 + i)
        )

    out = _remap_deepseekv3_expert_state_dict(state)

    gate_up = out["mlp.experts.gate_up_proj"]
    down = out["mlp.experts.down_proj"]

    assert gate_up.shape == (4, 8, 8)
    assert down.shape == (4, 8, 4)

    for i in range(4):
        torch.testing.assert_close(
            gate_up[i, :4],
            torch.full((4, 8), float(10 + i)),
        )
        torch.testing.assert_close(
            gate_up[i, 4:],
            torch.full((4, 8), float(20 + i)),
        )
        torch.testing.assert_close(
            down[i],
            torch.full((8, 4), float(30 + i)),
        )


def test_already_packed_beam_experts_are_unchanged():
    gate_up = torch.randn(4, 8, 8)
    down = torch.randn(4, 8, 4)

    state = {
        "mlp.experts.gate_up_proj": gate_up,
        "mlp.experts.down_proj": down,
    }

    out = _remap_deepseekv3_expert_state_dict(state)

    assert out["mlp.experts.gate_up_proj"] is gate_up
    assert out["mlp.experts.down_proj"] is down
