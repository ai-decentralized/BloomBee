"""Int8 drop-in replacement for HF's DeepseekV3Experts.

DeepseekV3Experts batches all routed experts' weights into two 3D nn.Parameter
tensors (gate_up_proj, down_proj) and indexes a slice per active expert at forward
time -- see modeling_deepseek_v3.py. This module keeps those two tensors quantized
to int8 (bloombee.models.deepseekv3.expert_quant) and only dequantizes the slice for
an expert once it's actually selected for the current batch of tokens, mirroring the
reference forward loop exactly except for that one substitution.

The router (DeepseekV3TopkRouter / `.gate`) is untouched -- it's <1% of the block's
parameters, and quantizing it would risk corrupting routing decisions for a
disproportionate hit to output quality relative to the storage it would save.
"""
import torch
import torch.nn as nn

from bloombee.models.deepseekv3.expert_quant import dequantize_expert_weight, quantize_expert_weight


class QuantizedDeepseekV3Experts(nn.Module):
    def __init__(self, experts: nn.Module, group_size: int = 128):
        super().__init__()
        self.num_experts = experts.num_experts
        self.hidden_dim = experts.hidden_dim
        self.intermediate_dim = experts.intermediate_dim
        self.act_fn = experts.act_fn

        gate_up_data, gate_up_scale, gate_up_group_size = quantize_expert_weight(
            experts.gate_up_proj.data, group_size
        )
        down_data, down_scale, down_group_size = quantize_expert_weight(experts.down_proj.data, group_size)

        self.register_buffer("gate_up_data", gate_up_data)
        self.register_buffer("gate_up_scale", gate_up_scale)
        self.gate_up_group_size = gate_up_group_size

        self.register_buffer("down_data", down_data)
        self.register_buffer("down_scale", down_scale)
        self.down_group_size = down_group_size

    def _apply(self, fn, recurse=True):
        # Preserve scale precision even when a parent module calls .half()/.to(dtype).
        scales = {name: self._buffers[name] for name in ("gate_up_scale", "down_scale")}
        result = super()._apply(fn, recurse=recurse)
        for name, scale in scales.items():
            self._buffers[name] = scale.to(device=self._buffers[name].device, dtype=torch.float32)
        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            gate_up_w = dequantize_expert_weight(
                self.gate_up_data, self.gate_up_scale, expert_idx, self.gate_up_group_size, current_state.dtype
            )
            gate, up = nn.functional.linear(current_state, gate_up_w).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up

            down_w = dequantize_expert_weight(
                self.down_data, self.down_scale, expert_idx, self.down_group_size, current_state.dtype
            )
            current_hidden_states = nn.functional.linear(current_hidden_states, down_w)

            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states
