from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from hivemind.utils.logging import get_logger
from transformers import PretrainedConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from bloombee.flexgen_utils.ExecutionEnv import ExecutionEnv
from bloombee.flexgen_utils.policy import Policy
from bloombee.flexgen_utils.pytorch_backend import TorchDevice, TorchMixedDevice, TorchTensor
from bloombee.flexgen_utils.llama_config import download_llama_weights

logger = get_logger(__name__)
DUMMY_WEIGHT = "_DUMMY_"


def _build_score_attention_mask(
    batch_size: int,
    query_length: int,
    past_key_values_length: int,
    device: torch.device,
) -> torch.Tensor:
    total_length = query_length + past_key_values_length
    scores = torch.full(
        (batch_size, 1, query_length, total_length),
        -65504.0,
        dtype=torch.float32,
        device=device,
    )
    for i in range(query_length):
        scores[:, :, i, : past_key_values_length + i + 1] = 0.0
    return scores


def _infer_llama_model_name(config: PretrainedConfig) -> str:
    for attr in ("name", "name_or_path", "_name_or_path"):
        value = getattr(config, attr, None)
        if value and isinstance(value, str):
            model_name = os.path.basename(value.rstrip("/"))
            if model_name.endswith("-hf"):
                model_name = model_name[:-3]
            return model_name

    h = getattr(config, "hidden_size", 0)
    layers = getattr(config, "num_hidden_layers", 0)
    intermediate = getattr(config, "intermediate_size", 0)
    if (h, layers) == (4096, 32):
        return "llama-7b"
    if (h, layers) == (5120, 40):
        return "llama-13b"
    if (h, layers) == (6656, 60):
        return "llama-30b"
    if (h, layers) == (8192, 80):
        return "llama-70b" if intermediate == 28672 else "llama-65b"
    return "llama-7b"


def _resolve_expanded_path(config: PretrainedConfig, path: str) -> str:
    model_name = _infer_llama_model_name(config)
    expanded_path = os.path.abspath(os.path.expanduser(os.path.join(path, f"{model_name}-np")))
    check_path = os.path.join(expanded_path, "embed_tokens.weight")
    if not os.path.exists(check_path) and DUMMY_WEIGHT not in check_path:
        download_llama_weights(model_name, path)
    return expanded_path


def _load_array_slice(filename: str, row_slice: Optional[slice] = None, col_slice: Optional[slice] = None) -> np.ndarray:
    array = np.load(filename, mmap_mode="r")
    if row_slice is not None and col_slice is not None:
        array = array[row_slice, col_slice]
    elif row_slice is not None:
        array = array[row_slice]
    elif col_slice is not None:
        array = array[:, col_slice]
    return np.ascontiguousarray(array)


def _load_weight_into_tensor(
    tensor: TorchTensor,
    filename: str,
    *,
    row_slice: Optional[slice] = None,
    col_slice: Optional[slice] = None,
) -> None:
    if DUMMY_WEIGHT in filename:
        tensor.load_from_np(np.ones(tensor.shape, dtype=np.float16))
        return
    sliced = _load_array_slice(filename, row_slice=row_slice, col_slice=col_slice)
    tensor.load_from_np(sliced.astype(np.float16, copy=False))


def _create_shard_env(device: torch.device, base_env: ExecutionEnv) -> ExecutionEnv:
    gpu = TorchDevice(str(device))
    cpu = base_env.cpu
    disk = base_env.disk
    mixed = TorchMixedDevice([gpu, cpu, disk])
    return ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=mixed)


def _as_torch_tensor(value):
    try:
        from bloombee.flexgen_utils.pytorch_backend import DeviceType

        if hasattr(value, "device") and (
            getattr(getattr(value, "device", None), "device_type", None) == DeviceType.COMPRESSED
            or (hasattr(value, "data") and isinstance(getattr(value, "data"), tuple) and len(getattr(value, "data")) == 3)
        ):
            return value.device.decompress(value)
    except Exception:
        pass
    return getattr(value, "data", value)


@dataclass(frozen=True)
class _ShardLayout:
    rank: int
    q_head_start: int
    q_head_end: int
    q_hidden_start: int
    q_hidden_end: int
    ffn_start: int
    ffn_end: int

    @property
    def local_heads(self) -> int:
        return self.q_head_end - self.q_head_start

    @property
    def local_hidden(self) -> int:
        return self.q_hidden_end - self.q_hidden_start

    @property
    def local_intermediate(self) -> int:
        return self.ffn_end - self.ffn_start


class _FlexgenLlamaShard(nn.Module):
    def __init__(
        self,
        *,
        config: PretrainedConfig,
        policy: Policy,
        base_env: ExecutionEnv,
        layer_idx: int,
        device: torch.device,
        layout: _ShardLayout,
        expanded_path: str,
    ):
        super().__init__()
        self.config = config
        self.policy = policy
        self.layer_idx = int(layer_idx)
        self.device = torch.device(device)
        self.layout = layout
        self.env = _create_shard_env(self.device, base_env)
        self.compute = self.env.gpu
        self.weight_load_dst = self.compute.compressed_device if policy.compress_weight else self.compute
        self.attention_compute = self.env.cpu if self.policy.cpu_cache_compute else self.env.gpu
        self.expanded_path = expanded_path

        self.attn_weight_home = self._init_attention_weights()
        self.mlp_weight_home = self._init_mlp_weights()

    def _layer_path(self) -> str:
        return os.path.join(self.expanded_path, f"layers.{self.layer_idx}.")

    def _init_attention_weights(self):
        from bloombee.models.llama.flex_llama import init_weight_list

        h = self.config.hidden_size
        local_hidden = self.layout.local_hidden
        specs = [
            ((local_hidden, h), np.float16, DUMMY_WEIGHT),
            ((local_hidden, h), np.float16, DUMMY_WEIGHT),
            ((local_hidden, h), np.float16, DUMMY_WEIGHT),
            ((h, local_hidden), np.float16, DUMMY_WEIGHT),
            ((h,), np.float16, DUMMY_WEIGHT),
            ((h // self.config.num_attention_heads // 2,), np.float16, DUMMY_WEIGHT),
        ]
        weights = init_weight_list(specs, self.policy, self.env)
        path = self._layer_path()
        row_slice = slice(self.layout.q_hidden_start, self.layout.q_hidden_end)
        col_slice = slice(self.layout.q_hidden_start, self.layout.q_hidden_end)
        _load_weight_into_tensor(weights[0], path + "self_attn.q_proj.weight", row_slice=row_slice)
        _load_weight_into_tensor(weights[1], path + "self_attn.k_proj.weight", row_slice=row_slice)
        _load_weight_into_tensor(weights[2], path + "self_attn.v_proj.weight", row_slice=row_slice)
        _load_weight_into_tensor(weights[3], path + "self_attn.o_proj.weight", col_slice=col_slice)
        _load_weight_into_tensor(weights[4], path + "input_layernorm.weight")
        _load_weight_into_tensor(weights[5], path + "self_attn.rotary_emb.inv_freq")
        return tuple(weights)

    def _init_mlp_weights(self):
        from bloombee.models.llama.flex_llama import init_weight_list

        h = self.config.hidden_size
        local_intermediate = self.layout.local_intermediate
        specs = [
            ((local_intermediate, h), np.float16, DUMMY_WEIGHT),
            ((h, local_intermediate), np.float16, DUMMY_WEIGHT),
            ((local_intermediate, h), np.float16, DUMMY_WEIGHT),
            ((h,), np.float16, DUMMY_WEIGHT),
        ]
        weights = init_weight_list(specs, self.policy, self.env)
        path = self._layer_path()
        row_slice = slice(self.layout.ffn_start, self.layout.ffn_end)
        col_slice = slice(self.layout.ffn_start, self.layout.ffn_end)
        _load_weight_into_tensor(weights[0], path + "mlp.gate_proj.weight", row_slice=row_slice)
        _load_weight_into_tensor(weights[1], path + "mlp.down_proj.weight", col_slice=col_slice)
        _load_weight_into_tensor(weights[2], path + "mlp.up_proj.weight", row_slice=row_slice)
        _load_weight_into_tensor(weights[3], path + "post_attention_layernorm.weight")
        return tuple(weights)

    def _copy_weight_bundle(self, home_weights, compute_mask):
        live_weights = []
        owned_weights = []
        for weight, use_compute in zip(home_weights, compute_mask):
            dst = self.compute if use_compute else self.weight_load_dst
            copied, owned = weight.smart_copy(dst)
            live_weights.append(copied)
            owned_weights.append((copied, owned))
        return tuple(live_weights), owned_weights

    def _cleanup_owned_weights(self, owned_weights) -> None:
        for copied, owned in owned_weights:
            if owned and hasattr(copied, "delete"):
                copied.delete()

    def load_attention_weights(self):
        return self._copy_weight_bundle(self.attn_weight_home, (False, False, False, False, True, True))

    def load_mlp_weights(self):
        return self._copy_weight_bundle(self.mlp_weight_home, (False, False, False, True))

    def _local_layer_past(
        self,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[Tuple[TorchTensor, TorchTensor]]:
        if layer_past is None:
            return None
        key_states, value_states = layer_past
        if key_states.dim() == 3 and value_states.dim() == 3:
            total_heads = self.config.num_attention_heads
            head_dim = self.config.hidden_size // total_heads
            bh, d1, d2 = key_states.shape
            if total_heads <= 0 or bh % total_heads != 0:
                raise ValueError(
                    f"Unexpected Bloom-style cache shape for TP layer past: key={tuple(key_states.shape)} "
                    f"value={tuple(value_states.shape)} total_heads={total_heads}"
                )
            batch_size = bh // total_heads
            if d2 == head_dim:
                seq_len = d1
                key_bhsd = key_states
            elif d1 == head_dim:
                seq_len = d2
                key_bhsd = key_states.permute(0, 2, 1)
            else:
                raise ValueError(
                    f"Unable to infer head_dim={head_dim} from key cache shape {tuple(key_states.shape)}"
                )

            if value_states.shape[-1] == head_dim:
                value_bhsd = value_states
            elif value_states.shape[1] == head_dim:
                value_bhsd = value_states.permute(0, 2, 1)
            else:
                raise ValueError(
                    f"Unable to infer head_dim={head_dim} from value cache shape {tuple(value_states.shape)}"
                )

            key_states = key_bhsd.view(batch_size, total_heads, seq_len, head_dim)
            value_states = value_bhsd.view(batch_size, total_heads, seq_len, head_dim)
        local_key = key_states[:, self.layout.q_head_start : self.layout.q_head_end].to(self.device, non_blocking=True)
        local_value = value_states[:, self.layout.q_head_start : self.layout.q_head_end].to(self.device, non_blocking=True)
        bsz, local_heads, seq_len, head_dim = local_key.shape
        key_sbhd = local_key.permute(2, 0, 1, 3).reshape(seq_len, bsz * local_heads, head_dim)
        value_sbhd = local_value.permute(2, 0, 1, 3).reshape(seq_len, bsz * local_heads, head_dim)
        return (
            TorchTensor.create_from_torch(key_sbhd, self.compute),
            TorchTensor.create_from_torch(value_sbhd, self.compute),
        )

    def attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        rotary_position_ids: Optional[torch.LongTensor],
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]],
        live_weights,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_tt = TorchTensor.create_from_torch(hidden_states.to(self.device, non_blocking=True), self.compute)
        mask_tt = TorchTensor.create_from_torch(attention_mask.to(self.device, non_blocking=True), self.attention_compute)
        rotary_ids = rotary_position_ids.to(self.device, non_blocking=True) if rotary_position_ids is not None else None
        donate = [False] * 16
        w_q, w_k, w_v, w_out, input_layernorm, rotary_emb_inv_freq = live_weights

        local_past = self._local_layer_past(layer_past)
        if local_past is None:
            output_tt, k_new, v_new = self.compute.mha_llama(
                hidden_tt,
                mask_tt,
                w_q,
                w_k,
                w_v,
                w_out,
                self.layout.local_heads,
                donate,
                self.policy.compress_cache,
                self.policy.comp_cache_config,
                input_layernorm,
                rotary_emb_inv_freq,
                rotary_ids,
            )
        else:
            k_cache, v_cache = local_past
            output_tt, k_new, v_new = self.compute.mha_gen_llama(
                hidden_tt,
                mask_tt,
                w_q,
                w_k,
                w_v,
                w_out,
                self.layout.local_heads,
                k_cache,
                v_cache,
                donate,
                self.policy.attn_sparsity,
                self.policy.compress_cache,
                self.policy.comp_cache_config,
                input_layernorm,
                rotary_emb_inv_freq,
                rotary_ids,
            )

        return _as_torch_tensor(output_tt), _as_torch_tensor(k_new), _as_torch_tensor(v_new)

    def mlp_forward(self, hidden_states: torch.Tensor, live_weights) -> torch.Tensor:
        hidden_tt = TorchTensor.create_from_torch(hidden_states.to(self.device, non_blocking=True), self.compute)
        gate, down, up, post_attention_layernorm = live_weights
        donate = [False] * 9
        output_tt = self.compute.mlp_llama(
            hidden_tt,
            gate,
            down,
            up,
            donate,
            self.config,
            post_attention_layernorm,
        )
        return _as_torch_tensor(output_tt)


class FlexgenLlamaTensorParallel(nn.Module):
    def __init__(
        self,
        block_template: nn.Module,
        config: PretrainedConfig,
        devices: Sequence[torch.device],
        output_device: torch.device,
        *,
        policy: Policy,
    ):
        super().__init__()
        from bloombee.models.llama.flex_llama import FLEX_LlamaRMSNorm, SimpleLMHead

        if len(devices) < 2:
            raise ValueError("FlexgenLlamaTensorParallel requires at least two devices")

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        if num_kv_heads != config.num_attention_heads:
            raise ValueError(
                "FlexGen-native TP currently requires num_key_value_heads == num_attention_heads "
                f"(got kv={num_kv_heads}, attn={config.num_attention_heads})"
            )
        if config.num_attention_heads % len(devices) != 0:
            raise ValueError(
                f"num_attention_heads={config.num_attention_heads} must be divisible by tp_world_size={len(devices)}"
            )
        if config.intermediate_size % len(devices) != 0:
            raise ValueError(
                f"intermediate_size={config.intermediate_size} must be divisible by tp_world_size={len(devices)}"
            )

        self.config = config
        self.policy = policy
        self.output_device = torch.device(output_device)
        self.devices = (self.output_device,)
        self.output_device_index = 0
        self.shard_num_heads = (config.num_attention_heads,)
        self.need_delayed_init = False
        self.layer_idx = int(getattr(block_template, "layer_id", getattr(block_template, "layer_idx", 0)))
        self.expanded_path = _resolve_expanded_path(config, getattr(block_template, "path", "."))

        local_heads = config.num_attention_heads // len(devices)
        head_dim = config.hidden_size // config.num_attention_heads
        local_hidden = local_heads * head_dim
        local_intermediate = config.intermediate_size // len(devices)

        shard_layouts = []
        for rank in range(len(devices)):
            q_head_start = rank * local_heads
            q_head_end = q_head_start + local_heads
            q_hidden_start = rank * local_hidden
            q_hidden_end = q_hidden_start + local_hidden
            ffn_start = rank * local_intermediate
            ffn_end = ffn_start + local_intermediate
            shard_layouts.append(
                _ShardLayout(
                    rank=rank,
                    q_head_start=q_head_start,
                    q_head_end=q_head_end,
                    q_hidden_start=q_hidden_start,
                    q_hidden_end=q_hidden_end,
                    ffn_start=ffn_start,
                    ffn_end=ffn_end,
                )
            )

        self.tp_shards = nn.ModuleList(
            [
                _FlexgenLlamaShard(
                    config=config,
                    policy=policy,
                    base_env=block_template.env,
                    layer_idx=self.layer_idx,
                    device=torch.device(device),
                    layout=layout,
                    expanded_path=self.expanded_path,
                )
                for device, layout in zip(devices, shard_layouts)
            ]
        )
        self.module_shards = nn.ModuleList([self.tp_shards[0]])

        self.input_layernorm = FLEX_LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = SimpleLMHead(hidden_size=config.hidden_size, vocab_size=config.vocab_size)
        self.requires_grad_(False)

        logger.info(
            "[FLEXGEN_TP] enabled for LLaMA block %s across devices=%s local_heads=%s local_intermediate=%s",
            self.layer_idx,
            [str(device) for device in devices],
            local_heads,
            local_intermediate,
        )

    def _synchronize_weight_loads(self) -> None:
        first_shard = self.tp_shards[0]
        first_shard.env.disk.synchronize()
        for shard in self.tp_shards:
            if shard.device.type == "cuda":
                torch.cuda.synchronize(shard.device)

    def _reduce_partials(self, partials: List[torch.Tensor], residual_input: torch.Tensor) -> torch.Tensor:
        reduced = None
        for partial in partials:
            partial_out = partial.to(self.output_device, non_blocking=True)
            reduced = partial_out if reduced is None else reduced + partial_out
        return reduced - (len(partials) - 1) * residual_input.to(self.output_device, non_blocking=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.LongTensor] = None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        del position_ids, kwargs
        if attention_mask is None:
            batch_size, seq_length, _ = hidden_states.shape
            past_key_values_length = 0
            if layer_past is not None:
                key_states, _ = layer_past
                past_key_values_length = key_states.shape[-1] if key_states.dim() == 3 else key_states.shape[2]
            attention_mask = _build_score_attention_mask(
                batch_size=batch_size,
                query_length=seq_length,
                past_key_values_length=past_key_values_length,
                device=hidden_states.device,
            )
        elif attention_mask.dim() == 3:
            # FlexGen prefill expects a singleton head axis so mask broadcasts over local heads.
            attention_mask = attention_mask.unsqueeze(1)
        elif attention_mask.dim() != 4:
            batch_size, seq_length, _ = hidden_states.shape
            past_key_values_length = 0
            if layer_past is not None:
                key_states, _ = layer_past
                past_key_values_length = key_states.shape[2]
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask=attention_mask,
                input_shape=(batch_size, seq_length),
                inputs_embeds=hidden_states,
                past_key_values_length=past_key_values_length,
            )

        attn_live = []
        attn_owned = []
        for shard in self.tp_shards:
            live_weights, owned = shard.load_attention_weights()
            attn_live.append(live_weights)
            attn_owned.append(owned)
        self._synchronize_weight_loads()

        attn_partials: List[torch.Tensor] = []
        k_parts: List[torch.Tensor] = []
        v_parts: List[torch.Tensor] = []
        try:
            for shard, live_weights in zip(self.tp_shards, attn_live):
                partial_hidden, local_k, local_v = shard.attention_forward(
                    hidden_states,
                    attention_mask,
                    rotary_position_ids,
                    layer_past,
                    live_weights,
                )
                attn_partials.append(partial_hidden)
                k_parts.append(local_k)
                v_parts.append(local_v)
        finally:
            for shard, owned in zip(self.tp_shards, attn_owned):
                shard._cleanup_owned_weights(owned)

        attn_output = self._reduce_partials(attn_partials, hidden_states)

        mlp_live = []
        mlp_owned = []
        for shard in self.tp_shards:
            live_weights, owned = shard.load_mlp_weights()
            mlp_live.append(live_weights)
            mlp_owned.append(owned)
        self._synchronize_weight_loads()

        mlp_partials: List[torch.Tensor] = []
        try:
            for shard, live_weights in zip(self.tp_shards, mlp_live):
                mlp_partials.append(shard.mlp_forward(attn_output, live_weights))
        finally:
            for shard, owned in zip(self.tp_shards, mlp_owned):
                shard._cleanup_owned_weights(owned)

        output_hidden_states = self._reduce_partials(mlp_partials, attn_output)

        if not use_cache:
            return output_hidden_states, None

        k_cat = torch.cat([k_part.to(self.output_device, non_blocking=True) for k_part in k_parts], dim=1)
        v_cat = torch.cat([v_part.to(self.output_device, non_blocking=True) for v_part in v_parts], dim=1)
        key = k_cat.permute(1, 2, 0).contiguous()
        value = v_cat.permute(1, 0, 2).contiguous()
        return output_hidden_states, (key, value)

    def rms_norm(self, hidden_states: torch.Tensor):
        return self.input_layernorm.forward(hidden_states)

    def lm_head_forward(self, hidden_states: torch.Tensor):
        return self.lm_head.forward(hidden_states)

    def load_lm_head(self):
        self.input_layernorm.load_weight(self.expanded_path)
        self.lm_head.load_weight(self.expanded_path)
