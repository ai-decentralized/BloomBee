from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from hivemind.utils.logging import get_logger
from torch.nn.parallel import parallel_apply
from transformers.cache_utils import DynamicCache
from transformers import PretrainedConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from tensor_parallel import TensorParallel
from tensor_parallel.cross_device_ops import broadcast_coalesced
from tensor_parallel.slicer_wrapper import TENSOR_PARALLEL_USE_NATIVE
from tensor_parallel.tensor_parallel import parallel_apply_simple
from tensor_parallel.utils import nested_flatten, nested_pack

logger = get_logger(__name__)


def _get_choice(cur_percent, percents, choices):
    cum = np.cumsum(percents)
    for i, boundary in enumerate(cum):
        if cur_percent < boundary:
            return choices[i]
    return choices[-1]


def _assign_param_devices(module: nn.Module, policy, gpu_device: torch.device) -> Dict[str, torch.device]:
    cpu_device = torch.device("cpu")
    param_list = list(module.named_parameters())
    if not param_list:
        return {}

    sizes = np.array([p.numel() for _, p in param_list], dtype=np.float64)
    sizes_cumsum = np.cumsum(sizes)
    total = sizes_cumsum[-1]

    effective_cpu = float(getattr(policy, "w_cpu_percent", 0) + getattr(policy, "w_disk_percent", 0))
    effective_gpu = float(getattr(policy, "w_gpu_percent", 100))
    dev_percents = [0.0, effective_cpu, effective_gpu]
    dev_choices = [cpu_device, cpu_device, gpu_device]

    param_devices = {}
    for i, (name, _) in enumerate(param_list):
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / total * 100
        param_devices[name] = _get_choice(mid_percent, dev_percents, dev_choices)
    return param_devices


class SparseDynamicCache(DynamicCache):
    """DynamicCache that supports sparse/non-zero layer indices."""

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) <= layer_idx:
            missing = layer_idx + 1 - len(self.key_cache)
            self.key_cache.extend([None] * missing)
            self.value_cache.extend([None] * missing)

        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class LlamaTensorParallelAdapter(nn.Module):
    """
    Adapt a tensor_parallel-wrapped HF LLaMA decoder layer to BloomBee's server-side
    block interface.

    BloomBee's current backend expects:
    - one logical output device per block;
    - packed KV cache tensors in Bloom/Petals layout;
    - a `module.devices/module_shards`-like interface.

    The wrapped TP module still computes across multiple GPUs, but the backend sees a
    single logical cache/output device and we reconstruct per-shard HF caches on entry.
    """

    def __init__(self, tp_module: TensorParallel, config: PretrainedConfig, layer_idx: int, policy=None):
        super().__init__()
        self.tp_module = tp_module
        self.config = config
        self.layer_idx = int(layer_idx)
        self.policy = policy
        self._pin_weight = bool(getattr(policy, "pin_weight", False))
        self._param_device_plan = []
        self._shard_has_cpu_offload = []
        self._placement_initialized = False
        self._offload_log_emitted = False

        output_device = tp_module.devices[tp_module.output_device_index]
        self.devices = (output_device,)
        self.output_device = output_device
        self.output_device_index = 0

        # Backend only needs a stable module list for bookkeeping. The actual TP module
        # keeps all internal shards and performs the real cross-device execution.
        self.module_shards = nn.ModuleList([tp_module.module_shards[tp_module.output_device_index]])
        self.shard_num_heads = (config.num_attention_heads,)

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
        del rotary_position_ids  # HF LLaMA computes rotary embeddings from position_ids/cache state.

        batch_size, seq_length, _ = hidden_states.shape
        past_key_values_length = 0
        if layer_past is not None:
            key_states, _ = layer_past
            key_states = getattr(key_states, "data", key_states)
            if torch.is_tensor(key_states) and key_states.ndim >= 3:
                past_key_values_length = key_states.shape[-1]

        if attention_mask is None:
            base_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=hidden_states.device)
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask=base_mask,
                input_shape=(batch_size, seq_length),
                inputs_embeds=hidden_states,
                past_key_values_length=past_key_values_length,
            )
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        elif attention_mask.dim() != 4:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask=attention_mask,
                input_shape=(batch_size, seq_length),
                inputs_embeds=hidden_states,
                past_key_values_length=past_key_values_length,
            )

        outputs = self._parallel_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            layer_past=layer_past,
            **kwargs,
        )
        output_hidden_states, output_cache = outputs[self.tp_module.output_device_index]
        new_kvs = (
            self._hf_cache_to_bloom(
                output_cache,
                batch_size=hidden_states.shape[0],
                new_seq_len=hidden_states.shape[1],
            )
            if use_cache
            else None
        )
        return output_hidden_states, new_kvs

    def _parallel_forward(self, hidden_states: torch.Tensor, **kwargs):
        self._initialize_param_placement()
        self._materialize_cpu_offloaded_params()

        args_and_kwargs = ((hidden_states,), kwargs)
        flat_tensors = [obj for obj in nested_flatten(args_and_kwargs) if isinstance(obj, torch.Tensor)]
        flat_tensors_replicated = broadcast_coalesced(flat_tensors, self.tp_module.devices, all_cuda=self.tp_module.all_cuda)

        next_tensor_index = 0
        args_and_kwargs_replicated = [list() for _ in self.tp_module.device_ids]
        for obj in nested_flatten(args_and_kwargs):
            if isinstance(obj, torch.Tensor):
                for idx in range(len(self.tp_module.module_shards)):
                    args_and_kwargs_replicated[idx].append(flat_tensors_replicated[idx][next_tensor_index])
                next_tensor_index += 1
            else:
                for idx in range(len(self.tp_module.module_shards)):
                    args_and_kwargs_replicated[idx].append(obj)

        for idx in range(len(self.tp_module.module_shards)):
            args_and_kwargs_replicated[idx] = nested_pack(args_and_kwargs_replicated[idx], args_and_kwargs)

        inputs, kwargs_tup = zip(*args_and_kwargs_replicated)
        kwargs_tup = list(kwargs_tup)

        shard_caches = self._build_shard_caches(kwargs.get("layer_past"), batch_size=hidden_states.shape[0])
        for idx in range(len(kwargs_tup)):
            shard_kwargs = dict(kwargs_tup[idx])
            shard_kwargs.pop("layer_past", None)
            shard_kwargs["past_key_value"] = shard_caches[idx]
            kwargs_tup[idx] = shard_kwargs

        try:
            if self.tp_module.all_cuda and not TENSOR_PARALLEL_USE_NATIVE:
                return parallel_apply(self.tp_module.module_shards, inputs, kwargs_tup, self.tp_module.devices)
            return parallel_apply_simple(self.tp_module.module_shards, inputs, kwargs_tup, self.tp_module.devices)
        finally:
            self._restore_cpu_offloaded_params()

    def _build_shard_caches(
        self,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]],
        *,
        batch_size: int,
    ) -> Sequence[DynamicCache]:
        if layer_past is None:
            return [SparseDynamicCache() for _ in self.tp_module.devices]

        key_states, value_states = layer_past
        key_states = getattr(key_states, "data", key_states)
        value_states = getattr(value_states, "data", value_states)
        key_hf, value_hf = self._bloom_cache_to_hf(key_states, value_states, batch_size=batch_size)

        shard_caches = []
        for shard_index, device in enumerate(self.tp_module.devices):
            cache = SparseDynamicCache()
            cache.update(
                key_hf.to(device, non_blocking=True),
                value_hf.to(device, non_blocking=True),
                layer_idx=self._shard_layer_idx(shard_index),
            )
            shard_caches.append(cache)
        return shard_caches

    def _bloom_cache_to_hf(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states.dim() == 4 and value_states.dim() == 4:
            return key_states, value_states

        if key_states.dim() != 3 or value_states.dim() != 3:
            raise ValueError(f"Unsupported packed cache shapes: key={tuple(key_states.shape)}, value={tuple(value_states.shape)}")

        seq_length = value_states.shape[1]
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        num_kv_heads = getattr(self.config, "num_key_value_heads", self.config.num_attention_heads)
        batch_heads = batch_size * num_kv_heads

        if key_states.shape[0] != batch_heads or value_states.shape[0] != batch_heads:
            raise ValueError(
                "Packed cache batch-head mismatch for TP LLaMA block: "
                f"expected BH={batch_heads}, got key={key_states.shape[0]}, value={value_states.shape[0]}"
            )

        if key_states.shape[1] == head_dim and key_states.shape[2] == seq_length:
            key_bhsd = key_states.permute(0, 2, 1)
        elif key_states.shape[1] == seq_length and key_states.shape[2] == head_dim:
            key_bhsd = key_states
        else:
            raise ValueError(f"Unexpected packed key cache shape: {tuple(key_states.shape)}")

        if value_states.shape[1] == seq_length and value_states.shape[2] == head_dim:
            value_bhsd = value_states
        elif value_states.shape[1] == head_dim and value_states.shape[2] == seq_length:
            value_bhsd = value_states.permute(0, 2, 1)
        else:
            raise ValueError(f"Unexpected packed value cache shape: {tuple(value_states.shape)}")

        key_hf = key_bhsd.view(batch_size, num_kv_heads, seq_length, head_dim)
        value_hf = value_bhsd.view(batch_size, num_kv_heads, seq_length, head_dim)
        return key_hf, value_hf

    def _hf_cache_to_bloom(
        self,
        cache: Optional[DynamicCache],
        *,
        batch_size: int,
        new_seq_len: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        layer_idx = self._shard_layer_idx(self.tp_module.output_device_index)
        if cache is None or not cache.key_cache or len(cache.key_cache) <= layer_idx:
            return None

        key_states = cache.key_cache[layer_idx]
        value_states = cache.value_cache[layer_idx]
        if key_states is None or value_states is None:
            return None
        if key_states.dim() != 4 or value_states.dim() != 4:
            raise ValueError(f"Unexpected HF cache shapes: key={tuple(key_states.shape)}, value={tuple(value_states.shape)}")

        if new_seq_len > 0:
            key_states = key_states[:, :, -new_seq_len:, :]
            value_states = value_states[:, :, -new_seq_len:, :]

        _, num_kv_heads, seq_length, head_dim = key_states.shape
        key_states = key_states.reshape(batch_size * num_kv_heads, seq_length, head_dim).permute(0, 2, 1).contiguous().clone()
        value_states = value_states.reshape(batch_size * num_kv_heads, seq_length, head_dim).contiguous().clone()
        return key_states, value_states

    def _shard_layer_idx(self, shard_index: int) -> int:
        shard = self.tp_module.module_shards[shard_index]
        self_attn = getattr(shard, "self_attn", None)
        return int(getattr(self_attn, "layer_idx", self.layer_idx))

    def _initialize_param_placement(self):
        if self._placement_initialized:
            return

        self._param_device_plan = []
        self._shard_has_cpu_offload = []

        for shard, device in zip(self.tp_module.module_shards, self.tp_module.devices):
            if self.policy is None:
                shard.to(device)
                self._param_device_plan.append({})
                self._shard_has_cpu_offload.append(False)
                continue

            param_devices = _assign_param_devices(shard, self.policy, device)
            has_cpu_offload = any(target.type == "cpu" for target in param_devices.values())

            if not has_cpu_offload:
                shard.to(device)
            else:
                with torch.no_grad():
                    for name, param in shard.named_parameters():
                        target = param_devices[name]
                        if target.type == "cpu":
                            cpu_tensor = param.data.detach().to("cpu", non_blocking=False)
                            if self._pin_weight and device.type == "cuda":
                                cpu_tensor = cpu_tensor.pin_memory()
                            param.data = cpu_tensor
                        else:
                            param.data = param.data.detach().to(device, non_blocking=False)

                    for buf_name, buf in list(shard.named_buffers()):
                        if buf is None:
                            continue
                        parts = buf_name.split(".")
                        submod = shard
                        for part in parts[:-1]:
                            submod = getattr(submod, part)
                        submod.register_buffer(parts[-1], buf.to(device), persistent=False)

            self._param_device_plan.append(param_devices)
            self._shard_has_cpu_offload.append(has_cpu_offload)

        if any(self._shard_has_cpu_offload) and not self._offload_log_emitted:
            cpu_param_counts = []
            gpu_param_counts = []
            for plan in self._param_device_plan:
                cpu_count = sum(1 for target in plan.values() if target.type == "cpu")
                cpu_param_counts.append(cpu_count)
                gpu_param_counts.append(len(plan) - cpu_count)
            logger.info(
                "[TP_WEIGHT_OFFLOAD] enabled for TP LLaMA block: cpu_params_per_shard=%s gpu_params_per_shard=%s",
                cpu_param_counts,
                gpu_param_counts,
            )
            self._offload_log_emitted = True

        self.tp_module.need_delayed_init = False
        self._placement_initialized = True

    def _materialize_cpu_offloaded_params(self):
        devices_to_sync = set()
        with torch.no_grad():
            for shard, device, plan, has_cpu_offload in zip(
                self.tp_module.module_shards,
                self.tp_module.devices,
                self._param_device_plan,
                self._shard_has_cpu_offload,
            ):
                if not has_cpu_offload:
                    continue
                for name, param in shard.named_parameters():
                    if plan.get(name, device).type == "cpu" and param.data.device != device:
                        param.data = param.data.detach().to(device, non_blocking=True)
                        devices_to_sync.add(device)

        for device in devices_to_sync:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    def _restore_cpu_offloaded_params(self):
        with torch.no_grad():
            for shard, device, plan, has_cpu_offload in zip(
                self.tp_module.module_shards,
                self.tp_module.devices,
                self._param_device_plan,
                self._shard_has_cpu_offload,
            ):
                if not has_cpu_offload:
                    continue
                for name, param in shard.named_parameters():
                    if plan.get(name, device).type != "cpu" or param.data.device.type == "cpu":
                        continue
                    cpu_tensor = param.data.detach().to("cpu", non_blocking=False)
                    if self._pin_weight and device.type == "cuda":
                        cpu_tensor = cpu_tensor.pin_memory()
                    param.data = cpu_tensor

    def load_lm_head(self, *args, **kwargs):
        return None

    def rms_norm(self, hidden_states: torch.Tensor):
        raise NotImplementedError("Speculative decoding is not supported on the LLaMA tensor-parallel server path yet")

    def lm_head_forward(self, hidden_states: torch.Tensor):
        raise NotImplementedError("Speculative decoding is not supported on the LLaMA tensor-parallel server path yet")
