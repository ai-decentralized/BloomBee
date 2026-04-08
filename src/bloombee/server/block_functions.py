"""
This module implements server-side computations on served blocks: forward, backward and inference; used by handler
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional, Sequence, Tuple, Union
import os

import torch
from hivemind.moe.expert_uid import ExpertUID
from hivemind.proto import runtime_pb2
from hivemind.utils.logging import get_logger
from hivemind.utils.nested import nested_flatten

from bloombee.data_structures import Handle, InferenceMetadata
from bloombee.server.backend import TransformerBackend
from bloombee.server.task_pool import PrioritizedTaskPool
from bloombee.server.task_prioritizer import TaskPrioritizerBase
from bloombee.utils.convert_block import QuantType
from bloombee.utils.lossless_transport import deserialize_torch_tensor, serialize_torch_tensor
from bloombee.utils.misc import DUMMY, is_dummy
from bloombee.utils.packaging import unpack_args_kwargs
from bloombee.server.speculative_pruner.pruner_manager import SpeculativePrunerManager
from bloombee.utils.microbatch_config import (
    is_microbatch_enabled,
    get_micro_batch_size,
    should_split_batch,
    compute_micro_batch_ranges,
    split_tensor_to_microbatches,
    merge_microbatch_outputs,
    log_microbatch_split,
    log_microbatch_merge,
    log_stage_timing,
    get_timing_tracker,
    MBPIPE_LOG_PREFIX,
)
from bloombee.utils.microbatch_schema import (
    fill_microbatch_defaults,
    RequestContext,
    create_microbatch_result_metadata,
    MBPIPE_SCHEMA_PREFIX,
)
from bloombee.utils.debug import dprint
import traceback

# [MBPIPE] Cross-stage streaming push support
_cross_stage_push_callback = None  # Will be set by handler for cross-stage streaming


from time import perf_counter
from datetime import datetime, timezone  
def print_time_now(s):
    # Get the current time in UTC  
    current_utc_datetime = datetime.now(timezone.utc)  
    # Format the datetime to the desired string format  
    formatted_utc_time = current_utc_datetime.strftime('%Y-%m-%d %H:%M:%S.%f %Z')  
    dprint('\t\t\t'+s+" UTC Time: "+ str(formatted_utc_time) )  
    

# We prioritize short inference requests and make them use a *merged* inference pool,
# so they are processed without interruptions and extra overheads
# Note: NF4 refers to FlexGen's 4-bit group quantization, not bitsandbytes
MAX_SHORT_INFERENCE_TOKENS = 8192000
MAX_NF4_SHORT_INFERENCE_TOKENS = 1

logger = get_logger(__name__)

# Create dedicated offloading debug logger
import logging
offload_logger = logging.getLogger('bloombee.offloading')
offload_logger.setLevel(logging.INFO)


def _mbpipe_verbose_enabled() -> bool:
    """Enable full per-micro-batch logs via BLOOMBEE_MBPIPE_VERBOSE=1."""
    return os.environ.get("BLOOMBEE_MBPIPE_VERBOSE", "0") == "1"


def _should_log_mb_detail(step_metadata: Optional[Dict[str, Any]]) -> bool:
    """
    Control per-step micro-batch detail logs.
    - verbose mode: always on
    - default mode: sampled by BLOOMBEE_MBPIPE_LOG_EVERY_STEPS (default=16)
    """
    if _mbpipe_verbose_enabled():
        return True
    try:
        every_n = max(1, int(os.environ.get("BLOOMBEE_MBPIPE_LOG_EVERY_STEPS", "16")))
    except Exception:
        every_n = 16
    if not isinstance(step_metadata, dict):
        return True
    pos = step_metadata.get("start_from_position")
    if not isinstance(pos, int):
        return True
    return pos <= 2 or (pos % every_n == 0)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_python_bool(value: Any) -> bool:
    """Safely normalize scalar/tensor flags to a Python bool."""
    if value is None:
        return False
    if torch.is_tensor(value):
        if is_dummy(value):
            return False
        if value.numel() == 0:
            return False
        return bool(value.bool().any().item())
    return bool(value)


def _effective_token_increment(
    hidden_states: torch.Tensor,
    kv_cache_position_ids: Any,
    is_spec_dec: Any,
) -> int:
    """
    Compute logical token increment for session position accounting.
    For speculative decoding, align with client-side logic:
    use kv_cache_position_ids[0].numel() when available.
    """
    default_inc = int(hidden_states.shape[1]) if torch.is_tensor(hidden_states) and hidden_states.ndim >= 2 else 0
    if not _as_python_bool(is_spec_dec):
        return default_inc
    if kv_cache_position_ids is None or is_dummy(kv_cache_position_ids):
        return default_inc
    if not torch.is_tensor(kv_cache_position_ids):
        try:
            kv_cache_position_ids = torch.as_tensor(kv_cache_position_ids)
        except Exception:
            return default_inc
    if kv_cache_position_ids.numel() == 0:
        return 0
    return 1


def _slice_batch_aligned(
    value: Any,
    mb_start: int,
    mb_end: int,
    full_batch_size: int,
) -> Any:
    """
    Slice tensor-like request fields only if they are batch-aligned.
    Non-tensor / scalar / already-global fields are returned as-is.
    """
    if value is None or not torch.is_tensor(value):
        return value
    if is_dummy(value):
        return value
    if value.ndim == 0:
        return value
    if value.shape[0] == full_batch_size:
        return value[mb_start:mb_end].contiguous()
    return value

def _create_tree_position_ids_with_invalid_cache(
    width: int,
    depth: int,
    prefill_length: torch.Tensor,           # (B,)
    kv_cache_position_ids: Optional[torch.Tensor],  # (B, max_pos_len) 或 None, -1 是 padding
    batch_offset,
    device: torch.device,
    target_seq_len: Optional[int] = None,
) -> torch.Tensor:
    B = prefill_length.shape[0]
    device = torch.device(device)

    # 1. 生成 Tree 模板
    tree_position_ids_list = []
    def dfs_generate(node_depth, current_depth):
        tree_position_ids_list.append(node_depth)
        if current_depth < depth:
            for _ in range(width):
                dfs_generate(node_depth + 1, current_depth + 1)
    dfs_generate(0, 0)
    tree_len = len(tree_position_ids_list)
    tree_position_ids = torch.tensor(tree_position_ids_list, dtype=torch.long, device=device)
    
    # 2. 判断是否为 Prefill 阶段
    is_prefill = kv_cache_position_ids is None or kv_cache_position_ids.numel() == 0
    prefill_length = prefill_length.to(device)

    if is_prefill:
        # ── Prefill 阶段 ──────────────────────────────────────────────────
        max_prefill_len = prefill_length.max().item()
        if target_seq_len is not None:
            total_len = target_seq_len
            prompt_part_len = target_seq_len - tree_len
        else:
            total_len = max_prefill_len + tree_len
            prompt_part_len = max_prefill_len

        full_position_ids = torch.zeros(B, total_len, dtype=torch.long, device=device)
        
        
        # Prompt 部分的 position ids

        # Prompt 部分的 position ids
        if prompt_part_len > 0:
            prefill_positions = torch.arange(prompt_part_len, dtype=torch.long, device=device)
            full_position_ids[:, :prompt_part_len] = prefill_positions.unsqueeze(0)

        tree_base = prefill_length.unsqueeze(1)  # (B, 1)
        full_position_ids[:, prompt_part_len:] = tree_base + tree_position_ids.unsqueeze(0)

        return full_position_ids

    else:
        kv_cache_position_ids = kv_cache_position_ids.to(device)
        kv_cache_position_ids = _slice_batch_aligned(
            kv_cache_position_ids, batch_offset, batch_offset + B, kv_cache_position_ids.shape[0]
        )
        valid_mask = kv_cache_position_ids >= 0  # (B, max_pos_len)

        batch_indices = torch.arange(B, device=device)
        has_valid = valid_mask.any(dim=1)
        first_valid_idx = valid_mask.int().argmax(dim=1)
        root_positions = kv_cache_position_ids[batch_indices, first_valid_idx]
        root_positions = root_positions * has_valid

        tree_valid_counts = valid_mask.sum(dim=1)
        base_positions = root_positions + tree_valid_counts        # (B,)

        # 处理 target_seq_len（不修改传入的 tensor）
        actual_tree_ids = tree_position_ids
        if target_seq_len is not None and target_seq_len != tree_len:
            pad = target_seq_len - tree_len
            if pad > 0:
                last_val = tree_position_ids[-1].item()
                actual_tree_ids = torch.cat([
                    tree_position_ids,
                    torch.full((pad,), last_val, dtype=torch.long, device=device)
                ])
            else:
                actual_tree_ids = tree_position_ids[:target_seq_len]

        position_ids = base_positions.unsqueeze(1) + actual_tree_ids.unsqueeze(0)

        return position_ids
        


def _unpack_inference_submit_result(result: Any) -> Tuple[torch.Tensor, Any, Optional[Dict[str, float]]]:
    """
    Backward-compatible unpack for inference_pool.submit_task():
    - legacy shape: (hidden_states, keep_indices)
    - extended shape: (hidden_states, keep_indices, runtime_kv_timing_dict)
    """
    if not isinstance(result, (tuple, list)) or len(result) < 2:
        raise RuntimeError(f"Unexpected inference_pool result: type={type(result)}, value={result!r}")
    hidden_states = result[0]
    keep_indices = result[1]
    runtime_kv_timing = result[2] if len(result) >= 3 and isinstance(result[2], dict) else None
    return hidden_states, keep_indices, runtime_kv_timing


async def run_rpc_forward(
    *flat_tensors: torch.Tensor,
    requested_backends: Sequence[TransformerBackend],
    active_adapter: str = "",
    prioritizer: TaskPrioritizerBase,
    points: int = 0,
    args_structure: Any = None,
) -> torch.Tensor:
    """
    Run forward pass on deserialized inputs and prompts, used by rpc_forward and rpc_forward_stream

    :param flat_tensors: a list of tensors that includes first layer inputs, optional prompts and extra tensors
    :note: some input tensors can be missing, in which case they will be replaced with dummy tensors (see is_dummy)
    :param requested_backends: a sequence of transformer blocks in the same order as they appear in forward pass
    :returns: hidden states after the last layer [batch_size, seq_length, hid_size]
    """
    # Start timing for Cross-GPU Transfer Latency measurement
    cross_gpu_start_time = perf_counter()
    
    if args_structure is not None:
        # TODO: kwargs currently is unused, it can be used later for peft-like adaptation
        flat_tensors, kwargs = unpack_args_kwargs(flat_tensors, args_structure)
    hidden_states, prompts, *_ = flat_tensors

    # Fix for bus error in cross-machine setups: ensure tensors are contiguous
    # Deserialized tensors from network buffers may not be properly aligned,
    # especially for certain batch sizes (e.g., batch_size=16)
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    if prompts is not None and not is_dummy(prompts) and not prompts.is_contiguous():
        prompts = prompts.contiguous()

    dtype = requested_backends[0].dtype
    # check parse input tensors and cast dtypes
    hidden_states = hidden_states.to(dtype)
    assert hidden_states.ndim == 3
    if prompts is None or is_dummy(prompts):
        prompts = [DUMMY] * len(requested_backends)
    else:
        prompts = [p.squeeze(0) for p in prompts.to(requested_backends[0].dtype).split(1, dim=0)]

    # Track S1->S2 transfer latency specifically
    s1_to_s2_transfer_times = []
    backend_processing_times = []
    
    # Run a chain of requested backends
    for i, (backend, prompt) in enumerate(zip(requested_backends, prompts)):
        backend_start_time = perf_counter()
        
        if not is_dummy(prompt):
            hidden_states[:, : prompt.shape[1]] += prompt

        assert isinstance(backend.inference_pool, PrioritizedTaskPool), "petals support only prioritized pools"
        priority = prioritizer.prioritize(
            hidden_states, points=points / len(requested_backends), backend=backend, type="forward"
        )
        
        # Submit task and measure processing time
        task_start_time = perf_counter()
        (hidden_states,) = await backend.forward_pool.submit_task(
            hidden_states,
            active_adapter,
            priority=priority,
        )
        task_end_time = perf_counter()
        task_processing_time = (task_end_time - task_start_time) * 1000  # Convert to milliseconds
        
        backend_end_time = perf_counter()
        backend_total_time = (backend_end_time - backend_start_time) * 1000
        
        # Track individual backend processing times
        backend_processing_times.append(task_processing_time)
        
        # Estimate S1->S2 transfer time (this is an approximation)
        # The transfer time is roughly the total time minus pure processing time
        if i > 0:  # Only measure transfer between different backends
            estimated_transfer_time = backend_total_time - task_processing_time
            s1_to_s2_transfer_times.append(estimated_transfer_time)
            logger.debug(f"[S1_TO_S2_TRANSFER] Backend {i} | "
                       f"Estimated Transfer Time: {estimated_transfer_time:.2f}ms | "
                       f"Total Backend Time: {backend_total_time:.2f}ms | "
                       f"Pure Processing: {task_processing_time:.2f}ms")
        
        logger.debug(f"[PROCESSING_LATENCY] Backend {i} | "
                   f"Task Processing: {task_processing_time:.2f}ms | "
                   f"Total Backend Time: {backend_total_time:.2f}ms | "
                   f"Hidden States Shape: {hidden_states.shape}")
        
        assert isinstance(hidden_states, torch.Tensor)
        assert (
            hidden_states.ndim == 3
        ), f"inputs to {type(backend)} must be a list with a single 3d tensor of hidden states"

    cross_gpu_end_time = perf_counter()
    cross_gpu_latency = (cross_gpu_end_time - cross_gpu_start_time) * 1000
    
    if s1_to_s2_transfer_times:
        s1_to_s2_mean = sum(s1_to_s2_transfer_times) / len(s1_to_s2_transfer_times)
        s1_to_s2_total = sum(s1_to_s2_transfer_times)
        logger.debug(f"[S1_TO_S2_TRANSFER_SUMMARY] "
                   f"Average Transfer: {s1_to_s2_mean:.2f}ms | "
                   f"Total Transfer: {s1_to_s2_total:.2f}ms | "
                   f"Transfer Count: {len(s1_to_s2_transfer_times)}")
    
    logger.debug(f"[CROSS_GPU_TRANSFER_LATENCY] Total: {cross_gpu_latency:.2f}ms | "
               f"Backends: {len(requested_backends)} | "
               f"Output Shape: {hidden_states.shape}")

    return hidden_states


async def run_rpc_backward(
    *flat_tensors: torch.Tensor,
    requested_backends: Sequence[TransformerBackend],
    active_adapter: str = "",
    prioritizer: TaskPrioritizerBase,
    points: int = 0,
    args_structure: Any = None,
) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    if args_structure is not None:
        # TODO: kwargs currently is unused, it can be used later for peft-like adaptation
        flat_tensors, kwargs = unpack_args_kwargs(flat_tensors, args_structure)
    inputs, grad_outputs, prompts, *_ = flat_tensors

    # Fix for bus error in cross-machine setups: ensure tensors are contiguous
    # Deserialized tensors from network buffers may not be properly aligned,
    # especially for certain batch sizes (e.g., batch_size=16)
    if not inputs.is_contiguous():
        inputs = inputs.contiguous()
    if not grad_outputs.is_contiguous():
        grad_outputs = grad_outputs.contiguous()
    if prompts is not None and not is_dummy(prompts) and not prompts.is_contiguous():
        prompts = prompts.contiguous()

    # Cast inputs & grad outputs to backend dtype
    inputs = inputs.to(requested_backends[0].dtype)
    grad_outputs = grad_outputs.to(requested_backends[-1].dtype)

    if prompts is None or is_dummy(prompts):
        prompts = [DUMMY] * len(requested_backends)
    else:
        prompts = [p.squeeze(0) for p in prompts.to(requested_backends[0].dtype).split(1, dim=0)]

    # Run a forward chain to collect intermediate inputs
    # Note that we do not forward for the last module since we do not need its output
    inter_inputs = []
    for backend, prompt in zip(requested_backends[:-1], prompts[:-1]):
        assert inputs.ndim == 3, f"inputs to {type(backend)} must be a single 3d tensor of hidden states"
        if not is_dummy(prompt):
            inputs[:, : prompt.shape[1]] += prompt
        inter_inputs.append(inputs)
        assert isinstance(backend.inference_pool, PrioritizedTaskPool), "petals support only prioritized pools"
        priority = prioritizer.prioritize(
            inputs, points=points / len(requested_backends), backend=backend, type="forward_in_backward"
        )
        (inputs,) = await backend.forward_pool.submit_task(inputs, active_adapter, priority=priority)

        assert isinstance(inputs, torch.Tensor)

    if not is_dummy(prompts[-1]):
        inputs[:, : prompts[-1].shape[1]] += prompts[-1]
    inter_inputs.append(inputs)

    assert len(inter_inputs) == len(prompts) == len(requested_backends), "internal shape error during backward"
    grad_prompts_reversed = []
    # Run a chain of requested backends
    for inp, prompt, backend in zip(*map(reversed, (inter_inputs, prompts, requested_backends))):
        assert isinstance(backend.inference_pool, PrioritizedTaskPool), "petals support only prioritized pools"
        priority = prioritizer.prioritize(
            inp, grad_outputs, points=points / len(requested_backends), backend=backend, type="backward"
        )
        (grad_outputs,) = await backend.backward_pool.submit_task(inp, grad_outputs, active_adapter, priority=priority)

        assert isinstance(grad_outputs, torch.Tensor)
        if not is_dummy(prompt):
            grad_prompts_reversed.append(grad_outputs[:, : prompt.shape[1]].unsqueeze(0))

    grad_prompts = torch.cat(grad_prompts_reversed[::-1], dim=0) if grad_prompts_reversed else DUMMY
    return [grad_outputs] if is_dummy(grad_prompts) else [grad_outputs, grad_prompts]  # TODO un-duct-tape

    
def restore_hidden_states(
    flattened_hidden_states: torch.Tensor,  # [N_total_valid, hidden_size]
    keep_indices: torch.Tensor,  # [B, max_keep_len]，padding 为 -1
    original_seq_len: int,  # 原始序列长度
) -> torch.Tensor:
    """
    将铺平的 hidden states 还原为 [B, original_seq_len, hidden_size]
    
    Args:
        flattened_hidden_states: [N_total_valid, hidden_size] 铺平后的有效 hidden states
        keep_indices: [B, max_keep_len] 每个 batch 的 keep indices，padding 为 -1
        original_seq_len: 原始序列长度
    
    Returns:
        restored_hidden_states: [B, original_seq_len, hidden_size]，无效位置用 0 填充
    """
    batch_size, max_keep_len = keep_indices.shape
    device = flattened_hidden_states.device
    dtype = flattened_hidden_states.dtype
    
    # 处理不同维度的输入
    if flattened_hidden_states.ndim == 2:
        # [N_total_valid, hidden_size] -> 直接使用
        flat_hidden = flattened_hidden_states
        hidden_size = flattened_hidden_states.shape[-1]
    elif flattened_hidden_states.ndim == 3:
        # [num_micro_batches, N_valid_per_mb, hidden_size] -> 合并前两维
        num_mb, n_valid_per_mb, hidden_size = flattened_hidden_states.shape
        flat_hidden = flattened_hidden_states.reshape(-1, hidden_size)  # [num_mb * N_valid_per_mb, hidden_size]
    else:
        raise ValueError(f"Unexpected flattened_hidden_states dim: {flattened_hidden_states.ndim}")
    
    # 创建输出 tensor，用 0 填充
    restored_hidden_states = torch.zeros(
        batch_size, original_seq_len, hidden_size,
        dtype=dtype, device=device
    )
    
    # 创建有效 mask: [B, max_keep_len]
    valid_mask = keep_indices >= 0
    
    # 创建 batch 索引: [B, max_keep_len]
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(keep_indices)
    
    # 取出有效部分的索引
    valid_batch_idx = batch_idx[valid_mask]      # [N_total_valid]
    valid_seq_idx = keep_indices[valid_mask]     # [N_total_valid]
    
    # 验证维度匹配
    n_total_valid = valid_mask.sum().item()
    if flat_hidden.shape[0] != n_total_valid:
        raise ValueError(
            f"Dimension mismatch: flattened_hidden_states has {flat_hidden.shape[0]} elements, "
            f"but keep_indices has {n_total_valid} valid entries"
        )
    
    # 写入还原位置
    restored_hidden_states[valid_batch_idx, valid_seq_idx, :] = flat_hidden
    
    return restored_hidden_states

def ensure_tensors(flat_tensors):
    result = []
    for i, t in enumerate(flat_tensors):
        if t is None:
            result.append(torch.tensor(0))
        elif isinstance(t, torch.Tensor):
            result.append(t)
        elif isinstance(t, (list, tuple)):
            t_clean = [x for x in t if x is not None]
            if len(t_clean) == 0:
                result.append(torch.tensor(0))
            elif isinstance(t_clean[0], torch.Tensor):
                result.append(torch.stack(t_clean))
            else:
                result.append(torch.tensor(t_clean))
        elif isinstance(t, (int, float, bool)):
            result.append(torch.tensor(t))
        else:
            raise TypeError(f"flat_tensors[{i}] cant trans to tensor: type={type(t)}, value={t}")
    return tuple(result)

async def iterate_rpc_inference(
    requested_uids: Sequence[ExpertUID],
    requested_backends: Sequence[TransformerBackend],
    active_adapter: Optional[str],
    input_iterator: AsyncIterator[Tuple[runtime_pb2.ExpertRequest, dict]],
    cache_handles: Sequence[Sequence[Handle]],
    pruner_manager: SpeculativePrunerManager,
    *,
    max_length: int,
    prioritizer: TaskPrioritizerBase,
    points: int,
    quant_type: QuantType,
    args_structure: Any = None,
    cross_stage_push_fn=None,  # [MBPIPE] Optional callback for cross-stage micro-batch streaming
) -> AsyncIterator[Tuple[Sequence[runtime_pb2.Tensor], bool, Dict]]:
    assert len(cache_handles) == len(requested_backends)

    start_iterate_rpc_infer_time = perf_counter()
    prefix_length = 0
    point_per_piece = points / max_length if max_length > 0 else 0.0
    
    # [MBPIPE] Request context for caching mb0 data across micro-batches
    request_context: Optional[RequestContext] = None

    async for request, step_metadata in input_iterator:
        step_receive_time = perf_counter()
        queue_wait_ms = _to_float(step_metadata.get("_queue_wait_ms"), 0.0)
        queue_source = str(step_metadata.get("_queue_source", "unknown"))
        step_id_for_log = step_metadata.get("step_id", "unknown")
        if "start_from_position" in step_metadata:
            start_from_position = step_metadata["start_from_position"]
            # assert (
            #     prefix_length >= start_from_position,
            # ), f"prefix_length={prefix_length}, start_from_position={start_from_position}"
            prefix_length = start_from_position

        deserialize_start = perf_counter()
        flat_tensors = tuple(deserialize_torch_tensor(tensor) for tensor in request.tensors)
        deserialize_time = (perf_counter() - deserialize_start) * 1000.0
        if args_structure is not None:
            flat_tensors, kwargs = unpack_args_kwargs(flat_tensors, args_structure)

        hidden_states, keep_indices, need_pruning1, tree_attention_mask, kv_cache_position_ids, draft_tokens, prefill_length, is_spec_dec1, prompts, hypo_ids, *_ = flat_tensors
        draft_tokens = draft_tokens if draft_tokens is not None and not is_dummy(draft_tokens) else None

        # Fix for bus error in cross-machine setups: ensure tensors are contiguous
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        if prompts is not None and not is_dummy(prompts) and not prompts.is_contiguous():
            prompts = prompts.contiguous()
        if not hypo_ids.is_contiguous():
            hypo_ids = hypo_ids.contiguous()
        if tree_attention_mask is not None and not is_dummy(tree_attention_mask) and not tree_attention_mask.is_contiguous():
            tree_attention_mask = tree_attention_mask.contiguous()
        if kv_cache_position_ids is not None and not is_dummy(kv_cache_position_ids) and not kv_cache_position_ids.is_contiguous():
            kv_cache_position_ids = kv_cache_position_ids.contiguous()
        if draft_tokens is not None and not is_dummy(draft_tokens) and not draft_tokens.is_contiguous():
            draft_tokens = draft_tokens.contiguous()
        if keep_indices is not None and not is_dummy(keep_indices) and not keep_indices.is_contiguous():
            keep_indices = keep_indices.contiguous()
        if need_pruning1 is not None and not is_dummy(need_pruning1) and not need_pruning1.is_contiguous():
            need_pruning1 = need_pruning1.contiguous()
        if is_spec_dec1 is not None and not is_dummy(is_spec_dec1) and not is_spec_dec1.is_contiguous():
            is_spec_dec1 = is_spec_dec1.contiguous()
            
        need_pruning = _as_python_bool(need_pruning1 == 1) if need_pruning1 is not None and not is_dummy(need_pruning1) else False
        is_spec_dec = _as_python_bool(is_spec_dec1 == 1) if is_spec_dec1 is not None and not is_dummy(is_spec_dec1) else False
        if not is_spec_dec and _as_python_bool(step_metadata.get("is_spec_dec", 0)):
            is_spec_dec = True
            logger.info(
                f"{MBPIPE_LOG_PREFIX} Full-batch spec override from metadata for step_id={step_metadata.get('step_id')}"
            )
        if not need_pruning and _as_python_bool(step_metadata.get("need_pruning", 0)):
            need_pruning = True

        if is_spec_dec and draft_tokens is not None and draft_tokens.shape[0] != hidden_states.shape[0]:
            hidden_states = restore_hidden_states(hidden_states, keep_indices, draft_tokens.shape[-1])
            
        batch_size, length_increment, _ = hidden_states.shape
                
        token_increment = _effective_token_increment(hidden_states, kv_cache_position_ids, is_spec_dec)
        
        if is_spec_dec:
            rotary_position_ids = _create_tree_position_ids_with_invalid_cache(
                width=2,
                depth=3,
                prefill_length=prefill_length - 1,
                kv_cache_position_ids=kv_cache_position_ids,
                batch_offset=0,
                device=hidden_states.device,
                target_seq_len=length_increment
            )
        else:
            rotary_position_ids = None
        
        # Cast inputs to backend dtype
        hidden_states = hidden_states.to(requested_backends[0].dtype)
        assert hypo_ids.dtype == torch.int64, f"hypo ids must be int64, got {hypo_ids.dtype}"
        
        # Add Cross-GPU Transfer Latency measurement
        cross_gpu_start_time = perf_counter()
        start_compute_time = perf_counter()  # Initialize compute time tracking
        compute_time = 0.0
        execution_mode = "unknown"

        # parse deep prompts (optional argument)
        has_prompts = prompts is not None and not is_dummy(prompts)
        if not has_prompts:
            prompts = [None] * len(requested_backends)
        else:
            prompts = [p.squeeze(0) for p in prompts.to(requested_backends[0].dtype).split(1, dim=0)]
            prompts = [prompt if not is_dummy(prompt) else None for prompt in prompts]
        # print('has_prompts', has_prompts)
        # print('prompts ', prompts)
        if not (len(requested_backends) == len(prompts)):
            raise ValueError(f"Received {len(prompts)} prompts for {len(requested_backends)} backends")

        if prefix_length + token_increment > max_length:
            raise ValueError(
                f"Maximum length exceeded: prefix {prefix_length} + current {token_increment}"
                f" exceeds pre-allocated maximum {max_length}"
            )

        # Note: quant_type is always NONE (quantization CLI removed), so always use standard threshold
        merge_max_tokens = MAX_SHORT_INFERENCE_TOKENS
        can_merge_pools = batch_size * length_increment <= merge_max_tokens
        # print('-=-=-=-=-=-=-=-==-=- can merge pools : ', can_merge_pools)
        priority = prioritizer.prioritize(
            hidden_states,
            hypo_ids,
            points=point_per_piece,
            requested_uids=requested_uids,
            type="inference",
        )
        # print('after priority = prioritizer.prioritize( )')
        #print_time_now('')
        # A client may pass a tensor with 0 tokens. This is a special case that occurs, e.g.
        # when user wants to pre-allocate cache or check that server *can* allocate that cache.
        if hidden_states.numel() > 0:
            assert hidden_states.ndim == 3, f"hidden states must be a single 3d tensor"
            last_uid = len(requested_uids) - 1
            if can_merge_pools:
                # Merged pools path: all blocks processed in one call
                # [MBPIPE] Check if we should split into micro-batches with pipeline overlap
                
                execution_mode = "merged_fullbatch"
                # Legacy path: process entire batch at once
                inference_infos = tuple(
                    InferenceMetadata(uid, prefix_length, tuple(handles), active_adapter, tree_attention_mask=tree_attention_mask, kv_cache_position_ids=kv_cache_position_ids, draft_tokens=draft_tokens, prefill_length=prefill_length, keep_indices=keep_indices, need_pruning=need_pruning, is_spec_dec=is_spec_dec, rotary_position_ids=rotary_position_ids)
                    for i, (uid, handles) in enumerate(zip(requested_uids, cache_handles))
                )
                submit_result = await requested_backends[0].inference_pool.submit_task(
                    hidden_states, hypo_ids, inference_infos, *prompts, priority=priority
                )
                hidden_states, keep_indices, _ = _unpack_inference_submit_result(submit_result)
                
            
            
            # offload_logger.info(f" Inference computation completed - step {prefix_length}")
            end_compute_time = perf_counter()
            compute_time = (end_compute_time - start_compute_time) * 1000  # ms
            # print('the inference computing time ', end_compute_time - start_compute_time)
            # print_time_now('')
        # serialize and send last layer outputs
        if keep_indices is not None:
            if not torch.is_tensor(keep_indices):
                keep_indices = torch.tensor(keep_indices, dtype=torch.int64, device=hidden_states.device)
            else:
                keep_indices = keep_indices.to(dtype=torch.int64, device=hidden_states.device)
        else:
            keep_indices = torch.arange(
                hidden_states.shape[1],
                dtype=torch.int64,
                device=hidden_states.device
            ).unsqueeze(0).expand(hidden_states.shape[0], -1)
        
        serialize_start = perf_counter()
        need_pruning_next = torch.tensor(0)
        
        flat_tensors = (hidden_states, keep_indices, need_pruning_next, tree_attention_mask, kv_cache_position_ids, draft_tokens)
        flat_tensors = ensure_tensors(flat_tensors)
        output_tensors = [
            serialize_torch_tensor(result.to(proto.dtype), proto.compression, allow_inplace=True)
            for result, proto in zip(flat_tensors, nested_flatten(requested_backends[-1].outputs_schema))
        ]

        serialize_end = perf_counter()
        serialize_time = (serialize_end - serialize_start) * 1000  # ms
        # print('after serialize and send last layer outputs ', )
        # print_time_now('')
        # print('hidden_states ', hidden_states)
        # print('type of hidden_states ', )
        # print('shape of hidden_states ', hidden_states.size())
        # # hidden_size_in_bytes = hidden_states.element_size() * output_tensors.numel()  
        # # print(f"Size of the hidden state in bytes: {size_in_bytes}")  
        # print()
        
        can_push = not has_prompts
        
        # Calculate Cross-GPU Transfer receive time
        cross_gpu_end_time = perf_counter()
        cross_gpu_receive_time = (cross_gpu_end_time - cross_gpu_start_time) * 1000  # ms
        
        # Calculate total step time
        step_end_time = perf_counter()
        step_total_time = (step_end_time - step_receive_time) * 1000  # ms
        step_residual_ms = step_total_time - (deserialize_time + compute_time + serialize_time)
        step_total_with_queue_ms = step_total_time + queue_wait_ms

        # Critical path analysis: compute vs communication ratio
        t_gpu2cpu_ms = serialize_time
        t_cpu2gpu_ms = deserialize_time
        data_bytes = hidden_states.nelement() * hidden_states.element_size()
        compute_pct = (compute_time / step_total_time * 100) if step_total_time > 0 else 0.0
        comm_overhead_ms = t_gpu2cpu_ms + t_cpu2gpu_ms
        comm_overhead_pct = (comm_overhead_ms / step_total_time * 100) if step_total_time > 0 else 0.0
        bw_gpu2cpu_gbps = (data_bytes / (t_gpu2cpu_ms / 1000) / 1e9) if t_gpu2cpu_ms > 0 else 0.0

        logger.info(
            f"[STEP_TIMING_BREAKDOWN] step_id={step_id_for_log} mode={execution_mode} "
            f"queue_wait={queue_wait_ms:.2f}ms queue_source={queue_source} "
            f"t_cpu2gpu={t_cpu2gpu_ms:.2f}ms compute={compute_time:.2f}ms "
            f"t_gpu2cpu={t_gpu2cpu_ms:.2f}ms residual={step_residual_ms:.2f}ms "
            f"step_total={step_total_time:.2f}ms total_with_queue={step_total_with_queue_ms:.2f}ms "
            f"compute_pct={compute_pct:.1f}% mem_copy_pct={comm_overhead_pct:.1f}% "
            f"data_bytes={data_bytes} bw_gpu2cpu={bw_gpu2cpu_gbps:.2f}GB/s "
            f"batch={batch_size} seq_inc={token_increment} raw_seq={length_increment} is_spec_dec={int(bool(is_spec_dec))}"
        )

        # Pass timing data to handler for [COMM_BREAKDOWN] log
        if isinstance(step_metadata, dict):
            step_metadata["_serialize_ms"] = t_gpu2cpu_ms
            step_metadata["_compute_ms"] = compute_time
            step_metadata["_data_bytes"] = data_bytes
            step_metadata["_step_total_ms"] = step_total_time
        
        # [MBPIPE] Record stage timing for cross-stage overlap decisions
        try:
            first_uid = str(requested_uids[0]) if requested_uids else "unknown"
            last_uid = str(requested_uids[-1]) if requested_uids else "unknown"
            first_idx = first_uid.split('.')[-1] if '.' in first_uid else "0"
            last_idx = last_uid.split('.')[-1] if '.' in last_uid else "0"
            stage_id = f"blocks_{first_idx}_{last_idx}"
            
            log_stage_timing(
                logger, stage_id,
                compute_time_ms=compute_time,
                comm_time_ms=comm_overhead_ms,
                component="iterate_rpc_inference"
            )
        except Exception as e:
            logger.debug(f"{MBPIPE_LOG_PREFIX} Failed to log stage timing: {e}")
        
        yield output_tensors, can_push, step_metadata
        # print('output_tensors ',output_tensors)
        # prepare for next step
        prefix_length += token_increment

    end_iterate_rpc_infer_time = perf_counter()#######
    # print('iterate (all steps) rpc infer time cost (sec): ', end_iterate_rpc_infer_time - start_iterate_rpc_infer_time)########
    # #print_time_now('')
    # print()
