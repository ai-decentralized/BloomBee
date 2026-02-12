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

# [MBPIPE] Cross-stage streaming push support
_cross_stage_push_callback = None  # Will be set by handler for cross-stage streaming


from time import perf_counter
from datetime import datetime, timezone  
def print_time_now(s):
    # Get the current time in UTC  
    current_utc_datetime = datetime.now(timezone.utc)  
    # Format the datetime to the desired string format  
    formatted_utc_time = current_utc_datetime.strftime('%Y-%m-%d %H:%M:%S.%f %Z')  
    print('\t\t\t'+s+" UTC Time: "+ str(formatted_utc_time) )  
    

# We prioritize short inference requests and make them use a *merged* inference pool,
# so they are processed without interruptions and extra overheads
# Note: NF4 refers to FlexGen's 4-bit group quantization, not bitsandbytes
MAX_SHORT_INFERENCE_TOKENS = 128
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
    if kv_cache_position_ids.ndim >= 2:
        return int(kv_cache_position_ids[0].numel())
    return int(kv_cache_position_ids.numel())


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
            logger.info(f"[S1_TO_S2_TRANSFER] Backend {i} | "
                       f"Estimated Transfer Time: {estimated_transfer_time:.2f}ms | "
                       f"Total Backend Time: {backend_total_time:.2f}ms | "
                       f"Pure Processing: {task_processing_time:.2f}ms")
        
        # Log processing latency for each backend
        logger.info(f"[PROCESSING_LATENCY] Backend {i} | "
                   f"Task Processing: {task_processing_time:.2f}ms | "
                   f"Total Backend Time: {backend_total_time:.2f}ms | "
                   f"Hidden States Shape: {hidden_states.shape}")
        
        assert isinstance(hidden_states, torch.Tensor)
        assert (
            hidden_states.ndim == 3
        ), f"inputs to {type(backend)} must be a list with a single 3d tensor of hidden states"

    # Calculate total Cross-GPU Transfer Latency
    cross_gpu_end_time = perf_counter()
    cross_gpu_latency = (cross_gpu_end_time - cross_gpu_start_time) * 1000
    
    # Calculate S1->S2 transfer statistics
    if s1_to_s2_transfer_times:
        s1_to_s2_mean = sum(s1_to_s2_transfer_times) / len(s1_to_s2_transfer_times)
        s1_to_s2_total = sum(s1_to_s2_transfer_times)
        logger.info(f"[S1_TO_S2_TRANSFER_SUMMARY] "
                   f"Average Transfer: {s1_to_s2_mean:.2f}ms | "
                   f"Total Transfer: {s1_to_s2_total:.2f}ms | "
                   f"Transfer Count: {len(s1_to_s2_transfer_times)}")
    
    logger.info(f"[CROSS_GPU_TRANSFER_LATENCY] Total: {cross_gpu_latency:.2f}ms | "
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
    hidden_size = flattened_hidden_states.shape[-1]
    device = flattened_hidden_states.device
    dtype = flattened_hidden_states.dtype
    
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
    
    # 写入还原位置
    restored_hidden_states[valid_batch_idx, valid_seq_idx, :] = flattened_hidden_states
    
    return restored_hidden_states

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


        # ========== [MBPIPE] MICRO-BATCH BRANCH ==========
        # Handle cross-stage micro-batches with 2-tensor format
        # Process each immediately (for pipeline overlap) but accumulate results
        if step_metadata.get("type") == "micro_batch":
            import time as _time
            receive_timestamp_us = int(_time.time() * 1_000_000)  # Absolute timestamp for correlation
            log_mb_detail = _should_log_mb_detail(step_metadata)
            queue_wait_ms = _to_float(step_metadata.get("_queue_wait_ms"), 0.0)
            queue_source = str(step_metadata.get("_queue_source", "unknown"))
            
            mb_idx = step_metadata.get("mb_idx", 0)
            expected_num_mb = step_metadata.get("expected_num_mb", 1)
            mb_offset = step_metadata.get("offset", 0)
            mb_size = step_metadata.get("size", 1)
            full_batch_size = step_metadata.get("full_batch_size", mb_size)
            step_id = step_metadata.get("step_id")
            session_id = step_metadata.get("session_id", "unknown")
            start_from_position = step_metadata.get("start_from_position")
            micro_meta_msg = (
                f"{MBPIPE_LOG_PREFIX} micro-batch metadata: step_id={step_id}, mb_idx={mb_idx}, "
                f"start_from_position={start_from_position}"
            )
            if log_mb_detail:
                logger.info(micro_meta_msg)
            else:
                logger.debug(micro_meta_msg)
            
            # [CROSS_STAGE] Calculate overlap metrics
            push_timestamp_us = step_metadata.get("stage_push_timestamp_us", 0)
            compute_start_timestamp_us = step_metadata.get("stage_compute_start_timestamp_us", 0)
            compute_end_timestamp_us = step_metadata.get("stage_compute_end_timestamp_us", 0)
            sender_to_receiver_clock_offset_us = _to_int(
                step_metadata.get("sender_to_receiver_clock_offset_us"), 0
            )
            sender_to_receiver_clock_rtt_us = _to_int(
                step_metadata.get("sender_to_receiver_clock_rtt_us"), 0
            )
            sender_to_receiver_clock_samples = _to_int(
                step_metadata.get("sender_to_receiver_clock_samples"), 0
            )
            transfer_latency_us = receive_timestamp_us - push_timestamp_us if push_timestamp_us > 0 else 0
            
            # [CROSS_STAGE] Pipeline overlap: If previous stage started MB1 compute before we receive MB0,
            # there's potential for overlap (prev stage works on MB1 while we work on MB0)
            if log_mb_detail:
                logger.info(
                    f"[CROSS_STAGE] Stage receives MB{mb_idx} at t={receive_timestamp_us}, "
                    f"push_at={push_timestamp_us}, transfer_latency={transfer_latency_us/1000:.2f}ms"
                )
            
            # [CROSS_STAGE_OVERLAP] Track timing data for overlap calculation
            # Key insight: True cross-stage overlap means Stage1 processes MB_{n+1} while Stage2 processes MB_n
            # We store: prev_stage_compute_start (when prev stage STARTED computing this MB)
            #           this_stage_receive_time (when we received this MB)
            overlap_tracking_key = (step_metadata.get("session_id", "unknown"), step_metadata.get("step_id"))
            if not hasattr(iterate_rpc_inference, '_cross_stage_overlap_data'):
                iterate_rpc_inference._cross_stage_overlap_data = {}
            if overlap_tracking_key not in iterate_rpc_inference._cross_stage_overlap_data:
                iterate_rpc_inference._cross_stage_overlap_data[overlap_tracking_key] = {}
            
            iterate_rpc_inference._cross_stage_overlap_data[overlap_tracking_key][mb_idx] = {
                'prev_stage_compute_start_us': compute_start_timestamp_us,
                'prev_stage_compute_end_us': compute_end_timestamp_us,
                'prev_stage_clock_offset_us': sender_to_receiver_clock_offset_us,
                'prev_stage_clock_rtt_us': sender_to_receiver_clock_rtt_us,
                'prev_stage_clock_samples': sender_to_receiver_clock_samples,
                'this_stage_receive_us': receive_timestamp_us,
                'transfer_latency_us': transfer_latency_us,
            }
            
            if compute_start_timestamp_us > 0:
                if log_mb_detail:
                    logger.info(
                        f"[CROSS_STAGE] Prev stage started MB{mb_idx} compute at t={compute_start_timestamp_us}, "
                        f"we can now start processing (overlap opportunity exists!)"
                    )
            processing_msg = (
                f"{MBPIPE_LOG_PREFIX} iterate_rpc_inference: processing micro-batch "
                f"mb_idx={mb_idx}/{expected_num_mb}, offset={mb_offset}, size={mb_size}, full_batch={full_batch_size}"
            )
            if log_mb_detail:
                logger.info(processing_msg)
            else:
                logger.debug(processing_msg)
            
            # [MB_DEBUG] Log micro-batch state
            logger.debug(f"[MB_DEBUG] === MICRO-BATCH {mb_idx} PROCESSING ===")
            logger.debug(f"[MB_DEBUG] mb_offset={mb_offset}, mb_size={mb_size}, full_batch_size={full_batch_size}")
            logger.debug(f"[MB_DEBUG] step_id={step_id}, session_id={session_id}")
            
            # Deserialize only the 2 tensors from micro-batch push
            mb_deserialize_start = perf_counter()
            flat_tensors = tuple(deserialize_torch_tensor(tensor) for tensor in request.tensors)
            mb_deserialize_ms = (perf_counter() - mb_deserialize_start) * 1000.0
            
            # [MB_DEBUG] Log deserialized tensors
            logger.debug(f"[MB_DEBUG] Deserialized {len(flat_tensors)} tensors from request")
            for i, t in enumerate(flat_tensors):
                if hasattr(t, 'shape'):
                    logger.debug(f"[MB_DEBUG] flat_tensors[{i}]: shape={t.shape}, dtype={t.dtype}, device={t.device}")
                else:
                    logger.debug(f"[MB_DEBUG] flat_tensors[{i}]: type={type(t)}")
            
            if len(flat_tensors) >= 2:
                mb_hidden_states = flat_tensors[0]
                mb_keep_indices = flat_tensors[1] if len(flat_tensors) > 1 else None
            else:
                mb_hidden_states = flat_tensors[0] if flat_tensors else None
                mb_keep_indices = None
            
            # [MB_DEBUG] Log extracted tensors
            logger.debug(f"[MB_DEBUG] mb_hidden_states: shape={mb_hidden_states.shape if mb_hidden_states is not None else 'None'}")
            logger.debug(f"[MB_DEBUG] mb_keep_indices: shape={mb_keep_indices.shape if mb_keep_indices is not None and hasattr(mb_keep_indices, 'shape') else 'None'}")
            
            if mb_hidden_states is None:
                logger.warning(f"{MBPIPE_SCHEMA_PREFIX} Empty micro-batch received, skipping")
                continue
            
            # Ensure contiguous
            if not mb_hidden_states.is_contiguous():
                mb_hidden_states = mb_hidden_states.contiguous()
                logger.debug(f"[MB_DEBUG] Made mb_hidden_states contiguous")
            if mb_keep_indices is not None and not mb_keep_indices.is_contiguous():
                mb_keep_indices = mb_keep_indices.contiguous()
                logger.debug(f"[MB_DEBUG] Made mb_keep_indices contiguous")
            
            # [MBPIPE_FIX] RequestContext must be scoped by (session_id, step_id).
            # Reusing context across steps can leak stale fields (e.g. prefill_length/hypo_ids).
            if (
                request_context is None
                or request_context.request_id != session_id
                or request_context.step_id != step_id
            ):
                request_context = RequestContext(session_id, step_id)
                if log_mb_detail:
                    logger.info(
                        f"[MB_DEBUG] Reset RequestContext for micro-batch step: "
                        f"session_id={session_id}, step_id={step_id}"
                    )
                else:
                    logger.debug(
                        f"[MB_DEBUG] Reset RequestContext for micro-batch step: "
                        f"session_id={session_id}, step_id={step_id}"
                    )
            else:
                logger.debug(f"[MB_DEBUG] Reusing RequestContext for current micro-batch step")

            # Cross-stage micro-batch pushes only carry hidden_states/keep_indices.
            # Seed a per-step context explicitly so schema fallback does not emit warnings.
            if not request_context.is_initialized:
                spec_from_metadata = _as_python_bool(step_metadata.get("is_spec_dec", 0))
                pruning_from_metadata = _as_python_bool(step_metadata.get("need_pruning", 0))
                request_context.cache_from_mb0(
                    prompts=[None] * len(requested_backends),
                    hypo_ids=torch.arange(mb_size, dtype=torch.int64, device=mb_hidden_states.device),
                    tree_attention_mask=None,
                    kv_cache_position_ids=None,
                    draft_tokens=None,
                    prefill_length=int(step_metadata.get("prefill_length", 0) or 0),
                    is_spec_dec=spec_from_metadata,
                    need_pruning=pruning_from_metadata,
                    num_backends=len(requested_backends),
                )
                logger.debug(
                    f"{MBPIPE_SCHEMA_PREFIX} Initialized default RequestContext for "
                    f"micro-batch step session_id={session_id}, step_id={step_id}, mb_idx={mb_idx}"
                )
            
            # Fill missing fields using schema defaults
            (
                hidden_states, keep_indices, prompts, hypo_ids,
                tree_attention_mask, kv_cache_position_ids, draft_tokens,
                prefill_length, is_spec_dec, need_pruning
            ) = fill_microbatch_defaults(
                mb_hidden_states, mb_keep_indices, request_context, len(requested_backends)
            )
            
            # [MB_DEBUG] Log after fill_microbatch_defaults
            logger.debug(f"[MB_DEBUG] After fill_microbatch_defaults:")
            logger.debug(f"[MB_DEBUG]   hidden_states.shape={hidden_states.shape}")
            if hidden_states.numel() > 0:
                logger.debug(f"[MB_DEBUG]   hidden_states.abs().mean()={hidden_states.abs().mean().item():.6f}")
            logger.debug(f"[MB_DEBUG]   keep_indices={keep_indices}")
            logger.debug(f"[MB_DEBUG]   tree_attention_mask={tree_attention_mask is not None}")
            logger.debug(f"[MB_DEBUG]   kv_cache_position_ids={kv_cache_position_ids}")
            logger.debug(f"[MB_DEBUG]   prefill_length={prefill_length}, is_spec_dec={is_spec_dec}, need_pruning={need_pruning}")
            
            # Ensure prompts/hypo_ids match the current micro-batch.
            if not isinstance(prompts, list) or len(prompts) != len(requested_backends):
                prompts = [None] * len(requested_backends)
            if hypo_ids is None or not torch.is_tensor(hypo_ids) or hypo_ids.numel() != mb_size:
                hypo_ids = torch.arange(mb_size, dtype=torch.int64, device=hidden_states.device)
            else:
                hypo_ids = hypo_ids.to(dtype=torch.int64, device=hidden_states.device)
            
            # [MB_DEBUG] Log prompts and hypo_ids
            logger.debug(f"[MB_DEBUG] prompts_count={len(prompts)}")
            logger.debug(f"[MB_DEBUG] hypo_ids.shape={hypo_ids.shape}, values={hypo_ids.tolist()}")
            
            # [DEBUG] Log prompts info for debugging
            logger.debug(
                f"{MBPIPE_LOG_PREFIX} [DEBUG] After fix: "
                f"prompts type={type(prompts)}, len={len(prompts)}, "
                f"hypo_ids.shape={hypo_ids.shape}, num_backends={len(requested_backends)}"
            )
            
            # For micro-batch processing
            batch_size = mb_size
            length_increment = hidden_states.shape[1]
            token_increment = _effective_token_increment(hidden_states, kv_cache_position_ids, is_spec_dec)
            
            # Cast to backend dtype
            hidden_states = hidden_states.to(requested_backends[0].dtype)

            
            # ========== PROCESS THIS MICRO-BATCH IMMEDIATELY ==========
            # This is where pipeline overlap happens - we process each micro-batch
            # as it arrives while upstream is still computing the next one
            
            process_start_time = perf_counter()
            
            # Process through backends (simplified path for micro-batch)
            merge_max_tokens = MAX_SHORT_INFERENCE_TOKENS
            can_merge_pools = batch_size * length_increment <= merge_max_tokens
            priority = prioritizer.prioritize(
                hidden_states, hypo_ids, points=point_per_piece, 
                requested_uids=requested_uids, type="inference"
            )
            
            # [DEBUG] Log before submitting task
            logger.debug(
                f"{MBPIPE_LOG_PREFIX} [DEBUG] Before submit_task: "
                f"hidden_states.shape={hidden_states.shape}, "
                f"hypo_ids.shape={hypo_ids.shape if hasattr(hypo_ids, 'shape') else 'N/A'}, "
                f"can_merge_pools={can_merge_pools}, "
                f"prompts unpacked would be {len(prompts)} items"
            )
            
            if hidden_states.numel() > 0:
                if can_merge_pools:
                    # Use merged pool for this micro-batch
                    inference_infos = tuple(
                        InferenceMetadata(
                            uid, prefix_length, tuple(handles), active_adapter,
                            tree_attention_mask=tree_attention_mask,
                            kv_cache_position_ids=kv_cache_position_ids,
                            draft_tokens=draft_tokens,
                            prefill_length=prefill_length,
                            keep_indices=keep_indices,
                            need_pruning=need_pruning,
                            is_spec_dec=is_spec_dec,
                            batch_offset=mb_offset,
                            full_batch_size=full_batch_size,
                            micro_batch_size=mb_size
                        )
                        for uid, handles in zip(requested_uids, cache_handles)
                    )
                    # [MB_DEBUG] Log InferenceMetadata details
                    logger.debug(f"[MB_DEBUG] === INFERENCE METADATA ===")
                    for i, info in enumerate(inference_infos):
                        logger.debug(f"[MB_DEBUG] inference_infos[{i}]: uid={info.uid}, prefix_length={info.prefix_length}")
                        logger.debug(f"[MB_DEBUG]   batch_offset={info.batch_offset}, full_batch_size={info.full_batch_size}, micro_batch_size={info.micro_batch_size}")
                        logger.debug(f"[MB_DEBUG]   cache_handles count={len(info.cache_handles) if info.cache_handles else 0}")
                    
                    # [DEBUG] Log inference_infos
                    logger.debug(
                        f"{MBPIPE_LOG_PREFIX} [DEBUG] inference_infos count={len(inference_infos)}, "
                        f"submitting with {len(prompts)} prompts"
                    )
                    
                    # [MB_DEBUG] Log before submit_task
                    logger.debug(f"[MB_DEBUG] === SUBMIT TASK ===")
                    logger.debug(f"[MB_DEBUG] hidden_states.shape={hidden_states.shape}")
                    logger.debug(f"[MB_DEBUG] hypo_ids.shape={hypo_ids.shape}, values={hypo_ids.tolist()}")
                    
                    submit_start = perf_counter()
                    submit_result = await requested_backends[0].inference_pool.submit_task(
                        hidden_states, hypo_ids, inference_infos, *prompts, priority=priority
                    )
                    hidden_states, keep_indices, _ = _unpack_inference_submit_result(submit_result)
                    submit_time_ms = (perf_counter() - submit_start) * 1000
                    
                    # [MB_DEBUG] Log after submit_task
                    logger.debug(f"[MB_DEBUG] submit_task completed in {submit_time_ms:.2f}ms")
                    logger.debug(f"[MB_DEBUG] output hidden_states.shape={hidden_states.shape}")
                    logger.debug(f"[MB_DEBUG] output keep_indices={keep_indices}")

                else:
                    # Process through backends sequentially
                    for backend, uid, handles, prompt in zip(
                        requested_backends, requested_uids, cache_handles, prompts
                    ):
                        inference_infos = (InferenceMetadata(
                            uid, prefix_length, tuple(handles), active_adapter,
                            tree_attention_mask=tree_attention_mask,
                            kv_cache_position_ids=kv_cache_position_ids,
                            draft_tokens=draft_tokens,
                            prefill_length=prefill_length,
                            keep_indices=keep_indices,
                            need_pruning=need_pruning,
                            is_spec_dec=is_spec_dec,
                            batch_offset=mb_offset,
                            full_batch_size=full_batch_size,
                            micro_batch_size=mb_size
                        ),)
                        submit_result = await backend.inference_pool.submit_task(
                            hidden_states, hypo_ids, inference_infos, prompt, priority=priority
                        )
                        hidden_states, keep_indices, _ = _unpack_inference_submit_result(submit_result)
            
            process_time_ms = (perf_counter() - process_start_time) * 1000
            compute_done_timestamp_us = int(_time.time() * 1_000_000)
            compute_start_timestamp_us_this = int((perf_counter() - (process_time_ms / 1000)) * 1_000_000)  # Approximate
            
            # [CROSS_STAGE] Log compute completion for overlap analysis
            # This shows when Stage i+1 finishes computing MB n
            # Overlap = (Stage i starts MB n+1) while (Stage i+1 still processing MB n)
            if log_mb_detail:
                logger.info(
                    f"[CROSS_STAGE] Stage completes MB{mb_idx} at t={compute_done_timestamp_us}, "
                    f"process_time={process_time_ms:.1f}ms"
                )
            
            # [CROSS_STAGE_OVERLAP] Record this stage's compute times and calculate overlap
            overlap_tracking_key = (step_metadata.get("session_id", "unknown"), step_metadata.get("step_id"))
            if hasattr(iterate_rpc_inference, '_cross_stage_overlap_data') and \
               overlap_tracking_key in iterate_rpc_inference._cross_stage_overlap_data and \
               mb_idx in iterate_rpc_inference._cross_stage_overlap_data[overlap_tracking_key]:
                
                overlap_data = iterate_rpc_inference._cross_stage_overlap_data[overlap_tracking_key][mb_idx]
                overlap_data['this_stage_compute_start_us'] = compute_done_timestamp_us - int(process_time_ms * 1000)
                overlap_data['this_stage_compute_end_us'] = compute_done_timestamp_us
                overlap_data['this_stage_process_time_ms'] = process_time_ms
                
                # Calculate cross-stage overlap for THIS micro-batch:
                # True overlap happens when: prev_stage was still computing THIS MB's successor (MB_n+1)
                # while this stage was computing THIS MB (MB_n)
                # 
                # For MB_n:
                #   - Prev stage started computing MB_n at: prev_stage_compute_start_us
                #   - Prev stage sent MB_n and started MB_{n+1}: we can infer from next MB's data
                #   - This stage received MB_n at: this_stage_receive_us
                #   - This stage completed MB_n at: this_stage_compute_end_us
                #
                # Overlap = max(0, min(this_stage_compute_end, prev_stage_next_mb_compute_end) - max(this_stage_compute_start, prev_stage_next_mb_compute_start))
                # 
                # Simplified: If prev stage started MB_n+1 BEFORE this stage finished MB_n, there's overlap
                
                prev_compute_start = overlap_data['prev_stage_compute_start_us']
                this_receive = overlap_data['this_stage_receive_us']
                this_compute_end = overlap_data['this_stage_compute_end_us']
                
                # Log per-MB cross-stage timing
                latency_to_receive = (this_receive - prev_compute_start) / 1000 if prev_compute_start > 0 else 0
                if log_mb_detail:
                    logger.info(
                        f"[CROSS_STAGE_OVERLAP] MB{mb_idx}: prev_stage_started={prev_compute_start}, "
                        f"this_stage_received={this_receive} (+{latency_to_receive:.1f}ms from prev compute start), "
                        f"this_stage_done={this_compute_end} (compute={process_time_ms:.1f}ms)"
                    )
            if log_mb_detail:
                logger.info(
                    f"{MBPIPE_LOG_PREFIX} Micro-batch {mb_idx} processed in {process_time_ms:.1f}ms, "
                    f"output shape: {hidden_states.shape}"
                )
            else:
                logger.debug(
                    f"{MBPIPE_LOG_PREFIX} Micro-batch {mb_idx} processed in {process_time_ms:.1f}ms, "
                    f"output shape: {hidden_states.shape}"
                )
            if hidden_states.numel() > 0:
                logger.debug(f"[MB_DEBUG] Output hidden_states.abs().mean()={hidden_states.abs().mean().item():.6f} for mb_idx={mb_idx}")
            
            # ========== [MBPIPE_STREAMING] CROSS-STAGE PUSH FOR DECODE OVERLAP ==========
            # Push this micro-batch to downstream stage IMMEDIATELY after processing
            # This enables Stage_{i+1} to start processing MB_n while Stage_i computes MB_{n+1}
            next_servers = step_metadata.get("next_servers", None)
            if cross_stage_push_fn is not None and next_servers is not None:
                import asyncio
                push_timestamp_us = int(_time.time() * 1_000_000)
                
                push_metadata = step_metadata.copy()
                push_metadata["micro_batch_idx"] = mb_idx
                push_metadata["micro_batch_offset"] = mb_offset
                push_metadata["micro_batch_size"] = mb_size
                push_metadata["full_batch_size"] = full_batch_size
                push_metadata["total_micro_batches"] = expected_num_mb
                push_metadata["stage_push_timestamp_us"] = push_timestamp_us
                compute_start_abs_us = compute_done_timestamp_us - int(process_time_ms * 1000)
                push_metadata["stage_compute_start_timestamp_us"] = compute_start_abs_us
                push_metadata["stage_compute_end_timestamp_us"] = compute_done_timestamp_us
                push_metadata["is_streaming_decode"] = True  # Mark as streaming decode
                
                # [MBPIPE_FIX] Clone tensors before fire-and-forget async push.
                # Without cloning, underlying buffers may be mutated/reused before serialization finishes.
                push_hidden = hidden_states.detach().clone()
                push_keep = keep_indices.detach().clone() if torch.is_tensor(keep_indices) else keep_indices

                # Fire-and-forget async push - don't wait for completion
                asyncio.create_task(
                    cross_stage_push_fn(push_hidden, push_keep, push_metadata)
                )
                
                if log_mb_detail:
                    logger.info(
                        f"[MBPIPE_STREAMING] Cross-stage push: MB{mb_idx} sent to downstream at t={push_timestamp_us}, "
                        f"compute_time={process_time_ms:.1f}ms (decode overlap enabled!)"
                    )

            
            # ========== ACCUMULATE MICRO-BATCH RESULTS ==========
            # Store this micro-batch result in accumulator
            mb_accum_key = (session_id, step_id)
            if not hasattr(iterate_rpc_inference, '_mb_accumulators'):
                iterate_rpc_inference._mb_accumulators = {}
            
            if mb_accum_key not in iterate_rpc_inference._mb_accumulators:
                iterate_rpc_inference._mb_accumulators[mb_accum_key] = {
                    'expected': expected_num_mb,
                    'results': {},  # mb_idx -> (hidden_states, keep_indices, offset)
                    'full_batch_size': full_batch_size,
                    'step_metadata': step_metadata.copy(),
                    'step_start_time': step_receive_time,
                    'queue_wait_ms_sum': 0.0,
                    'queue_wait_pre_ms': 0.0,
                    'queue_wait_inter_ms': 0.0,
                    'deserialize_ms_sum': 0.0,
                    'compute_ms_sum': 0.0,
                    'queue_source_counts': {},
                    'token_increment': int(token_increment),
                }
            
            accum = iterate_rpc_inference._mb_accumulators[mb_accum_key]
            if int(accum.get('token_increment', token_increment)) != int(token_increment):
                logger.warning(
                    f"{MBPIPE_LOG_PREFIX} token_increment mismatch within step_id={step_id}: "
                    f"accum={accum.get('token_increment')} current={token_increment}, using max()"
                )
                accum['token_increment'] = max(int(accum.get('token_increment', 0)), int(token_increment))
            accum['results'][mb_idx] = (hidden_states.clone(), keep_indices, mb_offset)
            accum['queue_wait_ms_sum'] += queue_wait_ms
            if mb_idx == 0:
                accum['queue_wait_pre_ms'] = queue_wait_ms
            else:
                accum['queue_wait_inter_ms'] += queue_wait_ms
            accum['deserialize_ms_sum'] += mb_deserialize_ms
            accum['compute_ms_sum'] += process_time_ms
            accum['queue_source_counts'][queue_source] = accum['queue_source_counts'].get(queue_source, 0) + 1
            if log_mb_detail:
                logger.info(
                    f"[STEP_TIMING_MB] step_id={step_id} mb_idx={mb_idx} "
                    f"queue_wait={queue_wait_ms:.2f}ms queue_source={queue_source} "
                    f"deserialize={mb_deserialize_ms:.2f}ms compute={process_time_ms:.2f}ms"
                )
            
            if log_mb_detail:
                logger.info(
                    f"{MBPIPE_LOG_PREFIX} Accumulated mb_idx={mb_idx}, "
                    f"have {len(accum['results'])}/{accum['expected']} micro-batches"
                )
            
            # Check if we have all micro-batches
            if len(accum['results']) >= accum['expected']:
                logger.info(
                    f"{MBPIPE_LOG_PREFIX} All {accum['expected']} micro-batches received, merging..."
                )
                
                # Sort by mb_idx and merge
                sorted_indices = sorted(accum['results'].keys())
                merged_hidden_list = []
                for idx in sorted_indices:
                    h, k, offset = accum['results'][idx]
                    merged_hidden_list.append(h)
                
                # Merge hidden states
                merged_hidden_states = torch.cat(merged_hidden_list, dim=0)
                
                # Use last keep_indices (they should all be the same per-token)
                _, merged_keep_indices, _ = accum['results'][sorted_indices[-1]]
                
                if log_mb_detail:
                    logger.info(
                        f"{MBPIPE_LOG_PREFIX} Merged output shape: {merged_hidden_states.shape}"
                    )
                if merged_hidden_states.numel() > 0:
                     logger.debug(f"[MB_DEBUG] Merged hidden_states.abs().mean()={merged_hidden_states.abs().mean().item():.6f}")
                
                # [CROSS_STAGE_OVERLAP] Calculate and log final overlap summary
                if hasattr(iterate_rpc_inference, '_cross_stage_overlap_data') and \
                   overlap_tracking_key in iterate_rpc_inference._cross_stage_overlap_data:
                    
                    overlap_summary = iterate_rpc_inference._cross_stage_overlap_data[overlap_tracking_key]
                    total_overlap_ms = 0.0
                    total_stage2_compute_ms = 0.0
                    total_comparable_stage2_compute_ms = 0.0
                    missing_pair_count = 0
                    invalid_pair_count = 0
                    clock_corrected_pair_count = 0
                    clock_uncorrected_pair_count = 0
                    clock_offset_abs_sum_ms = 0.0
                    clock_rtt_max_ms = 0.0
                    
                    # For each pair (MB_n on this stage, MB_{n+1} on previous stage),
                    # compute strict overlap as interval intersection:
                    # overlap = len([s2_n_start, s2_n_end] ∩ [s1_{n+1}_start, s1_{n+1}_end])
                    
                    sorted_mb_indices = sorted(overlap_summary.keys())
                    
                    for i, mb_n in enumerate(sorted_mb_indices):
                        mb_data = overlap_summary[mb_n]
                        this_compute_ms = float(mb_data.get('this_stage_process_time_ms', 0) or 0.0)
                        this_compute_start_us = _to_int(mb_data.get('this_stage_compute_start_us'), 0)
                        this_compute_end_us = _to_int(mb_data.get('this_stage_compute_end_us'), 0)
                        total_stage2_compute_ms += this_compute_ms
                        
                        # Check overlap with next MB
                        if i + 1 < len(sorted_mb_indices):
                            mb_n_plus_1 = sorted_mb_indices[i + 1]
                            next_mb_data = overlap_summary.get(mb_n_plus_1, {})
                            
                            stage1_next_start_us = _to_int(next_mb_data.get('prev_stage_compute_start_us'), 0)
                            stage1_next_end_us = _to_int(next_mb_data.get('prev_stage_compute_end_us'), 0)
                            stage1_next_clock_offset_us = _to_int(next_mb_data.get('prev_stage_clock_offset_us'), 0)
                            stage1_next_clock_rtt_us = max(0, _to_int(next_mb_data.get('prev_stage_clock_rtt_us'), 0))
                            stage1_next_clock_samples = _to_int(next_mb_data.get('prev_stage_clock_samples'), 0)
                            
                            if (
                                this_compute_start_us <= 0
                                or this_compute_end_us <= 0
                                or stage1_next_start_us <= 0
                                or stage1_next_end_us <= 0
                            ):
                                missing_pair_count += 1
                                if log_mb_detail:
                                    logger.info(
                                        f"[CROSS_STAGE_OVERLAP] step={overlap_tracking_key[1]} MB{mb_n}→MB{mb_n_plus_1}: "
                                        f"skipped (missing timestamps for strict overlap)"
                                    )
                                continue

                            # Convert previous-stage timestamps to this host clock domain.
                            # local_time ~= remote_time + offset_us, where offset_us = local - remote.
                            if stage1_next_clock_samples > 0:
                                stage1_next_start_us += stage1_next_clock_offset_us
                                stage1_next_end_us += stage1_next_clock_offset_us
                                clock_corrected_pair_count += 1
                                clock_offset_abs_sum_ms += abs(stage1_next_clock_offset_us) / 1000.0
                                clock_rtt_max_ms = max(clock_rtt_max_ms, stage1_next_clock_rtt_us / 1000.0)
                            else:
                                clock_uncorrected_pair_count += 1

                            if stage1_next_end_us < stage1_next_start_us or this_compute_end_us < this_compute_start_us:
                                invalid_pair_count += 1
                                if log_mb_detail:
                                    logger.info(
                                        f"[CROSS_STAGE_OVERLAP] step={overlap_tracking_key[1]} MB{mb_n}→MB{mb_n_plus_1}: "
                                        f"skipped (invalid timestamp interval)"
                                    )
                                continue

                            total_comparable_stage2_compute_ms += this_compute_ms

                            overlap_start_us = max(this_compute_start_us, stage1_next_start_us)
                            overlap_end_us = min(this_compute_end_us, stage1_next_end_us)

                            if overlap_end_us > overlap_start_us:
                                actual_overlap_us = overlap_end_us - overlap_start_us
                                actual_overlap_ms = actual_overlap_us / 1000.0
                                total_overlap_ms += actual_overlap_ms

                                mb_overlap_pct = (actual_overlap_ms / this_compute_ms * 100) if this_compute_ms > 0 else 0
                                if log_mb_detail:
                                    logger.info(
                                        f"[CROSS_STAGE_OVERLAP] step={overlap_tracking_key[1]} MB{mb_n}→MB{mb_n_plus_1}: "
                                        f"strict_overlap={actual_overlap_ms:.1f}ms ({mb_overlap_pct:.0f}% of MB{mb_n} compute) ✓"
                                    )
                            else:
                                if log_mb_detail:
                                    if stage1_next_start_us >= this_compute_end_us:
                                        gap_ms = (stage1_next_start_us - this_compute_end_us) / 1000.0
                                        reason = f"no_overlap (gap={gap_ms:.1f}ms, Stage1_next_start after Stage2_end)"
                                    elif this_compute_start_us >= stage1_next_end_us:
                                        gap_ms = (this_compute_start_us - stage1_next_end_us) / 1000.0
                                        reason = f"no_overlap (gap={gap_ms:.1f}ms, Stage2_start after Stage1_next_end)"
                                    else:
                                        reason = "no_overlap (zero-length interval intersection)"
                                    logger.info(
                                        f"[CROSS_STAGE_OVERLAP] step={overlap_tracking_key[1]} MB{mb_n}→MB{mb_n_plus_1}: {reason}"
                                    )
                    
                    # Calculate overlap efficiency
                    if total_stage2_compute_ms > 0:
                        overlap_efficiency = (total_overlap_ms / total_stage2_compute_ms) * 100
                        strict_efficiency = (
                            (total_overlap_ms / total_comparable_stage2_compute_ms) * 100
                            if total_comparable_stage2_compute_ms > 0
                            else 0.0
                        )
                        avg_abs_clock_offset_ms = (
                            (clock_offset_abs_sum_ms / clock_corrected_pair_count)
                            if clock_corrected_pair_count > 0
                            else 0.0
                        )
                        clock_sync_pairs = clock_corrected_pair_count + clock_uncorrected_pair_count
                        clock_sync_coverage_pct = (
                            (clock_corrected_pair_count / clock_sync_pairs) * 100
                            if clock_sync_pairs > 0
                            else 0.0
                        )
                        logger.info(
                            f"[CROSS_STAGE_OVERLAP_SUMMARY] step={overlap_tracking_key[1]} overlap={total_overlap_ms:.1f}ms, "
                            f"Stage2_compute={total_stage2_compute_ms:.1f}ms, "
                            f"efficiency={overlap_efficiency:.1f}%, "
                            f"strict_efficiency={strict_efficiency:.1f}%, "
                            f"comparable_compute={total_comparable_stage2_compute_ms:.1f}ms, "
                            f"missing_pairs={missing_pair_count}, invalid_pairs={invalid_pair_count}, "
                            f"clock_corrected_pairs={clock_corrected_pair_count}, "
                            f"clock_uncorrected_pairs={clock_uncorrected_pair_count}, "
                            f"clock_sync_coverage={clock_sync_coverage_pct:.1f}%, "
                            f"avg_abs_clock_offset={avg_abs_clock_offset_ms:.2f}ms, "
                            f"max_clock_rtt={clock_rtt_max_ms:.2f}ms"
                        )
                    
                    # Cleanup overlap tracking data
                    del iterate_rpc_inference._cross_stage_overlap_data[overlap_tracking_key]
                
                # Now serialize and yield the merged result
                if merged_keep_indices is not None:
                    if not torch.is_tensor(merged_keep_indices):
                        merged_keep_indices = torch.tensor(merged_keep_indices, dtype=torch.int64, device=merged_hidden_states.device)
                    else:
                        merged_keep_indices = merged_keep_indices.to(dtype=torch.int64, device=merged_hidden_states.device)
                else:
                    merged_keep_indices = torch.arange(
                        merged_hidden_states.shape[1],
                        dtype=torch.int64,
                        device=merged_hidden_states.device
                    )
                
                need_pruning_next = torch.tensor(0)
                merge_serialize_start = perf_counter()
                output_tensors = [
                    serialize_torch_tensor(result.to(proto.dtype), proto.compression, allow_inplace=True)
                    for result, proto in zip(
                        (merged_hidden_states, merged_keep_indices, need_pruning_next), 
                        nested_flatten(requested_backends[-1].outputs_schema)
                    )
                ]
                merge_serialize_ms = (perf_counter() - merge_serialize_start) * 1000.0
                merge_total_ms = (perf_counter() - accum.get('step_start_time', step_receive_time)) * 1000.0
                queue_wait_sum_ms = float(accum.get('queue_wait_ms_sum', 0.0))
                queue_wait_pre_ms = float(accum.get('queue_wait_pre_ms', 0.0))
                queue_wait_inter_ms = float(accum.get('queue_wait_inter_ms', 0.0))
                deserialize_sum_ms = float(accum.get('deserialize_ms_sum', 0.0))
                compute_sum_ms = float(accum.get('compute_ms_sum', 0.0))
                merge_residual_ms = merge_total_ms - (
                    deserialize_sum_ms + compute_sum_ms + merge_serialize_ms
                )
                total_with_pre_wait_ms = merge_total_ms + queue_wait_pre_ms
                queue_source_counts = accum.get("queue_source_counts", {})
                queue_source_stats = ",".join(
                    f"{k}:{v}" for k, v in sorted(queue_source_counts.items())
                ) or "unknown:0"
                logger.info(
                    f"[STEP_TIMING_BREAKDOWN_MB] step_id={step_id} mode=micro_batch "
                    f"expected_mb={accum.get('expected', 0)} recv_mb={len(accum.get('results', {}))} "
                    f"queue_wait_sum={queue_wait_sum_ms:.2f}ms "
                    f"deserialize_sum={deserialize_sum_ms:.2f}ms "
                    f"compute_sum={compute_sum_ms:.2f}ms "
                    f"serialize={merge_serialize_ms:.2f}ms "
                    f"residual={merge_residual_ms:.2f}ms "
                    f"total={merge_total_ms:.2f}ms "
                    f"queue_sources={queue_source_stats} "
                    f"queue_wait_pre={queue_wait_pre_ms:.2f}ms "
                    f"queue_wait_inter={queue_wait_inter_ms:.2f}ms "
                    f"total_with_pre_wait={total_with_pre_wait_ms:.2f}ms"
                )
                # Cleanup accumulator
                del iterate_rpc_inference._mb_accumulators[mb_accum_key]
                
                can_push = True  # Micro-batch results don't have prompts
                yield output_tensors, can_push, accum['step_metadata']
                prefix_length += int(accum.get('token_increment', length_increment))
            
            # Continue to wait for more micro-batches (don't process normal path)
            continue

        # [MERGED] Use upstream's standard tensor unpacking for full-batch path
        deserialize_start = perf_counter()
        flat_tensors = tuple(deserialize_torch_tensor(tensor) for tensor in request.tensors)
        deserialize_time = (perf_counter() - deserialize_start) * 1000.0
        if args_structure is not None:
            flat_tensors, kwargs = unpack_args_kwargs(flat_tensors, args_structure)

        hidden_states, keep_indices, need_pruning1, prompts, hypo_ids, tree_attention_mask, kv_cache_position_ids, draft_tokens, prefill_length, is_spec_dec1, *_ = flat_tensors
        draft_tokens = draft_tokens if draft_tokens is not None and not is_dummy(draft_tokens) else None
        batch_size, length_increment, _ = hidden_states.shape

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
        token_increment = _effective_token_increment(hidden_states, kv_cache_position_ids, is_spec_dec)
        
        if is_spec_dec and draft_tokens is not None and draft_tokens.shape[0] != hidden_states.shape[0]:
            hidden_states = restore_hidden_states(hidden_states, keep_indices, draft_tokens.shape[-1])
        
        # if is_spec_dec and not need_pruning:
            
        #     prefill_len = draft_tokens.shape[-1] - tree_attention_mask.shape[-1]
        #     attention_mask_indices = keep_indices[:, prefill_len:] - prefill_len
        #     logger.info(f"keep_indices: {keep_indices}")
        #     logger.info(f"draft_tokens: {draft_tokens}, tree_attention_mask: {tree_attention_mask.shape}")
        #     logger.info(f"prefill_length: {prefill_length}, attention_mask_indices: {attention_mask_indices}")
        #     idx = attention_mask_indices
        #     idx_2d = idx.unsqueeze(-1).expand(-1, -1, tree_attention_mask.size(-1))
        #     tree_attention_mask = torch.gather(tree_attention_mask, dim=1, index=idx_2d)
        #     idx_3d = idx.unsqueeze(1).expand(-1, tree_attention_mask.size(1), -1)
        #     tree_attention_mask = torch.gather(tree_attention_mask, dim=2, index=idx_3d)
        #     logger.info(f"tree_attention_mask: {tree_attention_mask.shape}")

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
            # print('before merge pools ')
            #print_time_now('')
            
            # Add offloading debug information
            # offload_logger.info(f" Inference computation started - step {prefix_length}")
            # offload_logger.info(f"   - Batch size: {batch_size}")
            # offload_logger.info(f"   - Length increment: {length_increment}")
            # offload_logger.info(f"   - Prefix length: {prefix_length}")
            # offload_logger.info(f"   - Max length: {max_length}")
            
            # # Check cache usage
            # for i, (backend, handles) in enumerate(zip(requested_backends, cache_handles)):
            #     cache_manager = backend.cache_manager
            #     offload_logger.info(f"   - Backend {i}: {len(handles)} cache handles")
            #     offload_logger.info(f"     GPU cache ratio: {cache_manager.offloading_policy.cache_gpu_percent}%")
            #     offload_logger.info(f"     CPU cache ratio: {cache_manager.offloading_policy.cache_cpu_percent}%")
            #     offload_logger.info(f"     CPU cache compute: {cache_manager.offloading_policy.cpu_cache_compute}")
            last_uid = len(requested_uids) - 1
            if can_merge_pools:
                # Merged pools path: all blocks processed in one call
                # [MBPIPE] Check if we should split into micro-batches with pipeline overlap
                if should_split_batch(batch_size) and not is_spec_dec:
                    execution_mode = "merged_microbatch"
                    # Micro-batch pipeline path: split batch, process with overlap
                    import asyncio
                    from time import perf_counter as _perf_counter
                    log_mb_detail = _should_log_mb_detail(step_metadata)
                    verbose_mb = _mbpipe_verbose_enabled()
                    
                    micro_ranges = compute_micro_batch_ranges(batch_size)
                    log_microbatch_split(logger, batch_size, len(micro_ranges), "iterate_rpc_inference.merged_pools")
                    
                    # Track timing for overlap statistics
                    mb_timings = []  # [(start_time, end_time), ...]
                    cross_stage_push_tasks = []  # [MBPIPE] Track cross-stage push tasks
                    pipeline_start_time = _perf_counter()
                    
                    # [MBPIPE] Get next_servers from step_metadata for cross-stage streaming
                    next_servers = step_metadata.get("next_servers", None) if step_metadata else None
                    enable_cross_stage = (
                        cross_stage_push_fn is not None
                        and next_servers is not None
                        and not is_spec_dec
                    )
                    if (
                        is_spec_dec
                        and cross_stage_push_fn is not None
                        and next_servers is not None
                        and log_mb_detail
                    ):
                        logger.info(
                            f"{MBPIPE_LOG_PREFIX} Cross-stage streaming disabled for speculative decoding "
                            f"(preserve full spec context)"
                        )
                    if enable_cross_stage and log_mb_detail:
                        logger.info(f"{MBPIPE_LOG_PREFIX} Cross-stage streaming enabled: {len(next_servers)} downstream stages")
                    
                    async def process_microbatch_merged(mb_idx: int, mb_start: int, mb_end: int, total_mb: int):
                        """Process a single micro-batch through merged pool with GPU memory reuse."""
                        mb_start_time = _perf_counter()
                        
                        mb_size = mb_end - mb_start
                        
                        # [MBPIPE_MEM_DEBUG] Log memory only in verbose mode to reduce noise.
                        if verbose_mb:
                            try:
                                from bloombee.utils.memory_usage import log_mbpipe_memory, get_gpu_memory_mb
                                mem_before = get_gpu_memory_mb()
                                logger.debug(f"[MBPIPE_MEM_DEBUG] ===== MB{mb_idx} START (size={mb_size}, offset={mb_start}) =====")
                                log_mbpipe_memory(f"MB{mb_idx}_START", f"size={mb_size}, total_mb={total_mb}")
                            except Exception as e:
                                mem_before = 0
                                logger.debug(f"[MBPIPE_MEM_DEBUG] Memory logging failed: {e}")
                        else:
                            mem_before = 0
                        
                        # [CROSS_STAGE] Record absolute timestamp when compute starts for overlap analysis
                        import time as _time
                        compute_start_timestamp_us = int(_time.time() * 1_000_000)
                        if log_mb_detail:
                            logger.info(f"[CROSS_STAGE] Stage starts MB{mb_idx} compute at t={compute_start_timestamp_us}")
                        
                        mb_hidden = hidden_states[mb_start:mb_end].clone()
                        mb_hypo = hypo_ids[mb_start:mb_end] if hypo_ids is not None and not is_dummy(hypo_ids) else hypo_ids
                        mb_tree_mask = _slice_batch_aligned(tree_attention_mask, mb_start, mb_end, batch_size)
                        mb_kv_position_ids = _slice_batch_aligned(kv_cache_position_ids, mb_start, mb_end, batch_size)
                        mb_draft_tokens = _slice_batch_aligned(draft_tokens, mb_start, mb_end, batch_size)
                        mb_prefill_length = _slice_batch_aligned(prefill_length, mb_start, mb_end, batch_size)
                        mb_keep_idx = keep_indices[mb_start:mb_end] if keep_indices is not None and not is_dummy(keep_indices) and keep_indices.dim() > 0 and keep_indices.shape[0] == batch_size else keep_indices
                        
                        # [MBPIPE_MULTIPLEX] GPU Memory Multiplexing Mode:
                        # When micro-batching is enabled, GPU cache is sized for micro_batch_size only.
                        # All micro-batches REUSE the same GPU cache slots (offset=0).
                        # This achieves true GPU memory savings!
                        #
                        # Key insight: 
                        # - GPU cache shape: (S, micro_batch_size * H, D)
                        # - Each micro-batch writes to offset=0 (not mb_start)
                        # - offload() saves data to CPU after compute
                        # - prefetch() loads data from CPU before compute
                        from bloombee.utils.microbatch_config import is_microbatch_enabled, get_micro_batch_size
                        
                        # [MBPIPE_FIX] Always pass logical offsets to cache manager
                        # The cache manager will detect if we are in Multiplexing Mode (cache_size == mb_size)
                        # or Legacy Mode (cache_size == full_batch_size) and handle the physical offset mapping internally.
                        #
                        # Previously, we forced offset=0 here, which broke Legacy Mode where the cache 
                        # is large enough for the full batch.
                        cache_batch_offset = mb_start
                        cache_full_batch_size = batch_size
                        
                        if is_microbatch_enabled() and total_mb > 1:
                            logger.debug(f"[MBPIPE_DEBUG] Micro-batch enabled: MB{mb_idx}/{total_mb} uses logical offset={cache_batch_offset}")
                        else:
                            logger.debug(f"[MBPIPE_DEBUG] MB{mb_idx}: Legacy/Single mode, cache_batch_offset={cache_batch_offset}")
                        
                        mb_inference_infos = tuple(
                            InferenceMetadata(uid, prefix_length, tuple(handles), active_adapter,
                                tree_attention_mask=mb_tree_mask,
                                kv_cache_position_ids=mb_kv_position_ids,
                                draft_tokens=mb_draft_tokens,
                                prefill_length=mb_prefill_length,
                                keep_indices=mb_keep_idx,
                                need_pruning=need_pruning,
                                is_spec_dec=is_spec_dec,
                                batch_offset=cache_batch_offset,
                                full_batch_size=cache_full_batch_size,
                                micro_batch_size=mb_size)
                            for uid, handles in zip(requested_uids, cache_handles)
                        )
                        
                        # [KVCACHE_OFFLOAD] STEP 2: Compute - forward pass through model
                        compute_start = _perf_counter()
                        submit_result = await requested_backends[0].inference_pool.submit_task(
                            mb_hidden, mb_hypo, mb_inference_infos, *prompts, priority=priority
                        )
                        mb_out_hidden, mb_out_keep, runtime_kv_timing = _unpack_inference_submit_result(submit_result)
                        compute_time_ms = (_perf_counter() - compute_start) * 1000
                        
                        mb_end_time = _perf_counter()
                        mb_timings.append((mb_idx, mb_start_time, mb_end_time))
                        
                        # [MBPIPE_MEM_DEBUG] Log memory after compute in verbose mode only.
                        if verbose_mb:
                            try:
                                log_mbpipe_memory(f"MB{mb_idx}_AFTER_COMPUTE", f"compute_time={compute_time_ms:.1f}ms")
                            except Exception:
                                pass
                        
                        # Use runtime-side KV timing delta returned by merged inference step.
                        kv_timing = runtime_kv_timing or {}
                        if runtime_kv_timing is None:
                            logger.debug("[KVCACHE_TIMING] Runtime KV timing not returned, using zero fallback")
                        prefetch_wait_time_ms = float(kv_timing.get("prefetch_wait_ms", 0.0))
                        offload_wait_time_ms = float(kv_timing.get("offload_wait_ms", 0.0))
                        prefetch_launch_time_ms = float(kv_timing.get("prefetch_launch_ms", 0.0))
                        offload_launch_time_ms = float(kv_timing.get("offload_launch_ms", 0.0))
                        prefetch_wait_calls = int(kv_timing.get("prefetch_wait_calls", 0))
                        offload_wait_calls = int(kv_timing.get("offload_wait_calls", 0))
                        prefetch_launch_calls = int(kv_timing.get("prefetch_launch_calls", 0))
                        offload_launch_calls = int(kv_timing.get("offload_launch_calls", 0))
                        
                        # [MBPIPE_MEM_DEBUG] End-of-micro-batch memory summary in verbose mode only.
                        if verbose_mb:
                            try:
                                mem_after = get_gpu_memory_mb()
                                mem_delta = mem_after - mem_before
                                logger.debug(f"[MBPIPE_MEM_DEBUG] ===== MB{mb_idx} END =====")
                                logger.debug(f"[MBPIPE_MEM_DEBUG] MB{mb_idx} Memory Delta: {mem_delta:+.1f}MB (before={mem_before:.1f}MB, after={mem_after:.1f}MB)")
                                if abs(mem_delta) < 1.0:
                                    logger.debug(f"[MBPIPE_MEM_DEBUG] NOTE: No significant memory change - offload/prefetch NOT reducing GPU memory!")
                            except Exception:
                                pass
                        
                        # [KVCACHE_OFFLOAD] Log detailed timing breakdown for this micro-batch
                        total_mb_time_ms = (mb_end_time - mb_start_time) * 1000
                        transfer_wait_time_ms = prefetch_wait_time_ms + offload_wait_time_ms
                        transfer_launch_time_ms = prefetch_launch_time_ms + offload_launch_time_ms
                        if log_mb_detail:
                            logger.info(
                                f"[KVCACHE_TIMING] MB{mb_idx}: compute={compute_time_ms:.2f}ms, "
                                f"prefetch_wait={prefetch_wait_time_ms:.2f}ms({prefetch_wait_calls}), "
                                f"offload_wait={offload_wait_time_ms:.2f}ms({offload_wait_calls}), "
                                f"prefetch_launch={prefetch_launch_time_ms:.2f}ms({prefetch_launch_calls}), "
                                f"offload_launch={offload_launch_time_ms:.2f}ms({offload_launch_calls}), "
                                f"wait_total={transfer_wait_time_ms:.2f}ms, launch_total={transfer_launch_time_ms:.2f}ms, "
                                f"total={total_mb_time_ms:.2f}ms"
                            )
                        
                        # Store timing in list for later analysis
                        kv_timing_stats.append({
                            'mb_idx': mb_idx,
                            'prefetch_wait_ms': prefetch_wait_time_ms,
                            'offload_wait_ms': offload_wait_time_ms,
                            'prefetch_wait_calls': prefetch_wait_calls,
                            'offload_wait_calls': offload_wait_calls,
                            'prefetch_launch_ms': prefetch_launch_time_ms,
                            'offload_launch_ms': offload_launch_time_ms,
                            'prefetch_launch_calls': prefetch_launch_calls,
                            'offload_launch_calls': offload_launch_calls,
                            # Backward-compat aliases for existing diagnostics logic.
                            'prefetch_sync_ms': prefetch_wait_time_ms,
                            'offload_ms': offload_launch_time_ms,
                            'next_prefetch_start_ms': prefetch_launch_time_ms,
                            'compute_ms': compute_time_ms,
                            'total_ms': total_mb_time_ms
                        })
                        
                        # [MBPIPE] Cross-stage streaming: push ALL micro-batches to next stage immediately
                        # This enables Server2 to start processing before Server1 finishes all micro-batches
                        if enable_cross_stage:
                            compute_end_timestamp_us = int(_time.time() * 1_000_000)
                            push_timestamp_us = int(_time.time() * 1_000_000)  # Absolute timestamp for cross-server correlation
                            push_metadata = step_metadata.copy()
                            push_metadata["micro_batch_idx"] = mb_idx
                            push_metadata["micro_batch_offset"] = mb_start
                            push_metadata["micro_batch_size"] = mb_size
                            push_metadata["full_batch_size"] = batch_size
                            push_metadata["total_micro_batches"] = len(micro_ranges)
                            push_metadata["stage_push_timestamp_us"] = push_timestamp_us  # For overlap analysis
                            push_metadata["stage_compute_start_timestamp_us"] = compute_start_timestamp_us  # When this MB started computing
                            push_metadata["stage_compute_end_timestamp_us"] = compute_end_timestamp_us

                            # [MBPIPE_FIX] Clone tensors before async push to avoid buffer reuse races.
                            push_hidden = mb_out_hidden.detach().clone()
                            push_keep = mb_out_keep.detach().clone() if torch.is_tensor(mb_out_keep) else mb_out_keep

                            push_task = asyncio.create_task(
                                cross_stage_push_fn(push_hidden, push_keep, push_metadata)
                            )
                            cross_stage_push_tasks.append(push_task)
                            if log_mb_detail:
                                logger.info(f"[CROSS_STAGE] Stage sends MB{mb_idx} at t={push_timestamp_us}, "
                                           f"compute_started_at={compute_start_timestamp_us}, compute_time={compute_time_ms:.2f}ms")
                        
                        return mb_out_hidden, mb_out_keep
                    
                    # [KVCACHE_OFFLOAD] Track timing statistics for overlap analysis
                    kv_timing_stats = []
                    
                    # [KVCACHE_OFFLOAD] Process micro-batches SEQUENTIALLY for GPU memory reuse
                    # Parallel processing won't work because GPU only has space for ONE micro-batch
                    results = []
                    total_mb = len(micro_ranges)
                    for mb_idx, (mb_start, mb_end) in enumerate(micro_ranges):
                        result = await process_microbatch_merged(mb_idx, mb_start, mb_end, total_mb)
                        results.append(result)
                    
                    # [KVCACHE_OFFLOAD] State clearing moved to handler.py to persist across streaming steps
                    # cache_manager = requested_backends[0].cache_manager if requested_backends else None
                    # if cache_manager is not None and hasattr(cache_manager, 'clear_offload_state'):
                    #     cache_manager.clear_offload_state()
                    
                    pipeline_end_time = _perf_counter()
                    
                    # [MBPIPE] Wait for cross-stage push tasks (fire-and-forget style, don't block)
                    if cross_stage_push_tasks:
                        # Don't await - let them complete in background
                        logger.debug(f"{MBPIPE_LOG_PREFIX} {len(cross_stage_push_tasks)} cross-stage push tasks running in background")
                        # Mark that cross-stage push is handling the data transfer
                        if step_metadata is not None:
                            step_metadata["cross_stage_pushed"] = True
                    
                    # Merge results
                    micro_hidden_list = [r[0] for r in results]
                    micro_keep_list = [r[1] for r in results]
                    
                    hidden_states = merge_microbatch_outputs(micro_hidden_list, dim=0)
                    keep_indices = micro_keep_list[-1] if micro_keep_list else None
                    
                    # Calculate overlap statistics
                    total_pipeline_time = (pipeline_end_time - pipeline_start_time) * 1000  # ms
                    sum_mb_times = sum((end - start) * 1000 for _, start, end in mb_timings)
                    overlap_time = max(0, sum_mb_times - total_pipeline_time)
                    overlap_ratio = (overlap_time / sum_mb_times * 100) if sum_mb_times > 0 else 0
                    
                    if log_mb_detail:
                        # Local micro-batch timing window only (single-stage), not cross-stage overlap.
                        logger.debug(
                            f"{MBPIPE_LOG_PREFIX} LocalWindow stats (not cross-stage): "
                            f"total={total_pipeline_time:.1f}ms, "
                            f"sum_mb={sum_mb_times:.1f}ms, overlap={overlap_time:.1f}ms ({overlap_ratio:.1f}%)"
                        )
                    
                    # [KVCACHE_OFFLOAD] Calculate and log detailed KV cache overlap statistics
                    if kv_timing_stats:
                        total_prefetch_wait = sum(s['prefetch_wait_ms'] for s in kv_timing_stats)
                        total_offload_wait = sum(s['offload_wait_ms'] for s in kv_timing_stats)
                        total_prefetch_launch = sum(s['prefetch_launch_ms'] for s in kv_timing_stats)
                        total_offload_launch = sum(s['offload_launch_ms'] for s in kv_timing_stats)
                        total_prefetch_wait_calls = sum(s['prefetch_wait_calls'] for s in kv_timing_stats)
                        total_offload_wait_calls = sum(s['offload_wait_calls'] for s in kv_timing_stats)
                        total_prefetch_launch_calls = sum(s['prefetch_launch_calls'] for s in kv_timing_stats)
                        total_offload_launch_calls = sum(s['offload_launch_calls'] for s in kv_timing_stats)
                        total_compute = sum(s['compute_ms'] for s in kv_timing_stats)
                        total_wait_overhead = total_prefetch_wait + total_offload_wait
                        total_launch_overhead = total_prefetch_launch + total_offload_launch
                        total_elapsed = sum(s['total_ms'] for s in kv_timing_stats)

                        # Lower wait-overhead ratio means better overlap between transfer and compute.
                        compute_efficiency = (total_compute / total_elapsed * 100) if total_elapsed > 0 else 0
                        wait_overhead_ratio = (total_wait_overhead / total_elapsed * 100) if total_elapsed > 0 else 0
                        launch_overhead_ratio = (total_launch_overhead / total_elapsed * 100) if total_elapsed > 0 else 0
                        
                        start_pos_for_log = step_metadata.get("start_from_position") if isinstance(step_metadata, dict) else "unknown"
                        logger.info(
                            f"[MBPIPE_SUMMARY] step={start_pos_for_log} mb={len(kv_timing_stats)} "
                            f"compute={total_compute:.1f}ms elapsed={total_elapsed:.1f}ms "
                            f"wait={total_wait_overhead:.1f}ms({wait_overhead_ratio:.1f}%) "
                            f"launch={total_launch_overhead:.1f}ms({launch_overhead_ratio:.1f}%) "
                            f"efficiency={compute_efficiency:.1f}%"
                        )
                        if log_mb_detail:
                            logger.info(f"[MBPIPE_SUMMARY] Total prefetch wait: {total_prefetch_wait:.1f}ms ({total_prefetch_wait_calls} calls)")
                            logger.info(f"[MBPIPE_SUMMARY] Total offload wait: {total_offload_wait:.1f}ms ({total_offload_wait_calls} calls)")
                            logger.info(f"[MBPIPE_SUMMARY] Total prefetch launch: {total_prefetch_launch:.1f}ms ({total_prefetch_launch_calls} calls)")
                            logger.info(f"[MBPIPE_SUMMARY] Total offload launch: {total_offload_launch:.1f}ms ({total_offload_launch_calls} calls)")
                        if verbose_mb:
                            logger.info(f"[MBPIPE_SUMMARY] ----- Per Micro-Batch Breakdown -----")
                            for stat in kv_timing_stats:
                                mb_eff = (stat['compute_ms'] / stat['total_ms'] * 100) if stat['total_ms'] > 0 else 0
                                logger.info(f"[MBPIPE_SUMMARY] MB{stat['mb_idx']}: "
                                           f"compute={stat['compute_ms']:.1f}ms, "
                                           f"prefetch_wait={stat['prefetch_wait_ms']:.1f}ms({stat['prefetch_wait_calls']}), "
                                           f"offload_wait={stat['offload_wait_ms']:.1f}ms({stat['offload_wait_calls']}), "
                                           f"prefetch_launch={stat['prefetch_launch_ms']:.1f}ms({stat['prefetch_launch_calls']}), "
                                           f"offload_launch={stat['offload_launch_ms']:.1f}ms({stat['offload_launch_calls']}), "
                                           f"efficiency={mb_eff:.1f}%")
                    
                    log_microbatch_merge(logger, len(micro_ranges), hidden_states.shape[0], "iterate_rpc_inference.merged_pools")
                    
                else:
                    execution_mode = "merged_fullbatch"
                    # Legacy path: process entire batch at once
                    inference_infos = tuple(
                        InferenceMetadata(uid, prefix_length, tuple(handles), active_adapter, tree_attention_mask=tree_attention_mask, kv_cache_position_ids=kv_cache_position_ids, draft_tokens=draft_tokens, prefill_length=prefill_length, keep_indices=keep_indices, need_pruning=need_pruning, is_spec_dec=is_spec_dec)
                        for i, (uid, handles) in enumerate(zip(requested_uids, cache_handles))
                    )
                    submit_result = await requested_backends[0].inference_pool.submit_task(
                        hidden_states, hypo_ids, inference_infos, *prompts, priority=priority
                    )
                    hidden_states, keep_indices, _ = _unpack_inference_submit_result(submit_result)
                
            else:
                # Separate pools path: process backends one by one
                # [MBPIPE] Check if we should split into micro-batches with pipeline overlap
                if should_split_batch(batch_size) and not is_spec_dec:
                    execution_mode = "separate_microbatch"
                    # Micro-batch pipeline path: split batch, process with overlap
                    micro_ranges = compute_micro_batch_ranges(batch_size)
                    log_microbatch_split(logger, batch_size, len(micro_ranges), "iterate_rpc_inference.separate_pools")
                    
                    # Process micro-batches with pipeline overlap using asyncio
                    import asyncio
                    from time import perf_counter as _perf_counter
                    
                    # Track timing for overlap statistics
                    mb_timings = []
                    cross_stage_push_tasks = []  # [MBPIPE] Track cross-stage push tasks
                    pipeline_start_time = _perf_counter()
                    
                    # [MBPIPE] Get next_servers from step_metadata for cross-stage streaming
                    next_servers = step_metadata.get("next_servers", None) if step_metadata else None
                    enable_cross_stage = (
                        cross_stage_push_fn is not None
                        and next_servers is not None
                        and not is_spec_dec
                    )
                    if is_spec_dec and cross_stage_push_fn is not None and next_servers is not None:
                        logger.info(
                            f"{MBPIPE_LOG_PREFIX} Cross-stage streaming disabled (separate_pools) for speculative decoding "
                            f"(preserve full spec context)"
                        )
                    if enable_cross_stage:
                        logger.info(f"{MBPIPE_LOG_PREFIX} Cross-stage streaming enabled (separate_pools): {len(next_servers)} downstream stages")
                    
                    async def process_microbatch(mb_idx: int, mb_start: int, mb_end: int):
                        """Process a single micro-batch through all backends."""
                        mb_start_time = _perf_counter()
                        if enable_cross_stage:
                            import time as _time
                            compute_start_timestamp_us = int(_time.time() * 1_000_000)
                        else:
                            compute_start_timestamp_us = 0
                        
                        mb_hidden = hidden_states[mb_start:mb_end].clone()
                        mb_hypo = hypo_ids[mb_start:mb_end] if hypo_ids is not None and not is_dummy(hypo_ids) else hypo_ids
                        mb_tree_mask = _slice_batch_aligned(tree_attention_mask, mb_start, mb_end, batch_size)
                        mb_kv_position_ids = _slice_batch_aligned(kv_cache_position_ids, mb_start, mb_end, batch_size)
                        mb_draft_tokens = _slice_batch_aligned(draft_tokens, mb_start, mb_end, batch_size)
                        mb_prefill_length = _slice_batch_aligned(prefill_length, mb_start, mb_end, batch_size)
                        mb_keep_idx = keep_indices[mb_start:mb_end] if keep_indices is not None and not is_dummy(keep_indices) and keep_indices.dim() > 0 and keep_indices.shape[0] == batch_size else keep_indices
                        
                        mb_size = mb_end - mb_start
                        
                        for i, (backend, uid, handles, prompt) in enumerate(zip(requested_backends, requested_uids, cache_handles, prompts)):
                            # [MBPIPE] Pass batch_offset, full_batch_size and micro_batch_size for KV cache slicing
                            inference_info = (InferenceMetadata(
                                uid, prefix_length, tuple(handles), active_adapter,
                                tree_attention_mask=mb_tree_mask,
                                kv_cache_position_ids=mb_kv_position_ids,
                                draft_tokens=mb_draft_tokens,
                                prefill_length=mb_prefill_length,
                                keep_indices=mb_keep_idx,
                                need_pruning=need_pruning,
                                is_spec_dec=is_spec_dec,
                                batch_offset=mb_start,
                                full_batch_size=batch_size,
                                micro_batch_size=mb_size,
                            ),)
                            
                            submit_result = await backend.inference_pool.submit_task(
                                mb_hidden, mb_hypo, inference_info, prompt, priority=priority
                            )
                            mb_hidden, mb_keep_idx, _ = _unpack_inference_submit_result(submit_result)
                        
                        mb_end_time = _perf_counter()
                        mb_timings.append((mb_idx, mb_start_time, mb_end_time))
                        
                        # [MBPIPE] Cross-stage streaming: push ALL micro-batches to next stage immediately
                        if enable_cross_stage:
                            compute_end_timestamp_us = int(_time.time() * 1_000_000)
                            push_timestamp_us = int(_time.time() * 1_000_000)
                            push_metadata = step_metadata.copy()
                            push_metadata["micro_batch_idx"] = mb_idx
                            push_metadata["micro_batch_offset"] = mb_start
                            push_metadata["micro_batch_size"] = mb_size
                            push_metadata["full_batch_size"] = batch_size
                            push_metadata["total_micro_batches"] = len(micro_ranges)
                            push_metadata["stage_compute_start_timestamp_us"] = compute_start_timestamp_us
                            push_metadata["stage_compute_end_timestamp_us"] = compute_end_timestamp_us
                            push_metadata["stage_push_timestamp_us"] = push_timestamp_us
                            # [MBPIPE_FIX] Clone tensors before async push to avoid buffer reuse races.
                            push_hidden = mb_hidden.detach().clone()
                            push_keep = mb_keep_idx.detach().clone() if torch.is_tensor(mb_keep_idx) else mb_keep_idx
                            push_task = asyncio.create_task(
                                cross_stage_push_fn(push_hidden, push_keep, push_metadata)
                            )
                            cross_stage_push_tasks.append(push_task)
                            logger.info(f"{MBPIPE_LOG_PREFIX} Cross-stage push: micro-batch {mb_idx+1}/{len(micro_ranges)} sent to next stage")
                        
                        return mb_hidden, mb_keep_idx
                    
                    # Run sequentially for KV cache correctness.
                    # In separate_pools mode, backends still share multiplexed KV slots per step;
                    # concurrent micro-batches can race on cache prefetch/write/offload.
                    logger.info(
                        f"{MBPIPE_LOG_PREFIX} separate_pools micro-batches run sequentially for KV safety"
                    )
                    results = []
                    for mb_idx, (mb_start, mb_end) in enumerate(micro_ranges):
                        result = await process_microbatch(mb_idx, mb_start, mb_end)
                        results.append(result)

                    pipeline_end_time = _perf_counter()
                    
                    # [MBPIPE] Wait for cross-stage push tasks (fire-and-forget style)
                    if cross_stage_push_tasks:
                        logger.debug(f"{MBPIPE_LOG_PREFIX} {len(cross_stage_push_tasks)} cross-stage push tasks running in background")
                        # Mark that cross-stage push is handling the data transfer
                        if step_metadata is not None:
                            step_metadata["cross_stage_pushed"] = True
                    
                    # Merge results
                    micro_hidden_list = [r[0] for r in results]
                    micro_keep_list = [r[1] for r in results]
                    
                    hidden_states = merge_microbatch_outputs(micro_hidden_list, dim=0)
                    keep_indices = micro_keep_list[-1] if micro_keep_list else None
                    
                    # Calculate overlap statistics
                    total_pipeline_time = (pipeline_end_time - pipeline_start_time) * 1000  # ms
                    sum_mb_times = sum((end - start) * 1000 for _, start, end in mb_timings)
                    overlap_time = max(0, sum_mb_times - total_pipeline_time)
                    overlap_ratio = (overlap_time / sum_mb_times * 100) if sum_mb_times > 0 else 0
                    
                    # Local micro-batch timing window only (single-stage), not cross-stage overlap.
                    logger.debug(
                        f"{MBPIPE_LOG_PREFIX} LocalWindow stats (not cross-stage): "
                        f"total={total_pipeline_time:.1f}ms, "
                        f"sum_mb={sum_mb_times:.1f}ms, overlap={overlap_time:.1f}ms ({overlap_ratio:.1f}%)"
                    )
                    
                    log_microbatch_merge(logger, len(micro_ranges), hidden_states.shape[0], "iterate_rpc_inference.separate_pools")
                    
                else:
                    execution_mode = "separate_fullbatch"
                    # Legacy path: process entire batch through all backends sequentially
                    # Track S1->S2 transfer latency specifically
                    s1_to_s2_transfer_times = []
                    backend_processing_times = []
                    
                    for i, (backend, uid, handles, prompt) in enumerate(zip(requested_backends, requested_uids, cache_handles, prompts)):
                        backend_start_time = perf_counter()
                        
                        inference_infos = (InferenceMetadata(uid, prefix_length, tuple(handles), active_adapter, tree_attention_mask=tree_attention_mask, kv_cache_position_ids=kv_cache_position_ids, draft_tokens=draft_tokens, prefill_length=prefill_length, keep_indices=keep_indices, need_pruning=need_pruning, is_spec_dec=is_spec_dec),)
                        
                        # Submit task and measure processing time
                        task_start_time = perf_counter()
                        submit_result = await backend.inference_pool.submit_task(
                            hidden_states, hypo_ids, inference_infos, prompt, priority=priority
                        )
                        hidden_states, keep_indices, _ = _unpack_inference_submit_result(submit_result)
                        task_end_time = perf_counter()
                        task_processing_time = (task_end_time - task_start_time) * 1000
                        
                        backend_end_time = perf_counter()
                        backend_total_time = (backend_end_time - backend_start_time) * 1000
                        
                        backend_processing_times.append(task_processing_time)
                        
                        if i > 0:
                            estimated_transfer_time = backend_total_time - task_processing_time
                            s1_to_s2_transfer_times.append(estimated_transfer_time)
                            logger.info(f"[S1_TO_S2_TRANSFER] Backend {i} | "
                                       f"Estimated Transfer Time: {estimated_transfer_time:.2f}ms | "
                                       f"Total Backend Time: {backend_total_time:.2f}ms | "
                                       f"Pure Processing: {task_processing_time:.2f}ms")
                        
                        logger.info(f"[PROCESSING_LATENCY] Backend {i} | "
                                   f"Task Processing: {task_processing_time:.2f}ms | "
                                   f"Total Backend Time: {backend_total_time:.2f}ms | "
                                   f"Hidden States Shape: {hidden_states.shape}")
                    
                    if s1_to_s2_transfer_times:
                        s1_to_s2_mean = sum(s1_to_s2_transfer_times) / len(s1_to_s2_transfer_times)
                        s1_to_s2_total = sum(s1_to_s2_transfer_times)
                        logger.info(f"[S1_TO_S2_TRANSFER_SUMMARY] "
                                   f"Average Transfer: {s1_to_s2_mean:.2f}ms | "
                                   f"Total Transfer: {s1_to_s2_total:.2f}ms | "
                                   f"Transfer Count: {len(s1_to_s2_transfer_times)}")
                    
                    cross_gpu_end_time = perf_counter()
                    cross_gpu_latency = (cross_gpu_end_time - cross_gpu_start_time) * 1000
                    
                    logger.info(f"[CROSS_GPU_TRANSFER_LATENCY] Total: {cross_gpu_latency:.2f}ms | "
                               f"Backends: {len(requested_backends)} | "
                               f"Output Shape: {hidden_states.shape}")
            
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
        output_tensors = [
            serialize_torch_tensor(result.to(proto.dtype), proto.compression, allow_inplace=True)
            for result, proto in zip((hidden_states, keep_indices, need_pruning_next), nested_flatten(requested_backends[-1].outputs_schema))
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
        logger.info(
            f"[STEP_TIMING_BREAKDOWN] step_id={step_id_for_log} mode={execution_mode} "
            f"queue_wait={queue_wait_ms:.2f}ms queue_source={queue_source} "
                f"deserialize={deserialize_time:.2f}ms compute={compute_time:.2f}ms "
                f"serialize={serialize_time:.2f}ms residual={step_residual_ms:.2f}ms "
                f"step_total={step_total_time:.2f}ms total_with_queue={step_total_with_queue_ms:.2f}ms "
                f"cross_gpu_window={cross_gpu_receive_time:.2f}ms "
                f"batch={batch_size} seq_inc={token_increment} raw_seq={length_increment} is_spec_dec={int(bool(is_spec_dec))}"
            )
        
        # [MBPIPE] Record stage timing for cross-stage overlap decisions
        try:
            # Extract stage ID safely from UIDs (format: "prefix.block_idx")
            first_uid = str(requested_uids[0]) if requested_uids else "unknown"
            last_uid = str(requested_uids[-1]) if requested_uids else "unknown"
            # Try to extract block indices
            first_idx = first_uid.split('.')[-1] if '.' in first_uid else "0"
            last_idx = last_uid.split('.')[-1] if '.' in last_uid else "0"
            stage_id = f"blocks_{first_idx}_{last_idx}"
            
            log_stage_timing(
                logger, stage_id,
                compute_time_ms=step_total_time - cross_gpu_receive_time,  # Compute time (excluding comm)
                comm_time_ms=cross_gpu_receive_time,  # Communication time (if any)
                component="iterate_rpc_inference"
            )
        except Exception as e:
            # Don't let timing logging break the main flow
            logger.debug(f"{MBPIPE_LOG_PREFIX} Failed to log stage timing: {e}")
        
        yield output_tensors, can_push, step_metadata
        # print('output_tensors ',output_tensors)
        # prepare for next step
        prefix_length += token_increment

    end_iterate_rpc_infer_time = perf_counter()#######
    # print('iterate (all steps) rpc infer time cost (sec): ', end_iterate_rpc_infer_time - start_iterate_rpc_infer_time)########
    # #print_time_now('')
    # print()
