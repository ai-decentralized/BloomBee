"""
Micro-batch Pipeline Configuration Module.

The feature is controlled via environment variables:
- BLOOMBEE_ENABLE_MICROBATCH_PIPELINE: "1" to enable, "0" to disable (default)
- BLOOMBEE_MICRO_BATCH_SIZE: positive integer micro-batch size
- BLOOMBEE_MICRO_ENABLE_GPU_MULTIPLEXING: "1" to enable bounded GPU working-slot
  reuse. Default is "0", which keeps micro-batching in overlap-only mode so
  FlexGen cache offload remains available.

Non-positive micro-batch sizes are treated as disabled to make local toggling safe.
All logs from this feature use the prefix [MBPIPE].
"""

import os
import logging
from typing import Tuple, List, Optional, Sequence, Any

import torch

from bloombee.utils.debug_config import is_log_channel_enabled

# Prefix for all micro-batch pipeline logs
MBPIPE_LOG_PREFIX = "[MBPIPE]"


def mbpipe_info_logs_enabled() -> bool:
    return is_log_channel_enabled("microbatch_logs")

# Environment variable names
ENV_ENABLE_MICROBATCH = "BLOOMBEE_ENABLE_MICROBATCH_PIPELINE"
ENV_MICRO_BATCH_SIZE = "BLOOMBEE_MICRO_BATCH_SIZE"
ENV_ENABLE_GPU_MULTIPLEXING = "BLOOMBEE_MICRO_ENABLE_GPU_MULTIPLEXING"

# Default values
# Micro-batch size for pipeline overlap. Each micro-batch writes to its own slice of the KV cache.
DEFAULT_MICRO_BATCH_SIZE = 0 # Default micro-batch size for pipeline overlap


def _is_microbatch_flag_enabled() -> bool:
    """
    Check if micro-batch pipeline is enabled via environment variable.

    Returns:
        True only if BLOOMBEE_ENABLE_MICROBATCH_PIPELINE is set to "1".
        False by default.
    """
    env_value = os.environ.get(ENV_ENABLE_MICROBATCH, "0")  # Default: disabled
    return env_value == "1"


def get_micro_batch_size() -> int:
    """
    Get the configured micro-batch size from environment variable.
    
    Returns:
        The micro-batch size. If not set or invalid, returns DEFAULT_MICRO_BATCH_SIZE.
    """
    if not _is_microbatch_flag_enabled():
        return 0

    env_value = os.environ.get(ENV_MICRO_BATCH_SIZE, "")
    if not env_value:
        return DEFAULT_MICRO_BATCH_SIZE if DEFAULT_MICRO_BATCH_SIZE > 0 else 0

    try:
        size = int(env_value)
        if size < 1:
            return 0
        return size
    except ValueError:
        return DEFAULT_MICRO_BATCH_SIZE if DEFAULT_MICRO_BATCH_SIZE > 0 else 0


def is_microbatch_enabled() -> bool:
    """
    Check if micro-batch pipeline is effectively enabled.

    A non-positive micro-batch size is treated as disabled.
    """
    return _is_microbatch_flag_enabled() and get_micro_batch_size() > 0


def is_microbatch_gpu_multiplexing_enabled() -> bool:
    """
    Check whether micro-batching should also shrink active GPU KV capacity.

    This is intentionally disabled by default. The default micro-batch behavior
    is overlap-only: split execution for pipeline overlap while keeping a full
    logical KV cache allocation, which preserves FlexGen static cache offload.
    """
    env_value = os.environ.get(ENV_ENABLE_GPU_MULTIPLEXING, "0")
    return env_value == "1"


def get_micro_batch_config() -> dict:
    """
    Get the complete micro-batch configuration as a dictionary.
    
    Returns:
        A dictionary with:
        - 'enabled': bool - whether micro-batching is enabled
        - 'micro_batch_size': int - the configured micro-batch size
        - 'gpu_multiplexing': bool - whether to shrink active GPU KV working set
        - 'mode': str - one of "legacy", "overlap_only", or "multiplexing"
    """
    enabled = is_microbatch_enabled()
    gpu_multiplexing = is_microbatch_gpu_multiplexing_enabled()
    if not enabled:
        mode = "legacy"
    elif gpu_multiplexing:
        mode = "multiplexing"
    else:
        mode = "overlap_only"
    return {
        'enabled': enabled,
        'micro_batch_size': get_micro_batch_size(),
        'gpu_multiplexing': gpu_multiplexing,
        'mode': mode,
    }


def get_current_path() -> str:
    """
    Get the current execution path name.
    
    Returns:
        One of: "legacy", "overlap_only", or "multiplexing".
    """
    return get_micro_batch_config()["mode"]


def get_config_summary() -> str:
    """
    Get a summary string of the current micro-batch pipeline configuration.
    
    Returns:
        A formatted string describing the current configuration.
    """
    enabled = is_microbatch_enabled()
    micro_batch_size = get_micro_batch_size()
    path = get_current_path()
    
    return (
        f"enabled={enabled}, "
        f"micro_batch_size={micro_batch_size}, "
        f"path={path}, "
        f"gpu_multiplexing={path == 'multiplexing'}"
    )


def log_config(logger: logging.Logger, context: str = "") -> None:
    """
    Log the current micro-batch pipeline configuration.
    
    Args:
        logger: The logger to use for output.
        context: Optional context string to include in the log message.
    """
    if not mbpipe_info_logs_enabled():
        return
    enabled = is_microbatch_enabled()
    micro_batch_size = get_micro_batch_size()
    path = get_current_path()
    
    context_str = f" ({context})" if context else ""
    logger.info(
        f"{MBPIPE_LOG_PREFIX} Config{context_str}: "
        f"enabled={enabled}, micro_batch_size={micro_batch_size}, "
        f"path={path}, gpu_multiplexing={path == 'multiplexing'}"
    )


def log_memory_savings_diagnosis(logger: logging.Logger, batch_size: int = 8) -> None:
    """
    Log a diagnosis of whether micro-batching will actually reduce GPU memory.
    
    This function helps debug why micro-batching may not be reducing memory as expected.
    
    Args:
        logger: The logger to use for output.
        batch_size: The client's batch size for analysis.
    """
    if not mbpipe_info_logs_enabled():
        return
    enabled = is_microbatch_enabled()
    micro_batch_size = get_micro_batch_size()
    gpu_multiplexing = is_microbatch_gpu_multiplexing_enabled()
    
    logger.info(f"{MBPIPE_LOG_PREFIX} ===== MEMORY SAVINGS DIAGNOSIS =====")
    logger.info(f"{MBPIPE_LOG_PREFIX} Client batch_size: {batch_size}")
    logger.info(f"{MBPIPE_LOG_PREFIX} Micro-batch enabled: {enabled}")
    logger.info(f"{MBPIPE_LOG_PREFIX} Micro-batch size: {micro_batch_size}")
    
    if not enabled or micro_batch_size <= 0:
        logger.info(f"{MBPIPE_LOG_PREFIX} Result: NO memory savings (micro-batching disabled)")
        return
    
    if micro_batch_size >= batch_size:
        logger.info(f"{MBPIPE_LOG_PREFIX} Result: NO memory savings (micro_batch_size >= batch_size)")
        return

    if not gpu_multiplexing:
        logger.info(f"{MBPIPE_LOG_PREFIX} ")
        logger.info(f"{MBPIPE_LOG_PREFIX} Current behavior (overlap-only micro-batching):")
        logger.info(f"{MBPIPE_LOG_PREFIX}   1. Requests are split into micro-batches for pipeline overlap")
        logger.info(f"{MBPIPE_LOG_PREFIX}   2. KV cache is still allocated for the FULL logical batch")
        logger.info(f"{MBPIPE_LOG_PREFIX}   3. FlexGen static cache offload remains available")
        logger.info(f"{MBPIPE_LOG_PREFIX}   4. GPU KV memory is NOT reduced by micro_batch_size")
        logger.info(f"{MBPIPE_LOG_PREFIX} ")
        logger.info(f"{MBPIPE_LOG_PREFIX} Result: NO GPU memory savings (overlap-only mode)")
        logger.info(f"{MBPIPE_LOG_PREFIX} ===== END DIAGNOSIS =====")
        return
    
    logger.info(f"{MBPIPE_LOG_PREFIX} ")
    logger.info(f"{MBPIPE_LOG_PREFIX} Current behavior (GPU multiplexing):")
    logger.info(f"{MBPIPE_LOG_PREFIX}   1. KV cache is allocated for MICRO batch ({micro_batch_size} items)")
    logger.info(f"{MBPIPE_LOG_PREFIX}   2. Each micro-batch reuses the same GPU slots (offset=0)")
    logger.info(f"{MBPIPE_LOG_PREFIX}   3. offload/prefetch swaps micro-batch KV state via CPU staging")
    logger.info(f"{MBPIPE_LOG_PREFIX}   4. GPU KV memory is controlled by micro_batch_size")
    logger.info(f"{MBPIPE_LOG_PREFIX} ")
    logger.info(f"{MBPIPE_LOG_PREFIX} Expected memory:")
    logger.info(f"{MBPIPE_LOG_PREFIX}   - GPU cache for {micro_batch_size} items (micro-batch)")
    logger.info(f"{MBPIPE_LOG_PREFIX}   - CPU staging for {batch_size} items (all micro-batches)")
    logger.info(f"{MBPIPE_LOG_PREFIX}   - Savings: {(1 - micro_batch_size/batch_size)*100:.1f}% GPU memory reduction")
    logger.info(f"{MBPIPE_LOG_PREFIX} ===== END DIAGNOSIS =====")


def log_path_entry(logger: logging.Logger, component: str, batch_size: int = 0) -> None:
    """
    Log entry into a specific path (legacy or microbatch).
    
    Args:
        logger: The logger to use for output.
        component: Name of the component logging this entry (e.g., "handler", "backend").
        batch_size: Optional batch size being processed.
    """
    if not mbpipe_info_logs_enabled():
        return
    path = get_current_path()
    micro_batch_size = get_micro_batch_size()
    
    batch_info = f", batch_size={batch_size}" if batch_size > 0 else ""
    logger.info(
        f"{MBPIPE_LOG_PREFIX} {component}: entering {path} path, "
        f"micro_batch_size={micro_batch_size}{batch_info}"
    )


def log_microbatch_runtime_info(
    logger: logging.Logger,
    batch_size: int,
    seq_len: int,
    num_blocks: int,
    context: str = ""
) -> None:
    """
    Log comprehensive micro-batch runtime information.
    
    Args:
        logger: The logger to use.
        batch_size: Total batch size from client.
        seq_len: Sequence length.
        num_blocks: Number of transformer blocks.
        context: Optional context string.
    """
    if not mbpipe_info_logs_enabled():
        return
    enabled = is_microbatch_enabled()
    micro_batch_size = get_micro_batch_size()
    gpu_multiplexing = is_microbatch_gpu_multiplexing_enabled()
    
    context_str = f" ({context})" if context else ""
    
    logger.info(f"{MBPIPE_LOG_PREFIX} ===== MICRO-BATCH RUNTIME INFO{context_str} =====")
    logger.info(f"{MBPIPE_LOG_PREFIX} Enabled: {enabled}")
    logger.info(f"{MBPIPE_LOG_PREFIX} Global batch_size: {batch_size}")
    logger.info(f"{MBPIPE_LOG_PREFIX} Micro-batch size: {micro_batch_size}")
    
    if enabled and micro_batch_size > 0 and micro_batch_size < batch_size:
        num_microbatches = (batch_size + micro_batch_size - 1) // micro_batch_size
        logger.info(f"{MBPIPE_LOG_PREFIX} Number of micro-batches: {num_microbatches}")
        if not gpu_multiplexing:
            logger.info(f"{MBPIPE_LOG_PREFIX} GPU memory mode: OVERLAP_ONLY (cache sized for full batch)")
            logger.info(f"{MBPIPE_LOG_PREFIX} FlexGen static cache offload remains active")
            logger.info(f"{MBPIPE_LOG_PREFIX} ===========================================")
            return
        logger.info(f"{MBPIPE_LOG_PREFIX} GPU memory mode: MULTIPLEXING (cache sized for {micro_batch_size})")
        
        # Estimate memory
        # KV cache per block: 2 * seq_len * batch * heads * head_dim * dtype_size
        # Assuming LLaMA-7B: hidden=4096, heads=32, head_dim=128, dtype=fp16 (2 bytes)
        kv_per_block_full = 2 * seq_len * batch_size * 32 * 128 * 2 / (1024 * 1024)  # MB
        kv_per_block_micro = 2 * seq_len * micro_batch_size * 32 * 128 * 2 / (1024 * 1024)  # MB
        
        total_kv_full = kv_per_block_full * num_blocks
        total_kv_micro = kv_per_block_micro * num_blocks
        savings = total_kv_full - total_kv_micro
        savings_pct = (savings / total_kv_full * 100) if total_kv_full > 0 else 0
        
        logger.info(f"{MBPIPE_LOG_PREFIX} Estimated KV cache (full batch): {total_kv_full:.1f} MB")
        logger.info(f"{MBPIPE_LOG_PREFIX} Estimated KV cache (micro-batch): {total_kv_micro:.1f} MB")
        logger.info(f"{MBPIPE_LOG_PREFIX} Estimated savings: {savings:.1f} MB ({savings_pct:.1f}%)")
    else:
        logger.info(f"{MBPIPE_LOG_PREFIX} GPU memory mode: LEGACY (no multiplexing)")
    
    logger.info(f"{MBPIPE_LOG_PREFIX} ===========================================")




# =============================================================================
# Backward-compatible re-exports.
# The tensor ops and the timing/buffer machinery used to live in this file;
# they moved to microbatch_ops / pipeline_timing. Importers may still pull
# them from here.
# =============================================================================
from bloombee.utils.microbatch_ops import (  # noqa: E402,F401
    should_split_batch,
    compute_micro_batch_ranges,
    split_tensor_to_microbatches,
    merge_microbatch_outputs,
    merge_microbatch_keep_indices,
    log_microbatch_split,
    log_microbatch_merge,
)
from bloombee.utils.pipeline_timing import (  # noqa: E402,F401
    StageTimingStats,
    PipelineTimingTracker,
    get_timing_tracker,
    log_stage_timing,
    AsyncOutputBuffer,
)
