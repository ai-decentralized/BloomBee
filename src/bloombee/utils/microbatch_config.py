"""
Micro-batch Pipeline Configuration Module

This module provides configuration and utilities for the micro-batch pipeline feature.
The feature is controlled via environment variables:

- BLOOMBEE_ENABLE_MICROBATCH_PIPELINE: Set to "1" to enable (default), "0" to disable
- BLOOMBEE_MICRO_BATCH_SIZE: Set the micro-batch size, default is 4

All logs from this feature use the prefix [MBPIPE] for easy filtering.
"""

import os
import logging
from typing import Tuple, List, Optional, Sequence, Any

import torch

# Prefix for all micro-batch pipeline logs
MBPIPE_LOG_PREFIX = "[MBPIPE]"

# Environment variable names
ENV_ENABLE_MICROBATCH = "BLOOMBEE_ENABLE_MICROBATCH_PIPELINE"
ENV_MICRO_BATCH_SIZE = "BLOOMBEE_MICRO_BATCH_SIZE"

# Default values
# Micro-batch size for pipeline overlap. Each micro-batch writes to its own slice of the KV cache.
DEFAULT_MICRO_BATCH_SIZE = 2  # Default micro-batch size for pipeline overlap


def is_microbatch_enabled() -> bool:
    """
    Check if micro-batch pipeline is enabled via environment variable.
    
    Returns:
        True by default, or if BLOOMBEE_ENABLE_MICROBATCH_PIPELINE is set to "1".
        False only if explicitly set to "0".
    """
    env_value = os.environ.get(ENV_ENABLE_MICROBATCH, "1")  # Default: enabled
    return env_value == "1"


def get_micro_batch_size() -> int:
    """
    Get the configured micro-batch size from environment variable.
    
    Returns:
        The micro-batch size. If not set or invalid, returns DEFAULT_MICRO_BATCH_SIZE.
    """
    if not is_microbatch_enabled():
        return DEFAULT_MICRO_BATCH_SIZE
    
    env_value = os.environ.get(ENV_MICRO_BATCH_SIZE, "")
    if not env_value:
        return DEFAULT_MICRO_BATCH_SIZE
    
    try:
        size = int(env_value)
        if size < 1:
            return DEFAULT_MICRO_BATCH_SIZE
        return size
    except ValueError:
        return DEFAULT_MICRO_BATCH_SIZE


def get_micro_batch_config() -> dict:
    """
    Get the complete micro-batch configuration as a dictionary.
    
    Returns:
        A dictionary with:
        - 'enabled': bool - whether micro-batching is enabled
        - 'micro_batch_size': int - the configured micro-batch size
    """
    return {
        'enabled': is_microbatch_enabled(),
        'micro_batch_size': get_micro_batch_size()
    }


def get_current_path() -> str:
    """
    Get the current execution path name.
    
    Returns:
        "microbatch" if micro-batch pipeline is enabled, "legacy" otherwise.
    """
    return "microbatch" if is_microbatch_enabled() else "legacy"


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
        f"path={path}"
    )


def log_config(logger: logging.Logger, context: str = "") -> None:
    """
    Log the current micro-batch pipeline configuration.
    
    Args:
        logger: The logger to use for output.
        context: Optional context string to include in the log message.
    """
    enabled = is_microbatch_enabled()
    micro_batch_size = get_micro_batch_size()
    path = get_current_path()
    
    context_str = f" ({context})" if context else ""
    logger.info(
        f"{MBPIPE_LOG_PREFIX} Config{context_str}: "
        f"enabled={enabled}, micro_batch_size={micro_batch_size}, path={path}"
    )


def log_memory_savings_diagnosis(logger: logging.Logger, batch_size: int = 8) -> None:
    """
    Log a diagnosis of whether micro-batching will actually reduce GPU memory.
    
    This function helps debug why micro-batching may not be reducing memory as expected.
    
    Args:
        logger: The logger to use for output.
        batch_size: The client's batch size for analysis.
    """
    enabled = is_microbatch_enabled()
    micro_batch_size = get_micro_batch_size()
    
    logger.info(f"{MBPIPE_LOG_PREFIX} ===== MEMORY SAVINGS DIAGNOSIS =====")
    logger.info(f"{MBPIPE_LOG_PREFIX} Client batch_size: {batch_size}")
    logger.info(f"{MBPIPE_LOG_PREFIX} Micro-batch enabled: {enabled}")
    logger.info(f"{MBPIPE_LOG_PREFIX} Micro-batch size: {micro_batch_size}")
    
    if not enabled:
        logger.info(f"{MBPIPE_LOG_PREFIX} Result: NO memory savings (micro-batching disabled)")
        return
    
    if micro_batch_size >= batch_size:
        logger.info(f"{MBPIPE_LOG_PREFIX} Result: NO memory savings (micro_batch_size >= batch_size)")
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
    enabled = is_microbatch_enabled()
    micro_batch_size = get_micro_batch_size()
    
    context_str = f" ({context})" if context else ""
    
    logger.info(f"{MBPIPE_LOG_PREFIX} ===== MICRO-BATCH RUNTIME INFO{context_str} =====")
    logger.info(f"{MBPIPE_LOG_PREFIX} Enabled: {enabled}")
    logger.info(f"{MBPIPE_LOG_PREFIX} Global batch_size: {batch_size}")
    logger.info(f"{MBPIPE_LOG_PREFIX} Micro-batch size: {micro_batch_size}")
    
    if enabled and micro_batch_size < batch_size:
        num_microbatches = (batch_size + micro_batch_size - 1) // micro_batch_size
        logger.info(f"{MBPIPE_LOG_PREFIX} Number of micro-batches: {num_microbatches}")
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
# Micro-batch Splitting and Merging Utilities
# =============================================================================

def should_split_batch(batch_size: int) -> bool:
    """
    Determine if the batch should be split into micro-batches.
    
    Args:
        batch_size: The size of the incoming batch.
        
    Returns:
        True if batch should be split, False otherwise.
    """
    if not is_microbatch_enabled():
        return False
    
    micro_batch_size = get_micro_batch_size()
    # If micro_batch_size <= 0, don't split (disabled)
    if micro_batch_size <= 0:
        return False
    # Only split if batch is larger than micro-batch size
    return batch_size > micro_batch_size



def compute_micro_batch_ranges(batch_size: int) -> List[Tuple[int, int]]:
    """
    Compute the (start, end) ranges for each micro-batch.
    
    Args:
        batch_size: The total batch size.
        
    Returns:
        A list of (start, end) tuples for each micro-batch.
    """
    micro_batch_size = get_micro_batch_size()
    
    ranges = []
    start = 0
    while start < batch_size:
        end = min(start + micro_batch_size, batch_size)
        ranges.append((start, end))
        start = end
    
    return ranges


def split_tensor_to_microbatches(
    tensor: torch.Tensor,
    dim: int = 0
) -> List[torch.Tensor]:
    """
    Split a tensor along the batch dimension into micro-batches.
    
    Args:
        tensor: The input tensor to split. Expected shape: [batch_size, ...].
        dim: The dimension to split along (default: 0, batch dimension).
        
    Returns:
        A list of tensor chunks, one per micro-batch.
    """
    if tensor is None:
        return [None]
    
    batch_size = tensor.shape[dim]
    
    if not should_split_batch(batch_size):
        # No splitting needed, return as single-element list
        return [tensor]
    
    ranges = compute_micro_batch_ranges(batch_size)
    
    chunks = []
    for start, end in ranges:
        if dim == 0:
            chunk = tensor[start:end]
        elif dim == 1:
            chunk = tensor[:, start:end]
        else:
            # Generic case using narrow
            chunk = tensor.narrow(dim, start, end - start)
        chunks.append(chunk)
    
    return chunks


def merge_microbatch_outputs(
    outputs: List[torch.Tensor],
    dim: int = 0
) -> torch.Tensor:
    """
    Merge micro-batch outputs back into a single tensor.
    
    Args:
        outputs: A list of output tensors from each micro-batch.
        dim: The dimension to concatenate along (default: 0, batch dimension).
        
    Returns:
        A single merged tensor.
    """
    if len(outputs) == 1:
        return outputs[0]
    
    # Filter out None values
    valid_outputs = [o for o in outputs if o is not None]
    if not valid_outputs:
        return None
    
    return torch.cat(valid_outputs, dim=dim)


def log_microbatch_split(
    logger: logging.Logger,
    batch_size: int,
    num_microbatches: int,
    component: str = ""
) -> None:
    """
    Log micro-batch splitting information.
    
    Args:
        logger: The logger to use.
        batch_size: Original batch size.
        num_microbatches: Number of micro-batches created.
        component: Optional component name for context.
    """
    micro_batch_size = get_micro_batch_size()
    context = f" ({component})" if component else ""
    logger.info(
        f"{MBPIPE_LOG_PREFIX} Split{context}: "
        f"batch_size={batch_size} -> {num_microbatches} micro-batches "
        f"(micro_batch_size={micro_batch_size})"
    )


def log_microbatch_merge(
    logger: logging.Logger,
    num_microbatches: int,
    merged_batch_size: int,
    component: str = ""
) -> None:
    """
    Log micro-batch merging information.
    
    Args:
        logger: The logger to use.
        num_microbatches: Number of micro-batches merged.
        merged_batch_size: Final merged batch size.
        component: Optional component name for context.
    """
    context = f" ({component})" if component else ""
    logger.info(
        f"{MBPIPE_LOG_PREFIX} Merge{context}: "
        f"{num_microbatches} micro-batches -> batch_size={merged_batch_size}"
    )


# =============================================================================
# Stage Timing Infrastructure for Cross-Stage Pipeline Overlap
# =============================================================================

from dataclasses import dataclass, field
from time import perf_counter
from collections import deque
from typing import Deque


@dataclass
class StageTimingStats:
    """Statistics for a single stage's timing measurements."""
    stage_id: str
    compute_times: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    comm_times: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def avg_compute_time(self) -> float:
        """Average compute time in ms."""
        if not self.compute_times:
            return 0.0
        return sum(self.compute_times) / len(self.compute_times)
    
    @property
    def avg_comm_time(self) -> float:
        """Average communication time in ms."""
        if not self.comm_times:
            return 0.0
        return sum(self.comm_times) / len(self.comm_times)
    
    def record_compute(self, time_ms: float):
        """Record a compute time measurement."""
        self.compute_times.append(time_ms)
    
    def record_comm(self, time_ms: float):
        """Record a communication time measurement."""
        self.comm_times.append(time_ms)


class PipelineTimingTracker:
    """
    Tracks timing statistics across pipeline stages for dynamic buffer decisions.
    
    This is a singleton that collects timing data from all stages to enable
    intelligent decisions about buffer placement and overlap strategies.
    
    Note: Uses simple dict operations without locks for asyncio compatibility.
    Some race conditions are acceptable for timing statistics.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._stages: dict = {}  # stage_id -> StageTimingStats
        self._global_comm_times: Deque[float] = deque(maxlen=100)
    
    def get_or_create_stage(self, stage_id: str) -> StageTimingStats:
        """Get or create timing stats for a stage."""
        if stage_id not in self._stages:
            self._stages[stage_id] = StageTimingStats(stage_id=stage_id)
        return self._stages[stage_id]
    
    def record_stage_compute(self, stage_id: str, time_ms: float):
        """Record compute time for a stage."""
        stage = self.get_or_create_stage(stage_id)
        stage.record_compute(time_ms)
    
    def record_stage_comm(self, stage_id: str, time_ms: float):
        """Record communication time for a stage."""
        stage = self.get_or_create_stage(stage_id)
        stage.record_comm(time_ms)
    
    def record_cross_stage_comm(self, time_ms: float):
        """Record cross-stage communication time."""
        self._global_comm_times.append(time_ms)
    
    @property
    def avg_cross_stage_comm(self) -> float:
        """Average cross-stage communication time in ms."""
        if not self._global_comm_times:
            return 0.0
        try:
            return sum(self._global_comm_times) / len(self._global_comm_times)
        except Exception:
            return 0.0
    
    def should_use_buffer(self) -> tuple[bool, str]:
        """
        Decide whether to use buffer based on timing statistics.
        
        Returns:
            (use_buffer, buffer_position): 
            - use_buffer: True if buffer should be used
            - buffer_position: "producer", "consumer", or "none"
        """
        try:
            if len(self._stages) < 1:
                return False, "none"
            
            # Get average times
            avg_comm = self.avg_cross_stage_comm
            if avg_comm == 0:
                return False, "none"
            
            # Get stage compute times
            stage_times = [(sid, s.avg_compute_time) for sid, s in list(self._stages.items()) if s.avg_compute_time > 0]
            if len(stage_times) < 2:
                # Not enough data, default to producer buffer if comm time is significant
                if avg_comm > 50:  # 50ms threshold
                    return True, "producer"
                return False, "none"
            
            # Sort by stage order (assuming stage_id is sortable)
            stage_times.sort(key=lambda x: x[0])
            t1 = stage_times[0][1]  # First stage
            t2 = stage_times[-1][1]  # Last stage
            
            # Compute ratio
            if t2 == 0:
                return False, "none"
            
            ratio = t1 / t2
            comm_ratio = avg_comm / max(t1, t2)
            
            # Decision logic
            if comm_ratio < 0.3:
                # Communication is fast enough, no buffer needed
                return False, "none"
            
            if ratio > 1.3:
                # Upstream slower, use producer buffer
                return True, "producer"
            elif ratio < 0.7:
                # Downstream slower, use consumer buffer
                return True, "consumer"
            else:
                # Balanced, use producer buffer by default
                return True, "producer"
        except Exception:
            return False, "none"
    
    def get_summary(self) -> dict:
        """Get a summary of timing statistics."""
        try:
            return {
                "stages": {
                    sid: {
                        "avg_compute_ms": s.avg_compute_time,
                        "avg_comm_ms": s.avg_comm_time,
                        "samples": len(s.compute_times)
                    }
                    for sid, s in list(self._stages.items())
                },
                "avg_cross_stage_comm_ms": self.avg_cross_stage_comm,
                "buffer_decision": self.should_use_buffer()
            }
        except Exception:
            return {"stages": {}, "avg_cross_stage_comm_ms": 0, "buffer_decision": (False, "none")}


# Global timing tracker instance
_timing_tracker: Optional[PipelineTimingTracker] = None


def get_timing_tracker() -> PipelineTimingTracker:
    """Get the global pipeline timing tracker."""
    global _timing_tracker
    if _timing_tracker is None:
        _timing_tracker = PipelineTimingTracker()
    return _timing_tracker


def log_stage_timing(
    logger: logging.Logger,
    stage_id: str,
    compute_time_ms: float,
    comm_time_ms: float = 0.0,
    component: str = ""
) -> None:
    """
    Log and record stage timing information.
    
    Args:
        logger: The logger to use.
        stage_id: Identifier for the stage (e.g., "0:16", "16:32").
        compute_time_ms: Compute time in milliseconds.
        comm_time_ms: Communication time in milliseconds (optional).
        component: Optional component name for context.
    """
    tracker = get_timing_tracker()
    tracker.record_stage_compute(stage_id, compute_time_ms)
    if comm_time_ms > 0:
        tracker.record_stage_comm(stage_id, comm_time_ms)
        tracker.record_cross_stage_comm(comm_time_ms)
    
    context = f" ({component})" if component else ""
    use_buffer, buffer_pos = tracker.should_use_buffer()
    
    logger.info(
        f"{MBPIPE_LOG_PREFIX} StageTiming{context}: stage={stage_id}, "
        f"compute={compute_time_ms:.1f}ms, comm={comm_time_ms:.1f}ms, "
        f"buffer_decision=({use_buffer}, {buffer_pos})"
    )


def log_timing_summary(logger: logging.Logger) -> None:
    """Log a summary of all timing statistics."""
    tracker = get_timing_tracker()
    summary = tracker.get_summary()
    
    logger.info(f"{MBPIPE_LOG_PREFIX} TimingSummary: {summary}")


# =============================================================================
# Step 2: Async Output Buffer for Compute/Communication Overlap
# =============================================================================

import asyncio
from typing import Callable, Any
from time import perf_counter as _perf_counter


class AsyncOutputBuffer:
    """
    Asynchronous output buffer for overlapping computation with communication.
    
    This class implements a producer-side buffer that allows:
    1. Non-blocking put() - computation can continue immediately after buffering
    2. Background async sending - communication happens in parallel with next computation
    3. Graceful shutdown - ensures all pending sends complete before closing
    
    Usage:
        buffer = AsyncOutputBuffer(max_pending=2, logger=logger)
        await buffer.start_sender(push_fn)  # Start background sender
        
        # In computation loop:
        await buffer.put(output_tensor, metadata)  # Non-blocking
        
        # When done:
        await buffer.flush()  # Wait for all pending sends
        await buffer.stop()   # Stop the sender
    """
    
    def __init__(
        self,
        max_pending: int = 2,
        logger: Optional[logging.Logger] = None,
        name: str = "default"
    ):
        """
        Initialize the async output buffer.
        
        Args:
            max_pending: Maximum number of outputs that can be buffered.
                        If queue is full, put() will wait (backpressure).
            logger: Optional logger for debugging.
            name: Name identifier for this buffer (for logging).
        """
        self.max_pending = max_pending
        self.logger = logger
        self.name = name
        
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_pending)
        self._send_task: Optional[asyncio.Task] = None
        self._push_fn: Optional[Callable] = None
        self._running = False
        self._stats = {
            "total_puts": 0,
            "total_sends": 0,
            "total_send_time_ms": 0.0,
            "max_queue_depth": 0,
        }
    
    async def start_sender(self, push_fn: Callable) -> None:
        """
        Start the background sender coroutine.
        
        Args:
            push_fn: Async function to call for sending data.
                    Signature: push_fn(tensor, metadata) -> None
        """
        if self._running:
            return
        
        self._push_fn = push_fn
        self._running = True
        self._send_task = asyncio.create_task(self._sender_loop())
        
        if self.logger:
            self.logger.debug(
                f"{MBPIPE_LOG_PREFIX} AsyncOutputBuffer[{self.name}] started "
                f"(max_pending={self.max_pending})"
            )
    
    async def _sender_loop(self) -> None:
        """Background loop that continuously sends buffered outputs."""
        while self._running or not self._queue.empty():
            try:
                # Wait for an item with a timeout to allow checking _running flag
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                
                send_start = _perf_counter()
                
                try:
                    # Item is passed directly to push_fn - let push_fn handle unpacking
                    await self._push_fn(item)
                    send_time = (_perf_counter() - send_start) * 1000
                    
                    self._stats["total_sends"] += 1
                    self._stats["total_send_time_ms"] += send_time
                    
                    if self.logger:
                        self.logger.debug(
                            f"{MBPIPE_LOG_PREFIX} AsyncOutputBuffer[{self.name}] "
                            f"sent item (send_time={send_time:.1f}ms, "
                            f"queue_size={self._queue.qsize()})"
                        )
                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            f"{MBPIPE_LOG_PREFIX} AsyncOutputBuffer[{self.name}] "
                            f"send failed: {e}"
                        )
                finally:
                    self._queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"{MBPIPE_LOG_PREFIX} AsyncOutputBuffer[{self.name}] "
                        f"sender error: {e}"
                    )
    
    async def put(self, item: Any, clone: bool = False) -> float:
        """
        Put an item into the buffer for async sending.
        
        Args:
            item: The item to send (will be passed directly to push_fn).
                  Can be a tensor, tuple, or any object that push_fn expects.
            clone: If True and item has a 'clone' method, clone it to avoid modification.
        
        Returns:
            Time spent waiting for queue space (ms). 0 if no wait was needed.
        """
        if not self._running:
            raise RuntimeError("Buffer not started. Call start_sender() first.")
        
        wait_start = _perf_counter()
        
        # Clone item if needed to avoid data races
        if clone and hasattr(item, 'clone'):
            item = item.clone()
        
        # Put into queue (may block if queue is full - backpressure)
        await self._queue.put(item)
        
        wait_time = (_perf_counter() - wait_start) * 1000
        
        self._stats["total_puts"] += 1
        current_depth = self._queue.qsize()
        if current_depth > self._stats["max_queue_depth"]:
            self._stats["max_queue_depth"] = current_depth
        
        if self.logger and wait_time > 1.0:  # Log if wait time > 1ms
            self.logger.debug(
                f"{MBPIPE_LOG_PREFIX} AsyncOutputBuffer[{self.name}] "
                f"put (wait_time={wait_time:.1f}ms, queue_size={current_depth})"
            )
        
        return wait_time
    
    async def flush(self) -> None:
        """Wait for all buffered items to be sent."""
        await self._queue.join()
        
        if self.logger:
            self.logger.debug(
                f"{MBPIPE_LOG_PREFIX} AsyncOutputBuffer[{self.name}] flushed"
            )
    
    async def stop(self) -> None:
        """Stop the background sender and wait for completion."""
        self._running = False
        
        if self._send_task:
            # Give sender time to finish remaining items
            await self.flush()
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass
            self._send_task = None
        
        if self.logger:
            avg_send_time = (
                self._stats["total_send_time_ms"] / max(1, self._stats["total_sends"])
            )
            self.logger.info(
                f"{MBPIPE_LOG_PREFIX} AsyncOutputBuffer[{self.name}] stopped: "
                f"puts={self._stats['total_puts']}, "
                f"sends={self._stats['total_sends']}, "
                f"avg_send={avg_send_time:.1f}ms, "
                f"max_depth={self._stats['max_queue_depth']}"
            )
    
    @property
    def stats(self) -> dict:
        """Get buffer statistics."""
        return self._stats.copy()
    
    @property
    def queue_size(self) -> int:
        """Current number of items in the queue."""
        return self._queue.qsize()
    
    @property
    def is_running(self) -> bool:
        """Whether the buffer is running."""
        return self._running


class BufferedPipelineManager:
    """
    Manager for multiple async output buffers in a pipeline.
    
    Manages buffer creation and lifecycle for multi-stage pipelines,
    with support for dynamic buffer decisions based on timing data.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger
        self._buffers: dict[str, AsyncOutputBuffer] = {}
        self._enabled = True
    
    def get_or_create_buffer(
        self,
        stage_id: str,
        max_pending: int = 2
    ) -> AsyncOutputBuffer:
        """Get or create a buffer for a specific stage."""
        if stage_id not in self._buffers:
            self._buffers[stage_id] = AsyncOutputBuffer(
                max_pending=max_pending,
                logger=self.logger,
                name=f"stage_{stage_id}"
            )
        return self._buffers[stage_id]
    
    async def start_all(self, push_fn_factory: Callable[[str], Callable]) -> None:
        """
        Start all buffers with their respective push functions.
        
        Args:
            push_fn_factory: Function that takes stage_id and returns push_fn.
        """
        for stage_id, buffer in self._buffers.items():
            push_fn = push_fn_factory(stage_id)
            await buffer.start_sender(push_fn)
    
    async def stop_all(self) -> None:
        """Stop all buffers."""
        for buffer in self._buffers.values():
            await buffer.stop()
        self._buffers.clear()
    
    def should_use_buffer_for_stage(self, stage_id: str) -> bool:
        """
        Determine if buffer should be used for a specific stage.
        Uses the global timing tracker for decision.
        """
        if not self._enabled:
            return False
        
        tracker = get_timing_tracker()
        use_buffer, _ = tracker.should_use_buffer()
        return use_buffer
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value


# Global pipeline buffer manager
_buffer_manager: Optional[BufferedPipelineManager] = None


def get_buffer_manager(logger: Optional[logging.Logger] = None) -> BufferedPipelineManager:
    """Get the global buffered pipeline manager."""
    global _buffer_manager
    if _buffer_manager is None:
        _buffer_manager = BufferedPipelineManager(logger=logger)
    return _buffer_manager
