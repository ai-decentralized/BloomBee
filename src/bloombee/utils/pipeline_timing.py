"""Pipeline timing tracker and async output buffer for compute/communication
overlap. Split out of microbatch_config (which keeps only switches and logging
helpers)."""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
from time import perf_counter as _perf_counter
from typing import Any, Callable, Deque, Optional

from bloombee.utils.microbatch_config import MBPIPE_LOG_PREFIX, mbpipe_info_logs_enabled

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
    if not mbpipe_info_logs_enabled():
        return
    context = f" ({component})" if component else ""
    use_buffer, buffer_pos = tracker.should_use_buffer()
    
    logger.info(
        f"{MBPIPE_LOG_PREFIX} StageTiming{context}: stage={stage_id}, "
        f"compute={compute_time_ms:.1f}ms, comm={comm_time_ms:.1f}ms, "
        f"buffer_decision=({use_buffer}, {buffer_pos})"
    )


# =============================================================================
# Step 2: Async Output Buffer for Compute/Communication Overlap
# =============================================================================

import asyncio
from typing import Callable
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
        self._first_error: Optional[Exception] = None
        self._stats = {
            "total_puts": 0,
            "total_sends": 0,
            "total_send_failures": 0,
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
        self._first_error = None
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
                    self._stats["total_send_failures"] += 1
                    if self._first_error is None:
                        self._first_error = e
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
        if self._first_error is not None:
            raise RuntimeError(
                f"AsyncOutputBuffer[{self.name}] has a prior send failure"
            ) from self._first_error
        
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
        self._raise_if_failed()
        
        if self.logger:
            self.logger.debug(
                f"{MBPIPE_LOG_PREFIX} AsyncOutputBuffer[{self.name}] flushed"
            )
    
    def _raise_if_failed(self) -> None:
        if self._first_error is not None:
            raise RuntimeError(
                f"AsyncOutputBuffer[{self.name}] send failed"
            ) from self._first_error

    async def stop(self, raise_on_error: bool = True) -> None:
        """Stop the background sender and wait for completion."""
        self._running = False
        
        if self._send_task:
            # Give sender time to finish remaining items.
            await self._queue.join()
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
                f"failures={self._stats['total_send_failures']}, "
                f"avg_send={avg_send_time:.1f}ms, "
                f"max_depth={self._stats['max_queue_depth']}"
            )
        if raise_on_error:
            self._raise_if_failed()
    
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
