"""Server-to-server flow control and link telemetry.

S2SLinkTelemetry keeps a sliding window of per-link latency/bandwidth/jitter
samples so experiments can tell real throughput changes from network variance.
AdaptivePushConcurrency self-tunes the number of in-flight cross-stage pushes
from runtime feedback. Both moved out of handler.py, which only wires them up.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import Deque, Optional

from bloombee.utils.microbatch_config import MBPIPE_LOG_PREFIX


class S2SLinkTelemetry:
    """
    Rolling transport telemetry for one server-to-server link.

    This is used to distinguish real throughput changes from network variance:
    we log latency, bandwidth and jitter over a sliding window so experiments
    can show whether the network stayed stable while throughput changed.
    """

    label: str
    window_size: int
    samples: int = 0
    total_bytes: int = 0
    clock_sync_samples: int = 0
    last_latency_ms: Optional[float] = None
    latency_ms_window: Deque[float] = field(init=False)
    bandwidth_mbps_window: Deque[float] = field(init=False)
    jitter_ms_window: Deque[float] = field(init=False)
    raw_latency_ms_window: Deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self.latency_ms_window = deque(maxlen=self.window_size)
        self.bandwidth_mbps_window = deque(maxlen=self.window_size)
        self.jitter_ms_window = deque(maxlen=self.window_size)
        self.raw_latency_ms_window = deque(maxlen=self.window_size)

    def record(
        self,
        *,
        latency_ms: float,
        raw_latency_ms: float,
        bandwidth_mbps: float,
        total_bytes: int,
        clock_sync_ok: bool,
    ) -> float:
        jitter_ms = 0.0
        if self.last_latency_ms is not None:
            jitter_ms = abs(float(latency_ms) - float(self.last_latency_ms))
        self.last_latency_ms = float(latency_ms)

        self.samples += 1
        self.total_bytes += max(0, int(total_bytes))
        if clock_sync_ok:
            self.clock_sync_samples += 1
        self.latency_ms_window.append(float(latency_ms))
        self.bandwidth_mbps_window.append(float(bandwidth_mbps))
        self.jitter_ms_window.append(float(jitter_ms))
        self.raw_latency_ms_window.append(float(raw_latency_ms))
        return jitter_ms

class AdaptivePushConcurrency:
    """
    Self-tuning limiter for cross-stage micro-batch pushes.

    The limiter adjusts in-flight push concurrency from runtime signals:
    - acquire wait (queue pressure inside sender)
    - RPC send duration (network pressure)
    - send failures (stability signal)

    No external tuning knobs are required; limits are bounded to keep behavior stable.
    """

    def __init__(
        self,
        *,
        logger_: logging.Logger,
        name: str,
        initial_limit: int = 4,
        min_limit: int = 2,
        max_limit: int = 12,
        ewma_alpha: float = 0.2,
        decision_interval: int = 8,
    ):
        self._logger = logger_
        self._name = name
        self._limit = max(min_limit, min(max_limit, int(initial_limit)))
        self._min_limit = int(min_limit)
        self._max_limit = int(max_limit)
        self._ewma_alpha = float(ewma_alpha)
        self._decision_interval = max(1, int(decision_interval))

        self._in_flight = 0
        self._cond = asyncio.Condition()

        self._ewma_wait_ms = 0.0
        self._ewma_send_ms = 0.0
        self._release_events = 0
        self._recent_failures = 0
        self._consecutive_failures = 0

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def _update_ewma(self, prev: float, sample: float) -> float:
        if prev <= 0.0:
            return sample
        a = self._ewma_alpha
        return prev * (1.0 - a) + sample * a

    async def acquire(self) -> float:
        wait_start = perf_counter()
        async with self._cond:
            while self._in_flight >= self._limit:
                await self._cond.wait()
            self._in_flight += 1
        wait_ms = (perf_counter() - wait_start) * 1000.0
        self._ewma_wait_ms = self._update_ewma(self._ewma_wait_ms, wait_ms)
        return wait_ms

    async def release(self, *, send_time_ms: float, success: bool) -> None:
        change_log = None
        async with self._cond:
            self._in_flight = max(0, self._in_flight - 1)
            self._ewma_send_ms = self._update_ewma(self._ewma_send_ms, max(0.0, float(send_time_ms)))

            if success:
                self._consecutive_failures = 0
                self._recent_failures = max(0, self._recent_failures - 1)
            else:
                self._consecutive_failures += 1
                self._recent_failures = min(16, self._recent_failures + 1)

            self._release_events += 1
            if self._release_events % self._decision_interval == 0:
                old_limit = self._limit
                reason = None

                # Stability first: back off quickly on repeated failures.
                if self._consecutive_failures >= 2 or self._recent_failures >= 3:
                    self._limit = max(self._min_limit, self._limit - 1)
                    self._consecutive_failures = 0
                    reason = "send_failures"
                # If local wait is non-trivial while network send remains moderate,
                # increase concurrency to reduce sender-side queue pressure.
                elif self._ewma_wait_ms > 8.0 and self._ewma_send_ms < 220.0 and self._in_flight >= max(1, self._limit - 1):
                    self._limit = min(self._max_limit, self._limit + 1)
                    reason = "queue_pressure"
                # If network send slows down a lot, decrease concurrency to avoid congestion collapse.
                elif self._ewma_send_ms > 320.0 and self._ewma_wait_ms < 2.0:
                    self._limit = max(self._min_limit, self._limit - 1)
                    reason = "network_backpressure"

                if self._limit != old_limit:
                    change_log = (
                        old_limit,
                        self._limit,
                        reason or "unspecified",
                        self._ewma_wait_ms,
                        self._ewma_send_ms,
                        self._in_flight,
                    )

            self._cond.notify_all()

        if change_log is not None:
            old_limit, new_limit, reason, ewma_wait_ms, ewma_send_ms, in_flight = change_log
            self._logger.info(
                f"{MBPIPE_LOG_PREFIX} [FLOW_CONTROL] adaptive_limit[{self._name}] "
                f"{old_limit}->{new_limit} reason={reason} "
                f"ewma_wait={ewma_wait_ms:.1f}ms ewma_send={ewma_send_ms:.1f}ms in_flight={in_flight}"
            )


