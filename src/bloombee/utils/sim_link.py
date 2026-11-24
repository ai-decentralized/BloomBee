"""Transfer accounting and an optional simulated network link for LoRA training RPCs.

BLOOMBEE_SIM_LINK="<latency_ms>,<bandwidth_mbit>" (e.g. "50,200") makes every LoRA
message wait latency + bytes / bandwidth before it is sent. It is meant for
benchmarks: GPUs on one machine can then stand in for peers on the internet.
Unset (the default) adds no delay.
"""

import asyncio
import os
import threading
import time

_lock = threading.Lock()
TRANSFER_STATS = {"bytes_sent": 0, "bytes_received": 0, "calls": 0}


def _parse(value):
    if not value:
        return None
    latency_ms, bandwidth_mbit = (float(x) for x in value.split(","))
    if latency_ms < 0 or bandwidth_mbit <= 0:
        raise ValueError("BLOOMBEE_SIM_LINK must be '<latency_ms>,<bandwidth_mbit>' with bandwidth > 0")
    return latency_ms / 1000.0, bandwidth_mbit * 1e6 / 8.0


_LINK = _parse(os.getenv("BLOOMBEE_SIM_LINK", ""))


_busy_until = {"up": 0.0, "down": 0.0}


def link_delay(nbytes, direction="up"):
    """Seconds to wait for `nbytes`: latency plus transmission on a link shared by all
    concurrent messages of this direction (they queue behind each other)."""
    if _LINK is None:
        return 0.0
    latency_s, bytes_per_s = _LINK
    with _lock:
        now = time.monotonic()
        start = max(now, _busy_until[direction])
        _busy_until[direction] = start + nbytes / bytes_per_s
        return (_busy_until[direction] - now) + latency_s


async def account_send(nbytes):
    """Record an outgoing message and apply the simulated link delay, if any."""
    with _lock:
        TRANSFER_STATS["bytes_sent"] += nbytes
        TRANSFER_STATS["calls"] += 1
    delay = link_delay(nbytes, "up")
    if delay:
        await asyncio.sleep(delay)


def account_receive(nbytes):
    with _lock:
        TRANSFER_STATS["bytes_received"] += nbytes


def reset_stats():
    with _lock:
        for key in TRANSFER_STATS:
            TRANSFER_STATS[key] = 0


def snapshot():
    with _lock:
        return dict(TRANSFER_STATS)
