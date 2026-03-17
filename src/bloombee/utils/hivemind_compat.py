"""Compatibility imports for Hivemind symbols that moved out of the package root."""

import asyncio

import hivemind.p2p.p2p_daemon as hivemind_p2p_daemon
import hivemind.p2p.p2p_daemon_bindings.control as hivemind_p2p_control
import hivemind.utils.asyncio as hivemind_asyncio
from hivemind.dht import DHT, DHTNode, DHTValue
from hivemind.dht.node import Blacklist
from hivemind.p2p import P2P, P2PContext, PeerID, ServicerBase, StubBase
from hivemind.utils import DHTExpiration, MAX_DHT_TIME_DISCREPANCY_SECONDS, MPFuture, TimedStorage, get_dht_time
from hivemind.utils.asyncio import anext
from hivemind.utils.logging import TextStyle, get_logger, use_hivemind_log_handler
from hivemind.utils.nested import nested_compare, nested_flatten, nested_pack
from hivemind.utils.serializer import MSGPackSerializer
from hivemind.utils.tensor_descr import BatchTensorDescriptor, TensorDescriptor


def _safe_cancel_task_if_running(task: asyncio.Task | None) -> None:
    """Cancel a task without depending on a current loop in the caller thread."""
    if task is None or task.done():
        return

    try:
        loop = task.get_loop()
    except RuntimeError:
        return

    if loop.is_closed():
        return

    if loop.is_running():
        loop.call_soon_threadsafe(task.cancel)


def _patch_hivemind_cleanup() -> None:
    if hivemind_asyncio.cancel_task_if_running is _safe_cancel_task_if_running:
        return

    hivemind_asyncio.cancel_task_if_running = _safe_cancel_task_if_running
    hivemind_p2p_control.cancel_task_if_running = _safe_cancel_task_if_running
    hivemind_p2p_daemon.cancel_task_if_running = _safe_cancel_task_if_running


# _patch_hivemind_cleanup()

__all__ = [
    "BatchTensorDescriptor",
    "Blacklist",
    "DHT",
    "DHTExpiration",
    "DHTNode",
    "DHTValue",
    "MAX_DHT_TIME_DISCREPANCY_SECONDS",
    "MPFuture",
    "MSGPackSerializer",
    "P2P",
    "P2PContext",
    "PeerID",
    "ServicerBase",
    "StubBase",
    "TensorDescriptor",
    "TextStyle",
    "TimedStorage",
    "anext",
    "get_dht_time",
    "get_logger",
    "nested_compare",
    "nested_flatten",
    "nested_pack",
    "use_hivemind_log_handler",
]
