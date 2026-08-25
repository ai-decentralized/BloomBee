from __future__ import annotations

from typing import Any

from hivemind.p2p.p2p_daemon_bindings.control import DEFAULT_MAX_MSG_SIZE


def get_default_p2p_max_msg_size() -> int:
    """Default libp2p persistent-connection message limit for BloomBee payloads."""
    import os

    value = os.getenv("BLOOMBEE_P2P_MAX_MSG_SIZE")
    return int(value) if isinstance(value, str) else 64 * 1024 * 1024


def apply_p2p_max_msg_size(p2p: Any, max_msg_size: int | None) -> None:
    """Apply BloomBee's P2P message limit to replicated hivemind control clients.

    hivemind's ``P2P.replicate`` connects to an existing daemon but creates a new
    Python control client with the library default 4 MiB limit. Large speculative
    batches can exceed that limit before the daemon sees the request, so each
    replicated wrapper must carry the same limit as the daemon.
    """
    if max_msg_size is None:
        return

    control = getattr(getattr(p2p, "_client", None), "control", None)
    if control is None:
        return

    control.persistent_conn_max_msg_size = max(int(max_msg_size), DEFAULT_MAX_MSG_SIZE)
