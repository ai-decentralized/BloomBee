"""
BloomBee debug print toggle.

For fine-grained groups, see bloombee.utils.debug_config.
"""

from __future__ import annotations

from typing import Optional

from bloombee.utils.debug_config import is_global_debug_enabled

_DEBUG_OVERRIDE: Optional[bool] = None


def is_debug() -> bool:
    """Return True if debug output is currently enabled."""
    if _DEBUG_OVERRIDE is not None:
        return _DEBUG_OVERRIDE
    return is_global_debug_enabled()


def set_debug(enabled: bool) -> None:
    """Programmatically enable or disable debug output."""
    global _DEBUG_OVERRIDE
    _DEBUG_OVERRIDE = bool(enabled)


def dprint(*args, **kwargs) -> None:
    """Debug print: only outputs when BLOOMBEE_DEBUG is enabled."""
    if is_debug():
        print(*args, **kwargs)
