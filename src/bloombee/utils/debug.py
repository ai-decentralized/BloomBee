"""
BloomBee Debug Toggle Module

Controls all debug/diagnostic output across the project.
Debug output is OFF by default for optimal inference performance.

Toggle ON via environment variable:
    export BLOOMBEE_DEBUG=1

Toggle OFF (default):
    export BLOOMBEE_DEBUG=0
    # or simply don't set BLOOMBEE_DEBUG

Can also be toggled programmatically:
    from bloombee.utils.debug import set_debug, is_debug
    set_debug(True)   # Enable debug output
    set_debug(False)  # Disable debug output
"""

import os

_DEBUG = os.environ.get("BLOOMBEE_DEBUG", "0").lower() in ("1", "true", "yes", "on")


def is_debug() -> bool:
    """Return True if debug output is currently enabled."""
    return _DEBUG


def set_debug(enabled: bool) -> None:
    """Programmatically enable or disable debug output."""
    global _DEBUG
    _DEBUG = bool(enabled)


def dprint(*args, **kwargs) -> None:
    """Debug print: only outputs when BLOOMBEE_DEBUG is enabled."""
    if _DEBUG:
        print(*args, **kwargs)
