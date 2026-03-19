"""
Central debug-group configuration for BloomBee.

Usage:
    export BLOOMBEE_DEBUG=1
        Enable all debug groups unless a group-specific env explicitly disables one.

    export BLOOMBEE_DEBUG_COMPRESSION=1
    export BLOOMBEE_DEBUG_KV_CACHE=1
    export BLOOMBEE_DEBUG_MICROBATCH=1
    export BLOOMBEE_DEBUG_INFERENCE=1
        Enable only the selected debug group.

Explicit per-feature env vars still win over these group toggles.
For example, BLOOMBEE_VERBOSE_KV_LOGS=0 disables KV verbose logs even if
BLOOMBEE_DEBUG_KV_CACHE=1 is set.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}

_GLOBAL_DEBUG_ENV = "BLOOMBEE_DEBUG"
_GROUP_ENVS = {
    "compression": ("BLOOMBEE_DEBUG_COMPRESSION", "BLOOMBEE_DEBUG_COMP"),
    "kv_cache": ("BLOOMBEE_DEBUG_KV_CACHE", "BLOOMBEE_DEBUG_KV"),
    "microbatch": ("BLOOMBEE_DEBUG_MICROBATCH", "BLOOMBEE_DEBUG_MB"),
    "inference": ("BLOOMBEE_DEBUG_INFERENCE", "BLOOMBEE_DEBUG_INF"),
}
_LOG_CHANNELS = {
    "microbatch_logs": {
        "envs": ("BLOOMBEE_MBPIPE_LOGS",),
        "groups": ("microbatch",),
    },
    "handler_step_timing_logs": {
        "envs": ("BLOOMBEE_HANDLER_STEP_TIMING_LOGS",),
        "groups": ("microbatch",),
    },
    "zipnn_logs": {
        "envs": ("BLOOMBEE_COMP_ZIPNN_LOGS",),
        "groups": ("compression",),
    },
    "client_inference_logs": {
        "envs": ("BLOOMBEE_CLIENT_INFERENCE_LOGS",),
        "groups": ("inference",),
    },
    "cross_gpu_transfer_logs": {
        "envs": ("BLOOMBEE_CROSS_GPU_TRANSFER_LOGS",),
        "groups": ("microbatch",),
    },
    "kv_source_probe_logs": {
        "envs": ("BLOOMBEE_KV_SOURCE_PROBE_LOGS",),
        "groups": ("kv_cache",),
    },
}


def _parse_optional_env_bool(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return None


def is_global_debug_enabled() -> bool:
    return bool(_parse_optional_env_bool(_GLOBAL_DEBUG_ENV))


def is_debug_group_enabled(group: str) -> bool:
    if group not in _GROUP_ENVS:
        raise KeyError(f"Unknown BloomBee debug group: {group}")
    for env_name in _GROUP_ENVS[group]:
        parsed = _parse_optional_env_bool(env_name)
        if parsed is not None:
            return parsed
    return is_global_debug_enabled()


def any_debug_group_enabled(groups: Iterable[str]) -> bool:
    return any(is_debug_group_enabled(group) for group in groups)


def get_env_bool_with_debug_fallback(
    name: str,
    *,
    default: bool = False,
    groups: Iterable[str] = (),
) -> bool:
    parsed = _parse_optional_env_bool(name)
    if parsed is not None:
        return parsed
    if groups and any_debug_group_enabled(groups):
        return True
    return bool(default)


def is_log_channel_enabled(channel: str) -> bool:
    if channel not in _LOG_CHANNELS:
        raise KeyError(f"Unknown BloomBee log channel: {channel}")
    config = _LOG_CHANNELS[channel]
    for env_name in config["envs"]:
        parsed = _parse_optional_env_bool(env_name)
        if parsed is not None:
            return parsed
    return any_debug_group_enabled(config["groups"])
