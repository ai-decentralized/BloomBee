from __future__ import annotations

import contextvars
import math
import os
import struct
import sys
import threading
import time
import zlib
from contextlib import contextmanager
from functools import lru_cache
from typing import AsyncIterator, Dict, Iterable, List, Optional

import torch
from hivemind.compression.serialization import (
    deserialize_torch_tensor as _deserialize_torch_tensor,
    serialize_torch_tensor as _serialize_torch_tensor,
)
from hivemind.proto import runtime_pb2
from hivemind.utils.logging import get_logger
from hivemind.utils.streaming import combine_from_streaming

from bloombee.utils.debug_config import get_env_bool_with_debug_fallback, is_log_channel_enabled

try:
    from bloombee.utils import lossless_wrapper_config as _lossless_cfg
except Exception:
    _lossless_cfg = None

logger = get_logger(__name__)

try:
    import zstandard as _zstd
except Exception:
    _zstd = None

try:
    from zipnn import ZipNN as _ZipNN
except Exception:
    _ZipNN = None


_MAGIC = b"BBLC"
_VERSION = 1
_ALGO_ZSTD = 1
_ALGO_ZLIB = 2
_ALGO_ZSTD_BYTE_SPLIT = 3
_ALGO_ZIPNN = 4
_HEADER_STRUCT = struct.Struct("!4sBBQ")
_HEADER_SIZE = _HEADER_STRUCT.size
_BYTE_SPLIT_PAYLOAD_STRUCT = struct.Struct("!BI")
_BYTE_SPLIT_PAYLOAD_SIZE = _BYTE_SPLIT_PAYLOAD_STRUCT.size

_MISSING_ZSTD_WARNING_EMITTED = False
_MISSING_ZIPNN_WARNING_EMITTED = False
_COMP_PROFILE_ENV = "BLOOMBEE_COMP_RATIO_PROFILE"
_COMP_TIMING_PROFILE_ENV = "BLOOMBEE_COMP_TIMING_PROFILE"
_COMP_DETAIL_PROFILE_ENV = "BLOOMBEE_COMP_DETAIL_PROFILE"
_COMP_RESEARCH_PROFILE_ENV = "BLOOMBEE_COMP_RESEARCH_PROFILE"
_COMP_BIT_PROFILE_ENV = "BLOOMBEE_COMP_BIT_PROFILE"
_COMP_STRIDE_PROFILE_ENV = "BLOOMBEE_COMP_STRIDE_PROFILE"
_ACT_DIST_PROFILE_ENV = "BLOOMBEE_ACT_DIST_PROFILE"
_COMP_ZIPNN_PROFILE_ENV = "BLOOMBEE_COMP_ZIPNN_PROFILE"
_DEBUG_TENSOR_NAMES_ENV = "BLOOMBEE_DEBUG_TENSOR_NAMES"
_WIRE_TRUNCATE_FP16_ENV = "BLOOMBEE_WIRE_TRUNCATE_FP16"
_WIRE_TRUNCATE_TARGETS_ENV = "BLOOMBEE_WIRE_TRUNCATE_TARGETS"
_WIRE_TRUNCATE_PHASES_ENV = "BLOOMBEE_WIRE_TRUNCATE_PHASES"
_LOSSLESS_LAYOUT_ENV = "BLOOMBEE_LOSSLESS_LAYOUT"
_LOSSLESS_LAYOUT_TARGETS_ENV = "BLOOMBEE_LOSSLESS_LAYOUT_TARGETS"
_TRANSPORT_PROFILE_CTX: contextvars.ContextVar[Optional[Dict[str, float]]] = contextvars.ContextVar(
    "bloombee_transport_profile", default=None
)
_COMP_RESEARCH_LOCK = threading.Lock()
_COMP_RESEARCH_TOTAL_SAMPLES = 0
_COMP_RESEARCH_BUCKETS: Dict[str, Dict[str, float]] = {}
_COMP_RESEARCH_BUCKET_ORDER = (
    "<4KB",
    "4KB-49KB",
    "49KB-80KB",
    "80KB-200KB",
    "200KB-1MB",
    ">=1MB",
)
_SER_LAYOUT_LOGGED_DTYPES: set[str] = set()
_COMP_LOG_CFG_EMITTED = False


def _get_env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _get_cfg(name: str, default):
    if _lossless_cfg is None:
        return default
    return getattr(_lossless_cfg, name, default)


def _allow_env_override() -> bool:
    cfg_val = _get_cfg("ALLOW_ENV_OVERRIDE", 1)
    try:
        return bool(int(cfg_val))
    except Exception:
        return bool(cfg_val)


def _warn_missing_zstd_once() -> None:
    global _MISSING_ZSTD_WARNING_EMITTED
    if not _MISSING_ZSTD_WARNING_EMITTED:
        logger.warning(
            "Lossless wrapper requested zstd, but 'zstandard' is unavailable; sending tensors without wrapper compression"
        )
        _MISSING_ZSTD_WARNING_EMITTED = True


def _warn_missing_zipnn_once() -> None:
    global _MISSING_ZIPNN_WARNING_EMITTED
    if not _MISSING_ZIPNN_WARNING_EMITTED:
        logger.warning("ZipNN compare requested, but 'zipnn' is unavailable; skipping ZipNN diagnostics")
        _MISSING_ZIPNN_WARNING_EMITTED = True


def _lossless_send_enabled() -> bool:
    cfg_val = _get_cfg("ENABLE_LOSSLESS_WRAPPER", 0)
    if _allow_env_override() and "BLOOMBEE_LOSSLESS_WRAPPER" in os.environ:
        return _get_env_bool("BLOOMBEE_LOSSLESS_WRAPPER", "0")
    try:
        return bool(int(cfg_val))
    except Exception:
        return bool(cfg_val)


def _lossless_algo() -> str:
    cfg_val = str(_get_cfg("LOSSLESS_ALGO", "zstd"))
    if _allow_env_override():
        cfg_val = os.environ.get("BLOOMBEE_LOSSLESS_ALGO", cfg_val)
    return str(cfg_val).strip().lower()


def _lossless_level() -> int:
    cfg_val = _get_cfg("LOSSLESS_LEVEL", 3)
    if _allow_env_override() and "BLOOMBEE_LOSSLESS_LEVEL" in os.environ:
        return _get_env_int("BLOOMBEE_LOSSLESS_LEVEL", 3)
    try:
        return int(cfg_val)
    except Exception:
        return 3


def _lossless_layout() -> str:
    cfg_val = str(_get_cfg("LOSSLESS_LAYOUT", "byte_split"))
    if _allow_env_override():
        cfg_val = os.environ.get(_LOSSLESS_LAYOUT_ENV, cfg_val)
    return str(cfg_val).strip().lower()


def _lossless_min_bytes() -> int:
    cfg_val = _get_cfg("LOSSLESS_MIN_BYTES", 4096)
    if _allow_env_override() and "BLOOMBEE_LOSSLESS_MIN_BYTES" in os.environ:
        return max(0, _get_env_int("BLOOMBEE_LOSSLESS_MIN_BYTES", 4096))
    try:
        return max(0, int(cfg_val))
    except Exception:
        return 4096


def _lossless_min_gain_bytes() -> int:
    cfg_val = _get_cfg("LOSSLESS_MIN_GAIN_BYTES", 32)
    if _allow_env_override() and "BLOOMBEE_LOSSLESS_MIN_GAIN_BYTES" in os.environ:
        return max(0, _get_env_int("BLOOMBEE_LOSSLESS_MIN_GAIN_BYTES", 32))
    try:
        return max(0, int(cfg_val))
    except Exception:
        return 32


def comp_ratio_profile_enabled() -> bool:
    """
    Enable per-tensor compression ratio profiling logs.
    BLOOMBEE_DEBUG=1 or BLOOMBEE_DEBUG_COMPRESSION=1 enables it by default.
    """
    return get_env_bool_with_debug_fallback(
        _COMP_PROFILE_ENV,
        default=False,
        groups=("compression",),
    )


def comp_timing_profile_enabled() -> bool:
    """
    Enable detailed compression/decompression timing profiling.
    """
    return get_env_bool_with_debug_fallback(
        _COMP_TIMING_PROFILE_ENV,
        default=False,
        groups=("compression",),
    )


def comp_detail_profile_enabled() -> bool:
    """
    Enable heavyweight per-tensor compression diagnostics.
    BLOOMBEE_DEBUG=1 or BLOOMBEE_DEBUG_COMPRESSION=1 enables it by default.
    """
    return get_env_bool_with_debug_fallback(
        _COMP_DETAIL_PROFILE_ENV,
        default=False,
        groups=("compression",),
    )


def comp_research_profile_enabled() -> bool:
    """
    Enable lightweight rolling compression scaling summaries.
    This is cheaper than COMP_DETAIL and intended for online cost/benefit analysis.
    """
    return get_env_bool_with_debug_fallback(
        _COMP_RESEARCH_PROFILE_ENV,
        default=False,
        groups=("compression",),
    )


def comp_bit_profile_enabled() -> bool:
    """
    Enable bit-level floating-point profiling (sign/exponent/mantissa) on sampled tensors.
    """
    return get_env_bool_with_debug_fallback(
        _COMP_BIT_PROFILE_ENV,
        default=False,
        groups=("compression",),
    )


def comp_stride_profile_enabled() -> bool:
    """
    Enable strided/chunk repeat diagnostics to inspect alignment-sensitive byte patterns.
    """
    return get_env_bool_with_debug_fallback(
        _COMP_STRIDE_PROFILE_ENV,
        default=False,
        groups=("compression",),
    )


def act_dist_profile_enabled() -> bool:
    """
    Enable activation magnitude decade histogram logs (10x bins).
    """
    return get_env_bool_with_debug_fallback(
        _ACT_DIST_PROFILE_ENV,
        default=False,
        groups=("compression",),
    )


def comp_zipnn_profile_enabled() -> bool:
    """
    Enable side-by-side ZipNN diagnostics on selected tensors without changing the actual wire format.
    """
    cfg_val = _get_cfg("COMP_ZIPNN_PROFILE", 0)
    try:
        default = "1" if bool(int(cfg_val)) else "0"
    except Exception:
        default = "1" if bool(cfg_val) else "0"
    return get_env_bool_with_debug_fallback(
        _COMP_ZIPNN_PROFILE_ENV,
        default=(default == "1"),
        groups=("compression",),
    )


@lru_cache(maxsize=1)
def _debug_tensor_names() -> set[str]:
    raw = os.environ.get(_DEBUG_TENSOR_NAMES_ENV, "hidden_states")
    names = {item.strip().lower() for item in raw.split(",") if item.strip()}
    return names or {"hidden_states"}


def _is_debug_target_tensor(
    tensor: Optional[torch.Tensor],
    debug_context: Optional[Dict[str, object]],
) -> bool:
    if tensor is None or not torch.is_tensor(tensor):
        return False
    if not torch.is_floating_point(tensor):
        return False
    if debug_context is None:
        return False
    tensor_name = str(debug_context.get("tensor_name", "")).strip().lower()
    names = _debug_tensor_names()
    return "*" in names or tensor_name in names


def wire_truncate_fp16_enabled() -> bool:
    """
    If enabled, selected FP32 tensors are truncated to FP16 before serialization.
    """
    return _get_env_bool(_WIRE_TRUNCATE_FP16_ENV, "0")


@lru_cache(maxsize=1)
def _wire_truncate_targets() -> List[str]:
    raw = os.environ.get(_WIRE_TRUNCATE_TARGETS_ENV, "client:rpc_inference:hidden_states")
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


@lru_cache(maxsize=1)
def _wire_truncate_phases() -> set[str]:
    raw = os.environ.get(_WIRE_TRUNCATE_PHASES_ENV, "prefill,decode,spec_decode")
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _match_target_pattern(actual: str, pattern: str) -> bool:
    return pattern in ("*", "") or actual == pattern


def _context_matches_target_specs(
    debug_context: Optional[Dict[str, object]],
    target_specs: List[str],
) -> bool:
    if not debug_context:
        return False
    source = str(debug_context.get("source", "")).strip().lower()
    channel = str(debug_context.get("channel", "")).strip().lower()
    tensor_name = str(debug_context.get("tensor_name", "")).strip().lower()
    for spec in target_specs:
        parts = spec.split(":")
        if len(parts) != 3:
            continue
        if (
            _match_target_pattern(source, parts[0])
            and _match_target_pattern(channel, parts[1])
            and _match_target_pattern(tensor_name, parts[2])
        ):
            return True
    return False


@lru_cache(maxsize=1)
def _lossless_layout_targets() -> List[str]:
    cfg_val = str(_get_cfg("LOSSLESS_LAYOUT_TARGETS", "*:*:hidden_states"))
    if _allow_env_override():
        cfg_val = os.environ.get(_LOSSLESS_LAYOUT_TARGETS_ENV, cfg_val)
    return [item.strip().lower() for item in cfg_val.split(",") if item.strip()]


def _context_matches_wire_target(debug_context: Optional[Dict[str, object]]) -> bool:
    phase = str(debug_context.get("phase", "")).strip().lower() if debug_context else ""
    phases = _wire_truncate_phases()
    if phases and phase and phase not in phases:
        return False
    return _context_matches_target_specs(debug_context, _wire_truncate_targets())


def _context_matches_lossless_layout_target(debug_context: Optional[Dict[str, object]]) -> bool:
    return _context_matches_target_specs(debug_context, _lossless_layout_targets())


def _should_wire_truncate_fp16(
    tensor: Optional[torch.Tensor],
    debug_context: Optional[Dict[str, object]],
) -> bool:
    if not wire_truncate_fp16_enabled():
        return False
    if tensor is None or not torch.is_tensor(tensor):
        return False
    if tensor.dtype != torch.float32:
        return False
    return _context_matches_wire_target(debug_context)


def _comp_research_log_every() -> int:
    return max(1, _get_env_int("BLOOMBEE_COMP_SUMMARY_LOG_EVERY", 128))


def _new_transport_profile() -> Dict[str, float]:
    return {
        "serialize_calls": 0.0,
        "deserialize_calls": 0.0,
        "serialize_core_ms": 0.0,
        "deserialize_core_ms": 0.0,
        "serialize_wrapper_ms": 0.0,
        "deserialize_unwrap_ms": 0.0,
        "compress_ms": 0.0,
        "decompress_ms": 0.0,
        "compress_calls": 0.0,
        "decompress_calls": 0.0,
        "serialize_wrapper_applied_calls": 0.0,
        "deserialize_wrapper_applied_calls": 0.0,
        "serialize_raw_bytes": 0.0,
        "serialize_wire_bytes": 0.0,
        "deserialize_wire_bytes": 0.0,
        "deserialize_raw_bytes": 0.0,
        "compress_input_bytes": 0.0,
        "compress_output_bytes": 0.0,
        "decompress_input_bytes": 0.0,
        "decompress_output_bytes": 0.0,
    }


def _record_transport_profile(key: str, value: float) -> None:
    stats = _TRANSPORT_PROFILE_CTX.get()
    if stats is None:
        return
    stats[key] = float(stats.get(key, 0.0) + value)


@contextmanager
def transport_profile_scope():
    """
    Capture lossless transport timing and volume counters for a logical step.
    """
    if not comp_timing_profile_enabled():
        yield None
        return

    profile = _new_transport_profile()
    token = _TRANSPORT_PROFILE_CTX.set(profile)
    try:
        yield profile
    finally:
        _TRANSPORT_PROFILE_CTX.reset(token)


def summarize_transport_profile(stats: Optional[Dict[str, float]]) -> Dict[str, float]:
    summary = _new_transport_profile()
    if stats:
        for k, v in stats.items():
            if k in summary:
                summary[k] = float(v)

    raw = max(float(summary["serialize_raw_bytes"]), 1.0)
    wire = float(summary["serialize_wire_bytes"])
    summary["serialize_ratio"] = wire / raw
    summary["serialize_savings"] = 1.0 - summary["serialize_ratio"]
    summary["serialize_saved_bytes"] = max(0.0, raw - wire)
    summary["compress_saved_kb_per_ms"] = (
        (summary["serialize_saved_bytes"] / 1024.0) / summary["compress_ms"]
        if summary["compress_ms"] > 0
        else 0.0
    )
    summary["compress_only_break_even_bw_mbps"] = _break_even_bandwidth_mbps(
        summary["serialize_saved_bytes"], summary["compress_ms"]
    )
    summary["wrapper_only_break_even_bw_mbps"] = _break_even_bandwidth_mbps(
        summary["serialize_saved_bytes"], summary["serialize_wrapper_ms"]
    )

    draw = max(float(summary["deserialize_wire_bytes"]), 1.0)
    draw_out = float(summary["deserialize_raw_bytes"])
    summary["deserialize_ratio"] = draw_out / draw
    return summary


def log_transport_profile_event(
    log,
    *,
    source: str,
    channel: str,
    blocks: str,
    step_id: str,
    batch_size: int,
    stats: Optional[Dict[str, float]],
    extra: Optional[Dict[str, object]] = None,
) -> None:
    """
    Emit a stable single-line timing/volume record for compression profiling.
    """
    if not comp_timing_profile_enabled():
        return
    summary = summarize_transport_profile(stats)
    msg = (
        "[COMP_TIMING] "
        f"source={source} "
        f"channel={channel} "
        f"blocks={blocks} "
        f"step_id={step_id} "
        f"batch={int(batch_size)} "
        f"serialize_core_ms={summary['serialize_core_ms']:.3f} "
        f"serialize_wrap_ms={summary['serialize_wrapper_ms']:.3f} "
        f"compress_ms={summary['compress_ms']:.3f} "
        f"deserialize_unwrap_ms={summary['deserialize_unwrap_ms']:.3f} "
        f"decompress_ms={summary['decompress_ms']:.3f} "
        f"deserialize_core_ms={summary['deserialize_core_ms']:.3f} "
        f"serialize_raw_bytes={int(summary['serialize_raw_bytes'])} "
        f"serialize_wire_bytes={int(summary['serialize_wire_bytes'])} "
        f"deserialize_wire_bytes={int(summary['deserialize_wire_bytes'])} "
        f"deserialize_raw_bytes={int(summary['deserialize_raw_bytes'])} "
        f"serialize_ratio={summary['serialize_ratio']:.6f} "
        f"serialize_savings={summary['serialize_savings']:.6f} "
        f"serialize_saved_bytes={int(summary['serialize_saved_bytes'])} "
        f"compress_saved_kb_per_ms={summary['compress_saved_kb_per_ms']:.6f} "
        f"compress_only_break_even_bw_mbps={summary['compress_only_break_even_bw_mbps']:.3f} "
        f"wrapper_only_break_even_bw_mbps={summary['wrapper_only_break_even_bw_mbps']:.3f} "
        f"compress_input_bytes={int(summary['compress_input_bytes'])} "
        f"compress_output_bytes={int(summary['compress_output_bytes'])} "
        f"decompress_input_bytes={int(summary['decompress_input_bytes'])} "
        f"decompress_output_bytes={int(summary['decompress_output_bytes'])} "
        f"serialize_wrapper_applied={int(summary['serialize_wrapper_applied_calls'])} "
        f"deserialize_wrapper_applied={int(summary['deserialize_wrapper_applied_calls'])} "
        f"compress_calls={int(summary['compress_calls'])} "
        f"decompress_calls={int(summary['decompress_calls'])}"
    )
    if extra:
        for k, v in extra.items():
            msg += f" {k}={v}"
    log.info(msg)


def tensor_raw_nbytes(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None or not torch.is_tensor(tensor):
        return 0
    try:
        return int(tensor.numel()) * int(tensor.element_size())
    except Exception:
        return 0


def tensor_nnz_ratio(tensor: Optional[torch.Tensor]) -> float:
    """
    Ratio of non-zero elements in a tensor, in [0, 1].
    """
    if tensor is None or not torch.is_tensor(tensor):
        return 0.0
    try:
        numel = int(tensor.numel())
        if numel <= 0:
            return 0.0
        nnz = int(torch.count_nonzero(tensor).item())
        return float(nnz) / float(numel)
    except Exception:
        return 0.0


def log_comp_ratio_event(
    log,
    *,
    source: str,
    channel: str,
    blocks: str,
    step_id: str,
    batch_size: int,
    tensor_name: str,
    raw_bytes: int,
    wire_bytes: int,
    nnz_ratio: float,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    """
    Emit a stable single-line record for compression-factor analysis.
    """
    if not comp_ratio_profile_enabled():
        return
    try:
        raw_i = int(max(0, raw_bytes))
        wire_i = int(max(0, wire_bytes))
        ratio = (float(wire_i) / float(raw_i)) if raw_i > 0 else 1.0
        savings = 1.0 - ratio
        nnz = max(0.0, min(1.0, float(nnz_ratio)))
        msg = (
            "[COMP_RATIO] "
            f"source={source} "
            f"channel={channel} "
            f"blocks={blocks} "
            f"step_id={step_id} "
            f"batch={int(batch_size)} "
            f"tensor={tensor_name} "
            f"raw_bytes={raw_i} "
            f"wire_bytes={wire_i} "
            f"ratio={ratio:.6f} "
            f"savings={savings:.6f} "
            f"nnz={nnz:.6f} "
            f"lossless={int(_lossless_send_enabled())} "
            f"algo={_lossless_algo()}"
        )
        if extra:
            for k, v in extra.items():
                msg += f" {k}={v}"
        log.info(msg)
    except Exception:
        # Profiling logs must never break inference.
        return


def _compression_type_name(compression_type: runtime_pb2.CompressionType) -> str:
    try:
        return runtime_pb2.CompressionType.Name(int(compression_type))
    except Exception:
        return str(int(compression_type)) if compression_type is not None else "unknown"


def _size_bucket_label(raw_bytes: int) -> str:
    raw_i = max(0, int(raw_bytes))
    if raw_i < 4 * 1024:
        return "<4KB"
    if raw_i < 49 * 1024:
        return "4KB-49KB"
    if raw_i < 80 * 1024:
        return "49KB-80KB"
    if raw_i < 200 * 1024:
        return "80KB-200KB"
    if raw_i < 1024 * 1024:
        return "200KB-1MB"
    return ">=1MB"


def _break_even_bandwidth_mbps(saved_bytes: float, overhead_ms: float) -> float:
    saved_f = max(0.0, float(saved_bytes))
    overhead_f = max(0.0, float(overhead_ms))
    if saved_f <= 0.0 or overhead_f <= 0.0:
        return 0.0
    return (saved_f * 8.0) / (overhead_f / 1000.0) / 1_000_000.0


def _new_comp_research_bucket() -> Dict[str, float]:
    return {
        "samples": 0.0,
        "applied": 0.0,
        "raw_bytes": 0.0,
        "wire_bytes": 0.0,
        "saved_bytes": 0.0,
        "compress_ms": 0.0,
        "wrapper_ms": 0.0,
        "break_even_bw_mbps": 0.0,
    }


def _record_comp_research_event(
    *,
    raw_bytes: int,
    wire_bytes: int,
    wrap_info: Dict[str, object],
) -> None:
    if not comp_research_profile_enabled():
        return
    raw_i = max(0, int(raw_bytes))
    wire_i = max(0, int(wire_bytes))
    bucket = _size_bucket_label(raw_i)
    compress_ms = max(0.0, float(wrap_info.get("compress_elapsed_ms", 0.0) or 0.0))
    wrapper_ms = max(0.0, float(wrap_info.get("wrapper_elapsed_ms", 0.0) or 0.0))
    saved_bytes = max(0, raw_i - wire_i)
    break_even_bw_mbps = _break_even_bandwidth_mbps(saved_bytes, compress_ms)

    with _COMP_RESEARCH_LOCK:
        global _COMP_RESEARCH_TOTAL_SAMPLES
        stats = _COMP_RESEARCH_BUCKETS.setdefault(bucket, _new_comp_research_bucket())
        stats["samples"] += 1.0
        stats["applied"] += float(int(wrap_info.get("applied", 0) or 0))
        stats["raw_bytes"] += float(raw_i)
        stats["wire_bytes"] += float(wire_i)
        stats["saved_bytes"] += float(saved_bytes)
        stats["compress_ms"] += compress_ms
        stats["wrapper_ms"] += wrapper_ms
        stats["break_even_bw_mbps"] += break_even_bw_mbps
        _COMP_RESEARCH_TOTAL_SAMPLES += 1

        if _COMP_RESEARCH_TOTAL_SAMPLES % _comp_research_log_every() != 0:
            return

        parts = []
        for label in _COMP_RESEARCH_BUCKET_ORDER:
            bucket_stats = _COMP_RESEARCH_BUCKETS.get(label)
            if not bucket_stats or bucket_stats["samples"] <= 0:
                continue
            sample_count = max(bucket_stats["samples"], 1.0)
            raw_sum = max(bucket_stats["raw_bytes"], 1.0)
            avg_ratio = bucket_stats["wire_bytes"] / raw_sum
            avg_saved_bytes = bucket_stats["saved_bytes"] / sample_count
            avg_comp_ms = bucket_stats["compress_ms"] / sample_count
            avg_wrapper_ms = bucket_stats["wrapper_ms"] / sample_count
            avg_break_even_bw_mbps = bucket_stats["break_even_bw_mbps"] / sample_count
            apply_rate = bucket_stats["applied"] / sample_count
            parts.append(
                f"{label}:n={int(sample_count)}"
                f",apply={apply_rate:.3f}"
                f",R={avg_ratio:.6f}"
                f",saved_kb={avg_saved_bytes / 1024.0:.2f}"
                f",comp_ms={avg_comp_ms:.3f}"
                f",wrap_ms={avg_wrapper_ms:.3f}"
                f",be_bw={avg_break_even_bw_mbps:.3f}"
            )
        if parts:
            logger.info(
                "[COMP_SCALING_SUMMARY] "
                f"total_samples={_COMP_RESEARCH_TOTAL_SAMPLES} "
                f"lossless={int(_lossless_send_enabled())} "
                f"algo={_lossless_algo()} "
                + " | ".join(parts)
            )


def _sample_tensor_value_profile(tensor: Optional[torch.Tensor], max_values: int = 4096) -> Dict[str, float]:
    profile = {
        "tensor_zero_ratio": 0.0,
        "tensor_pos_ratio": 0.0,
        "tensor_sample_unique_ratio": 0.0,
        "tensor_sample_mean": 0.0,
        "tensor_sample_std": 0.0,
        "tensor_sample_abs_mean": 0.0,
        "tensor_sample_min": 0.0,
        "tensor_sample_max": 0.0,
        "tensor_sample_abs_p50": 0.0,
        "tensor_sample_abs_p90": 0.0,
        "tensor_sample_abs_p99": 0.0,
        "fp16_roundtrip_mae": 0.0,
        "fp16_roundtrip_p99_abs_err": 0.0,
        "fp16_roundtrip_max_abs_err": 0.0,
        "fp16_roundtrip_rel_mae": 0.0,
        "tensor_sample_size": 0.0,
    }
    if tensor is None or not torch.is_tensor(tensor):
        return profile
    try:
        flat = tensor.detach().reshape(-1)
        if flat.numel() <= 0:
            return profile
        sample = flat[: min(int(flat.numel()), int(max_values))]
        is_floating = bool(torch.is_floating_point(sample))
        if sample.device.type != "cpu":
            sample = sample.to("cpu")
        sample = sample.to(torch.float32)
        sample_n = int(sample.numel())
        if sample_n <= 0:
            return profile
        abs_sample = sample.abs()
        zero_ratio = float((sample == 0).sum().item()) / float(sample_n)
        pos_ratio = float((sample > 0).sum().item()) / float(sample_n)
        if sample_n <= 512:
            unique_ratio = float(torch.unique(sample).numel()) / float(sample_n)
        else:
            rounded = torch.round(sample * 1000.0) / 1000.0
            unique_ratio = float(torch.unique(rounded).numel()) / float(sample_n)
        abs_p50 = float(torch.quantile(abs_sample, 0.50).item())
        abs_p90 = float(torch.quantile(abs_sample, 0.90).item())
        abs_p99 = float(torch.quantile(abs_sample, 0.99).item())
        fp16_roundtrip_mae = 0.0
        fp16_roundtrip_p99_abs_err = 0.0
        fp16_roundtrip_max_abs_err = 0.0
        fp16_roundtrip_rel_mae = 0.0
        if is_floating:
            fp16_roundtrip = sample.to(torch.float16).to(torch.float32)
            abs_err = (fp16_roundtrip - sample).abs()
            fp16_roundtrip_mae = float(abs_err.mean().item())
            fp16_roundtrip_p99_abs_err = float(torch.quantile(abs_err, 0.99).item())
            fp16_roundtrip_max_abs_err = float(abs_err.max().item())
            fp16_roundtrip_rel_mae = float((abs_err / abs_sample.clamp_min(1e-8)).mean().item())
        profile.update(
            {
                "tensor_zero_ratio": zero_ratio,
                "tensor_pos_ratio": pos_ratio,
                "tensor_sample_unique_ratio": unique_ratio,
                "tensor_sample_mean": float(sample.mean().item()),
                "tensor_sample_std": float(sample.std(unbiased=False).item()),
                "tensor_sample_abs_mean": float(sample.abs().mean().item()),
                "tensor_sample_min": float(sample.min().item()),
                "tensor_sample_max": float(sample.max().item()),
                "tensor_sample_abs_p50": abs_p50,
                "tensor_sample_abs_p90": abs_p90,
                "tensor_sample_abs_p99": abs_p99,
                "fp16_roundtrip_mae": fp16_roundtrip_mae,
                "fp16_roundtrip_p99_abs_err": fp16_roundtrip_p99_abs_err,
                "fp16_roundtrip_max_abs_err": fp16_roundtrip_max_abs_err,
                "fp16_roundtrip_rel_mae": fp16_roundtrip_rel_mae,
                "tensor_sample_size": float(sample_n),
            }
        )
    except Exception:
        return profile
    return profile


def _sample_buffer_profile(buffer: bytes, max_bytes: int = 4096) -> Dict[str, float]:
    profile = {
        "byte_sample_size": 0.0,
        "byte_zero_ratio": 0.0,
        "byte_unique_count": 0.0,
        "byte_unique_ratio": 0.0,
        "byte_top1_ratio": 0.0,
        "byte_adj_repeat_ratio": 0.0,
        "byte_entropy_bits": 0.0,
    }
    if not buffer:
        return profile
    sample = memoryview(buffer)[: min(len(buffer), int(max_bytes))]
    n = int(len(sample))
    if n <= 0:
        return profile
    counts = [0] * 256
    prev = None
    adj_repeats = 0
    for b in sample:
        val = int(b)
        counts[val] += 1
        if prev is not None and prev == val:
            adj_repeats += 1
        prev = val
    unique_count = sum(1 for c in counts if c > 0)
    top1 = max(counts)
    entropy = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = float(c) / float(n)
        entropy -= p * math.log2(p)
    profile.update(
        {
            "byte_sample_size": float(n),
            "byte_zero_ratio": float(counts[0]) / float(n),
            "byte_unique_count": float(unique_count),
            "byte_unique_ratio": float(unique_count) / float(n),
            "byte_top1_ratio": float(top1) / float(n),
            "byte_adj_repeat_ratio": float(adj_repeats) / float(max(1, n - 1)),
            "byte_entropy_bits": entropy,
        }
    )
    return profile


def _entropy_from_counts(counts: Iterable[int], total: int) -> float:
    if total <= 0:
        return 0.0
    entropy = 0.0
    denom = float(total)
    for c in counts:
        if c <= 0:
            continue
        p = float(c) / denom
        entropy -= p * math.log2(p)
    return entropy


def _sample_fp_bit_profile(tensor: Optional[torch.Tensor], max_values: int = 4096) -> Dict[str, object]:
    profile: Dict[str, object] = {
        "fp_bits": 0,
        "fp_sample_size": 0,
        "fp_sign_one_ratio": 0.0,
        "fp_exp_unique": 0,
        "fp_exp_top1": -1,
        "fp_exp_top1_ratio": 0.0,
        "fp_exp_entropy_bits": 0.0,
        "fp_exp_adj_repeat_ratio": 0.0,
        "fp_exp_zero_ratio": 0.0,
        "fp_exp_all_ones_ratio": 0.0,
        "fp_mantissa_zero_ratio": 0.0,
        "fp_exp_topk": "",
    }
    if tensor is None or not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
        return profile
    try:
        flat = tensor.detach().reshape(-1)
        if flat.numel() <= 0:
            return profile
        sample = flat[: min(int(flat.numel()), int(max_values))]
        if sample.device.type != "cpu":
            sample = sample.to("cpu")
        sample = sample.contiguous()
        sample_n = int(sample.numel())
        if sample_n <= 0:
            return profile

        if sample.dtype == torch.float32:
            bits = sample.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
            sign = (bits >> 31) & 0x1
            exponent = (bits >> 23) & 0xFF
            mantissa = bits & 0x7FFFFF
            exp_cardinality = 256
            exp_all_ones = 0xFF
            fp_bits = 32
        elif sample.dtype == torch.float16:
            bits = sample.view(torch.int16).to(torch.int64) & 0xFFFF
            sign = (bits >> 15) & 0x1
            exponent = (bits >> 10) & 0x1F
            mantissa = bits & 0x3FF
            exp_cardinality = 32
            exp_all_ones = 0x1F
            fp_bits = 16
        else:
            # We only do exact bit-lane attribution for FP16/FP32.
            return profile

        exp_counts = torch.bincount(exponent.to(torch.int64), minlength=exp_cardinality)
        exp_total = int(exp_counts.sum().item())
        exp_top_k = min(4, exp_cardinality)
        exp_top_vals, exp_top_idx = torch.topk(exp_counts, k=exp_top_k)
        exp_top_parts: List[str] = []
        for idx_i, cnt_i in zip(exp_top_idx.tolist(), exp_top_vals.tolist()):
            if cnt_i <= 0:
                continue
            exp_top_parts.append(f"{int(idx_i)}:{(float(cnt_i) / float(sample_n)):.4f}")

        adj_repeat = float((exponent[1:] == exponent[:-1]).sum().item()) / float(max(1, sample_n - 1))

        profile.update(
            {
                "fp_bits": fp_bits,
                "fp_sample_size": sample_n,
                "fp_sign_one_ratio": float(sign.float().mean().item()),
                "fp_exp_unique": int((exp_counts > 0).sum().item()),
                "fp_exp_top1": int(exp_top_idx[0].item()) if exp_top_vals.numel() > 0 else -1,
                "fp_exp_top1_ratio": (float(exp_top_vals[0].item()) / float(sample_n)) if exp_top_vals.numel() > 0 else 0.0,
                "fp_exp_entropy_bits": _entropy_from_counts(exp_counts.tolist(), exp_total),
                "fp_exp_adj_repeat_ratio": adj_repeat,
                "fp_exp_zero_ratio": float((exponent == 0).float().mean().item()),
                "fp_exp_all_ones_ratio": float((exponent == exp_all_ones).float().mean().item()),
                "fp_mantissa_zero_ratio": float((mantissa == 0).float().mean().item()),
                "fp_exp_topk": ",".join(exp_top_parts),
            }
        )
    except Exception:
        return profile
    return profile


def _fp_layout_description(dtype: Optional[torch.dtype]) -> str:
    if dtype == torch.float32:
        # For little-endian IEEE-754 FP32 values:
        # lane0=b0 (mantissa bits 0..7)
        # lane1=b1 (mantissa bits 8..15)
        # lane2=b2 (mantissa bits 16..22 + exponent bit 0)
        # lane3=b3 (exponent bits 1..7 + sign bit)
        return "fp32_le lanes:0=m[0:8],1=m[8:16],2=m[16:23]+e0,3=e[1:8]+s"
    if dtype == torch.float16:
        # For little-endian IEEE-754 FP16 values:
        # lane0=b0 (mantissa bits 0..7)
        # lane1=b1 (mantissa bits 8..9 + exponent bits 0..4 + sign bit)
        return "fp16_le lanes:0=m[0:8],1=m[8:10]+e[0:5]+s"
    return ""


def _log_serial_layout_once(dtype: Optional[torch.dtype]) -> None:
    if dtype is None:
        return
    dtype_name = str(dtype).replace("torch.", "")
    if dtype_name in _SER_LAYOUT_LOGGED_DTYPES:
        return
    layout = _fp_layout_description(dtype)
    if not layout:
        return
    _SER_LAYOUT_LOGGED_DTYPES.add(dtype_name)
    logger.info(
        "[SER_LAYOUT] "
        f"dtype={dtype_name} "
        f"byteorder={sys.byteorder} "
        f"mapping=\"{layout}\""
    )


def _sample_strided_pattern_profile(
    buffer: bytes,
    *,
    elem_size: int,
    max_bytes: int = 16384,
) -> Dict[str, object]:
    profile: Dict[str, object] = {
        "stride_elem_size": int(elem_size),
        "stride_sample_bytes": 0,
        "stride_best_repeat_offset": -1,
        "stride_best_repeat_ratio": 0.0,
        "stride_offset0_repeat_ratio": 0.0,
        "stride_repeat_by_offset": "",
        "stride_adj_eq_by_offset": "",
        "stride_lane_entropy": "",
        "stride_lane_zero_ratio": "",
    }
    if not buffer or elem_size <= 0:
        return profile
    sample = memoryview(buffer)[: min(len(buffer), int(max_bytes))]
    n = int(len(sample))
    if n < elem_size:
        return profile
    profile["stride_sample_bytes"] = n

    repeat_parts: List[str] = []
    adj_parts: List[str] = []
    lane_entropy_parts: List[str] = []
    lane_zero_parts: List[str] = []
    best_offset = -1
    best_repeat = -1.0

    for offset in range(elem_size):
        chunk_count = (n - offset) // elem_size
        if chunk_count <= 0:
            repeat_parts.append(f"o{offset}:0.0000")
            adj_parts.append(f"o{offset}:0.0000")
            continue
        seen: set[bytes] = set()
        prev_chunk: Optional[bytes] = None
        adj_eq = 0
        for i in range(chunk_count):
            st = offset + i * elem_size
            chunk = bytes(sample[st : st + elem_size])
            if prev_chunk is not None and chunk == prev_chunk:
                adj_eq += 1
            prev_chunk = chunk
            seen.add(chunk)
        repeat_ratio = 1.0 - (float(len(seen)) / float(chunk_count))
        adj_ratio = float(adj_eq) / float(max(1, chunk_count - 1))
        repeat_parts.append(f"o{offset}:{repeat_ratio:.4f}")
        adj_parts.append(f"o{offset}:{adj_ratio:.4f}")
        if repeat_ratio > best_repeat:
            best_repeat = repeat_ratio
            best_offset = offset

        lane = sample[offset:n:elem_size]
        lane_n = int(len(lane))
        if lane_n > 0:
            counts = [0] * 256
            for b in lane:
                counts[int(b)] += 1
            lane_entropy = _entropy_from_counts(counts, lane_n)
            lane_zero_ratio = float(counts[0]) / float(lane_n)
            lane_entropy_parts.append(f"o{offset}:{lane_entropy:.4f}")
            lane_zero_parts.append(f"o{offset}:{lane_zero_ratio:.4f}")

    profile["stride_best_repeat_offset"] = int(best_offset)
    profile["stride_best_repeat_ratio"] = max(0.0, float(best_repeat))
    profile["stride_offset0_repeat_ratio"] = float(best_repeat if elem_size == 1 else 0.0)
    if elem_size > 1:
        # Extract o0 from assembled parts.
        for part in repeat_parts:
            if part.startswith("o0:"):
                profile["stride_offset0_repeat_ratio"] = float(part.split(":")[1])
                break
    profile["stride_repeat_by_offset"] = ",".join(repeat_parts)
    profile["stride_adj_eq_by_offset"] = ",".join(adj_parts)
    profile["stride_lane_entropy"] = ",".join(lane_entropy_parts)
    profile["stride_lane_zero_ratio"] = ",".join(lane_zero_parts)
    return profile


def _sample_abs_decade_histogram(
    tensor: Optional[torch.Tensor],
    max_values: int = 4096,
) -> Dict[str, object]:
    profile: Dict[str, object] = {
        "act_sample_size": 0,
        "act_abs_zero_ratio": 0.0,
        "act_abs_lt_0_1_count": 0,
        "act_abs_ge_1000_count": 0,
        "act_abs_lt_0_1_ratio": 0.0,
        "act_abs_ge_1000_ratio": 0.0,
        "act_abs_bin_counts": "",
        "act_abs_bin_ratio": "",
        "act_abs_decade_hist": "",
        "act_abs_decade_peak": "",
    }
    if tensor is None or not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
        return profile
    try:
        flat = tensor.detach().reshape(-1)
        if flat.numel() <= 0:
            return profile
        sample = flat[: min(int(flat.numel()), int(max_values))]
        if sample.device.type != "cpu":
            sample = sample.to("cpu")
        sample = sample.to(torch.float32)
        abs_sample = sample.abs()
        sample_n = int(abs_sample.numel())
        if sample_n <= 0:
            return profile
        zero_count = int((abs_sample == 0).sum().item())
        bins = (
            ("0.1~1", 0.1, 1.0),
            ("1~10", 1.0, 10.0),
            ("10~100", 10.0, 100.0),
            ("100~1000", 100.0, 1000.0),
        )
        counts: List[int] = []
        labels: List[str] = []
        for label, lo, hi in bins:
            cnt = int(((abs_sample >= lo) & (abs_sample < hi)).sum().item())
            labels.append(label)
            counts.append(cnt)

        low_tail = int((abs_sample < 0.1).sum().item())
        high_tail = int((abs_sample >= 1000.0).sum().item())

        ratio_parts: List[str] = []
        count_parts: List[str] = []
        peak_idx = -1
        peak_val = -1
        denom = float(max(1, sample_n))
        for idx, (label, cnt) in enumerate(zip(labels, counts)):
            ratio = float(cnt) / denom
            ratio_parts.append(f"{label}:{ratio:.6f}")
            count_parts.append(f"{label}:{cnt}")
            if cnt > peak_val:
                peak_val = cnt
                peak_idx = idx

        profile.update(
            {
                "act_sample_size": sample_n,
                "act_abs_zero_ratio": float(zero_count) / float(sample_n),
                "act_abs_lt_0_1_count": low_tail,
                "act_abs_ge_1000_count": high_tail,
                "act_abs_lt_0_1_ratio": float(low_tail) / float(sample_n),
                "act_abs_ge_1000_ratio": float(high_tail) / float(sample_n),
                "act_abs_bin_counts": ",".join(count_parts),
                "act_abs_bin_ratio": ",".join(ratio_parts),
                "act_abs_decade_hist": ",".join(ratio_parts),
                "act_abs_decade_peak": labels[peak_idx] if peak_idx >= 0 else "",
            }
        )
    except Exception:
        return profile
    return profile


def _log_comp_bit_event(
    *,
    tensor: Optional[torch.Tensor],
    debug_context: Optional[Dict[str, object]],
) -> None:
    if not comp_bit_profile_enabled():
        return
    if not _is_debug_target_tensor(tensor, debug_context):
        return
    _log_serial_layout_once(tensor.dtype if torch.is_tensor(tensor) else None)
    fp = _sample_fp_bit_profile(tensor)
    if int(fp.get("fp_sample_size", 0) or 0) <= 0:
        return
    msg = (
        "[COMP_BIT] "
        f"fp_bits={int(fp.get('fp_bits', 0))} "
        f"fp_sample_size={int(fp.get('fp_sample_size', 0))} "
        f"fp_sign_one_ratio={float(fp.get('fp_sign_one_ratio', 0.0)):.6f} "
        f"fp_exp_unique={int(fp.get('fp_exp_unique', 0))} "
        f"fp_exp_top1={int(fp.get('fp_exp_top1', -1))} "
        f"fp_exp_top1_ratio={float(fp.get('fp_exp_top1_ratio', 0.0)):.6f} "
        f"fp_exp_entropy_bits={float(fp.get('fp_exp_entropy_bits', 0.0)):.6f} "
        f"fp_exp_adj_repeat_ratio={float(fp.get('fp_exp_adj_repeat_ratio', 0.0)):.6f} "
        f"fp_exp_zero_ratio={float(fp.get('fp_exp_zero_ratio', 0.0)):.6f} "
        f"fp_exp_all_ones_ratio={float(fp.get('fp_exp_all_ones_ratio', 0.0)):.6f} "
        f"fp_mantissa_zero_ratio={float(fp.get('fp_mantissa_zero_ratio', 0.0)):.6f} "
        f"fp_exp_topk={fp.get('fp_exp_topk', '')}"
    )
    if debug_context:
        for k, v in debug_context.items():
            msg += f" {k}={v}"
    logger.info(msg)


def _log_comp_config_once() -> None:
    global _COMP_LOG_CFG_EMITTED
    if _COMP_LOG_CFG_EMITTED:
        return
    _COMP_LOG_CFG_EMITTED = True
    logger.info(
        "[COMP_LOG_CFG] "
        f"module={__file__} "
        f"lossless_enabled={int(_lossless_send_enabled())} "
        f"algo={_lossless_algo()} "
        f"layout={_lossless_layout()} "
        f"detail={int(comp_detail_profile_enabled())} "
        f"bit={int(comp_bit_profile_enabled())} "
        f"stride={int(comp_stride_profile_enabled())} "
        f"act={int(act_dist_profile_enabled())} "
        f"zipnn={int(comp_zipnn_profile_enabled())} "
        f"zipnn_available={int(_ZipNN is not None)} "
        f"debug_tensors={','.join(sorted(_debug_tensor_names()))}"
    )


def _new_zipnn_compare_info() -> Dict[str, object]:
    return {
        "attempted": 0,
        "available": int(_ZipNN is not None),
        "supported": 0,
        "lossless_verified": 0,
        "reason": "disabled",
        "dtype": "",
        "payload_bytes": 0,
        "wrapped_bytes": 0,
        "saved_bytes": 0,
        "ratio": 1.0,
        "elapsed_ms": 0.0,
        "break_even_bw_mbps": 0.0,
        "selected_wire_bytes": 0,
        "selected_delta_bytes": 0,
        "better_than_selected": 0,
    }
    

def _zipnn_dtype_name(tensor: Optional[torch.Tensor]) -> Optional[str]:
    if tensor is None or not torch.is_tensor(tensor):
        return None
    if tensor.dtype == torch.float16:
        return "float16"
    if tensor.dtype == torch.bfloat16:
        return "bfloat16"
    if tensor.dtype == torch.float32:
        return "float32"
    return None


@lru_cache(maxsize=8)
def _get_zipnn_compressor(dtype_name: str, zstd_level: int):
    if _ZipNN is None:
        return None
    return _ZipNN(
        method="ZSTD",
        input_format="byte",
        bytearray_dtype=dtype_name,
        compression_threshold=1.0,
        zstd_level=zstd_level,
    )


@lru_cache(maxsize=1)
def _get_zipnn_decompressor():
    if _ZipNN is None:
        return None
    return _ZipNN(method="ZSTD", input_format="byte")


@lru_cache(maxsize=8)
def _zipnn_lossless_dtype_supported(dtype_name: str) -> bool:
    compressor = _get_zipnn_compressor(dtype_name, 3)
    decompressor = _get_zipnn_decompressor()
    if compressor is None or decompressor is None:
        return False
    raw = bytes(((idx * 73) + 11) & 0xFF for idx in range(4096))
    try:
        compressed = compressor.compress(raw)
        restored = decompressor.decompress(compressed)
    except Exception:
        return False
    return bytes(restored) == raw


def _supports_zipnn_compare(
    tensor: Optional[torch.Tensor],
    compression_type: runtime_pb2.CompressionType,
    raw_size: int,
    debug_context: Optional[Dict[str, object]],
) -> bool:
    if _ZipNN is None:
        return False
    if compression_type != runtime_pb2.CompressionType.NONE:
        return False
    if tensor is None or not torch.is_tensor(tensor):
        return False
    dtype_name = _zipnn_dtype_name(tensor)
    if dtype_name is None:
        return False
    if raw_size != tensor.numel() * tensor.element_size():
        return False
    if not _context_matches_lossless_layout_target(debug_context):
        return False
    return _zipnn_lossless_dtype_supported(dtype_name)


def _supports_zipnn_transport(
    tensor: Optional[torch.Tensor],
    compression_type: runtime_pb2.CompressionType,
    raw_size: int,
    debug_context: Optional[Dict[str, object]],
) -> bool:
    return _supports_zipnn_compare(tensor, compression_type, raw_size, debug_context)


def _zipnn_skip_reason(
    tensor: Optional[torch.Tensor],
    compression_type: runtime_pb2.CompressionType,
    raw_size: int,
    debug_context: Optional[Dict[str, object]],
) -> str:
    if _ZipNN is None:
        _warn_missing_zipnn_once()
        return "zipnn_unavailable"
    if compression_type != runtime_pb2.CompressionType.NONE:
        return "zipnn_hivemind_compressed"
    if tensor is None or not torch.is_tensor(tensor):
        return "zipnn_missing_tensor"
    dtype_name = _zipnn_dtype_name(tensor)
    if dtype_name is None:
        return "zipnn_unsupported_dtype"
    if raw_size != tensor.numel() * tensor.element_size():
        return "zipnn_size_mismatch"
    if not _context_matches_lossless_layout_target(debug_context):
        return "zipnn_target_mismatch"
    if not _zipnn_lossless_dtype_supported(dtype_name):
        return "zipnn_lossless_verification_failed"
    return "zipnn_unsupported"


def _profile_zipnn_candidate(
    *,
    tensor: Optional[torch.Tensor],
    compression_type: runtime_pb2.CompressionType,
    raw_buffer: bytes,
    selected_wire_bytes: int,
    debug_context: Optional[Dict[str, object]],
) -> Dict[str, object]:
    info = _new_zipnn_compare_info()
    info["selected_wire_bytes"] = int(max(0, selected_wire_bytes))
    if not comp_zipnn_profile_enabled():
        return info
    if _ZipNN is None:
        info["reason"] = "zipnn_unavailable"
        _warn_missing_zipnn_once()
        return info
    dtype_name = _zipnn_dtype_name(tensor)
    if dtype_name is None:
        info["reason"] = "unsupported_dtype"
        return info
    info["dtype"] = dtype_name
    info["lossless_verified"] = int(_zipnn_lossless_dtype_supported(dtype_name))
    if not _supports_zipnn_transport(tensor, compression_type, len(raw_buffer), debug_context):
        if compression_type != runtime_pb2.CompressionType.NONE:
            info["reason"] = "hivemind_compressed"
        elif not info["lossless_verified"]:
            info["reason"] = "lossless_verification_failed"
        elif not _context_matches_lossless_layout_target(debug_context):
            info["reason"] = "target_mismatch"
        else:
            info["reason"] = "unsupported"
        return info

    compressor = _get_zipnn_compressor(dtype_name, _lossless_level())
    if compressor is None:
        info["reason"] = "zipnn_unavailable"
        return info

    info["attempted"] = 1
    info["supported"] = 1
    t0 = time.perf_counter()
    try:
        payload = compressor.compress(raw_buffer)
    except Exception:
        info["reason"] = "compress_failed"
        return info
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    payload_bytes = len(payload)
    wrapped_bytes = _HEADER_SIZE + payload_bytes
    saved_bytes = max(0, len(raw_buffer) - wrapped_bytes)
    info.update(
        {
            "reason": "ok",
            "payload_bytes": int(payload_bytes),
            "wrapped_bytes": int(wrapped_bytes),
            "saved_bytes": int(saved_bytes),
            "ratio": (float(wrapped_bytes) / float(len(raw_buffer))) if raw_buffer else 1.0,
            "elapsed_ms": float(elapsed_ms),
            "break_even_bw_mbps": _break_even_bandwidth_mbps(saved_bytes, elapsed_ms),
            "selected_delta_bytes": int(max(0, selected_wire_bytes) - wrapped_bytes),
            "better_than_selected": int(wrapped_bytes < max(0, selected_wire_bytes)),
        }
    )
    return info


def _zipnn_info_from_selected_wrap(
    *,
    tensor: Optional[torch.Tensor],
    raw_buffer: bytes,
    wrap_info: Dict[str, object],
) -> Dict[str, object]:
    info = _new_zipnn_compare_info()
    dtype_name = _zipnn_dtype_name(tensor)
    wrapped_bytes = int(wrap_info.get("wrapped_bytes", 0) or 0)
    payload_bytes = int(wrap_info.get("compressed_bytes", 0) or 0)
    raw_len = len(raw_buffer)
    saved_bytes = max(0, raw_len - wrapped_bytes)
    info.update(
        {
            "attempted": 1,
            "supported": 1,
            "lossless_verified": int(bool(dtype_name) and _zipnn_lossless_dtype_supported(dtype_name)),
            "reason": "selected",
            "dtype": dtype_name or "",
            "payload_bytes": payload_bytes,
            "wrapped_bytes": wrapped_bytes,
            "saved_bytes": saved_bytes,
            "ratio": (float(wrapped_bytes) / float(raw_len)) if raw_len > 0 else 1.0,
            "elapsed_ms": float(wrap_info.get("compress_elapsed_ms", 0.0) or 0.0),
            "break_even_bw_mbps": _break_even_bandwidth_mbps(
                saved_bytes, float(wrap_info.get("compress_elapsed_ms", 0.0) or 0.0)
            ),
            "selected_wire_bytes": wrapped_bytes,
            "selected_delta_bytes": 0,
            "better_than_selected": 0,
        }
    )
    return info


def _log_zipnn_compare_event(
    *,
    raw_bytes: int,
    wrap_info: Dict[str, object],
    zipnn_info: Optional[Dict[str, object]],
    debug_context: Optional[Dict[str, object]],
) -> None:
    if not is_log_channel_enabled("zipnn_logs"):
        return
    selected_algo = str(wrap_info.get("algo_name", _lossless_algo()))
    if not comp_zipnn_profile_enabled() and selected_algo != "zipnn":
        return
    info = zipnn_info or _new_zipnn_compare_info()
    msg = (
        "[COMP_ZIPNN] "
        f"attempted={int(info.get('attempted', 0))} "
        f"available={int(info.get('available', 0))} "
        f"supported={int(info.get('supported', 0))} "
        f"lossless_verified={int(info.get('lossless_verified', 0))} "
        f"reason={info.get('reason', 'unknown')} "
        f"dtype={info.get('dtype', '')} "
        f"raw_bytes={int(max(0, raw_bytes))} "
        f"zipnn_payload_bytes={int(info.get('payload_bytes', 0))} "
        f"zipnn_wrapped_bytes={int(info.get('wrapped_bytes', 0))} "
        f"zipnn_saved_bytes={int(info.get('saved_bytes', 0))} "
        f"zipnn_ratio={float(info.get('ratio', 1.0)):.6f} "
        f"zipnn_elapsed_ms={float(info.get('elapsed_ms', 0.0)):.6f} "
        f"zipnn_break_even_bw_mbps={float(info.get('break_even_bw_mbps', 0.0)):.6f} "
        f"selected_algo={wrap_info.get('algo_name', _lossless_algo())} "
        f"selected_layout={wrap_info.get('layout', 'plain')} "
        f"selected_wire_bytes={int(info.get('selected_wire_bytes', 0))} "
        f"zipnn_delta_vs_selected_bytes={int(info.get('selected_delta_bytes', 0))} "
        f"zipnn_better_than_selected={int(info.get('better_than_selected', 0))}"
    )
    if debug_context:
        for k, v in debug_context.items():
            msg += f" {k}={v}"
    logger.info(msg)


def _log_stride_pattern_event(
    *,
    tensor: Optional[torch.Tensor],
    raw_buffer: bytes,
    debug_context: Optional[Dict[str, object]],
) -> None:
    if not comp_stride_profile_enabled():
        return
    if not _is_debug_target_tensor(tensor, debug_context):
        return
    if tensor is None or not torch.is_tensor(tensor):
        return
    elem_size = int(tensor.element_size())
    if elem_size <= 0:
        return
    stride = _sample_strided_pattern_profile(raw_buffer, elem_size=elem_size)
    msg = (
        "[COMP_ZSTD_STRIDE] "
        f"elem_size={int(stride.get('stride_elem_size', 0))} "
        f"sample_bytes={int(stride.get('stride_sample_bytes', 0))} "
        f"best_repeat_offset={int(stride.get('stride_best_repeat_offset', -1))} "
        f"best_repeat_ratio={float(stride.get('stride_best_repeat_ratio', 0.0)):.6f} "
        f"offset0_repeat_ratio={float(stride.get('stride_offset0_repeat_ratio', 0.0)):.6f} "
        f"repeat_by_offset={stride.get('stride_repeat_by_offset', '')} "
        f"adj_eq_by_offset={stride.get('stride_adj_eq_by_offset', '')} "
        f"lane_entropy={stride.get('stride_lane_entropy', '')} "
        f"lane_zero_ratio={stride.get('stride_lane_zero_ratio', '')}"
    )
    if debug_context:
        for k, v in debug_context.items():
            msg += f" {k}={v}"
    logger.info(msg)


def _log_activation_distribution_event(
    *,
    tensor: Optional[torch.Tensor],
    debug_context: Optional[Dict[str, object]],
) -> None:
    if not act_dist_profile_enabled():
        return
    if not _is_debug_target_tensor(tensor, debug_context):
        return
    dist = _sample_abs_decade_histogram(tensor)
    if int(dist.get("act_sample_size", 0) or 0) <= 0:
        return
    msg = (
        "[ACT_DIST] "
        f"act_sample_size={int(dist.get('act_sample_size', 0))} "
        f"act_abs_zero_ratio={float(dist.get('act_abs_zero_ratio', 0.0)):.6f} "
        f"act_abs_lt_0_1_count={int(dist.get('act_abs_lt_0_1_count', 0))} "
        f"act_abs_lt_0_1_ratio={float(dist.get('act_abs_lt_0_1_ratio', 0.0)):.6f} "
        f"act_abs_ge_1000_count={int(dist.get('act_abs_ge_1000_count', 0))} "
        f"act_abs_ge_1000_ratio={float(dist.get('act_abs_ge_1000_ratio', 0.0)):.6f} "
        f"act_abs_bin_counts={dist.get('act_abs_bin_counts', '')} "
        f"act_abs_bin_ratio={dist.get('act_abs_bin_ratio', '')} "
        f"act_abs_decade_peak={dist.get('act_abs_decade_peak', '')} "
        f"act_abs_decade_hist={dist.get('act_abs_decade_hist', '')}"
    )
    if debug_context:
        for k, v in debug_context.items():
            msg += f" {k}={v}"
    logger.info(msg)


def _log_comp_detail_event(
    *,
    tensor: Optional[torch.Tensor],
    compression_type: runtime_pb2.CompressionType,
    raw_buffer: bytes,
    wire_buffer: bytes,
    wrap_info: Dict[str, object],
    zipnn_info: Optional[Dict[str, object]] = None,
    debug_context: Optional[Dict[str, object]] = None,
) -> None:
    if not comp_detail_profile_enabled():
        return
    try:
        _log_comp_config_once()
        raw_len = int(len(raw_buffer))
        wire_len = int(len(wire_buffer))
        ratio = (float(wire_len) / float(raw_len)) if raw_len > 0 else 1.0
        savings = 1.0 - ratio
        saved_bytes = max(0, raw_len - wire_len)
        size_bucket = _size_bucket_label(raw_len)
        compress_elapsed_ms = max(0.0, float(wrap_info.get("compress_elapsed_ms", 0.0) or 0.0))
        wrapper_elapsed_ms = max(0.0, float(wrap_info.get("wrapper_elapsed_ms", 0.0) or 0.0))
        compress_saved_kb_per_ms = (
            (saved_bytes / 1024.0) / compress_elapsed_ms if compress_elapsed_ms > 0.0 else 0.0
        )
        compress_only_break_even_bw_mbps = _break_even_bandwidth_mbps(saved_bytes, compress_elapsed_ms)
        wrapper_only_break_even_bw_mbps = _break_even_bandwidth_mbps(saved_bytes, wrapper_elapsed_ms)
        tensor_stats = _sample_tensor_value_profile(tensor)
        byte_stats = _sample_buffer_profile(raw_buffer)
        shape = tuple(int(dim) for dim in tensor.shape) if torch.is_tensor(tensor) else ()
        dtype = str(tensor.dtype).replace("torch.", "") if torch.is_tensor(tensor) else "unknown"
        msg = (
            "[COMP_DETAIL] "
            f"hivemind_comp={_compression_type_name(compression_type)} "
            f"lossless_enabled={int(_lossless_send_enabled())} "
            f"lossless_applied={int(wrap_info.get('applied', 0))} "
            f"lossless_reason={wrap_info.get('reason', 'unknown')} "
            f"lossless_algo={wrap_info.get('algo_name', _lossless_algo())} "
            f"lossless_layout={wrap_info.get('layout', 'plain')} "
            f"lossless_raw_bytes={raw_len} "
            f"lossless_wire_bytes={wire_len} "
            f"lossless_saved_bytes={saved_bytes} "
            f"lossless_ratio={ratio:.6f} "
            f"lossless_savings={savings:.6f} "
            f"size_bucket={size_bucket} "
            f"compress_elapsed_ms={compress_elapsed_ms:.6f} "
            f"wrapper_elapsed_ms={wrapper_elapsed_ms:.6f} "
            f"compress_saved_kb_per_ms={compress_saved_kb_per_ms:.6f} "
            f"compress_only_break_even_bw_mbps={compress_only_break_even_bw_mbps:.6f} "
            f"wrapper_only_break_even_bw_mbps={wrapper_only_break_even_bw_mbps:.6f} "
            f"lossless_compressed_bytes={int(wrap_info.get('compressed_bytes', 0))} "
            f"lossless_wrapped_bytes={int(wrap_info.get('wrapped_bytes', wire_len))} "
            f"plain_wrapped_bytes={int(wrap_info.get('plain_wrapped_bytes', 0))} "
            f"byte_split_wrapped_bytes={int(wrap_info.get('byte_split_wrapped_bytes', 0))} "
            f"zipnn_compare_wrapped_bytes={int((zipnn_info or {}).get('wrapped_bytes', 0))} "
            f"zipnn_compare_elapsed_ms={float((zipnn_info or {}).get('elapsed_ms', 0.0)):.6f} "
            f"zipnn_compare_delta_vs_selected_bytes={int((zipnn_info or {}).get('selected_delta_bytes', 0))} "
            f"zipnn_compare_better_than_selected={int((zipnn_info or {}).get('better_than_selected', 0))} "
            f"lossless_net_gain_bytes={int(wrap_info.get('net_gain_bytes', max(0, raw_len - wire_len)))} "
            f"min_bytes={int(wrap_info.get('min_bytes', _lossless_min_bytes()))} "
            f"min_gain_bytes={int(wrap_info.get('min_gain_bytes', _lossless_min_gain_bytes()))} "
            f"dtype={dtype} "
            f"shape={shape} "
            f"numel={int(tensor.numel()) if torch.is_tensor(tensor) else 0} "
            f"tensor_zero_ratio={tensor_stats['tensor_zero_ratio']:.6f} "
            f"tensor_pos_ratio={tensor_stats['tensor_pos_ratio']:.6f} "
            f"tensor_sample_unique_ratio={tensor_stats['tensor_sample_unique_ratio']:.6f} "
            f"tensor_sample_mean={tensor_stats['tensor_sample_mean']:.6f} "
            f"tensor_sample_std={tensor_stats['tensor_sample_std']:.6f} "
            f"tensor_sample_abs_mean={tensor_stats['tensor_sample_abs_mean']:.6f} "
            f"tensor_sample_min={tensor_stats['tensor_sample_min']:.6f} "
            f"tensor_sample_max={tensor_stats['tensor_sample_max']:.6f} "
            f"tensor_sample_abs_p50={tensor_stats['tensor_sample_abs_p50']:.6f} "
            f"tensor_sample_abs_p90={tensor_stats['tensor_sample_abs_p90']:.6f} "
            f"tensor_sample_abs_p99={tensor_stats['tensor_sample_abs_p99']:.6f} "
            f"fp16_roundtrip_mae={tensor_stats['fp16_roundtrip_mae']:.6f} "
            f"fp16_roundtrip_p99_abs_err={tensor_stats['fp16_roundtrip_p99_abs_err']:.6f} "
            f"fp16_roundtrip_max_abs_err={tensor_stats['fp16_roundtrip_max_abs_err']:.6f} "
            f"fp16_roundtrip_rel_mae={tensor_stats['fp16_roundtrip_rel_mae']:.6f} "
            f"tensor_sample_size={int(tensor_stats['tensor_sample_size'])} "
            f"byte_sample_size={int(byte_stats['byte_sample_size'])} "
            f"byte_zero_ratio={byte_stats['byte_zero_ratio']:.6f} "
            f"byte_unique_count={int(byte_stats['byte_unique_count'])} "
            f"byte_unique_ratio={byte_stats['byte_unique_ratio']:.6f} "
            f"byte_top1_ratio={byte_stats['byte_top1_ratio']:.6f} "
            f"byte_adj_repeat_ratio={byte_stats['byte_adj_repeat_ratio']:.6f} "
            f"byte_entropy_bits={byte_stats['byte_entropy_bits']:.6f}"
        )
        if debug_context:
            for k, v in debug_context.items():
                msg += f" {k}={v}"
        logger.info(msg)
        _log_comp_bit_event(tensor=tensor, debug_context=debug_context)
        _log_stride_pattern_event(tensor=tensor, raw_buffer=raw_buffer, debug_context=debug_context)
        _log_activation_distribution_event(tensor=tensor, debug_context=debug_context)
    except Exception:
        return


def _supports_byte_split_layout(
    tensor: Optional[torch.Tensor],
    compression_type: runtime_pb2.CompressionType,
    raw_size: int,
    debug_context: Optional[Dict[str, object]],
) -> bool:
    if _lossless_layout() != "byte_split":
        return False
    if _lossless_algo() != "zstd":
        return False
    if sys.byteorder != "little":
        return False
    if compression_type != runtime_pb2.CompressionType.NONE:
        return False
    if tensor is None or not torch.is_tensor(tensor):
        return False
    if tensor.dtype not in (torch.float16, torch.float32):
        return False
    if raw_size != tensor.numel() * tensor.element_size():
        return False
    return _context_matches_lossless_layout_target(debug_context)


def _split_high_byte_lane(raw: bytes, elem_size: int) -> tuple[bytes, bytes]:
    if elem_size == 2:
        return raw[1::2], raw[0::2]
    if elem_size == 4:
        extracted = raw[3::4]
        remaining = bytearray(len(raw) - len(extracted))
        remaining[0::3] = raw[0::4]
        remaining[1::3] = raw[1::4]
        remaining[2::3] = raw[2::4]
        return extracted, bytes(remaining)
    raise ValueError(f"Unsupported byte-split elem_size={elem_size}")


def _reconstruct_high_byte_lane(extracted: bytes, remaining: bytes, elem_size: int, original_size: int) -> bytes:
    if elem_size <= 0 or original_size % elem_size != 0:
        raise ValueError(f"Invalid byte-split original_size={original_size} elem_size={elem_size}")
    numel = original_size // elem_size
    extracted_size = numel
    remaining_size = original_size - extracted_size
    if len(extracted) != extracted_size:
        raise ValueError(f"Invalid extracted byte-split size: expected {extracted_size}, got {len(extracted)}")
    if len(remaining) != remaining_size:
        raise ValueError(f"Invalid remaining byte-split size: expected {remaining_size}, got {len(remaining)}")

    if elem_size == 2:
        out = bytearray(original_size)
        out[0::2] = remaining
        out[1::2] = extracted
        return bytes(out)

    if elem_size == 4:
        out = bytearray(original_size)
        out[0::4] = remaining[0::3]
        out[1::4] = remaining[1::3]
        out[2::4] = remaining[2::3]
        out[3::4] = extracted
        return bytes(out)

    raise ValueError(f"Unsupported byte-split elem_size={elem_size}")


@lru_cache(maxsize=16)
def _get_zstd_compressor(level: int):
    if _zstd is None:
        return None
    return _zstd.ZstdCompressor(level=level)


@lru_cache(maxsize=1)
def _get_zstd_decompressor():
    if _zstd is None:
        return None
    return _zstd.ZstdDecompressor()


def _compress_with_algo(raw: bytes, *, algo: str, level: int) -> tuple[int, bytes]:
    if algo in ("", "none", "off", "disabled"):
        return 0, raw

    if algo == "zstd":
        compressor = _get_zstd_compressor(level)
        if compressor is None:
            _warn_missing_zstd_once()
            return 0, raw
        t0 = time.perf_counter()
        compressed = compressor.compress(raw)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        _record_transport_profile("compress_calls", 1.0)
        _record_transport_profile("compress_ms", dt_ms)
        _record_transport_profile("compress_input_bytes", float(len(raw)))
        _record_transport_profile("compress_output_bytes", float(len(compressed)))
        return _ALGO_ZSTD, compressed

    if algo == "zlib":
        zlib_level = max(-1, min(level, 9))
        t0 = time.perf_counter()
        compressed = zlib.compress(raw, level=zlib_level)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        _record_transport_profile("compress_calls", 1.0)
        _record_transport_profile("compress_ms", dt_ms)
        _record_transport_profile("compress_input_bytes", float(len(raw)))
        _record_transport_profile("compress_output_bytes", float(len(compressed)))
        return _ALGO_ZLIB, compressed

    logger.warning(f"Unknown lossless wrapper algorithm={algo!r}, disabling wrapper compression")
    return 0, raw


def _compress_buffer(raw: bytes) -> tuple[int, bytes]:
    return _compress_with_algo(raw, algo=_lossless_algo(), level=_lossless_level())


def _build_zipnn_wrapper(raw: bytes, *, tensor: torch.Tensor) -> bytes:
    dtype_name = _zipnn_dtype_name(tensor)
    if dtype_name is None:
        raise ValueError("ZipNN wrapper requires float16/bfloat16/float32 tensor metadata")
    compressor = _get_zipnn_compressor(dtype_name, _lossless_level())
    if compressor is None:
        _warn_missing_zipnn_once()
        raise RuntimeError("ZipNN wrapper requires zipnn")
    t0 = time.perf_counter()
    payload = bytes(compressor.compress(raw))
    dt_ms = (time.perf_counter() - t0) * 1000.0
    _record_transport_profile("compress_calls", 1.0)
    _record_transport_profile("compress_ms", dt_ms)
    _record_transport_profile("compress_input_bytes", float(len(raw)))
    _record_transport_profile("compress_output_bytes", float(len(payload)))
    return _HEADER_STRUCT.pack(_MAGIC, _VERSION, _ALGO_ZIPNN, len(raw)) + payload


def _decompress_with_algo(algo_id: int, payload: bytes, original_size: int) -> bytes:
    t0 = time.perf_counter()
    if algo_id == _ALGO_ZSTD:
        decompressor = _get_zstd_decompressor()
        if decompressor is None:
            raise RuntimeError("Received zstd-wrapped tensor, but 'zstandard' is not installed")
        raw = decompressor.decompress(payload, max_output_size=original_size)
    elif algo_id == _ALGO_ZLIB:
        raw = zlib.decompress(payload)
    elif algo_id == _ALGO_ZIPNN:
        decompressor = _get_zipnn_decompressor()
        if decompressor is None:
            raise RuntimeError("Received ZipNN-wrapped tensor, but 'zipnn' is not installed")
        raw = bytes(decompressor.decompress(payload))
    else:
        raise ValueError(f"Unknown lossless wrapper algorithm id: {algo_id}")

    if len(raw) != original_size:
        raise ValueError(f"Lossless wrapper size mismatch: expected {original_size}, got {len(raw)}")
    dt_ms = (time.perf_counter() - t0) * 1000.0
    _record_transport_profile("decompress_calls", 1.0)
    _record_transport_profile("decompress_ms", dt_ms)
    _record_transport_profile("decompress_input_bytes", float(len(payload)))
    _record_transport_profile("decompress_output_bytes", float(len(raw)))
    return raw


def _decompress_buffer(algo_id: int, payload: bytes, original_size: int) -> bytes:
    return _decompress_with_algo(algo_id, payload, original_size)


def _build_plain_wrapper(raw: bytes) -> tuple[int, bytes]:
    algo_id, compressed = _compress_buffer(raw)
    if algo_id == 0:
        return 0, raw
    wrapped = _HEADER_STRUCT.pack(_MAGIC, _VERSION, algo_id, len(raw)) + compressed
    return algo_id, wrapped


def _build_zstd_byte_split_wrapper(raw: bytes, *, elem_size: int) -> bytes:
    extracted_raw, remaining_raw = _split_high_byte_lane(raw, elem_size)
    extracted_algo, extracted_comp = _compress_with_algo(extracted_raw, algo="zstd", level=_lossless_level())
    remaining_algo, remaining_comp = _compress_with_algo(remaining_raw, algo="zstd", level=_lossless_level())
    if extracted_algo != _ALGO_ZSTD or remaining_algo != _ALGO_ZSTD:
        raise RuntimeError("Byte-split wrapper requires zstd")

    payload = (
        _BYTE_SPLIT_PAYLOAD_STRUCT.pack(elem_size, len(extracted_comp))
        + extracted_comp
        + remaining_comp
    )
    return _HEADER_STRUCT.pack(_MAGIC, _VERSION, _ALGO_ZSTD_BYTE_SPLIT, len(raw)) + payload


def _decode_zstd_byte_split_payload(payload: bytes, original_size: int) -> bytes:
    if len(payload) < _BYTE_SPLIT_PAYLOAD_SIZE:
        raise ValueError("Byte-split payload is truncated")
    elem_size, extracted_comp_size = _BYTE_SPLIT_PAYLOAD_STRUCT.unpack_from(payload, 0)
    extracted_start = _BYTE_SPLIT_PAYLOAD_SIZE
    extracted_end = extracted_start + int(extracted_comp_size)
    if extracted_end > len(payload):
        raise ValueError("Byte-split payload extracted segment is truncated")

    extracted_comp = payload[extracted_start:extracted_end]
    remaining_comp = payload[extracted_end:]
    if original_size % max(1, elem_size) != 0:
        raise ValueError(f"Invalid byte-split size/original_size combination: {elem_size}, {original_size}")

    extracted_raw_size = original_size // elem_size
    remaining_raw_size = original_size - extracted_raw_size
    extracted_raw = _decompress_with_algo(_ALGO_ZSTD, extracted_comp, extracted_raw_size)
    remaining_raw = _decompress_with_algo(_ALGO_ZSTD, remaining_comp, remaining_raw_size)
    return _reconstruct_high_byte_lane(extracted_raw, remaining_raw, elem_size, original_size)


def _parse_wrapper(buffer: bytes, *, strict: bool = True):
    if len(buffer) < _HEADER_SIZE:
        return None
    if not buffer.startswith(_MAGIC):
        return None

    magic, version, algo_id, original_size = _HEADER_STRUCT.unpack_from(buffer, 0)
    if magic != _MAGIC:
        return None
    if version != _VERSION:
        if strict:
            raise ValueError(f"Unsupported lossless wrapper version: {version}")
        return None

    payload = buffer[_HEADER_SIZE:]
    return algo_id, original_size, payload


def _wrap_serialized_tensor_impl(
    serialized_tensor: runtime_pb2.Tensor,
    *,
    tensor: Optional[torch.Tensor] = None,
    compression_type: runtime_pb2.CompressionType = runtime_pb2.CompressionType.NONE,
    debug_context: Optional[Dict[str, object]] = None,
) -> tuple[runtime_pb2.Tensor, Dict[str, object]]:
    info: Dict[str, object] = {
        "applied": 0,
        "reason": "disabled",
        "algo_name": _lossless_algo(),
        "layout": "plain",
        "compressed_bytes": 0,
        "wrapped_bytes": len(serialized_tensor.buffer) if serialized_tensor is not None else 0,
        "net_gain_bytes": 0,
        "min_bytes": _lossless_min_bytes(),
        "min_gain_bytes": _lossless_min_gain_bytes(),
        "compress_elapsed_ms": 0.0,
        "wrapper_elapsed_ms": 0.0,
        "plain_wrapped_bytes": 0,
        "byte_split_wrapped_bytes": 0,
        "zipnn_wrapped_bytes": 0,
    }
    wrap_t0 = time.perf_counter()
    try:
        if not _lossless_send_enabled():
            info["reason"] = "disabled"
            return serialized_tensor, info

        raw = serialized_tensor.buffer
        if not raw:
            info["reason"] = "empty"
            return serialized_tensor, info
        if len(raw) < _lossless_min_bytes():
            info["reason"] = "below_min_bytes"
            return serialized_tensor, info
        if _parse_wrapper(raw, strict=False) is not None:
            info["reason"] = "already_wrapped"
            return serialized_tensor, info

        compress_t0 = time.perf_counter()
        candidates: List[tuple[str, int, bytes]] = []
        selected_algo = _lossless_algo()

        if selected_algo == "zipnn":
            if _supports_zipnn_transport(tensor, compression_type, len(raw), debug_context):
                try:
                    zipnn_wrapped = _build_zipnn_wrapper(raw, tensor=tensor)
                    candidates.append(("zipnn", _ALGO_ZIPNN, zipnn_wrapped))
                    info["zipnn_wrapped_bytes"] = int(len(zipnn_wrapped))
                except Exception:
                    info["zipnn_wrapped_bytes"] = 0
            if not candidates:
                info["reason"] = _zipnn_skip_reason(tensor, compression_type, len(raw), debug_context)
                return serialized_tensor, info
        else:
            plain_algo_id, plain_wrapped = _build_plain_wrapper(raw)
            if plain_algo_id != 0:
                candidates.append(("plain", plain_algo_id, plain_wrapped))
                info["plain_wrapped_bytes"] = int(len(plain_wrapped))

            if _supports_byte_split_layout(tensor, compression_type, len(raw), debug_context):
                try:
                    split_wrapped = _build_zstd_byte_split_wrapper(raw, elem_size=int(tensor.element_size()))
                    candidates.append(("byte_split", _ALGO_ZSTD_BYTE_SPLIT, split_wrapped))
                    info["byte_split_wrapped_bytes"] = int(len(split_wrapped))
                except Exception:
                    info["byte_split_wrapped_bytes"] = 0

        info["compress_elapsed_ms"] = (time.perf_counter() - compress_t0) * 1000.0
        if not candidates:
            info["reason"] = "compressor_not_applied"
            return serialized_tensor, info

        layout, algo_id, wrapped_buffer = min(candidates, key=lambda item: len(item[2]))
        info["layout"] = layout
        info["wrapped_bytes"] = int(len(wrapped_buffer))
        info["net_gain_bytes"] = int(max(0, len(raw) - len(wrapped_buffer)))
        info["compressed_bytes"] = int(max(0, len(wrapped_buffer) - _HEADER_SIZE))
        if algo_id == _ALGO_ZSTD_BYTE_SPLIT:
            info["algo_name"] = "zstd_byte_split"
        elif algo_id == _ALGO_ZIPNN:
            info["algo_name"] = "zipnn"
            info["layout"] = "zipnn"

        # Skip compression if it does not reduce payload enough to amortize header/CPU overhead.
        if len(wrapped_buffer) + _lossless_min_gain_bytes() >= len(raw):
            info["reason"] = "min_gain_not_met"
            return serialized_tensor, info

        wrapped = runtime_pb2.Tensor()
        wrapped.CopyFrom(serialized_tensor)
        wrapped.buffer = wrapped_buffer
        info["applied"] = 1
        info["reason"] = "applied"
        _record_transport_profile("serialize_wrapper_applied_calls", 1.0)
        return wrapped, info
    finally:
        wrapper_elapsed_ms = (time.perf_counter() - wrap_t0) * 1000.0
        info["wrapper_elapsed_ms"] = wrapper_elapsed_ms
        _record_transport_profile("serialize_wrapper_ms", wrapper_elapsed_ms)


def wrap_serialized_tensor(serialized_tensor: runtime_pb2.Tensor) -> runtime_pb2.Tensor:
    """
    Optionally wrap runtime_pb2.Tensor.buffer with a lossless compression header.
    This only affects transport bytes; tensor protobuf fields (dtype/shape/compression) stay intact.
    """
    wrapped, _ = _wrap_serialized_tensor_impl(serialized_tensor)
    return wrapped


def unwrap_serialized_tensor(serialized_tensor: runtime_pb2.Tensor) -> runtime_pb2.Tensor:
    """
    Backward-compatible transport decoder.
    - Wrapped tensor: decode and restore original buffer.
    - Legacy/raw tensor: returned unchanged.
    """
    unwrap_t0 = time.perf_counter()
    try:
        wrapped_buffer = serialized_tensor.buffer
        _record_transport_profile("deserialize_wire_bytes", float(len(wrapped_buffer)))

        parsed = _parse_wrapper(wrapped_buffer, strict=True)
        if parsed is None:
            _record_transport_profile("deserialize_raw_bytes", float(len(wrapped_buffer)))
            return serialized_tensor

        algo_id, original_size, payload = parsed
        if algo_id == _ALGO_ZSTD_BYTE_SPLIT:
            raw_buffer = _decode_zstd_byte_split_payload(payload, original_size)
        else:
            raw_buffer = _decompress_buffer(algo_id, payload, original_size)

        unwrapped = runtime_pb2.Tensor()
        unwrapped.CopyFrom(serialized_tensor)
        unwrapped.buffer = raw_buffer
        _record_transport_profile("deserialize_wrapper_applied_calls", 1.0)
        _record_transport_profile("deserialize_raw_bytes", float(len(raw_buffer)))
        return unwrapped
    finally:
        _record_transport_profile("deserialize_unwrap_ms", (time.perf_counter() - unwrap_t0) * 1000.0)


def serialize_torch_tensor(
    tensor: torch.Tensor,
    compression_type: runtime_pb2.CompressionType = runtime_pb2.CompressionType.NONE,
    info=None,
    allow_inplace: bool = False,
    debug_context: Optional[Dict[str, object]] = None,
    **kwargs,
) -> runtime_pb2.Tensor:
    effective_tensor = tensor
    effective_debug_context = dict(debug_context) if debug_context else None
    if _should_wire_truncate_fp16(tensor, debug_context):
        try:
            effective_tensor = tensor.to(torch.float16)
            if effective_debug_context is None:
                effective_debug_context = {}
            effective_debug_context["wire_cast"] = "fp32_to_fp16_truncate"
            effective_debug_context["wire_src_dtype"] = str(tensor.dtype).replace("torch.", "")
            effective_debug_context["wire_dst_dtype"] = "float16"
            if debug_context:
                logger.info(
                    "[WIRE_CAST] "
                    "action=fp32_to_fp16_truncate "
                    f"source={debug_context.get('source', '')} "
                    f"channel={debug_context.get('channel', '')} "
                    f"tensor_name={debug_context.get('tensor_name', '')} "
                    f"phase={debug_context.get('phase', '')} "
                    f"shape={tuple(int(dim) for dim in tensor.shape)}"
                )
        except Exception:
            effective_tensor = tensor

    t0 = time.perf_counter()
    serialized = _serialize_torch_tensor(
        effective_tensor,
        compression_type=compression_type,
        info=info,
        allow_inplace=allow_inplace,
        **kwargs,
    )
    _record_transport_profile("serialize_calls", 1.0)
    _record_transport_profile("serialize_core_ms", (time.perf_counter() - t0) * 1000.0)
    _record_transport_profile("serialize_raw_bytes", float(len(serialized.buffer)))
    wrapped, wrap_info = _wrap_serialized_tensor_impl(
        serialized,
        tensor=effective_tensor,
        compression_type=compression_type,
        debug_context=effective_debug_context,
    )
    if wrap_info.get("algo_name") == "zipnn":
        zipnn_info = _zipnn_info_from_selected_wrap(
            tensor=effective_tensor,
            raw_buffer=serialized.buffer,
            wrap_info=wrap_info,
        )
    else:
        zipnn_info = _profile_zipnn_candidate(
            tensor=effective_tensor,
            compression_type=compression_type,
            raw_buffer=serialized.buffer,
            selected_wire_bytes=len(wrapped.buffer),
            debug_context=effective_debug_context,
        )
    _record_transport_profile("serialize_wire_bytes", float(len(wrapped.buffer)))
    _record_comp_research_event(
        raw_bytes=len(serialized.buffer),
        wire_bytes=len(wrapped.buffer),
        wrap_info=wrap_info,
    )
    _log_comp_detail_event(
        tensor=effective_tensor,
        compression_type=compression_type,
        raw_buffer=serialized.buffer,
        wire_buffer=wrapped.buffer,
        wrap_info=wrap_info,
        zipnn_info=zipnn_info,
        debug_context=effective_debug_context,
    )
    _log_zipnn_compare_event(
        raw_bytes=len(serialized.buffer),
        wrap_info=wrap_info,
        zipnn_info=zipnn_info,
        debug_context=effective_debug_context,
    )
    return wrapped


def deserialize_torch_tensor(serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
    _record_transport_profile("deserialize_calls", 1.0)
    unwrapped = unwrap_serialized_tensor(serialized_tensor)
    t0 = time.perf_counter()
    tensor = _deserialize_torch_tensor(unwrapped)
    _record_transport_profile("deserialize_core_ms", (time.perf_counter() - t0) * 1000.0)
    return tensor


async def deserialize_tensor_stream(
    stream: AsyncIterator[Iterable[runtime_pb2.Tensor]],
) -> List[torch.Tensor]:
    """
    Streaming-compatible deserializer with transport wrapper support.
    """
    tensors: List[torch.Tensor] = []
    tensor_parts: List[runtime_pb2.Tensor] = []

    async for parts in stream:
        for part in parts:
            if part.dtype and tensor_parts:
                tensors.append(deserialize_torch_tensor(combine_from_streaming(tensor_parts)))
                tensor_parts = []
            tensor_parts.append(part)

    if tensor_parts:
        tensors.append(deserialize_torch_tensor(combine_from_streaming(tensor_parts)))

    return tensors
