from __future__ import annotations

import contextvars
import math
import os
import struct
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

try:
    from bloombee.utils import lossless_wrapper_config as _lossless_cfg
except Exception:
    _lossless_cfg = None

logger = get_logger(__name__)

try:
    import zstandard as _zstd
except Exception:
    _zstd = None


_MAGIC = b"BBLC"
_VERSION = 1
_ALGO_ZSTD = 1
_ALGO_ZLIB = 2
_HEADER_STRUCT = struct.Struct("!4sBBQ")
_HEADER_SIZE = _HEADER_STRUCT.size

_MISSING_ZSTD_WARNING_EMITTED = False
_COMP_PROFILE_ENV = "BLOOMBEE_COMP_RATIO_PROFILE"
_COMP_TIMING_PROFILE_ENV = "BLOOMBEE_COMP_TIMING_PROFILE"
_COMP_DETAIL_PROFILE_ENV = "BLOOMBEE_COMP_DETAIL_PROFILE"
_COMP_RESEARCH_PROFILE_ENV = "BLOOMBEE_COMP_RESEARCH_PROFILE"
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
    Default is enabled for research runs.
    """
    return _get_env_bool(_COMP_PROFILE_ENV, "1")


def comp_timing_profile_enabled() -> bool:
    """
    Enable detailed compression/decompression timing profiling.
    """
    return _get_env_bool(_COMP_TIMING_PROFILE_ENV, "1")


def comp_detail_profile_enabled() -> bool:
    """
    Enable heavyweight per-tensor compression diagnostics.
    Default is off because this can generate a lot of logs and a small amount of CPU overhead.
    """
    return _get_env_bool(_COMP_DETAIL_PROFILE_ENV, "0")


def comp_research_profile_enabled() -> bool:
    """
    Enable lightweight rolling compression scaling summaries.
    This is cheaper than COMP_DETAIL and intended for online cost/benefit analysis.
    """
    return _get_env_bool(_COMP_RESEARCH_PROFILE_ENV, "1")


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


def _log_comp_detail_event(
    *,
    tensor: Optional[torch.Tensor],
    compression_type: runtime_pb2.CompressionType,
    raw_buffer: bytes,
    wire_buffer: bytes,
    wrap_info: Dict[str, object],
    debug_context: Optional[Dict[str, object]] = None,
) -> None:
    if not comp_detail_profile_enabled():
        return
    try:
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
    except Exception:
        return


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


def _compress_buffer(raw: bytes) -> tuple[int, bytes]:
    algo = _lossless_algo()
    level = _lossless_level()

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

    logger.warning(f"Unknown BLOOMBEE_LOSSLESS_ALGO={algo!r}, disabling wrapper compression")
    return 0, raw


def _decompress_buffer(algo_id: int, payload: bytes, original_size: int) -> bytes:
    t0 = time.perf_counter()
    if algo_id == _ALGO_ZSTD:
        decompressor = _get_zstd_decompressor()
        if decompressor is None:
            raise RuntimeError("Received zstd-wrapped tensor, but 'zstandard' is not installed")
        raw = decompressor.decompress(payload, max_output_size=original_size)
    elif algo_id == _ALGO_ZLIB:
        raw = zlib.decompress(payload)
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
) -> tuple[runtime_pb2.Tensor, Dict[str, object]]:
    info: Dict[str, object] = {
        "applied": 0,
        "reason": "disabled",
        "algo_name": _lossless_algo(),
        "compressed_bytes": 0,
        "wrapped_bytes": len(serialized_tensor.buffer) if serialized_tensor is not None else 0,
        "net_gain_bytes": 0,
        "min_bytes": _lossless_min_bytes(),
        "min_gain_bytes": _lossless_min_gain_bytes(),
        "compress_elapsed_ms": 0.0,
        "wrapper_elapsed_ms": 0.0,
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
        algo_id, compressed = _compress_buffer(raw)
        info["compress_elapsed_ms"] = (time.perf_counter() - compress_t0) * 1000.0
        info["compressed_bytes"] = int(len(compressed))
        if algo_id == 0:
            info["reason"] = "compressor_not_applied"
            return serialized_tensor, info

        wrapped_buffer = _HEADER_STRUCT.pack(_MAGIC, _VERSION, algo_id, len(raw)) + compressed
        info["wrapped_bytes"] = int(len(wrapped_buffer))
        info["net_gain_bytes"] = int(max(0, len(raw) - len(wrapped_buffer)))

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
    t0 = time.perf_counter()
    serialized = _serialize_torch_tensor(
        tensor,
        compression_type=compression_type,
        info=info,
        allow_inplace=allow_inplace,
        **kwargs,
    )
    _record_transport_profile("serialize_calls", 1.0)
    _record_transport_profile("serialize_core_ms", (time.perf_counter() - t0) * 1000.0)
    _record_transport_profile("serialize_raw_bytes", float(len(serialized.buffer)))
    wrapped, wrap_info = _wrap_serialized_tensor_impl(serialized)
    _record_transport_profile("serialize_wire_bytes", float(len(wrapped.buffer)))
    _record_comp_research_event(
        raw_bytes=len(serialized.buffer),
        wire_bytes=len(wrapped.buffer),
        wrap_info=wrap_info,
    )
    _log_comp_detail_event(
        tensor=tensor,
        compression_type=compression_type,
        raw_buffer=serialized.buffer,
        wire_buffer=wrapped.buffer,
        wrap_info=wrap_info,
        debug_context=debug_context,
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
