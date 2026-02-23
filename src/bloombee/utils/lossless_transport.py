from __future__ import annotations

import contextvars
import os
import struct
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
_TRANSPORT_PROFILE_CTX: contextvars.ContextVar[Optional[Dict[str, float]]] = contextvars.ContextVar(
    "bloombee_transport_profile", default=None
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


def wrap_serialized_tensor(serialized_tensor: runtime_pb2.Tensor) -> runtime_pb2.Tensor:
    """
    Optionally wrap runtime_pb2.Tensor.buffer with a lossless compression header.
    This only affects transport bytes; tensor protobuf fields (dtype/shape/compression) stay intact.
    """
    wrap_t0 = time.perf_counter()
    try:
        if not _lossless_send_enabled():
            return serialized_tensor

        raw = serialized_tensor.buffer
        if not raw:
            return serialized_tensor
        if len(raw) < _lossless_min_bytes():
            return serialized_tensor
        if _parse_wrapper(raw, strict=False) is not None:
            return serialized_tensor

        algo_id, compressed = _compress_buffer(raw)
        if algo_id == 0:
            return serialized_tensor

        wrapped_buffer = _HEADER_STRUCT.pack(_MAGIC, _VERSION, algo_id, len(raw)) + compressed

        # Skip compression if it does not reduce payload enough to amortize header/CPU overhead.
        if len(wrapped_buffer) + _lossless_min_gain_bytes() >= len(raw):
            return serialized_tensor

        wrapped = runtime_pb2.Tensor()
        wrapped.CopyFrom(serialized_tensor)
        wrapped.buffer = wrapped_buffer
        _record_transport_profile("serialize_wrapper_applied_calls", 1.0)
        return wrapped
    finally:
        _record_transport_profile("serialize_wrapper_ms", (time.perf_counter() - wrap_t0) * 1000.0)


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
    wrapped = wrap_serialized_tensor(serialized)
    _record_transport_profile("serialize_wire_bytes", float(len(wrapped.buffer)))
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
