from __future__ import annotations

import os
import struct
import zlib
from functools import lru_cache
from typing import AsyncIterator, Iterable, List

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
        return _ALGO_ZSTD, compressor.compress(raw)

    if algo == "zlib":
        zlib_level = max(-1, min(level, 9))
        return _ALGO_ZLIB, zlib.compress(raw, level=zlib_level)

    logger.warning(f"Unknown BLOOMBEE_LOSSLESS_ALGO={algo!r}, disabling wrapper compression")
    return 0, raw


def _decompress_buffer(algo_id: int, payload: bytes, original_size: int) -> bytes:
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
    return wrapped


def unwrap_serialized_tensor(serialized_tensor: runtime_pb2.Tensor) -> runtime_pb2.Tensor:
    """
    Backward-compatible transport decoder.
    - Wrapped tensor: decode and restore original buffer.
    - Legacy/raw tensor: returned unchanged.
    """
    wrapped_buffer = serialized_tensor.buffer
    parsed = _parse_wrapper(wrapped_buffer, strict=True)
    if parsed is None:
        return serialized_tensor

    algo_id, original_size, payload = parsed
    raw_buffer = _decompress_buffer(algo_id, payload, original_size)

    unwrapped = runtime_pb2.Tensor()
    unwrapped.CopyFrom(serialized_tensor)
    unwrapped.buffer = raw_buffer
    return unwrapped


def serialize_torch_tensor(
    tensor: torch.Tensor,
    compression_type: runtime_pb2.CompressionType = runtime_pb2.CompressionType.NONE,
    info=None,
    allow_inplace: bool = False,
    **kwargs,
) -> runtime_pb2.Tensor:
    serialized = _serialize_torch_tensor(
        tensor,
        compression_type=compression_type,
        info=info,
        allow_inplace=allow_inplace,
        **kwargs,
    )
    return wrap_serialized_tensor(serialized)


def deserialize_torch_tensor(serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
    unwrapped = unwrap_serialized_tensor(serialized_tensor)
    return _deserialize_torch_tensor(unwrapped)


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
