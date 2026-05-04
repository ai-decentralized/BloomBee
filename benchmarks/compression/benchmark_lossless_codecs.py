#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

try:
    import zstandard as zstd
except Exception:
    zstd = None

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bloombee.utils import lossless_transport as lt

BBLC_HEADER_BYTES = lt._HEADER_SIZE
BYTE_SPLIT_PAYLOAD_HEADER_BYTES = lt._BYTE_SPLIT_PAYLOAD_SIZE
DEFAULT_CODECS = (
    "raw",
    "plain_zstd",
    "current_byte_split_zstd",
    "current_byte_split_zstd_reusectx",
    "zipnn",
    "bit_group_zstd",
    "byte_split_zstd_high_raw_low",
    "byte_split_selective_zstd",
    "sample_guided_selective_byte_split_zstd",
    "axis_transpose_byte_split_zstd",
    "blocked_axis_byte_split_zstd_32",
    "blocked_axis_byte_split_zstd_64",
    "blocked_axis_byte_split_zstd_128",
    "zstd_dict_byte_split_zstd",
    "token_xor_byte_split_zstd",
    "token_xor_zstd_high_raw_low",
    "adaptive_selector",
)
CSV_FIELDS = (
    "sample",
    "codec",
    "available",
    "reason",
    "roundtrip_ok",
    "raw_bytes",
    "wire_bytes",
    "ratio",
    "saved_bytes",
    "compress_ms",
    "decompress_ms",
    "break_even_bw_mbps",
    "dtype",
    "shape",
    "phase",
    "source",
    "direction",
    "channel",
    "level",
    "batch_size",
    "seq_len",
    "prompt_len",
    "blocks",
    "compute_dtype",
    "schema_dtype",
    "wire_dtype",
    "selected_codec",
)


@dataclass
class TensorSample:
    name: str
    path: Path
    tensor: torch.Tensor
    metadata: Dict[str, object]


@dataclass
class CodecResult:
    codec: str
    available: int
    reason: str
    roundtrip_ok: int
    raw_bytes: int
    wire_bytes: int
    compress_ms: float
    decompress_ms: float
    level: str = ""
    selected_codec: str = ""

    @property
    def ratio(self) -> float:
        return (float(self.wire_bytes) / float(self.raw_bytes)) if self.raw_bytes > 0 else 1.0

    @property
    def saved_bytes(self) -> int:
        return max(0, int(self.raw_bytes) - int(self.wire_bytes))

    @property
    def break_even_bw_mbps(self) -> float:
        total_ms = max(0.0, float(self.compress_ms) + float(self.decompress_ms))
        if self.saved_bytes <= 0 or total_ms <= 0.0:
            return 0.0
        return (float(self.saved_bytes) * 8.0) / (total_ms * 1000.0)


def tensor_to_raw_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().cpu().contiguous()
    return cpu.view(torch.uint8).numpy().tobytes()


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def parse_codecs(raw: str) -> List[str]:
    if raw.strip().lower() in ("", "all"):
        return list(DEFAULT_CODECS)
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_filter_set(raw: str) -> Optional[set[str]]:
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def shape_name(shape: Sequence[int]) -> str:
    return "x".join(str(int(dim)) for dim in shape)


def timed_median(fn, repeat: int) -> tuple[object, float]:
    timings: List[float] = []
    value = None
    for idx in range(max(1, int(repeat)) + 1):
        t0 = time.perf_counter()
        value = fn()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if idx > 0:
            timings.append(elapsed_ms)
    return value, float(statistics.median(timings or [0.0]))


def zstd_compress(raw: bytes, level: int, repeat: int) -> tuple[Optional[bytes], float, str]:
    if zstd is None:
        return None, 0.0, "zstd_unavailable"
    compressor = zstd.ZstdCompressor(level=level)
    compressed, elapsed = timed_median(lambda: compressor.compress(raw), repeat)
    return bytes(compressed), elapsed, "ok"


def zstd_decompress(payload: bytes, repeat: int) -> tuple[bytes, float]:
    decompressor = zstd.ZstdDecompressor()
    restored, elapsed = timed_median(lambda: decompressor.decompress(payload), repeat)
    return bytes(restored), elapsed


SELECTIVE_MODE_HEADER_BYTES = 1
TOKEN_XOR_PAYLOAD_HEADER_BYTES = 8
AXIS_LAYOUT_HEADER_BYTES = 8
ZSTD_DICT_ID_HEADER_BYTES = 4
SAMPLE_GUIDED_LOW_SAMPLE_BYTES = 64 * 1024
SAMPLE_GUIDED_LOW_RATIO_THRESHOLD = 0.98
ZSTD_DICT_SIZE_BYTES = 64 * 1024
ZSTD_DICT_CHUNK_BYTES = 16 * 1024
_ZSTD_DICT = None


def codec_raw(
    raw: bytes,
    repeat: int,
    level: int,
    dtype: torch.dtype,
    shape: Optional[Sequence[int]] = None,
) -> CodecResult:
    return CodecResult("raw", 1, "ok", 1, len(raw), len(raw), 0.0, 0.0)


def codec_plain_zstd(
    raw: bytes,
    repeat: int,
    level: int,
    dtype: torch.dtype,
    shape: Optional[Sequence[int]] = None,
) -> CodecResult:
    payload, compress_ms, reason = zstd_compress(raw, level, repeat)
    if payload is None:
        return CodecResult("plain_zstd", 0, reason, 0, len(raw), len(raw), 0.0, 0.0, str(level))
    restored, decompress_ms = zstd_decompress(payload, repeat)
    return CodecResult(
        "plain_zstd",
        1,
        "ok",
        int(restored == raw),
        len(raw),
        BBLC_HEADER_BYTES + len(payload),
        compress_ms,
        decompress_ms,
        str(level),
    )


def codec_current_byte_split_zstd(
    raw: bytes,
    repeat: int,
    level: int,
    dtype: torch.dtype,
    shape: Optional[Sequence[int]] = None,
) -> CodecResult:
    if dtype not in (torch.float16, torch.float32):
        return CodecResult("current_byte_split_zstd", 0, "unsupported_dtype", 0, len(raw), len(raw), 0.0, 0.0, str(level))
    if zstd is None:
        return CodecResult("current_byte_split_zstd", 0, "zstd_unavailable", 0, len(raw), len(raw), 0.0, 0.0, str(level))
    elem_size = torch.empty((), dtype=dtype).element_size()

    def compress_once():
        extracted_raw, remaining_raw = lt._split_high_byte_lane(raw, elem_size)
        extracted = zstd.ZstdCompressor(level=level).compress(extracted_raw)
        remaining = zstd.ZstdCompressor(level=level).compress(remaining_raw)
        return bytes(extracted), bytes(remaining)

    (extracted_payload, remaining_payload), compress_ms = timed_median(compress_once, repeat)

    def decompress_once():
        extracted_raw_size = len(raw) // elem_size
        remaining_raw_size = len(raw) - extracted_raw_size
        extracted_raw = zstd.ZstdDecompressor().decompress(extracted_payload, max_output_size=extracted_raw_size)
        remaining_raw = zstd.ZstdDecompressor().decompress(remaining_payload, max_output_size=remaining_raw_size)
        return lt._reconstruct_high_byte_lane(bytes(extracted_raw), bytes(remaining_raw), elem_size, len(raw))

    restored, decompress_ms = timed_median(decompress_once, repeat)
    wire_bytes = BBLC_HEADER_BYTES + BYTE_SPLIT_PAYLOAD_HEADER_BYTES + len(extracted_payload) + len(remaining_payload)
    return CodecResult(
        "current_byte_split_zstd",
        1,
        "ok",
        int(restored == raw),
        len(raw),
        wire_bytes,
        compress_ms,
        decompress_ms,
        str(level),
    )


def codec_current_byte_split_zstd_reusectx(
    raw: bytes,
    repeat: int,
    level: int,
    dtype: torch.dtype,
    shape: Optional[Sequence[int]] = None,
) -> CodecResult:
    if dtype not in (torch.float16, torch.float32):
        return CodecResult(
            "current_byte_split_zstd_reusectx", 0, "unsupported_dtype", 0, len(raw), len(raw), 0.0, 0.0, str(level)
        )
    if zstd is None:
        return CodecResult(
            "current_byte_split_zstd_reusectx", 0, "zstd_unavailable", 0, len(raw), len(raw), 0.0, 0.0, str(level)
        )
    elem_size = torch.empty((), dtype=dtype).element_size()

    def compress_once():
        extracted_raw, remaining_raw = lt._split_high_byte_lane(raw, elem_size)
        compressor = zstd.ZstdCompressor(level=level)
        extracted = compressor.compress(extracted_raw)
        remaining = compressor.compress(remaining_raw)
        return bytes(extracted), bytes(remaining)

    (extracted_payload, remaining_payload), compress_ms = timed_median(compress_once, repeat)

    def decompress_once():
        extracted_raw_size = len(raw) // elem_size
        remaining_raw_size = len(raw) - extracted_raw_size
        decompressor = zstd.ZstdDecompressor()
        extracted_raw = decompressor.decompress(extracted_payload, max_output_size=extracted_raw_size)
        remaining_raw = decompressor.decompress(remaining_payload, max_output_size=remaining_raw_size)
        return lt._reconstruct_high_byte_lane(bytes(extracted_raw), bytes(remaining_raw), elem_size, len(raw))

    restored, decompress_ms = timed_median(decompress_once, repeat)
    wire_bytes = BBLC_HEADER_BYTES + BYTE_SPLIT_PAYLOAD_HEADER_BYTES + len(extracted_payload) + len(remaining_payload)
    return CodecResult(
        "current_byte_split_zstd_reusectx",
        1,
        "ok",
        int(restored == raw),
        len(raw),
        wire_bytes,
        compress_ms,
        decompress_ms,
        str(level),
    )


def _byte_split_selective_result(
    codec_name: str,
    raw: bytes,
    repeat: int,
    level: int,
    dtype: torch.dtype,
    *,
    compress_remaining: bool,
    probe_remaining: bool,
    payload_header_bytes: int = 0,
) -> CodecResult:
    if dtype not in (torch.float16, torch.float32):
        return CodecResult(codec_name, 0, "unsupported_dtype", 0, len(raw), len(raw), 0.0, 0.0, str(level))
    if zstd is None:
        return CodecResult(codec_name, 0, "zstd_unavailable", 0, len(raw), len(raw), 0.0, 0.0, str(level))
    elem_size = torch.empty((), dtype=dtype).element_size()

    def compress_once():
        extracted_raw, remaining_raw = lt._split_high_byte_lane(raw, elem_size)
        compressor = zstd.ZstdCompressor(level=level)
        extracted_payload = bytes(compressor.compress(extracted_raw))
        if compress_remaining or probe_remaining:
            remaining_zstd = bytes(compressor.compress(remaining_raw))
        else:
            remaining_zstd = b""
        use_remaining_zstd = bool(compress_remaining)
        if probe_remaining:
            use_remaining_zstd = len(remaining_zstd) < len(remaining_raw)
        remaining_payload = remaining_zstd if use_remaining_zstd else remaining_raw
        return extracted_payload, remaining_payload, use_remaining_zstd

    (extracted_payload, remaining_payload, remaining_is_zstd), compress_ms = timed_median(compress_once, repeat)

    def decompress_once():
        extracted_raw_size = len(raw) // elem_size
        extracted_raw = zstd.ZstdDecompressor().decompress(extracted_payload, max_output_size=extracted_raw_size)
        if remaining_is_zstd:
            remaining_raw_size = len(raw) - extracted_raw_size
            remaining_raw = zstd.ZstdDecompressor().decompress(remaining_payload, max_output_size=remaining_raw_size)
        else:
            remaining_raw = remaining_payload
        return lt._reconstruct_high_byte_lane(bytes(extracted_raw), bytes(remaining_raw), elem_size, len(raw))

    restored, decompress_ms = timed_median(decompress_once, repeat)
    wire_bytes = (
        BBLC_HEADER_BYTES
        + BYTE_SPLIT_PAYLOAD_HEADER_BYTES
        + SELECTIVE_MODE_HEADER_BYTES
        + payload_header_bytes
        + len(extracted_payload)
        + len(remaining_payload)
    )
    return CodecResult(
        codec_name,
        1,
        "ok",
        int(restored == raw),
        len(raw),
        wire_bytes,
        compress_ms,
        decompress_ms,
        str(level),
    )


def codec_byte_split_zstd_high_raw_low(
    raw: bytes,
    repeat: int,
    level: int,
    dtype: torch.dtype,
    shape: Optional[Sequence[int]] = None,
) -> CodecResult:
    return _byte_split_selective_result(
        "byte_split_zstd_high_raw_low",
        raw,
        repeat,
        level,
        dtype,
        compress_remaining=False,
        probe_remaining=False,
    )


def codec_byte_split_selective_zstd(
    raw: bytes,
    repeat: int,
    level: int,
    dtype: torch.dtype,
    shape: Optional[Sequence[int]] = None,
) -> CodecResult:
    return _byte_split_selective_result(
        "byte_split_selective_zstd",
        raw,
        repeat,
        level,
        dtype,
        compress_remaining=False,
        probe_remaining=True,
    )


def _shape3(shape: Optional[Sequence[int]]) -> Optional[tuple[int, int, int]]:
    if not shape or len(shape) != 3:
        return None
    b, t, h = (int(dim) for dim in shape)
    if b <= 0 or t <= 0 or h <= 0:
        return None
    return b, t, h


def _axis_transpose_lane(lane: bytes, shape: tuple[int, int, int]) -> bytes:
    b, t, h = shape
    arr = np.frombuffer(lane, dtype=np.uint8).reshape(b, t, h)
    return np.ascontiguousarray(arr.transpose(0, 2, 1)).tobytes()


def _axis_untranspose_lane(lane: bytes, shape: tuple[int, int, int]) -> bytes:
    b, t, h = shape
    arr = np.frombuffer(lane, dtype=np.uint8).reshape(b, h, t)
    return np.ascontiguousarray(arr.transpose(0, 2, 1)).tobytes()


def _blocked_axis_transpose_lane(lane: bytes, shape: tuple[int, int, int], block: int) -> bytes:
    b, t, h = shape
    arr = np.frombuffer(lane, dtype=np.uint8).reshape(b, t, h)
    chunks = [
        np.ascontiguousarray(arr[:, :, start : min(start + block, h)].transpose(0, 2, 1)).reshape(-1)
        for start in range(0, h, block)
    ]
    return np.concatenate(chunks).tobytes() if chunks else b""


def _blocked_axis_untranspose_lane(lane: bytes, shape: tuple[int, int, int], block: int) -> bytes:
    b, t, h = shape
    out = np.empty((b, t, h), dtype=np.uint8)
    offset = 0
    for start in range(0, h, block):
        width = min(block, h - start)
        size = b * width * t
        chunk = np.frombuffer(lane, dtype=np.uint8, count=size, offset=offset).reshape(b, width, t)
        out[:, :, start : start + width] = chunk.transpose(0, 2, 1)
        offset += size
    return np.ascontiguousarray(out).tobytes()


def _axis_layout_byte_split_result(
    codec_name: str,
    raw: bytes,
    repeat: int,
    level: int,
    dtype: torch.dtype,
    shape: Optional[Sequence[int]],
    *,
    block: Optional[int] = None,
) -> CodecResult:
    shape3 = _shape3(shape)
    if dtype != torch.float16:
        return CodecResult(codec_name, 0, "unsupported_dtype", 0, len(raw), len(raw), 0.0, 0.0, str(level))
    if shape3 is None:
        return CodecResult(codec_name, 0, "unsupported_shape", 0, len(raw), len(raw), 0.0, 0.0, str(level))
    if zstd is None:
        return CodecResult(codec_name, 0, "zstd_unavailable", 0, len(raw), len(raw), 0.0, 0.0, str(level))

    def transform_lane(lane: bytes) -> bytes:
        if block is None:
            return _axis_transpose_lane(lane, shape3)
        return _blocked_axis_transpose_lane(lane, shape3, block)

    def inverse_lane(lane: bytes) -> bytes:
        if block is None:
            return _axis_untranspose_lane(lane, shape3)
        return _blocked_axis_untranspose_lane(lane, shape3, block)

    def compress_once():
        high_raw, low_raw = lt._split_high_byte_lane(raw, 2)
        compressor = zstd.ZstdCompressor(level=level)
        high_payload = bytes(compressor.compress(transform_lane(high_raw)))
        low_payload = bytes(compressor.compress(transform_lane(low_raw)))
        return high_payload, low_payload

    (high_payload, low_payload), compress_ms = timed_median(compress_once, repeat)

    def decompress_once():
        lane_size = len(raw) // 2
        decompressor = zstd.ZstdDecompressor()
        high_axis = decompressor.decompress(high_payload, max_output_size=lane_size)
        low_axis = decompressor.decompress(low_payload, max_output_size=lane_size)
        high_raw = inverse_lane(bytes(high_axis))
        low_raw = inverse_lane(bytes(low_axis))
        return lt._reconstruct_high_byte_lane(high_raw, low_raw, 2, len(raw))

    restored, decompress_ms = timed_median(decompress_once, repeat)
    return CodecResult(
        codec_name,
        1,
        "ok",
        int(restored == raw),
        len(raw),
        BBLC_HEADER_BYTES + BYTE_SPLIT_PAYLOAD_HEADER_BYTES + AXIS_LAYOUT_HEADER_BYTES + len(high_payload) + len(low_payload),
        compress_ms,
        decompress_ms,
        str(level),
    )


def codec_axis_transpose_byte_split_zstd(
    raw: bytes,
    repeat: int,
    level: int,
    dtype: torch.dtype,
    shape: Optional[Sequence[int]] = None,
) -> CodecResult:
    return _axis_layout_byte_split_result("axis_transpose_byte_split_zstd", raw, repeat, level, dtype, shape)


def _blocked_axis_codec(
    codec_name: str,
    raw: bytes,
    repeat: int,
    level: int,
    dtype: torch.dtype,
    shape: Optional[Sequence[int]],
    block: int,
) -> CodecResult:
    return _axis_layout_byte_split_result(codec_name, raw, repeat, level, dtype, shape, block=block)


def codec_blocked_axis_byte_split_zstd(
    raw: bytes,
    repeat: int,
    level: int,
    dtype: torch.dtype,
    shape: Optional[Sequence[int]] = None,
) -> CodecResult:
    return _blocked_axis_codec("blocked_axis_byte_split_zstd", raw, repeat, level, dtype, shape, 64)


def codec_blocked_axis_byte_split_zstd_32(
    raw: bytes,
    repeat: int,
    level: int,
    dtype: torch.dtype,
    shape: Optional[Sequence[int]] = None,
) -> CodecResult:
    return _blocked_axis_codec("blocked_axis_byte_split_zstd_32", raw, repeat, level, dtype, shape, 32)


def codec_blocked_axis_byte_split_zstd_64(
    raw: bytes,
    repeat: int,
    level: int,
    dtype: torch.dtype,
    shape: Optional[Sequence[int]] = None,
) -> CodecResult:
    return _blocked_axis_codec("blocked_axis_byte_split_zstd_64", raw, repeat, level, dtype, shape, 64)


def codec_blocked_axis_byte_split_zstd_128(
    raw: bytes,
    repeat: int,
    level: int,
    dtype: torch.dtype,
    shape: Optional[Sequence[int]] = None,
) -> CodecResult:
    return _blocked_axis_codec("blocked_axis_byte_split_zstd_128", raw, repeat, level, dtype, shape, 128)


def codec_sample_guided_selective_byte_split_zstd(
    raw: bytes,
    repeat: int,
    level: int,
    dtype: torch.dtype,
    shape: Optional[Sequence[int]] = None,
) -> CodecResult:
    if dtype not in (torch.float16, torch.float32):
        return CodecResult("sample_guided_selective_byte_split_zstd", 0, "unsupported_dtype", 0, len(raw), len(raw), 0.0, 0.0, str(level))
    if zstd is None:
        return CodecResult("sample_guided_selective_byte_split_zstd", 0, "zstd_unavailable", 0, len(raw), len(raw), 0.0, 0.0, str(level))
    elem_size = torch.empty((), dtype=dtype).element_size()

    def compress_once():
        high_raw, low_raw = lt._split_high_byte_lane(raw, elem_size)
        compressor = zstd.ZstdCompressor(level=level)
        high_payload = bytes(compressor.compress(high_raw))
        sample = low_raw[: min(SAMPLE_GUIDED_LOW_SAMPLE_BYTES, len(low_raw))]
        sample_payload = bytes(compressor.compress(sample))
        use_low_zstd = len(sample_payload) < int(len(sample) * SAMPLE_GUIDED_LOW_RATIO_THRESHOLD)
        if use_low_zstd:
            low_payload = sample_payload if len(sample) == len(low_raw) else bytes(compressor.compress(low_raw))
        else:
            low_payload = low_raw
        return high_payload, low_payload, use_low_zstd

    (high_payload, low_payload, low_is_zstd), compress_ms = timed_median(compress_once, repeat)

    def decompress_once():
        high_raw_size = len(raw) // elem_size
        decompressor = zstd.ZstdDecompressor()
        high_raw = decompressor.decompress(high_payload, max_output_size=high_raw_size)
        if low_is_zstd:
            low_raw_size = len(raw) - high_raw_size
            low_raw = decompressor.decompress(low_payload, max_output_size=low_raw_size)
        else:
            low_raw = low_payload
        return lt._reconstruct_high_byte_lane(bytes(high_raw), bytes(low_raw), elem_size, len(raw))

    restored, decompress_ms = timed_median(decompress_once, repeat)
    wire_bytes = (
        BBLC_HEADER_BYTES
        + BYTE_SPLIT_PAYLOAD_HEADER_BYTES
        + SELECTIVE_MODE_HEADER_BYTES
        + len(high_payload)
        + len(low_payload)
    )
    return CodecResult(
        "sample_guided_selective_byte_split_zstd",
        1,
        "ok",
        int(restored == raw),
        len(raw),
        wire_bytes,
        compress_ms,
        decompress_ms,
        str(level),
    )


def _token_xor_raw(raw: bytes, shape: Optional[Sequence[int]], dtype: torch.dtype) -> tuple[Optional[bytes], str]:
    if dtype != torch.float16:
        return None, "unsupported_dtype"
    if not shape or len(shape) < 3:
        return None, "unsupported_shape"
    shape_tuple = tuple(int(dim) for dim in shape)
    if any(dim <= 0 for dim in shape_tuple):
        return None, "unsupported_shape"
    words = np.frombuffer(raw, dtype=np.uint16)
    expected = int(np.prod(shape_tuple))
    if expected != int(words.size):
        return None, "shape_mismatch"
    arr = words.reshape(shape_tuple)
    residual = arr.copy()
    if shape_tuple[1] > 1:
        residual[:, 1:, ...] = np.bitwise_xor(arr[:, 1:, ...], arr[:, :-1, ...])
    return residual.astype(np.uint16, copy=False).tobytes(), "ok"


def _token_xor_restore(residual_raw: bytes, shape: Sequence[int], dtype: torch.dtype) -> bytes:
    if dtype != torch.float16:
        raise ValueError("token_xor currently supports float16 only")
    shape_tuple = tuple(int(dim) for dim in shape)
    residual = np.frombuffer(residual_raw, dtype=np.uint16).reshape(shape_tuple)
    words = residual.copy()
    for token_idx in range(1, shape_tuple[1]):
        words[:, token_idx, ...] = np.bitwise_xor(words[:, token_idx, ...], words[:, token_idx - 1, ...])
    return words.astype(np.uint16, copy=False).tobytes()


def codec_token_xor_byte_split_zstd(
    raw: bytes,
    repeat: int,
    level: int,
    dtype: torch.dtype,
    shape: Optional[Sequence[int]] = None,
) -> CodecResult:
    residual_raw, reason = _token_xor_raw(raw, shape, dtype)
    if residual_raw is None:
        return CodecResult("token_xor_byte_split_zstd", 0, reason, 0, len(raw), len(raw), 0.0, 0.0, str(level))
    if zstd is None:
        return CodecResult("token_xor_byte_split_zstd", 0, "zstd_unavailable", 0, len(raw), len(raw), 0.0, 0.0, str(level))
    elem_size = torch.empty((), dtype=dtype).element_size()

    def compress_once():
        extracted_raw, remaining_raw = lt._split_high_byte_lane(residual_raw, elem_size)
        compressor = zstd.ZstdCompressor(level=level)
        return bytes(compressor.compress(extracted_raw)), bytes(compressor.compress(remaining_raw))

    (extracted_payload, remaining_payload), compress_ms = timed_median(compress_once, repeat)

    def decompress_once():
        extracted_raw_size = len(residual_raw) // elem_size
        remaining_raw_size = len(residual_raw) - extracted_raw_size
        decompressor = zstd.ZstdDecompressor()
        extracted_raw = decompressor.decompress(extracted_payload, max_output_size=extracted_raw_size)
        remaining_raw = decompressor.decompress(remaining_payload, max_output_size=remaining_raw_size)
        restored_residual = lt._reconstruct_high_byte_lane(bytes(extracted_raw), bytes(remaining_raw), elem_size, len(raw))
        return _token_xor_restore(restored_residual, shape or (), dtype)

    restored, decompress_ms = timed_median(decompress_once, repeat)
    wire_bytes = (
        BBLC_HEADER_BYTES
        + TOKEN_XOR_PAYLOAD_HEADER_BYTES
        + BYTE_SPLIT_PAYLOAD_HEADER_BYTES
        + len(extracted_payload)
        + len(remaining_payload)
    )
    return CodecResult(
        "token_xor_byte_split_zstd",
        1,
        "ok",
        int(restored == raw),
        len(raw),
        wire_bytes,
        compress_ms,
        decompress_ms,
        str(level),
    )


def codec_token_xor_zstd_high_raw_low(
    raw: bytes,
    repeat: int,
    level: int,
    dtype: torch.dtype,
    shape: Optional[Sequence[int]] = None,
) -> CodecResult:
    residual_raw, reason = _token_xor_raw(raw, shape, dtype)
    if residual_raw is None:
        return CodecResult("token_xor_zstd_high_raw_low", 0, reason, 0, len(raw), len(raw), 0.0, 0.0, str(level))
    if zstd is None:
        return CodecResult("token_xor_zstd_high_raw_low", 0, "zstd_unavailable", 0, len(raw), len(raw), 0.0, 0.0, str(level))

    elem_size = torch.empty((), dtype=dtype).element_size()

    def compress_once():
        extracted_raw, remaining_raw = lt._split_high_byte_lane(residual_raw, elem_size)
        extracted_payload = bytes(zstd.ZstdCompressor(level=level).compress(extracted_raw))
        return extracted_payload, remaining_raw

    (extracted_payload, remaining_payload), compress_ms = timed_median(compress_once, repeat)

    def decompress_once():
        extracted_raw_size = len(residual_raw) // elem_size
        extracted_raw = zstd.ZstdDecompressor().decompress(extracted_payload, max_output_size=extracted_raw_size)
        restored_residual = lt._reconstruct_high_byte_lane(bytes(extracted_raw), bytes(remaining_payload), elem_size, len(raw))
        return _token_xor_restore(restored_residual, shape or (), dtype)

    restored, decompress_ms = timed_median(decompress_once, repeat)
    wire_bytes = (
        BBLC_HEADER_BYTES
        + TOKEN_XOR_PAYLOAD_HEADER_BYTES
        + BYTE_SPLIT_PAYLOAD_HEADER_BYTES
        + SELECTIVE_MODE_HEADER_BYTES
        + len(extracted_payload)
        + len(remaining_payload)
    )
    return CodecResult(
        "token_xor_zstd_high_raw_low",
        1,
        "ok",
        int(restored == raw),
        len(raw),
        wire_bytes,
        compress_ms,
        decompress_ms,
        str(level),
    )


def _word_view(raw: bytes, dtype: torch.dtype) -> tuple[Optional[np.ndarray], int, int, int]:
    if dtype == torch.float16:
        return np.frombuffer(raw, dtype=np.uint16), 16, 5, 10
    if dtype == torch.bfloat16:
        return np.frombuffer(raw, dtype=np.uint16), 16, 8, 7
    if dtype == torch.float32:
        return np.frombuffer(raw, dtype=np.uint32), 32, 8, 23
    return None, 0, 0, 0


def _pack_values(values: np.ndarray, bits: int) -> bytes:
    if bits <= 0 or values.size == 0:
        return b""
    values = values.astype(np.uint64, copy=False).reshape(-1)
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint64)
    packed_bits = ((values[:, None] >> shifts[None, :]) & 1).astype(np.uint8, copy=False)
    return np.packbits(packed_bits.reshape(-1), bitorder="big").tobytes()


def _unpack_values(data: bytes, numel: int, bits: int) -> np.ndarray:
    if bits <= 0:
        return np.zeros(numel, dtype=np.uint64)
    bit_arr = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")[: numel * bits]
    bit_arr = bit_arr.reshape(numel, bits)
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint64)
    out = np.zeros(numel, dtype=np.uint64)
    for idx, shift in enumerate(shifts):
        out |= bit_arr[:, idx].astype(np.uint64, copy=False) << shift
    return out


def _split_bit_groups(raw: bytes, dtype: torch.dtype) -> tuple[Optional[List[tuple[str, int, bytes]]], Dict[str, int]]:
    words, elem_bits, exp_bits, mantissa_bits = _word_view(raw, dtype)
    if words is None:
        return None, {}
    numel = int(words.size)
    words64 = words.astype(np.uint64, copy=False)
    sign = (words64 >> (elem_bits - 1)) & 0x1
    exponent = (words64 >> mantissa_bits) & ((1 << exp_bits) - 1)
    mantissa = words64 & ((1 << mantissa_bits) - 1)

    if dtype == torch.float32:
        groups = [
            ("sign", 1, _pack_values(sign, 1)),
            ("exponent", exp_bits, _pack_values(exponent, exp_bits)),
            ("mantissa_high", 8, _pack_values(mantissa >> 15, 8)),
            ("mantissa_mid", 8, _pack_values((mantissa >> 7) & 0xFF, 8)),
            ("mantissa_low", 7, _pack_values(mantissa & 0x7F, 7)),
        ]
    else:
        high_bits = min(4, mantissa_bits)
        low_bits = mantissa_bits - high_bits
        groups = [
            ("sign", 1, _pack_values(sign, 1)),
            ("exponent", exp_bits, _pack_values(exponent, exp_bits)),
            ("mantissa_high", high_bits, _pack_values(mantissa >> low_bits, high_bits)),
            ("mantissa_low", low_bits, _pack_values(mantissa & ((1 << low_bits) - 1), low_bits)),
        ]
    return groups, {"numel": numel, "elem_bits": elem_bits, "exp_bits": exp_bits, "mantissa_bits": mantissa_bits}


def _reconstruct_bit_groups(group_payloads: Dict[str, tuple[int, bytes]], meta: Dict[str, int], dtype: torch.dtype) -> bytes:
    numel = int(meta["numel"])
    elem_bits = int(meta["elem_bits"])
    exp_bits = int(meta["exp_bits"])
    mantissa_bits = int(meta["mantissa_bits"])
    sign = _unpack_values(group_payloads["sign"][1], numel, group_payloads["sign"][0])
    exponent = _unpack_values(group_payloads["exponent"][1], numel, group_payloads["exponent"][0])
    if dtype == torch.float32:
        mantissa = (
            (_unpack_values(group_payloads["mantissa_high"][1], numel, 8) << 15)
            | (_unpack_values(group_payloads["mantissa_mid"][1], numel, 8) << 7)
            | _unpack_values(group_payloads["mantissa_low"][1], numel, 7)
        )
    else:
        high_bits = group_payloads["mantissa_high"][0]
        low_bits = group_payloads["mantissa_low"][0]
        mantissa = (
            _unpack_values(group_payloads["mantissa_high"][1], numel, high_bits) << low_bits
        ) | _unpack_values(group_payloads["mantissa_low"][1], numel, low_bits)
    words = (sign << (elem_bits - 1)) | (exponent << mantissa_bits) | mantissa
    out_dtype = np.uint32 if dtype == torch.float32 else np.uint16
    return words.astype(out_dtype, copy=False).tobytes()


def codec_bit_group_zstd(
    raw: bytes,
    repeat: int,
    level: int,
    dtype: torch.dtype,
    shape: Optional[Sequence[int]] = None,
) -> CodecResult:
    groups, meta = _split_bit_groups(raw, dtype)
    if groups is None:
        return CodecResult("bit_group_zstd", 0, "unsupported_dtype", 0, len(raw), len(raw), 0.0, 0.0, str(level))
    if zstd is None:
        return CodecResult("bit_group_zstd", 0, "zstd_unavailable", 0, len(raw), len(raw), 0.0, 0.0, str(level))

    def compress_once():
        compressed = {}
        for name, bits, payload in groups:
            compressed[name] = (bits, bytes(zstd.ZstdCompressor(level=level).compress(payload)))
        return compressed

    compressed_groups, compress_ms = timed_median(compress_once, repeat)

    def decompress_once():
        restored_groups = {}
        for name, (bits, payload) in compressed_groups.items():
            restored_groups[name] = (bits, bytes(zstd.ZstdDecompressor().decompress(payload)))
        return _reconstruct_bit_groups(restored_groups, meta, dtype)

    restored, decompress_ms = timed_median(decompress_once, repeat)
    group_header_bytes = 8 * len(compressed_groups)
    wire_bytes = BBLC_HEADER_BYTES + group_header_bytes + sum(len(payload) for _, payload in compressed_groups.values())
    return CodecResult(
        "bit_group_zstd",
        1,
        "ok",
        int(restored == raw),
        len(raw),
        wire_bytes,
        compress_ms,
        decompress_ms,
        str(level),
    )


def codec_zipnn(
    raw: bytes,
    repeat: int,
    level: int,
    dtype: torch.dtype,
    shape: Optional[Sequence[int]] = None,
) -> CodecResult:
    timings_c: List[float] = []
    timings_d: List[float] = []
    info: Dict[str, object] = {}
    for idx in range(max(1, int(repeat)) + 1):
        info = lt.zipnn_oracle_roundtrip(raw, dtype, level=level)
        if idx > 0:
            timings_c.append(float(info.get("compress_ms", 0.0)))
            timings_d.append(float(info.get("decompress_ms", 0.0)))
    available = int(info.get("available", 0)) and int(info.get("supported", 0))
    if not available:
        return CodecResult("zipnn", int(info.get("available", 0)), str(info.get("reason", "unavailable")), 0, len(raw), len(raw), 0.0, 0.0, str(level))
    return CodecResult(
        "zipnn",
        1,
        str(info.get("reason", "ok")),
        int(info.get("roundtrip_ok", 0)),
        len(raw),
        int(info.get("wire_bytes", 0)),
        float(statistics.median(timings_c or [float(info.get("compress_ms", 0.0))])),
        float(statistics.median(timings_d or [float(info.get("decompress_ms", 0.0))])),
        str(level),
    )


def _iter_zstd_dict_chunks(samples: Sequence[TensorSample]) -> List[bytes]:
    chunks: List[bytes] = []
    for sample in samples:
        dtype = sample.tensor.dtype
        if dtype not in (torch.float16, torch.float32):
            continue
        raw = tensor_to_raw_bytes(sample.tensor)
        elem_size = torch.empty((), dtype=dtype).element_size()
        try:
            lanes = lt._split_high_byte_lane(raw, elem_size)
        except Exception:
            continue
        for lane in lanes:
            for offset in range(0, len(lane), ZSTD_DICT_CHUNK_BYTES):
                chunk = lane[offset : offset + ZSTD_DICT_CHUNK_BYTES]
                if len(chunk) >= 1024:
                    chunks.append(chunk)
    return chunks


def configure_zstd_dict_from_samples(samples: Sequence[TensorSample]) -> bool:
    global _ZSTD_DICT
    _ZSTD_DICT = None
    if zstd is None:
        return False
    chunks = _iter_zstd_dict_chunks(samples)
    total = sum(len(chunk) for chunk in chunks)
    if len(chunks) < 8 or total < 8192:
        return False
    dict_size = min(ZSTD_DICT_SIZE_BYTES, max(1024, total // 8))
    try:
        _ZSTD_DICT = zstd.train_dictionary(dict_size, chunks)
    except Exception:
        _ZSTD_DICT = None
    return _ZSTD_DICT is not None


def codec_zstd_dict_byte_split_zstd(
    raw: bytes,
    repeat: int,
    level: int,
    dtype: torch.dtype,
    shape: Optional[Sequence[int]] = None,
) -> CodecResult:
    if dtype not in (torch.float16, torch.float32):
        return CodecResult("zstd_dict_byte_split_zstd", 0, "unsupported_dtype", 0, len(raw), len(raw), 0.0, 0.0, str(level))
    if zstd is None:
        return CodecResult("zstd_dict_byte_split_zstd", 0, "zstd_unavailable", 0, len(raw), len(raw), 0.0, 0.0, str(level))
    if _ZSTD_DICT is None:
        return CodecResult("zstd_dict_byte_split_zstd", 0, "dict_unavailable", 0, len(raw), len(raw), 0.0, 0.0, str(level))
    elem_size = torch.empty((), dtype=dtype).element_size()

    def compress_once():
        high_raw, low_raw = lt._split_high_byte_lane(raw, elem_size)
        compressor = zstd.ZstdCompressor(level=level, dict_data=_ZSTD_DICT)
        high_payload = bytes(compressor.compress(high_raw))
        low_payload = bytes(compressor.compress(low_raw))
        return high_payload, low_payload

    (high_payload, low_payload), compress_ms = timed_median(compress_once, repeat)

    def decompress_once():
        high_raw_size = len(raw) // elem_size
        low_raw_size = len(raw) - high_raw_size
        decompressor = zstd.ZstdDecompressor(dict_data=_ZSTD_DICT)
        high_raw = decompressor.decompress(high_payload, max_output_size=high_raw_size)
        low_raw = decompressor.decompress(low_payload, max_output_size=low_raw_size)
        return lt._reconstruct_high_byte_lane(bytes(high_raw), bytes(low_raw), elem_size, len(raw))

    restored, decompress_ms = timed_median(decompress_once, repeat)
    return CodecResult(
        "zstd_dict_byte_split_zstd",
        1,
        "ok",
        int(restored == raw),
        len(raw),
        BBLC_HEADER_BYTES
        + BYTE_SPLIT_PAYLOAD_HEADER_BYTES
        + ZSTD_DICT_ID_HEADER_BYTES
        + len(high_payload)
        + len(low_payload),
        compress_ms,
        decompress_ms,
        str(level),
    )


CODEC_FUNCS = {
    "raw": codec_raw,
    "plain_zstd": codec_plain_zstd,
    "current_byte_split_zstd": codec_current_byte_split_zstd,
    "current_byte_split_zstd_reusectx": codec_current_byte_split_zstd_reusectx,
    "zipnn": codec_zipnn,
    "bit_group_zstd": codec_bit_group_zstd,
    "byte_split_zstd_high_raw_low": codec_byte_split_zstd_high_raw_low,
    "byte_split_selective_zstd": codec_byte_split_selective_zstd,
    "sample_guided_selective_byte_split_zstd": codec_sample_guided_selective_byte_split_zstd,
    "axis_transpose_byte_split_zstd": codec_axis_transpose_byte_split_zstd,
    "blocked_axis_byte_split_zstd": codec_blocked_axis_byte_split_zstd,
    "blocked_axis_byte_split_zstd_32": codec_blocked_axis_byte_split_zstd_32,
    "blocked_axis_byte_split_zstd_64": codec_blocked_axis_byte_split_zstd_64,
    "blocked_axis_byte_split_zstd_128": codec_blocked_axis_byte_split_zstd_128,
    "zstd_dict_byte_split_zstd": codec_zstd_dict_byte_split_zstd,
    "token_xor_byte_split_zstd": codec_token_xor_byte_split_zstd,
    "token_xor_zstd_high_raw_low": codec_token_xor_zstd_high_raw_low,
}


def metadata_for_files(input_dir: Path) -> Dict[str, Dict[str, object]]:
    metadata_path = input_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    data = json.loads(metadata_path.read_text())
    samples = data.get("samples", []) if isinstance(data, dict) else []
    return {str(item.get("filename", "")): dict(item) for item in samples if isinstance(item, dict)}


def iter_samples(
    input_dir: Path,
    *,
    phases: Optional[set[str]] = None,
    directions: Optional[set[str]] = None,
    channels: Optional[set[str]] = None,
    shapes: Optional[set[str]] = None,
) -> Iterable[TensorSample]:
    meta_by_name = metadata_for_files(input_dir)
    for path in sorted(input_dir.glob("*.pt")):
        try:
            tensor = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            tensor = torch.load(path, map_location="cpu")
        if not torch.is_tensor(tensor):
            continue
        metadata = dict(meta_by_name.get(path.name, {}))
        shape = list(tensor.shape)
        metadata.setdefault("filename", path.name)
        metadata.setdefault("dtype", str(tensor.dtype))
        metadata.setdefault("shape", shape)
        metadata.setdefault("batch_size", int(shape[0]) if shape else 1)
        metadata.setdefault("seq_len", int(shape[1]) if len(shape) >= 2 else 1)
        metadata.setdefault("phase", "decode" if int(metadata["seq_len"]) == 1 else "prefill")
        metadata.setdefault("source", "")
        metadata.setdefault("direction", "")
        metadata.setdefault("channel", "")
        if phases is not None and str(metadata.get("phase", "")) not in phases:
            continue
        if directions is not None and str(metadata.get("direction", "")) not in directions:
            continue
        if channels is not None and str(metadata.get("channel", "")) not in channels:
            continue
        if shapes is not None and shape_name(shape) not in shapes:
            continue
        yield TensorSample(path.name, path, tensor, metadata)


def row_from_result(sample: TensorSample, result: CodecResult) -> Dict[str, object]:
    meta = sample.metadata
    return {
        "sample": sample.name,
        "codec": result.codec,
        "available": result.available,
        "reason": result.reason,
        "roundtrip_ok": result.roundtrip_ok,
        "raw_bytes": result.raw_bytes,
        "wire_bytes": result.wire_bytes,
        "ratio": f"{result.ratio:.6f}",
        "saved_bytes": result.saved_bytes,
        "compress_ms": f"{result.compress_ms:.6f}",
        "decompress_ms": f"{result.decompress_ms:.6f}",
        "break_even_bw_mbps": f"{result.break_even_bw_mbps:.6f}",
        "dtype": dtype_name(sample.tensor.dtype),
        "shape": shape_name(sample.tensor.shape),
        "phase": meta.get("phase", ""),
        "source": meta.get("source", ""),
        "direction": meta.get("direction", ""),
        "channel": meta.get("channel", ""),
        "level": result.level,
        "batch_size": meta.get("batch_size", ""),
        "seq_len": meta.get("seq_len", ""),
        "prompt_len": meta.get("prompt_len", meta.get("inference_prefix_length", "")),
        "blocks": meta.get("blocks", meta.get("block_uid", "")),
        "compute_dtype": meta.get("compute_dtype", ""),
        "schema_dtype": meta.get("schema_dtype", ""),
        "wire_dtype": meta.get("wire_dtype", ""),
        "selected_codec": result.selected_codec,
    }


def benchmark_sample(sample: TensorSample, codecs: Sequence[str], repeat: int, level: int) -> List[CodecResult]:
    raw = tensor_to_raw_bytes(sample.tensor)
    results: List[CodecResult] = []
    deferred_adaptive = "adaptive_selector" in codecs
    for codec in codecs:
        if codec == "adaptive_selector":
            continue
        func = CODEC_FUNCS.get(codec)
        if func is None:
            results.append(CodecResult(codec, 0, "unknown_codec", 0, len(raw), len(raw), 0.0, 0.0, str(level)))
            continue
        results.append(func(raw, repeat, level, sample.tensor.dtype, tuple(int(dim) for dim in sample.tensor.shape)))
    if deferred_adaptive:
        candidates = [
            item for item in results
            if item.codec != "raw" and item.available and item.roundtrip_ok and item.saved_bytes > 0
        ]
        selected = min(candidates, key=lambda item: (item.wire_bytes, item.compress_ms + item.decompress_ms), default=None)
        if selected is None:
            results.append(CodecResult("adaptive_selector", 1, "selected_raw", 1, len(raw), len(raw), 0.0, 0.0, selected_codec="raw"))
        else:
            results.append(
                CodecResult(
                    "adaptive_selector",
                    1,
                    "selected_codec",
                    selected.roundtrip_ok,
                    selected.raw_bytes,
                    selected.wire_bytes,
                    selected.compress_ms,
                    selected.decompress_ms,
                    selected.level,
                    selected.codec,
                )
            )
    return results


def run_benchmark(
    input_dir: Path,
    codecs: Sequence[str],
    repeat: int,
    level: int,
    *,
    phases: Optional[set[str]] = None,
    directions: Optional[set[str]] = None,
    channels: Optional[set[str]] = None,
    shapes: Optional[set[str]] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    samples = list(iter_samples(input_dir, phases=phases, directions=directions, channels=channels, shapes=shapes))
    if "zstd_dict_byte_split_zstd" in codecs:
        configure_zstd_dict_from_samples(samples)
    for sample in samples:
        for result in benchmark_sample(sample, codecs, repeat, level):
            rows.append(row_from_result(sample, result))
    return rows


def write_csv(rows: List[Dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark BloomBee lossless wire codecs on real activation dumps")
    parser.add_argument("--input_dir", required=True, help="Directory containing .pt tensors and optional metadata.json")
    parser.add_argument("--output_csv", default="", help="Output CSV path; stdout summary only if omitted")
    parser.add_argument("--repeat", type=int, default=5, help="Median timing repeats after one warmup")
    parser.add_argument("--codecs", default="all", help="Comma-separated codecs or 'all'")
    parser.add_argument("--zstd-level", type=int, default=1, help="zstd level for benchmark codecs")
    parser.add_argument("--phase", default="", help="Optional comma-separated phase filter, e.g. prefill,decode")
    parser.add_argument("--direction", default="", help="Optional comma-separated direction filter")
    parser.add_argument("--channel", default="", help="Optional comma-separated channel filter")
    parser.add_argument("--shape", default="", help="Optional comma-separated shape filter, e.g. 1x512x5120")
    parser.add_argument("--require-roundtrip", action="store_true", help="Fail if any available codec does not roundtrip")
    args = parser.parse_args()

    rows = run_benchmark(
        Path(args.input_dir),
        parse_codecs(args.codecs),
        args.repeat,
        args.zstd_level,
        phases=parse_filter_set(args.phase),
        directions=parse_filter_set(args.direction),
        channels=parse_filter_set(args.channel),
        shapes=parse_filter_set(args.shape),
    )
    if args.require_roundtrip:
        failed = [row for row in rows if int(row["available"]) and int(row["roundtrip_ok"]) == 0]
        if failed:
            raise SystemExit(f"{len(failed)} available codec rows failed roundtrip")
    if args.output_csv:
        write_csv(rows, Path(args.output_csv))
    print(f"benchmarked_rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
