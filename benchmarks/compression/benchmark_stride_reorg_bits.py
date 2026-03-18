#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from benchmark_stride_reorg import (
    DEFAULT_HEADER_BYTES,
    build_split_spec,
    iter_tensor_files,
    load_tensor,
    make_synthetic_tensor,
    parse_shape,
    ratio,
    reconstruct_interleaved_bytes,
    split_interleaved_bytes,
    zstd_compress,
)


@dataclass
class VariantSpec:
    name: str
    mode: str
    dtype_name: str
    elem_bits: int
    sign_exp_bits: int
    mantissa_high_bits: int
    extracted_bits_per_elem: int
    remaining_bits_per_elem: int
    extracted_label: str
    remaining_label: str


@dataclass
class VariantResult:
    name: str
    dtype: str
    shape: List[int]
    numel: int
    raw_bytes: int
    variant: str
    mode: str
    direct_zstd_bytes: int
    direct_zstd_ratio: float
    extracted_raw_bytes: int
    extracted_zstd_bytes: int
    extracted_zstd_ratio: float
    remaining_raw_bytes: int
    remaining_zstd_bytes: int
    remaining_zstd_ratio: float
    split_total_zstd_bytes: int
    split_total_with_header_bytes: int
    split_total_ratio: float
    split_total_with_header_ratio: float
    split_vs_direct_delta_bytes: int
    split_vs_direct_delta_pct: float
    direct_compress_ms: float
    split_compress_ms: float
    reconstruct_ok: bool
    extracted_byte_zero_ratio: float
    extracted_byte_entropy_bits: float
    remaining_byte_zero_ratio: float
    remaining_byte_entropy_bits: float
    extracted_symbol_unique: int
    extracted_symbol_top1_ratio: float
    extracted_symbol_entropy_bits: float
    spec: Dict[str, object]


def parse_int_list(raw: str) -> Tuple[int, ...]:
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if not parts:
        return (0, 2, 4)
    return tuple(int(item) for item in parts)


def byte_stats(data: bytes) -> Tuple[float, float]:
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size == 0:
        return 0.0, 0.0
    zero_ratio = float(np.mean(arr == 0))
    counts = np.bincount(arr, minlength=256).astype(np.float64)
    probs = counts[counts > 0] / float(arr.size)
    entropy = float(-(probs * np.log2(probs)).sum())
    return zero_ratio, entropy


def symbol_stats(values: np.ndarray, bits: int) -> Tuple[int, float, float]:
    values = np.asarray(values).reshape(-1)
    if values.size == 0:
        return 0, 0.0, 0.0

    if bits <= 18:
        counts = np.bincount(values.astype(np.int64, copy=False))
        counts = counts[counts > 0].astype(np.float64)
    else:
        _, counts = np.unique(values, return_counts=True)
        counts = counts.astype(np.float64)

    probs = counts / float(values.size)
    unique = int(counts.size)
    top1 = float(probs.max(initial=0.0))
    entropy = float(-(probs * np.log2(probs)).sum()) if probs.size else 0.0
    return unique, top1, entropy


def pack_fixed_width(values: np.ndarray, bits: int) -> bytes:
    values = np.asarray(values, dtype=np.uint64).reshape(-1)
    if bits == 0 or values.size == 0:
        return b""
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint64)
    bit_matrix = ((values[:, None] >> shifts[None, :]) & 1).astype(np.uint8, copy=False)
    packed = np.packbits(bit_matrix.reshape(-1), bitorder="big")
    return packed.tobytes()


def unpack_fixed_width(data: bytes, numel: int, bits: int) -> np.ndarray:
    if bits == 0:
        return np.zeros(numel, dtype=np.uint64)
    bit_arr = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")
    needed = numel * bits
    if bit_arr.size < needed:
        raise ValueError("packed buffer does not contain enough bits")
    bit_arr = bit_arr[:needed].reshape(numel, bits)
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint64)
    out = np.zeros(numel, dtype=np.uint64)
    for idx, shift in enumerate(shifts):
        out |= bit_arr[:, idx].astype(np.uint64, copy=False) << shift
    return out


def dtype_layout(dtype: np.dtype) -> Tuple[np.dtype, int, int, int]:
    if dtype == np.dtype(np.float16):
        return np.dtype(np.uint16), 16, 6, 10
    if dtype == np.dtype(np.float32):
        return np.dtype(np.uint32), 32, 9, 23
    raise ValueError(f"only float16/float32 are supported, got {dtype}")


def build_variants(dtype: np.dtype, mantissa_high_bits: Sequence[int]) -> List[VariantSpec]:
    _, elem_bits, sign_exp_bits, mantissa_bits = dtype_layout(dtype)
    dtype_name = str(np.dtype(dtype))

    variants = [
        VariantSpec(
            name="byte_high_baseline",
            mode="byte_baseline",
            dtype_name=dtype_name,
            elem_bits=elem_bits,
            sign_exp_bits=sign_exp_bits,
            mantissa_high_bits=max(0, 8 - sign_exp_bits),
            extracted_bits_per_elem=8,
            remaining_bits_per_elem=elem_bits - 8,
            extracted_label="current_high_byte",
            remaining_label="rest_bytes",
        )
    ]

    seen: set[Tuple[str, int]] = set()
    for mant_hi in mantissa_high_bits:
        if mant_hi < 0 or mant_hi > mantissa_bits:
            raise ValueError(
                f"mantissa_high_bits={mant_hi} is out of range for {dtype_name} (mantissa_bits={mantissa_bits})"
            )
        top_bits = sign_exp_bits + mant_hi
        key = ("packed_top_bits", top_bits)
        if key in seen:
            continue
        seen.add(key)
        variants.append(
            VariantSpec(
                name=f"packed_signexp_mhi{mant_hi}",
                mode="packed_top_bits",
                dtype_name=dtype_name,
                elem_bits=elem_bits,
                sign_exp_bits=sign_exp_bits,
                mantissa_high_bits=mant_hi,
                extracted_bits_per_elem=top_bits,
                remaining_bits_per_elem=elem_bits - top_bits,
                extracted_label=f"sign_exp_plus_mantissa_hi{mant_hi}_bits",
                remaining_label=f"remaining_low_{elem_bits - top_bits}_bits",
            )
        )
    return variants


def split_byte_baseline(raw: bytes, dtype: np.dtype) -> Tuple[bytes, bytes]:
    spec = build_split_spec(dtype)
    return split_interleaved_bytes(raw, spec)


def reconstruct_byte_baseline(extracted: bytes, remaining: bytes, dtype: np.dtype) -> bytes:
    spec = build_split_spec(dtype)
    return reconstruct_interleaved_bytes(extracted, remaining, spec)


def split_packed_top_bits(tensor: np.ndarray, top_bits: int) -> Tuple[bytes, bytes, np.ndarray]:
    tensor = np.ascontiguousarray(tensor)
    word_dtype, elem_bits, _, _ = dtype_layout(tensor.dtype)
    words = tensor.view(word_dtype).reshape(-1).astype(np.uint64, copy=False)
    remaining_bits = elem_bits - top_bits
    low_mask = np.uint64((1 << remaining_bits) - 1) if remaining_bits > 0 else np.uint64(0)
    extracted_vals = words >> remaining_bits
    remaining_vals = words & low_mask
    return (
        pack_fixed_width(extracted_vals, top_bits),
        pack_fixed_width(remaining_vals, remaining_bits),
        extracted_vals,
    )


def reconstruct_packed_top_bits(
    extracted: bytes,
    remaining: bytes,
    dtype: np.dtype,
    numel: int,
    top_bits: int,
) -> bytes:
    word_dtype, elem_bits, _, _ = dtype_layout(dtype)
    remaining_bits = elem_bits - top_bits
    extracted_vals = unpack_fixed_width(extracted, numel, top_bits)
    remaining_vals = unpack_fixed_width(remaining, numel, remaining_bits)
    words = (extracted_vals << remaining_bits) | remaining_vals
    words = words.astype(word_dtype, copy=False)
    return words.reshape(-1).tobytes()


def run_variant(
    name: str,
    tensor: np.ndarray,
    *,
    variant: VariantSpec,
    direct_zstd: bytes,
    direct_ms: float,
    zstd_level: int,
    header_bytes: int,
    zstd_bin: str,
) -> VariantResult:
    tensor = np.ascontiguousarray(tensor)
    raw = tensor.tobytes()

    if variant.mode == "byte_baseline":
        extracted_raw, remaining_raw = split_byte_baseline(raw, tensor.dtype)
        reconstructed = reconstruct_byte_baseline(extracted_raw, remaining_raw, tensor.dtype)
        extracted_symbol_vals = np.frombuffer(extracted_raw, dtype=np.uint8).astype(np.uint64, copy=False)
    elif variant.mode == "packed_top_bits":
        extracted_raw, remaining_raw, extracted_symbol_vals = split_packed_top_bits(
            tensor, top_bits=variant.extracted_bits_per_elem
        )
        reconstructed = reconstruct_packed_top_bits(
            extracted_raw,
            remaining_raw,
            tensor.dtype,
            tensor.size,
            top_bits=variant.extracted_bits_per_elem,
        )
    else:
        raise ValueError(f"unknown variant mode: {variant.mode}")

    reconstruct_ok = reconstructed == raw
    extracted_zstd, extracted_ms = zstd_compress(extracted_raw, level=zstd_level, zstd_bin=zstd_bin)
    remaining_zstd, remaining_ms = zstd_compress(remaining_raw, level=zstd_level, zstd_bin=zstd_bin)

    split_total = len(extracted_zstd) + len(remaining_zstd)
    split_total_with_header = split_total + max(0, int(header_bytes))
    direct_size = len(direct_zstd)
    delta_bytes = split_total_with_header - direct_size
    delta_pct = (float(delta_bytes) / float(direct_size) * 100.0) if direct_size > 0 else 0.0

    extracted_zero_ratio, extracted_entropy = byte_stats(extracted_raw)
    remaining_zero_ratio, remaining_entropy = byte_stats(remaining_raw)
    symbol_unique, symbol_top1, symbol_entropy = symbol_stats(
        extracted_symbol_vals, bits=variant.extracted_bits_per_elem
    )

    spec = asdict(variant)
    if variant.mode == "byte_baseline":
        byte_spec = build_split_spec(tensor.dtype)
        spec["extracted_offsets"] = list(byte_spec.extracted_offsets)
        spec["remaining_offsets"] = list(byte_spec.remaining_offsets)

    return VariantResult(
        name=name,
        dtype=str(tensor.dtype),
        shape=[int(dim) for dim in tensor.shape],
        numel=int(tensor.size),
        raw_bytes=len(raw),
        variant=variant.name,
        mode=variant.mode,
        direct_zstd_bytes=direct_size,
        direct_zstd_ratio=ratio(direct_size, len(raw)),
        extracted_raw_bytes=len(extracted_raw),
        extracted_zstd_bytes=len(extracted_zstd),
        extracted_zstd_ratio=ratio(len(extracted_zstd), len(extracted_raw)),
        remaining_raw_bytes=len(remaining_raw),
        remaining_zstd_bytes=len(remaining_zstd),
        remaining_zstd_ratio=ratio(len(remaining_zstd), len(remaining_raw)),
        split_total_zstd_bytes=split_total,
        split_total_with_header_bytes=split_total_with_header,
        split_total_ratio=ratio(split_total, len(raw)),
        split_total_with_header_ratio=ratio(split_total_with_header, len(raw)),
        split_vs_direct_delta_bytes=delta_bytes,
        split_vs_direct_delta_pct=delta_pct,
        direct_compress_ms=direct_ms,
        split_compress_ms=extracted_ms + remaining_ms,
        reconstruct_ok=reconstruct_ok,
        extracted_byte_zero_ratio=extracted_zero_ratio,
        extracted_byte_entropy_bits=extracted_entropy,
        remaining_byte_zero_ratio=remaining_zero_ratio,
        remaining_byte_entropy_bits=remaining_entropy,
        extracted_symbol_unique=symbol_unique,
        extracted_symbol_top1_ratio=symbol_top1,
        extracted_symbol_entropy_bits=symbol_entropy,
        spec=spec,
    )


def print_result(result: VariantResult) -> None:
    better = result.split_total_with_header_bytes < result.direct_zstd_bytes
    trend = "better" if better else "worse"
    print(
        f"[{result.name}] variant={result.variant} dtype={result.dtype} "
        f"shape={tuple(result.shape)} numel={result.numel}"
    )
    print(
        f"  raw={result.raw_bytes}B "
        f"direct_zstd={result.direct_zstd_bytes}B ({result.direct_zstd_ratio:.6f}) "
        f"split_total={result.split_total_zstd_bytes}B ({result.split_total_ratio:.6f}) "
        f"split_plus_header={result.split_total_with_header_bytes}B ({result.split_total_with_header_ratio:.6f})"
    )
    print(
        f"  extracted={result.extracted_raw_bytes}B -> {result.extracted_zstd_bytes}B ({result.extracted_zstd_ratio:.6f}) "
        f"remaining={result.remaining_raw_bytes}B -> {result.remaining_zstd_bytes}B ({result.remaining_zstd_ratio:.6f})"
    )
    print(
        f"  split_vs_direct={result.split_vs_direct_delta_bytes:+d}B "
        f"({result.split_vs_direct_delta_pct:+.2f}%) [{trend}] "
        f"reconstruct_ok={int(result.reconstruct_ok)}"
    )
    print(
        f"  extracted_symbols: unique={result.extracted_symbol_unique} "
        f"top1_ratio={result.extracted_symbol_top1_ratio:.6f} "
        f"entropy_bits={result.extracted_symbol_entropy_bits:.6f}"
    )
    print(
        f"  extracted_bytes: zero_ratio={result.extracted_byte_zero_ratio:.6f} "
        f"entropy_bits={result.extracted_byte_entropy_bits:.6f}; "
        f"remaining_bytes: zero_ratio={result.remaining_byte_zero_ratio:.6f} "
        f"entropy_bits={result.remaining_byte_entropy_bits:.6f}"
    )
    print(
        f"  spec: mode={result.mode} extracted_bits={result.spec['extracted_bits_per_elem']} "
        f"mantissa_hi_bits={result.spec['mantissa_high_bits']}"
    )


def summarize(results: Sequence[VariantResult]) -> None:
    buckets: Dict[str, List[VariantResult]] = defaultdict(list)
    for item in results:
        buckets[item.variant].append(item)

    print("\nSummary by variant:")
    for variant_name in sorted(buckets):
        group = buckets[variant_name]
        avg_delta_pct = np.mean([item.split_vs_direct_delta_pct for item in group])
        avg_split_ratio = np.mean([item.split_total_with_header_ratio for item in group])
        avg_sym_entropy = np.mean([item.extracted_symbol_entropy_bits for item in group])
        avg_sym_top1 = np.mean([item.extracted_symbol_top1_ratio for item in group])
        print(
            f"  {variant_name}: n={len(group)} "
            f"avg_split_vs_direct={avg_delta_pct:+.2f}% "
            f"avg_total_ratio={avg_split_ratio:.6f} "
            f"avg_extracted_symbol_entropy={avg_sym_entropy:.6f} "
            f"avg_extracted_symbol_top1={avg_sym_top1:.6f}"
        )


def iter_work_items(inputs: Sequence[str], synthetic: str | None, shape: str, seed: int) -> Iterable[Tuple[str, np.ndarray]]:
    emitted = False
    for path in iter_tensor_files(inputs):
        emitted = True
        yield str(path), load_tensor(path)
    if not emitted and synthetic is not None:
        yield f"synthetic:{synthetic}", make_synthetic_tensor(synthetic, parse_shape(shape), seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bit-level benchmark for exp-only and exp+mantissa-high-bits stride reorganization before Zstd"
    )
    parser.add_argument(
        "--input",
        nargs="*",
        default=[],
        help="Tensor files or directories (.pt/.npy/.npz). Directories are scanned recursively.",
    )
    parser.add_argument(
        "--synthetic",
        choices=["small_fp32", "mid_fp16"],
        default=None,
        help="Generate a synthetic tensor if no real tensor file is provided.",
    )
    parser.add_argument(
        "--shape",
        default="64,1,5120",
        help="Synthetic tensor shape, comma-separated.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Synthetic RNG seed.")
    parser.add_argument("--zstd-level", type=int, default=3, help="Zstd compression level.")
    parser.add_argument(
        "--zstd-bin",
        default=shutil.which("zstd") or "zstd",
        help="Path to zstd CLI binary.",
    )
    parser.add_argument(
        "--header-bytes",
        type=int,
        default=DEFAULT_HEADER_BYTES,
        help="Estimated framing overhead added to split buffers when comparing to direct Zstd.",
    )
    parser.add_argument(
        "--mantissa-high-bits",
        default="0,2,4",
        help="Comma-separated list of mantissa high-bit counts to try in packed variants.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    work = list(iter_work_items(args.input, args.synthetic, args.shape, args.seed))
    if not work:
        raise SystemExit("No tensor inputs found. Provide --input path/dir or --synthetic preset.")

    mantissa_high_bits = parse_int_list(args.mantissa_high_bits)
    results: List[VariantResult] = []

    for name, tensor in work:
        if not np.issubdtype(tensor.dtype, np.floating):
            raise ValueError(f"{name}: tensor must be floating point, got {tensor.dtype}")
        if tensor.dtype not in (np.float16, np.float32):
            raise ValueError(f"{name}: only float16/float32 supported, got {tensor.dtype}")

        tensor = np.ascontiguousarray(tensor)
        direct_zstd, direct_ms = zstd_compress(tensor.tobytes(), level=args.zstd_level, zstd_bin=args.zstd_bin)
        variants = build_variants(tensor.dtype, mantissa_high_bits)
        for variant in variants:
            result = run_variant(
                name,
                tensor,
                variant=variant,
                direct_zstd=direct_zstd,
                direct_ms=direct_ms,
                zstd_level=args.zstd_level,
                header_bytes=args.header_bytes,
                zstd_bin=args.zstd_bin,
            )
            results.append(result)
            print_result(result)

    summarize(results)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(item) for item in results]
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved JSON: {args.json}")


if __name__ == "__main__":
    main()
