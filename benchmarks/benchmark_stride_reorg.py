#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


SUPPORTED_EXTS = {".pt", ".npy", ".npz"}
DEFAULT_HEADER_BYTES = 32


@dataclass
class SplitSpec:
    dtype_name: str
    elem_size: int
    extracted_offsets: Tuple[int, ...]
    remaining_offsets: Tuple[int, ...]
    extracted_label: str
    remaining_label: str


@dataclass
class BenchmarkResult:
    name: str
    dtype: str
    shape: List[int]
    numel: int
    raw_bytes: int
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
    spec: Dict[str, object]


def parse_shape(raw: str) -> Tuple[int, ...]:
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if not parts:
        raise ValueError("shape must not be empty")
    return tuple(int(item) for item in parts)


def iter_tensor_files(inputs: Sequence[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for item in inputs:
        path = Path(item)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            if path not in seen:
                seen.add(path)
                yield path
            continue
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTS and child not in seen:
                    seen.add(child)
                    yield child


def _load_pt_tensor_via_subprocess(path: Path) -> np.ndarray:
    child_code = """
import io
import os
import sys
import numpy as np
import torch

obj = torch.load(sys.argv[1], map_location="cpu")
if not isinstance(obj, torch.Tensor):
    raise TypeError(f"{sys.argv[1]} does not contain a torch.Tensor")
arr = obj.detach().cpu().contiguous().numpy()
buf = io.BytesIO()
np.save(buf, arr, allow_pickle=False)
sys.stdout.buffer.write(buf.getvalue())
sys.stdout.buffer.flush()
os._exit(0)
""".strip()
    proc = subprocess.run(
        [sys.executable, "-c", child_code, str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return np.load(io.BytesIO(proc.stdout), allow_pickle=False)


def load_tensor(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".pt":
        return np.ascontiguousarray(_load_pt_tensor_via_subprocess(path))
    if suffix == ".npy":
        return np.ascontiguousarray(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        if "data" in data:
            return np.ascontiguousarray(data["data"])
        first_key = next(iter(data.files), None)
        if first_key is None:
            raise ValueError(f"{path} is empty")
        return np.ascontiguousarray(data[first_key])
    raise ValueError(f"unsupported file extension: {path.suffix}")


def make_synthetic_tensor(preset: str, shape: Tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if preset == "small_fp32":
        return rng.normal(loc=0.0, scale=0.015, size=shape).astype(np.float32)
    if preset == "mid_fp16":
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=shape)
        magnitudes = np.exp(rng.normal(loc=-0.25, scale=0.55, size=shape)).astype(np.float32)
        return (signs * magnitudes).astype(np.float16)
    raise ValueError(f"unknown synthetic preset: {preset}")


def build_split_spec(dtype: np.dtype) -> SplitSpec:
    if dtype == np.dtype(np.float32):
        return SplitSpec(
            dtype_name="float32",
            elem_size=4,
            extracted_offsets=(3,),
            remaining_offsets=(0, 1, 2),
            extracted_label="exp_sign_byte",
            remaining_label="mantissa_bytes",
        )
    if dtype == np.dtype(np.float16):
        return SplitSpec(
            dtype_name="float16",
            elem_size=2,
            extracted_offsets=(1,),
            remaining_offsets=(0,),
            extracted_label="exp_sign_hi2mant_byte",
            remaining_label="mantissa_lo_byte",
        )
    raise ValueError(f"only float16/float32 are supported, got {dtype}")


def split_interleaved_bytes(raw: bytes, spec: SplitSpec) -> Tuple[bytes, bytes]:
    if len(raw) % spec.elem_size != 0:
        raise ValueError(f"raw byte size {len(raw)} is not divisible by elem_size={spec.elem_size}")
    byte_matrix = np.frombuffer(raw, dtype=np.uint8).reshape(-1, spec.elem_size)
    extracted = byte_matrix[:, spec.extracted_offsets].reshape(-1)
    remaining = byte_matrix[:, spec.remaining_offsets].reshape(-1)
    return extracted.tobytes(), remaining.tobytes()


def reconstruct_interleaved_bytes(extracted: bytes, remaining: bytes, spec: SplitSpec) -> bytes:
    extracted_width = len(spec.extracted_offsets)
    remaining_width = len(spec.remaining_offsets)
    if len(extracted) % max(1, extracted_width) != 0:
        raise ValueError("extracted buffer length is invalid")
    numel = len(extracted) // max(1, extracted_width)
    if len(remaining) != numel * remaining_width:
        raise ValueError("remaining buffer length does not match extracted buffer length")

    out = np.empty((numel, spec.elem_size), dtype=np.uint8)
    out[:, spec.extracted_offsets] = np.frombuffer(extracted, dtype=np.uint8).reshape(numel, extracted_width)
    out[:, spec.remaining_offsets] = np.frombuffer(remaining, dtype=np.uint8).reshape(numel, remaining_width)
    return out.reshape(-1).tobytes()


def zstd_compress(data: bytes, level: int, zstd_bin: str) -> Tuple[bytes, float]:
    cmd = [zstd_bin, f"-{int(level)}", "-q", "-c"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return proc.stdout, elapsed_ms


def ratio(comp_bytes: int, raw_bytes: int) -> float:
    if raw_bytes <= 0:
        return 1.0
    return float(comp_bytes) / float(raw_bytes)


def run_single(
    name: str,
    tensor: np.ndarray,
    *,
    zstd_level: int,
    header_bytes: int,
    zstd_bin: str,
) -> BenchmarkResult:
    if not np.issubdtype(tensor.dtype, np.floating):
        raise ValueError(f"{name}: tensor must be floating point, got {tensor.dtype}")
    if tensor.dtype not in (np.float16, np.float32):
        raise ValueError(f"{name}: only float16/float32 supported, got {tensor.dtype}")

    tensor = np.ascontiguousarray(tensor)
    raw = tensor.tobytes()
    spec = build_split_spec(tensor.dtype)
    extracted_raw, remaining_raw = split_interleaved_bytes(raw, spec)
    reconstructed = reconstruct_interleaved_bytes(extracted_raw, remaining_raw, spec)
    reconstruct_ok = reconstructed == raw

    direct_zstd, direct_ms = zstd_compress(raw, level=zstd_level, zstd_bin=zstd_bin)
    extracted_zstd, extracted_ms = zstd_compress(extracted_raw, level=zstd_level, zstd_bin=zstd_bin)
    remaining_zstd, remaining_ms = zstd_compress(remaining_raw, level=zstd_level, zstd_bin=zstd_bin)

    split_total = len(extracted_zstd) + len(remaining_zstd)
    split_total_with_header = split_total + max(0, int(header_bytes))
    direct_size = len(direct_zstd)
    delta_bytes = split_total_with_header - direct_size
    delta_pct = (float(delta_bytes) / float(direct_size) * 100.0) if direct_size > 0 else 0.0

    return BenchmarkResult(
        name=name,
        dtype=str(tensor.dtype),
        shape=[int(dim) for dim in tensor.shape],
        numel=int(tensor.size),
        raw_bytes=len(raw),
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
        spec=asdict(spec),
    )


def print_result(result: BenchmarkResult) -> None:
    better = result.split_total_with_header_bytes < result.direct_zstd_bytes
    trend = "better" if better else "worse"
    print(f"[{result.name}] dtype={result.dtype} shape={tuple(result.shape)} numel={result.numel}")
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
        f"  spec: extracted_offsets={result.spec['extracted_offsets']} "
        f"remaining_offsets={result.spec['remaining_offsets']} "
        f"header={result.split_total_with_header_bytes - result.split_total_zstd_bytes}B"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline benchmark for stride-scan reorganization before Zstd"
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
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    work: List[Tuple[str, np.ndarray]] = []
    for path in iter_tensor_files(args.input):
        tensor = load_tensor(path)
        work.append((str(path), tensor))

    if not work and args.synthetic is not None:
        tensor = make_synthetic_tensor(args.synthetic, parse_shape(args.shape), seed=args.seed)
        work.append((f"synthetic:{args.synthetic}", tensor))

    if not work:
        raise SystemExit("No tensor inputs found. Provide --input path/dir or --synthetic preset.")

    results: List[BenchmarkResult] = []
    for name, tensor in work:
        result = run_single(
            name,
            tensor,
            zstd_level=args.zstd_level,
            header_bytes=args.header_bytes,
            zstd_bin=args.zstd_bin,
        )
        results.append(result)
        print_result(result)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(item) for item in results]
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved JSON: {args.json}")


if __name__ == "__main__":
    main()
