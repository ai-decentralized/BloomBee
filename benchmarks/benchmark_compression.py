#!/usr/bin/env python3
"""
Compression Benchmark for BloomBee Activation Data

This script tests various lossless and lossy compression algorithms
on activation tensors to evaluate compression ratio, speed, and accuracy impact.

Usage:
    # First generate or dump activations:
    python activation_dumper.py synthetic --output_dir /tmp/activations --num_samples 20

    # Then run compression benchmark:
    python benchmark_compression.py --input_dir /tmp/activations --output_report /tmp/compression_report.json

Dependencies:
    pip install lz4 zstandard tabulate
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import torch
import numpy as np

# Try to import compression libraries
try:
    import lz4.frame as lz4
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False
    warnings.warn("lz4 not installed. Install with: pip install lz4")

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
    warnings.warn("zstandard not installed. Install with: pip install zstandard")

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    warnings.warn("tabulate not installed. Install with: pip install tabulate")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hivemind import get_logger

logger = get_logger(__name__)


@dataclass
class CompressionResult:
    """Results from a single compression test."""
    algorithm: str
    category: str  # "lossless" or "lossy"
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    compress_time_ms: float
    decompress_time_ms: float
    total_time_ms: float
    
    # Lossy-specific metrics (None for lossless)
    mse: Optional[float] = None
    max_abs_error: Optional[float] = None
    relative_error_percent: Optional[float] = None
    cosine_similarity: Optional[float] = None
    snr_db: Optional[float] = None  # Signal-to-Noise Ratio


class CompressionBenchmark:
    """Benchmark suite for activation compression algorithms."""
    
    def __init__(self):
        self.results: List[CompressionResult] = []
    
    # ========================
    # Lossless Compression
    # ========================
    
    def compress_lz4(self, data: bytes, level: int = 0) -> Tuple[bytes, float]:
        """Compress using LZ4."""
        if not HAS_LZ4:
            raise ImportError("lz4 not available")
        
        t0 = time.perf_counter()
        compressed = lz4.compress(data, compression_level=level)
        elapsed = (time.perf_counter() - t0) * 1000
        return compressed, elapsed
    
    def decompress_lz4(self, data: bytes) -> Tuple[bytes, float]:
        """Decompress LZ4 data."""
        t0 = time.perf_counter()
        decompressed = lz4.decompress(data)
        elapsed = (time.perf_counter() - t0) * 1000
        return decompressed, elapsed
    
    def compress_zstd(self, data: bytes, level: int = 3) -> Tuple[bytes, float]:
        """Compress using Zstandard."""
        if not HAS_ZSTD:
            raise ImportError("zstandard not available")
        
        cctx = zstd.ZstdCompressor(level=level)
        t0 = time.perf_counter()
        compressed = cctx.compress(data)
        elapsed = (time.perf_counter() - t0) * 1000
        return compressed, elapsed
    
    def decompress_zstd(self, data: bytes) -> Tuple[bytes, float]:
        """Decompress Zstandard data."""
        dctx = zstd.ZstdDecompressor()
        t0 = time.perf_counter()
        decompressed = dctx.decompress(data)
        elapsed = (time.perf_counter() - t0) * 1000
        return decompressed, elapsed
    
    # ========================
    # Lossy Compression
    # ========================
    
    def compress_float16(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Convert to float16."""
        t0 = time.perf_counter()
        compressed = tensor.to(torch.float16)
        elapsed = (time.perf_counter() - t0) * 1000
        return compressed, elapsed
    
    def decompress_float16(self, tensor: torch.Tensor, original_dtype: torch.dtype) -> Tuple[torch.Tensor, float]:
        """Convert back from float16."""
        t0 = time.perf_counter()
        decompressed = tensor.to(original_dtype)
        elapsed = (time.perf_counter() - t0) * 1000
        return decompressed, elapsed
    
    def compress_int8_uniform(self, tensor: torch.Tensor) -> Tuple[Tuple[torch.Tensor, float, float], float]:
        """Uniform 8-bit quantization."""
        t0 = time.perf_counter()
        
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        scale = (max_val - min_val) / 255.0 if max_val != min_val else 1.0
        
        # Quantize
        quantized = ((tensor - min_val) / scale).round().clamp(0, 255).to(torch.uint8)
        
        elapsed = (time.perf_counter() - t0) * 1000
        return (quantized, min_val, scale), elapsed
    
    def decompress_int8_uniform(
        self, 
        compressed: Tuple[torch.Tensor, float, float],
        original_dtype: torch.dtype
    ) -> Tuple[torch.Tensor, float]:
        """Dequantize from uniform 8-bit."""
        quantized, min_val, scale = compressed
        
        t0 = time.perf_counter()
        decompressed = (quantized.to(original_dtype) * scale + min_val)
        elapsed = (time.perf_counter() - t0) * 1000
        
        return decompressed, elapsed
    
    def compress_int8_blockwise(
        self, 
        tensor: torch.Tensor, 
        block_size: int = 64
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], float]:
        """Block-wise 8-bit quantization (similar to bitsandbytes)."""
        t0 = time.perf_counter()
        
        # Flatten and pad to block size
        original_shape = tensor.shape
        flat = tensor.flatten()
        
        # Pad
        pad_len = (block_size - len(flat) % block_size) % block_size
        if pad_len > 0:
            flat = torch.cat([flat, torch.zeros(pad_len, dtype=flat.dtype)])
        
        # Reshape into blocks
        blocks = flat.view(-1, block_size)
        
        # Compute per-block min and max
        block_mins = blocks.min(dim=1, keepdim=True)[0]
        block_maxs = blocks.max(dim=1, keepdim=True)[0]
        block_scales = (block_maxs - block_mins) / 255.0
        block_scales = torch.where(block_scales == 0, torch.ones_like(block_scales), block_scales)
        
        # Quantize
        quantized = ((blocks - block_mins) / block_scales).round().clamp(0, 255).to(torch.uint8)
        
        elapsed = (time.perf_counter() - t0) * 1000
        
        # Store metadata
        metadata = {
            "original_shape": original_shape,
            "pad_len": pad_len,
            "block_size": block_size,
        }
        
        return (quantized, block_mins.squeeze(), block_scales.squeeze(), metadata), elapsed
    
    def decompress_int8_blockwise(
        self,
        compressed: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict],
        original_dtype: torch.dtype
    ) -> Tuple[torch.Tensor, float]:
        """Dequantize from block-wise 8-bit."""
        quantized, block_mins, block_scales, metadata = compressed
        
        t0 = time.perf_counter()
        
        # Dequantize
        blocks = quantized.to(original_dtype) * block_scales.unsqueeze(1) + block_mins.unsqueeze(1)
        
        # Flatten and remove padding
        flat = blocks.flatten()
        if metadata["pad_len"] > 0:
            flat = flat[:-metadata["pad_len"]]
        
        # Reshape
        decompressed = flat.view(metadata["original_shape"])
        
        elapsed = (time.perf_counter() - t0) * 1000
        return decompressed, elapsed
    
    def compress_int4_groupwise(
        self,
        tensor: torch.Tensor,
        group_size: int = 32
    ) -> Tuple[Any, float]:
        """4-bit group-wise quantization (FlexGen-style)."""
        t0 = time.perf_counter()
        
        original_shape = tensor.shape
        original_dtype = tensor.dtype
        
        # Flatten
        flat = tensor.flatten().float()
        
        # Pad to group size
        pad_len = (group_size - len(flat) % group_size) % group_size
        if pad_len > 0:
            flat = torch.cat([flat, torch.zeros(pad_len)])
        
        # Reshape into groups
        groups = flat.view(-1, group_size)
        
        # Compute per-group min and max
        group_mins = groups.min(dim=1, keepdim=True)[0]
        group_maxs = groups.max(dim=1, keepdim=True)[0]
        
        # 4-bit: 0-15
        B = 15
        group_scales = B / (group_maxs - group_mins)
        group_scales = torch.where(group_scales.isinf(), torch.ones_like(group_scales), group_scales)
        
        # Quantize
        quantized = ((groups - group_mins) * group_scales).round().clamp(0, B).to(torch.uint8)
        
        # Pack 4-bit values: two values per byte
        even = quantized[:, 0::2]
        odd = quantized[:, 1::2]
        packed = (even << 4) | odd
        
        elapsed = (time.perf_counter() - t0) * 1000
        
        metadata = {
            "original_shape": original_shape,
            "original_dtype": str(original_dtype),
            "pad_len": pad_len,
            "group_size": group_size,
        }
        
        return (packed, group_mins.squeeze(), group_scales.squeeze(), metadata), elapsed
    
    def decompress_int4_groupwise(
        self,
        compressed: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict],
        original_dtype: torch.dtype
    ) -> Tuple[torch.Tensor, float]:
        """Dequantize from 4-bit group-wise."""
        packed, group_mins, group_scales, metadata = compressed
        
        t0 = time.perf_counter()
        
        # Unpack
        even = (packed >> 4) & 0xF
        odd = packed & 0xF
        
        # Interleave
        unpacked = torch.zeros(
            packed.shape[0], packed.shape[1] * 2,
            dtype=torch.uint8
        )
        unpacked[:, 0::2] = even
        unpacked[:, 1::2] = odd
        
        # Dequantize
        B = 15
        groups = unpacked.float() / group_scales.unsqueeze(1) + group_mins.unsqueeze(1)
        
        # Flatten and remove padding
        flat = groups.flatten()
        if metadata["pad_len"] > 0:
            flat = flat[:-metadata["pad_len"]]
        
        # Reshape
        decompressed = flat.view(metadata["original_shape"]).to(original_dtype)
        
        elapsed = (time.perf_counter() - t0) * 1000
        return decompressed, elapsed
    
    # ========================
    # Metrics Calculation
    # ========================
    
    def compute_accuracy_metrics(
        self, 
        original: torch.Tensor, 
        reconstructed: torch.Tensor
    ) -> Dict[str, float]:
        """Compute accuracy metrics between original and reconstructed tensors."""
        original = original.float()
        reconstructed = reconstructed.float()
        
        diff = original - reconstructed
        
        # MSE
        mse = (diff ** 2).mean().item()
        
        # Max absolute error
        max_abs_error = diff.abs().max().item()
        
        # Relative error
        orig_norm = original.norm().item()
        diff_norm = diff.norm().item()
        relative_error = (diff_norm / orig_norm * 100) if orig_norm > 0 else 0
        
        # Cosine similarity
        orig_flat = original.flatten()
        recon_flat = reconstructed.flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            orig_flat.unsqueeze(0), 
            recon_flat.unsqueeze(0)
        ).item()
        
        # SNR (Signal-to-Noise Ratio) in dB
        signal_power = (original ** 2).mean().item()
        noise_power = mse
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        return {
            "mse": mse,
            "max_abs_error": max_abs_error,
            "relative_error_percent": relative_error,
            "cosine_similarity": cos_sim,
            "snr_db": snr_db,
        }
    
    # ========================
    # Main Benchmark Methods
    # ========================
    
    def benchmark_lossless(self, tensor: torch.Tensor) -> List[CompressionResult]:
        """Benchmark all lossless compression algorithms."""
        results = []
        
        # Convert tensor to bytes
        tensor_bytes = tensor.numpy().tobytes()
        original_size = len(tensor_bytes)
        
        algorithms = []
        
        if HAS_LZ4:
            algorithms.extend([
                ("LZ4-Fast", lambda d: self.compress_lz4(d, level=0), self.decompress_lz4),
                ("LZ4-HC", lambda d: self.compress_lz4(d, level=9), self.decompress_lz4),
            ])
        
        if HAS_ZSTD:
            algorithms.extend([
                ("Zstd-1", lambda d: self.compress_zstd(d, level=1), self.decompress_zstd),
                ("Zstd-3", lambda d: self.compress_zstd(d, level=3), self.decompress_zstd),
                ("Zstd-9", lambda d: self.compress_zstd(d, level=9), self.decompress_zstd),
            ])
        
        for name, compress_fn, decompress_fn in algorithms:
            try:
                compressed, compress_time = compress_fn(tensor_bytes)
                decompressed, decompress_time = decompress_fn(compressed)
                
                # Verify correctness
                assert decompressed == tensor_bytes, f"{name} decompression mismatch!"
                
                result = CompressionResult(
                    algorithm=name,
                    category="lossless",
                    original_size_bytes=original_size,
                    compressed_size_bytes=len(compressed),
                    compression_ratio=original_size / len(compressed),
                    compress_time_ms=compress_time,
                    decompress_time_ms=decompress_time,
                    total_time_ms=compress_time + decompress_time,
                )
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to benchmark {name}: {e}")
        
        return results
    
    def benchmark_lossy(self, tensor: torch.Tensor) -> List[CompressionResult]:
        """Benchmark all lossy compression algorithms."""
        results = []
        original_dtype = tensor.dtype
        original_size = tensor.numel() * tensor.element_size()
        
        # Float16
        if tensor.dtype != torch.float16:
            try:
                compressed, compress_time = self.compress_float16(tensor)
                decompressed, decompress_time = self.decompress_float16(compressed, original_dtype)
                
                metrics = self.compute_accuracy_metrics(tensor, decompressed)
                compressed_size = compressed.numel() * compressed.element_size()
                
                results.append(CompressionResult(
                    algorithm="Float16",
                    category="lossy",
                    original_size_bytes=original_size,
                    compressed_size_bytes=compressed_size,
                    compression_ratio=original_size / compressed_size,
                    compress_time_ms=compress_time,
                    decompress_time_ms=decompress_time,
                    total_time_ms=compress_time + decompress_time,
                    **metrics,
                ))
            except Exception as e:
                logger.warning(f"Float16 benchmark failed: {e}")
        
        # Int8 Uniform
        try:
            compressed, compress_time = self.compress_int8_uniform(tensor)
            decompressed, decompress_time = self.decompress_int8_uniform(compressed, original_dtype)
            
            metrics = self.compute_accuracy_metrics(tensor, decompressed)
            compressed_size = compressed[0].numel() * compressed[0].element_size()  # Just the data
            
            results.append(CompressionResult(
                algorithm="Int8-Uniform",
                category="lossy",
                original_size_bytes=original_size,
                compressed_size_bytes=compressed_size,
                compression_ratio=original_size / compressed_size,
                compress_time_ms=compress_time,
                decompress_time_ms=decompress_time,
                total_time_ms=compress_time + decompress_time,
                **metrics,
            ))
        except Exception as e:
            logger.warning(f"Int8-Uniform benchmark failed: {e}")
        
        # Int8 Block-wise
        try:
            compressed, compress_time = self.compress_int8_blockwise(tensor, block_size=64)
            decompressed, decompress_time = self.decompress_int8_blockwise(compressed, original_dtype)
            
            metrics = self.compute_accuracy_metrics(tensor, decompressed)
            compressed_size = compressed[0].numel() * compressed[0].element_size()
            
            results.append(CompressionResult(
                algorithm="Int8-Blockwise-64",
                category="lossy",
                original_size_bytes=original_size,
                compressed_size_bytes=compressed_size,
                compression_ratio=original_size / compressed_size,
                compress_time_ms=compress_time,
                decompress_time_ms=decompress_time,
                total_time_ms=compress_time + decompress_time,
                **metrics,
            ))
        except Exception as e:
            logger.warning(f"Int8-Blockwise benchmark failed: {e}")
        
        # Int4 Group-wise
        try:
            compressed, compress_time = self.compress_int4_groupwise(tensor, group_size=32)
            decompressed, decompress_time = self.decompress_int4_groupwise(compressed, original_dtype)
            
            metrics = self.compute_accuracy_metrics(tensor, decompressed)
            compressed_size = compressed[0].numel() * compressed[0].element_size()
            
            results.append(CompressionResult(
                algorithm="Int4-Groupwise-32",
                category="lossy",
                original_size_bytes=original_size,
                compressed_size_bytes=compressed_size,
                compression_ratio=original_size / compressed_size,
                compress_time_ms=compress_time,
                decompress_time_ms=decompress_time,
                total_time_ms=compress_time + decompress_time,
                **metrics,
            ))
        except Exception as e:
            logger.warning(f"Int4-Groupwise benchmark failed: {e}")
        
        return results
    
    def benchmark_tensor(self, tensor: torch.Tensor) -> List[CompressionResult]:
        """Run all benchmarks on a single tensor."""
        results = []
        
        # Ensure tensor is on CPU and contiguous
        tensor = tensor.detach().cpu().contiguous()
        
        # Lossless
        results.extend(self.benchmark_lossless(tensor))
        
        # Lossy
        results.extend(self.benchmark_lossy(tensor))
        
        return results
    
    def benchmark_directory(self, input_dir: str) -> Dict:
        """Benchmark all activation files in a directory."""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")
        
        # Find all activation files
        pt_files = list(input_path.glob("*.pt"))
        npz_files = list(input_path.glob("*.npz"))
        
        all_files = pt_files + npz_files
        if not all_files:
            raise ValueError(f"No activation files found in {input_dir}")
        
        logger.info(f"Found {len(all_files)} activation files")
        
        all_results = []
        
        for i, filepath in enumerate(all_files):
            logger.info(f"Processing [{i+1}/{len(all_files)}]: {filepath.name}")
            
            # Load tensor
            if filepath.suffix == ".pt":
                tensor = torch.load(filepath)
            else:
                data = np.load(filepath)
                tensor = torch.from_numpy(data["data"])
            
            # Benchmark
            results = self.benchmark_tensor(tensor)
            
            # Add file info
            for r in results:
                all_results.append({
                    "file": filepath.name,
                    **asdict(r),
                })
        
        return self.aggregate_results(all_results)
    
    def aggregate_results(self, all_results: List[Dict]) -> Dict:
        """Aggregate results across all files."""
        if not all_results:
            return {}
        
        # Group by algorithm
        by_algorithm = {}
        for r in all_results:
            algo = r["algorithm"]
            if algo not in by_algorithm:
                by_algorithm[algo] = []
            by_algorithm[algo].append(r)
        
        # Compute averages
        summary = []
        for algo, results in by_algorithm.items():
            avg_ratio = np.mean([r["compression_ratio"] for r in results])
            avg_compress = np.mean([r["compress_time_ms"] for r in results])
            avg_decompress = np.mean([r["decompress_time_ms"] for r in results])
            avg_total = np.mean([r["total_time_ms"] for r in results])
            
            entry = {
                "algorithm": algo,
                "category": results[0]["category"],
                "avg_compression_ratio": avg_ratio,
                "avg_compress_time_ms": avg_compress,
                "avg_decompress_time_ms": avg_decompress,
                "avg_total_time_ms": avg_total,
                "num_samples": len(results),
            }
            
            # Lossy metrics
            if results[0].get("mse") is not None:
                entry["avg_mse"] = np.mean([r["mse"] for r in results])
                entry["avg_max_abs_error"] = np.mean([r["max_abs_error"] for r in results])
                entry["avg_relative_error_percent"] = np.mean([r["relative_error_percent"] for r in results])
                entry["avg_cosine_similarity"] = np.mean([r["cosine_similarity"] for r in results])
                entry["avg_snr_db"] = np.mean([r["snr_db"] for r in results if r["snr_db"] != float('inf')])
            
            summary.append(entry)
        
        return {
            "summary": summary,
            "details": all_results,
        }


def print_results_table(results: Dict):
    """Print results as a formatted table."""
    summary = results.get("summary", [])
    
    if not summary:
        print("No results to display")
        return
    
    # Separate lossless and lossy
    lossless = [r for r in summary if r["category"] == "lossless"]
    lossy = [r for r in summary if r["category"] == "lossy"]
    
    if HAS_TABULATE:
        # Lossless table
        if lossless:
            print("\n" + "=" * 80)
            print("LOSSLESS COMPRESSION RESULTS")
            print("=" * 80)
            headers = ["Algorithm", "Ratio", "Compress(ms)", "Decompress(ms)", "Total(ms)"]
            rows = [[
                r["algorithm"],
                f"{r['avg_compression_ratio']:.2f}x",
                f"{r['avg_compress_time_ms']:.2f}",
                f"{r['avg_decompress_time_ms']:.2f}",
                f"{r['avg_total_time_ms']:.2f}",
            ] for r in lossless]
            print(tabulate(rows, headers=headers, tablefmt="grid"))
        
        # Lossy table
        if lossy:
            print("\n" + "=" * 80)
            print("LOSSY COMPRESSION RESULTS")
            print("=" * 80)
            headers = ["Algorithm", "Ratio", "Comp(ms)", "Decomp(ms)", "MSE", "Cosine Sim", "SNR(dB)"]
            rows = [[
                r["algorithm"],
                f"{r['avg_compression_ratio']:.2f}x",
                f"{r['avg_compress_time_ms']:.2f}",
                f"{r['avg_decompress_time_ms']:.2f}",
                f"{r.get('avg_mse', 'N/A'):.6f}" if r.get('avg_mse') else "N/A",
                f"{r.get('avg_cosine_similarity', 'N/A'):.6f}" if r.get('avg_cosine_similarity') else "N/A",
                f"{r.get('avg_snr_db', 'N/A'):.1f}" if r.get('avg_snr_db') else "N/A",
            ] for r in lossy]
            print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        # Simple fallback
        print("\nResults Summary:")
        for r in summary:
            print(f"  {r['algorithm']}: {r['avg_compression_ratio']:.2f}x | "
                  f"Compress: {r['avg_compress_time_ms']:.2f}ms | "
                  f"Decompress: {r['avg_decompress_time_ms']:.2f}ms")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark compression algorithms for activation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing activation .pt files"
    )
    parser.add_argument(
        "--output_report", type=str, default=None,
        help="Output JSON report file path"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed results"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not HAS_LZ4 and not HAS_ZSTD:
        print("WARNING: No lossless compression libraries found.")
        print("Install with: pip install lz4 zstandard")
    
    # Run benchmark
    benchmark = CompressionBenchmark()
    
    try:
        results = benchmark.benchmark_directory(args.input_dir)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1
    
    # Print table
    print_results_table(results)
    
    # Save report
    if args.output_report:
        with open(args.output_report, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nReport saved to: {args.output_report}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
