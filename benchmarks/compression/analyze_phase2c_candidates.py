#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

DEFAULT_BANDWIDTHS_MBPS = (50.0, 100.0, 150.0, 300.0, 1000.0)
GROUP_FIELDS = ("phase", "direction", "channel", "shape", "codec")
OUTPUT_FIELDS = (
    "phase",
    "direction",
    "channel",
    "shape",
    "codec",
    "n",
    "ratio_median",
    "compress_plus_decompress_ms_median",
    "wire_bytes_median",
    "bandwidth_mbps",
    "modeled_total_ms_median",
)


def _parse_bandwidths(raw: str) -> List[float]:
    if not raw.strip():
        return list(DEFAULT_BANDWIDTHS_MBPS)
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _modeled_total_ms(row: Dict[str, str], bandwidth_mbps: float) -> float:
    cpu_ms = float(row["compress_ms"]) + float(row["decompress_ms"])
    wire_bytes = float(row["wire_bytes"])
    return cpu_ms + (wire_bytes * 8.0 / (bandwidth_mbps * 1000.0))


def read_rows(input_csv: Path) -> List[Dict[str, str]]:
    with input_csv.open() as f:
        return list(csv.DictReader(f))


def aggregate(rows: Sequence[Dict[str, str]], bandwidths: Iterable[float]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(row.get("available", "0")) and int(row.get("roundtrip_ok", "0")):
            grouped[tuple(row.get(field, "") for field in GROUP_FIELDS)].append(row)

    output: List[Dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        ratio_median = statistics.median(float(row["ratio"]) for row in values)
        cpu_median = statistics.median(float(row["compress_ms"]) + float(row["decompress_ms"]) for row in values)
        wire_median = statistics.median(float(row["wire_bytes"]) for row in values)
        for bandwidth in bandwidths:
            output.append(
                {
                    **dict(zip(GROUP_FIELDS, key)),
                    "n": len(values),
                    "ratio_median": f"{ratio_median:.6f}",
                    "compress_plus_decompress_ms_median": f"{cpu_median:.6f}",
                    "wire_bytes_median": f"{wire_median:.1f}",
                    "bandwidth_mbps": f"{bandwidth:.0f}" if bandwidth.is_integer() else f"{bandwidth:.3f}",
                    "modeled_total_ms_median": f"{statistics.median(_modeled_total_ms(row, bandwidth) for row in values):.6f}",
                }
            )
    return output


def write_csv(rows: Sequence[Dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate Phase 2C codec candidates with bandwidth-modeled latency")
    parser.add_argument("--input_csv", required=True, help="CSV emitted by benchmark_lossless_codecs.py")
    parser.add_argument("--output_csv", required=True, help="Aggregated modeled-latency CSV")
    parser.add_argument(
        "--bandwidths-mbps",
        default=",".join(str(int(value)) for value in DEFAULT_BANDWIDTHS_MBPS),
        help="Comma-separated bandwidth grid in Mbps",
    )
    args = parser.parse_args()

    rows = aggregate(read_rows(Path(args.input_csv)), _parse_bandwidths(args.bandwidths_mbps))
    write_csv(rows, Path(args.output_csv))
    print(f"aggregated_rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
