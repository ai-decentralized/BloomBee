#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_json(path: Path):
    return json.loads(path.read_text())


def parse_layer(block_uid: str) -> Optional[int]:
    match = re.search(r"\.(\d+)$", block_uid or "")
    return int(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join activation metadata with compression benchmark results and summarize by layer"
    )
    parser.add_argument("--metadata", type=Path, required=True, help="Path to activation metadata.json")
    parser.add_argument("--report", type=Path, required=True, help="Path to benchmark JSON report")
    parser.add_argument(
        "--variant",
        default=None,
        help="Optional variant filter for bit-level reports, e.g. byte_high_baseline or packed_signexp_mhi0",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    args = parser.parse_args()

    metadata = load_json(args.metadata)
    report = load_json(args.report)
    if isinstance(report, dict):
        report = report.get("results", [])

    filtered_report: List[Dict[str, object]] = []
    for row in report:
        if args.variant is not None and row.get("variant") != args.variant:
            continue
        filtered_report.append(row)

    if not filtered_report:
        raise SystemExit("No matching rows in report. Check --variant or input report path.")

    by_filename = {Path(row["name"]).name: row for row in filtered_report}
    rows = []
    for sample in metadata.get("samples", []):
        filename = sample["filename"]
        result = by_filename.get(filename)
        if result is None:
            continue
        rows.append(
            {
                "layer": parse_layer(sample.get("block_uid", "")),
                "block_uid": sample.get("block_uid", ""),
                "filename": filename,
                "dtype": sample.get("dtype", ""),
                "shape": sample.get("shape", []),
                "direct_ratio": float(result["direct_zstd_ratio"]),
                "split_ratio": float(result["split_total_with_header_ratio"]),
                "delta_pct_vs_direct": float(result["split_vs_direct_delta_pct"]),
                "raw_bytes": int(result["raw_bytes"]),
                "direct_zstd_bytes": int(result["direct_zstd_bytes"]),
                "split_total_with_header_bytes": int(result["split_total_with_header_bytes"]),
            }
        )

    rows.sort(key=lambda item: (item["layer"] is None, item["layer"], item["block_uid"]))
    if not rows:
        raise SystemExit("No rows matched between metadata and report.")

    print("layer\tblock_uid\tdtype\tdirect_ratio\tsplit_ratio\tdelta_pct_vs_direct")
    for row in rows:
        print(
            f"{row['layer']}\t{row['block_uid']}\t{row['dtype']}\t"
            f"{row['direct_ratio']:.6f}\t{row['split_ratio']:.6f}\t{row['delta_pct_vs_direct']:+.2f}%"
        )

    layers = np.array([row["layer"] for row in rows if row["layer"] is not None], dtype=float)
    direct = np.array([row["direct_ratio"] for row in rows if row["layer"] is not None], dtype=float)
    split = np.array([row["split_ratio"] for row in rows if row["layer"] is not None], dtype=float)
    improve = np.array([-row["delta_pct_vs_direct"] for row in rows if row["layer"] is not None], dtype=float)
    if layers.size >= 2:
        direct_corr = float(np.corrcoef(layers, direct)[0, 1])
        split_corr = float(np.corrcoef(layers, split)[0, 1])
        improve_corr = float(np.corrcoef(layers, improve)[0, 1])
        direct_slope = float(np.polyfit(layers, direct, 1)[0])
        split_slope = float(np.polyfit(layers, split, 1)[0])
        improve_slope = float(np.polyfit(layers, improve, 1)[0])
        print()
        print(
            "trend\t"
            f"direct_ratio_corr={direct_corr:+.4f}\t"
            f"split_ratio_corr={split_corr:+.4f}\t"
            f"improvement_corr={improve_corr:+.4f}"
        )
        print(
            "slope_per_layer\t"
            f"direct_ratio={direct_slope:+.6f}\t"
            f"split_ratio={split_slope:+.6f}\t"
            f"improvement_pct={improve_slope:+.6f}"
        )

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "layer",
            "block_uid",
            "filename",
            "dtype",
            "direct_ratio",
            "split_ratio",
            "delta_pct_vs_direct",
            "raw_bytes",
            "direct_zstd_bytes",
            "split_total_with_header_bytes",
        ]
        lines = [",".join(header)]
        for row in rows:
            lines.append(
                ",".join(
                    [
                        str(row["layer"]),
                        str(row["block_uid"]),
                        str(row["filename"]),
                        str(row["dtype"]),
                        f"{row['direct_ratio']:.6f}",
                        f"{row['split_ratio']:.6f}",
                        f"{row['delta_pct_vs_direct']:+.2f}",
                        str(row["raw_bytes"]),
                        str(row["direct_zstd_bytes"]),
                        str(row["split_total_with_header_bytes"]),
                    ]
                )
            )
        args.csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Saved CSV: {args.csv}")


if __name__ == "__main__":
    main()
