#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

TARGET_BINS = ["0.1~1", "1~10", "10~100", "100~1000"]


def parse_kv_pairs(line: str) -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        pairs[key] = value
    return pairs


def parse_hist(hist_raw: str) -> List[Tuple[str, float]]:
    parts: List[Tuple[str, float]] = []
    for item in hist_raw.split(","):
        if ":" not in item:
            continue
        label, value = item.split(":", 1)
        try:
            parts.append((label, float(value)))
        except Exception:
            continue
    return parts


def parse_hist_counts(hist_raw: str) -> List[Tuple[str, int]]:
    parts: List[Tuple[str, int]] = []
    for item in hist_raw.split(","):
        if ":" not in item:
            continue
        label, value = item.split(":", 1)
        try:
            parts.append((label, int(value)))
        except Exception:
            continue
    return parts


def row_matches(row: Dict[str, str], args: argparse.Namespace) -> bool:
    if args.phase and row.get("phase") != args.phase:
        return False
    if args.source and row.get("source") != args.source:
        return False
    if args.channel and row.get("channel") != args.channel:
        return False
    if args.blocks and row.get("blocks") != args.blocks:
        return False
    if args.tensor_name and row.get("tensor_name") != args.tensor_name:
        return False
    if args.batch is not None and row.get("batch") != str(args.batch):
        return False
    return True


def aggregate_hist(lines: List[str], args: argparse.Namespace) -> Tuple[List[str], List[float], int]:
    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    matched = 0
    label_order: List[str] = []
    count_sums: Dict[str, float] = defaultdict(float)
    sample_size_sum = 0.0

    for line in lines:
        if "[ACT_DIST]" not in line:
            continue
        row = parse_kv_pairs(line)
        if not row_matches(row, args):
            continue
        if args.y == "count":
            hist_c = parse_hist_counts(row.get("act_abs_bin_counts", ""))
            sample_size_sum += float(row.get("act_sample_size", "0") or 0)
            hist = [(label, float(value)) for label, value in hist_c]
        else:
            hist = parse_hist(row.get("act_abs_bin_ratio", "")) or parse_hist(row.get("act_abs_decade_hist", ""))
        if not hist:
            continue
        matched += 1
        if not label_order:
            label_order = [label for label, _ in hist]
        for label, ratio in hist:
            sums[label] += ratio
            counts[label] += 1
            count_sums[label] += ratio

    if matched <= 0:
        return [], [], 0

    labels = [label for label in TARGET_BINS if label in sums]
    if not labels:
        labels = label_order or sorted(sums.keys())
    if args.y == "count":
        values = [count_sums[label] / float(matched) for label in labels]
    else:
        values = [sums[label] / max(1, counts[label]) for label in labels]
    return labels, values, matched


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BloomBee [ACT_DIST] decade histogram from logs")
    parser.add_argument("--log", required=True, type=Path, help="Path to log file")
    parser.add_argument("--phase", default="decode", help="Filter by phase")
    parser.add_argument("--source", default="client", help="Filter by source")
    parser.add_argument("--channel", default="rpc_inference", help="Filter by channel")
    parser.add_argument("--blocks", default="0:20", help="Filter by blocks")
    parser.add_argument("--tensor-name", default="hidden_states", help="Filter by tensor_name")
    parser.add_argument("--batch", type=int, default=None, help="Filter by batch size")
    parser.add_argument("--y", choices=["ratio", "count"], default="ratio", help="Y-axis type")
    parser.add_argument("--png", type=Path, default=None, help="Output png path")
    args = parser.parse_args()

    content = args.log.read_text(encoding="utf-8", errors="ignore").splitlines()
    labels, values, matched = aggregate_hist(content, args)
    if matched <= 0:
        print("No matching [ACT_DIST] lines found.")
        return

    print(f"Matched lines: {matched}")
    for label, value in zip(labels, values):
        if args.y == "count":
            print(f"{label}\t{int(round(value))}")
        else:
            print(f"{label}\t{value:.6f}")

    if args.png is None:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not available; skip plotting.")
        return

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Average Percentage" if args.y == "ratio" else "Average Count")
    plt.title(
        f"ACT_DIST avg ({matched} samples) "
        f"phase={args.phase} source={args.source} channel={args.channel} "
        f"blocks={args.blocks} tensor={args.tensor_name} y={args.y}"
    )
    plt.tight_layout()
    args.png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.png, dpi=160)
    print(f"Saved plot: {args.png}")


if __name__ == "__main__":
    main()
