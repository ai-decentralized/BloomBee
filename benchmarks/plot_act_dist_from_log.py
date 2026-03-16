#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

TARGET_BINS = ["0.1~1", "1~10", "10~100", "100~1000"]
TARGET_BINS_WITH_TAILS = ["<0.1", "0.1~1", "1~10", "10~100", "100~1000", ">=1000"]


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


def parse_blocks(blocks_raw: str) -> List[str]:
    blocks = [item.strip() for item in blocks_raw.split(",") if item.strip()]
    return blocks or ["0:20"]


def row_matches(row: Dict[str, str], args: argparse.Namespace, block: str) -> bool:
    if args.phase and row.get("phase") != args.phase:
        return False
    if args.source and row.get("source") != args.source:
        return False
    if args.channel and row.get("channel") != args.channel:
        return False
    if block and row.get("blocks") != block:
        return False
    if args.tensor_name and row.get("tensor_name") != args.tensor_name:
        return False
    if args.batch is not None and row.get("batch") != str(args.batch):
        return False
    return True


def _extract_hist_values(row: Dict[str, str], args: argparse.Namespace) -> Dict[str, float]:
    hist: Dict[str, float] = {}
    if args.y == "count":
        for label, value in parse_hist_counts(row.get("act_abs_bin_counts", "")):
            hist[label] = float(value)
        if args.include_tails:
            hist["<0.1"] = float(row.get("act_abs_lt_0_1_count", "0") or 0.0)
            hist[">=1000"] = float(row.get("act_abs_ge_1000_count", "0") or 0.0)
    else:
        parsed = parse_hist(row.get("act_abs_bin_ratio", "")) or parse_hist(row.get("act_abs_decade_hist", ""))
        for label, value in parsed:
            hist[label] = float(value)
        if args.include_tails:
            hist["<0.1"] = float(row.get("act_abs_lt_0_1_ratio", "0") or 0.0)
            hist[">=1000"] = float(row.get("act_abs_ge_1000_ratio", "0") or 0.0)
    return hist


def aggregate_hist(lines: List[str], args: argparse.Namespace, block: str) -> Tuple[List[str], List[float], int]:
    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    matched = 0

    for line in lines:
        if "[ACT_DIST]" not in line:
            continue
        row = parse_kv_pairs(line)
        if not row_matches(row, args, block):
            continue
        hist = _extract_hist_values(row, args)
        if not hist:
            continue
        matched += 1
        for label, value in hist.items():
            sums[label] += value
            counts[label] += 1

    if matched <= 0:
        return [], [], 0

    preferred_bins = TARGET_BINS_WITH_TAILS if args.include_tails else TARGET_BINS
    labels = [label for label in preferred_bins if label in sums]
    if not labels:
        labels = sorted(sums.keys())
    values = [sums[label] / max(1, counts[label]) for label in labels]
    return labels, values, matched


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BloomBee [ACT_DIST] decade histogram from logs")
    parser.add_argument("--log", required=True, type=Path, help="Path to log file")
    parser.add_argument("--phase", default="decode", help="Filter by phase")
    parser.add_argument("--source", default="client", help="Filter by source")
    parser.add_argument("--channel", default="rpc_inference", help="Filter by channel")
    parser.add_argument("--blocks", default="0:20", help="Filter by blocks; use comma to compare, e.g. 0:20,20:40")
    parser.add_argument("--tensor-name", default="hidden_states", help="Filter by tensor_name")
    parser.add_argument("--batch", type=int, default=None, help="Filter by batch size")
    parser.add_argument("--y", choices=["ratio", "count"], default="ratio", help="Y-axis type")
    parser.add_argument(
        "--include-tails",
        type=int,
        choices=[0, 1],
        default=1,
        help="Include <0.1 and >=1000 bins in addition to decade bins (default: 1)",
    )
    parser.add_argument("--png", type=Path, default=None, help="Output png path")
    args = parser.parse_args()
    args.include_tails = bool(args.include_tails)

    content = args.log.read_text(encoding="utf-8", errors="ignore").splitlines()
    block_list = parse_blocks(args.blocks)

    results: Dict[str, Tuple[List[str], List[float], int]] = {}
    for block in block_list:
        labels, values, matched = aggregate_hist(content, args, block)
        if matched > 0:
            results[block] = (labels, values, matched)

    if not results:
        print("No matching [ACT_DIST] lines found.")
        return

    for block in block_list:
        if block not in results:
            print(f"Block {block}: no matching lines")
            continue
        labels, values, matched = results[block]
        print(f"Block {block} matched lines: {matched}")
        for label, value in zip(labels, values):
            if args.y == "count":
                print(f"{label}\t{int(round(value))}")
            else:
                print(f"{label}\t{value:.6f}")
        print("")

    if args.png is None:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not available; skip plotting.")
        return

    preferred_bins = TARGET_BINS_WITH_TAILS if args.include_tails else TARGET_BINS
    all_labels = [label for label in preferred_bins if any(label in results[b][0] for b in results)]
    if not all_labels:
        all_labels = sorted({label for labels, _, _ in results.values() for label in labels})

    import numpy as np

    x = np.arange(len(all_labels))
    n_blocks = len([b for b in block_list if b in results])
    width = 0.8 / max(1, n_blocks)
    plt.figure(figsize=(12, 4.8))

    plotted_index = 0
    for block in block_list:
        if block not in results:
            continue
        labels, values, _ = results[block]
        value_by_label = dict(zip(labels, values))
        ys = [value_by_label.get(label, 0.0) for label in all_labels]
        shift = (plotted_index - (n_blocks - 1) / 2.0) * width
        plt.bar(x + shift, ys, width=width, label=f"blocks {block}")
        plotted_index += 1

    plt.xticks(x, all_labels, rotation=45, ha="right")
    plt.ylabel("Average Ratio" if args.y == "ratio" else "Average Count per sample")
    plt.title(
        f"ACT_DIST comparison "
        f"phase={args.phase} source={args.source} channel={args.channel} "
        f"blocks={','.join(block_list)} tensor={args.tensor_name} y={args.y}"
    )
    if n_blocks > 1:
        plt.legend()
    plt.tight_layout()
    args.png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.png, dpi=160)
    print(f"Saved plot: {args.png}")


if __name__ == "__main__":
    main()
