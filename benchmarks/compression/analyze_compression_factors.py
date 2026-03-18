#!/usr/bin/env python3
import argparse
import math
import re
from collections import defaultdict
from pathlib import Path


COMP_RATIO_RE = re.compile(
    r"\[COMP_RATIO\]\s+"
    r"source=(?P<source>\S+)\s+"
    r"channel=(?P<channel>\S+)\s+"
    r"blocks=(?P<blocks>\S+)\s+"
    r"step_id=(?P<step_id>\S+)\s+"
    r"batch=(?P<batch>\d+)\s+"
    r"tensor=(?P<tensor>\S+)\s+"
    r"raw_bytes=(?P<raw_bytes>\d+)\s+"
    r"wire_bytes=(?P<wire_bytes>\d+)\s+"
    r"ratio=(?P<ratio>-?[0-9.]+)\s+"
    r"savings=(?P<savings>-?[0-9.]+)\s+"
    r"nnz=(?P<nnz>-?[0-9.]+)"
)


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def _safe_mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _weighted_ratio(rows):
    raw_total = sum(r["raw_bytes"] for r in rows)
    wire_total = sum(r["wire_bytes"] for r in rows)
    return (wire_total / raw_total) if raw_total > 0 else 1.0


def _pearson(xs, ys):
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = _safe_mean(xs)
    my = _safe_mean(ys)
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        dy = y - my
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    den = math.sqrt(max(den_x * den_y, 0.0))
    return (num / den) if den > 0 else 0.0


def _parse_block_key(s: str):
    m = re.match(r"(?P<a>\d+):(?P<b>\d+)(?:->(?P<c>\d+):(?P<d>\d+))?", s)
    if not m:
        return (10**9, 10**9, s)
    return (int(m.group("a")), int(m.group("b")), s)


def parse_comp_rows(logs):
    rows = []
    for source_name, path in logs:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = COMP_RATIO_RE.search(line)
                if not m:
                    continue
                source_role = source_name.split(".")[-1]
                rows.append(
                    {
                        "source": source_role,
                        "source_tag": source_name,
                        "record_source": m.group("source"),
                        "channel": m.group("channel"),
                        "blocks": m.group("blocks"),
                        "step_id": m.group("step_id"),
                        "batch": int(m.group("batch")),
                        "tensor": m.group("tensor"),
                        "raw_bytes": int(m.group("raw_bytes")),
                        "wire_bytes": int(m.group("wire_bytes")),
                        "ratio": float(m.group("ratio")),
                        "nnz": float(m.group("nnz")),
                    }
                )
    return rows


def summarize_rows(rows):
    out = {
        "count": len(rows),
        "overall_weighted_ratio": _weighted_ratio(rows),
        "overall_mean_ratio": _safe_mean([r["ratio"] for r in rows]) if rows else 0.0,
        "overall_mean_nnz": _safe_mean([r["nnz"] for r in rows]) if rows else 0.0,
        "by_source": {},
        "by_channel": {},
        "by_source_channel": {},
        "by_blocks": {},
        "by_batch": {},
        "pearson_ratio_vs_nnz": 0.0,
        "median_nnz": 0.0,
        "low_nnz_weighted_ratio": 1.0,
        "high_nnz_weighted_ratio": 1.0,
    }
    if not rows:
        return out

    by_source = defaultdict(list)
    by_channel = defaultdict(list)
    by_source_channel = defaultdict(list)
    by_blocks = defaultdict(list)
    by_batch = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
        by_channel[r["channel"]].append(r)
        by_source_channel[(r["source"], r["channel"])].append(r)
        by_blocks[r["blocks"]].append(r)
        by_batch[r["batch"]].append(r)
    for k, group in by_source.items():
        out["by_source"][k] = {
            "count": len(group),
            "weighted_ratio": _weighted_ratio(group),
            "mean_ratio": _safe_mean([x["ratio"] for x in group]),
            "mean_nnz": _safe_mean([x["nnz"] for x in group]),
        }
    for k, group in by_channel.items():
        out["by_channel"][k] = {
            "count": len(group),
            "weighted_ratio": _weighted_ratio(group),
            "mean_ratio": _safe_mean([x["ratio"] for x in group]),
            "mean_nnz": _safe_mean([x["nnz"] for x in group]),
        }
    for k, group in by_source_channel.items():
        out["by_source_channel"][k] = {
            "count": len(group),
            "weighted_ratio": _weighted_ratio(group),
            "mean_ratio": _safe_mean([x["ratio"] for x in group]),
            "mean_nnz": _safe_mean([x["nnz"] for x in group]),
        }
    for k, group in by_blocks.items():
        out["by_blocks"][k] = {
            "count": len(group),
            "weighted_ratio": _weighted_ratio(group),
            "mean_ratio": _safe_mean([x["ratio"] for x in group]),
            "mean_nnz": _safe_mean([x["nnz"] for x in group]),
        }
    for k, group in by_batch.items():
        out["by_batch"][k] = {
            "count": len(group),
            "weighted_ratio": _weighted_ratio(group),
            "mean_ratio": _safe_mean([x["ratio"] for x in group]),
            "mean_nnz": _safe_mean([x["nnz"] for x in group]),
        }

    ratios = [r["ratio"] for r in rows]
    nnzs = [r["nnz"] for r in rows]
    out["pearson_ratio_vs_nnz"] = _pearson(nnzs, ratios)
    sorted_nnz = sorted(nnzs)
    out["median_nnz"] = sorted_nnz[len(sorted_nnz) // 2]
    low_rows = [r for r in rows if r["nnz"] <= out["median_nnz"]]
    high_rows = [r for r in rows if r["nnz"] > out["median_nnz"]]
    out["low_nnz_weighted_ratio"] = _weighted_ratio(low_rows)
    out["high_nnz_weighted_ratio"] = _weighted_ratio(high_rows)
    return out


def print_summary(title: str, summary):
    print("=" * 96)
    print(title)
    print("=" * 96)
    print("Method / formulas:")
    print("1) ratio = wire_bytes / raw_bytes  (lower is better)")
    print("2) weighted ratio (group) = sum(wire_bytes) / sum(raw_bytes)")
    print("3) nnz_ratio = nonzero_count / numel")
    print("4) Pearson r(ratio, nnz) = cov(ratio, nnz) / (std(ratio) * std(nnz))")
    print("-" * 96)
    if summary["count"] == 0:
        print("No [COMP_RATIO] rows found.")
        print()
        return

    print(
        f"overall: count={summary['count']}, "
        f"weighted_ratio={_fmt(summary['overall_weighted_ratio'])}, "
        f"mean_ratio={_fmt(summary['overall_mean_ratio'])}, "
        f"mean_nnz={_fmt(summary['overall_mean_nnz'])}"
    )
    print()
    print("By source file:")
    print(f"{'source':26} {'w_ratio':>10} {'mean_ratio':>12} {'mean_nnz':>10} {'count':>8}")
    for source in sorted(summary["by_source"]):
        item = summary["by_source"][source]
        print(
            f"{source:26} "
            f"{_fmt(item['weighted_ratio']):>10} "
            f"{_fmt(item['mean_ratio']):>12} "
            f"{_fmt(item['mean_nnz']):>10} "
            f"{item['count']:>8}"
        )
    print()
    print("By channel:")
    print(f"{'channel':26} {'w_ratio':>10} {'mean_ratio':>12} {'mean_nnz':>10} {'count':>8}")
    for channel in sorted(summary["by_channel"]):
        item = summary["by_channel"][channel]
        print(
            f"{channel:26} "
            f"{_fmt(item['weighted_ratio']):>10} "
            f"{_fmt(item['mean_ratio']):>12} "
            f"{_fmt(item['mean_nnz']):>10} "
            f"{item['count']:>8}"
        )
    print()
    print("By source+channel:")
    print(f"{'source':14} {'channel':20} {'w_ratio':>10} {'mean_ratio':>12} {'count':>8}")
    for source, channel in sorted(summary["by_source_channel"]):
        item = summary["by_source_channel"][(source, channel)]
        print(
            f"{source:14} {channel:20} "
            f"{_fmt(item['weighted_ratio']):>10} "
            f"{_fmt(item['mean_ratio']):>12} "
            f"{item['count']:>8}"
        )
    print()
    print("By layer/span:")
    print(f"{'blocks':26} {'w_ratio':>10} {'mean_ratio':>12} {'mean_nnz':>10} {'count':>8}")
    for blocks in sorted(summary["by_blocks"], key=_parse_block_key):
        item = summary["by_blocks"][blocks]
        print(
            f"{blocks:26} "
            f"{_fmt(item['weighted_ratio']):>10} "
            f"{_fmt(item['mean_ratio']):>12} "
            f"{_fmt(item['mean_nnz']):>10} "
            f"{item['count']:>8}"
        )
    print()
    print("By batch:")
    print(f"{'batch':>8} {'w_ratio':>10} {'mean_ratio':>12} {'mean_nnz':>10} {'count':>8}")
    for batch in sorted(summary["by_batch"]):
        item = summary["by_batch"][batch]
        print(
            f"{batch:>8} "
            f"{_fmt(item['weighted_ratio']):>10} "
            f"{_fmt(item['mean_ratio']):>12} "
            f"{_fmt(item['mean_nnz']):>10} "
            f"{item['count']:>8}"
        )
    print()
    print(
        f"nnz relation: pearson_r={_fmt(summary['pearson_ratio_vs_nnz'])}, "
        f"median_nnz={_fmt(summary['median_nnz'])}, "
        f"low_nnz_w_ratio={_fmt(summary['low_nnz_weighted_ratio'])}, "
        f"high_nnz_w_ratio={_fmt(summary['high_nnz_weighted_ratio'])}"
    )
    print()


def print_delta(before, after):
    print("=" * 96)
    print("Before/After Delta (after - before)")
    print("=" * 96)
    print(
        f"overall weighted_ratio: {before['overall_weighted_ratio']:.4f} -> {after['overall_weighted_ratio']:.4f} "
        f"(delta={after['overall_weighted_ratio']-before['overall_weighted_ratio']:+.4f})"
    )
    print(
        f"overall mean_ratio    : {before['overall_mean_ratio']:.4f} -> {after['overall_mean_ratio']:.4f} "
        f"(delta={after['overall_mean_ratio']-before['overall_mean_ratio']:+.4f})"
    )
    print(
        f"pearson(ratio,nnz)    : {before['pearson_ratio_vs_nnz']:.4f} -> {after['pearson_ratio_vs_nnz']:.4f} "
        f"(delta={after['pearson_ratio_vs_nnz']-before['pearson_ratio_vs_nnz']:+.4f})"
    )
    print("-" * 96)
    print("By source+channel (intersection):")
    skeys = sorted(set(before["by_source_channel"]).intersection(set(after["by_source_channel"])))
    if not skeys:
        print("No shared source+channel keys.")
    else:
        print(f"{'source':14} {'channel':20} {'before_w':>10} {'after_w':>10} {'delta':>10}")
        for key in skeys:
            source, channel = key
            b = before["by_source_channel"][key]["weighted_ratio"]
            a = after["by_source_channel"][key]["weighted_ratio"]
            print(f"{source:14} {channel:20} {b:>10.4f} {a:>10.4f} {a-b:>10.4f}")
    print("-" * 96)
    print("By layer/span (intersection):")
    keys = sorted(set(before["by_blocks"]).intersection(set(after["by_blocks"])), key=_parse_block_key)
    if not keys:
        print("No shared layer/span keys.")
    else:
        print(f"{'blocks':26} {'before_w':>10} {'after_w':>10} {'delta':>10}")
        for k in keys:
            b = before["by_blocks"][k]["weighted_ratio"]
            a = after["by_blocks"][k]["weighted_ratio"]
            print(f"{k:26} {b:>10.4f} {a:>10.4f} {a-b:>10.4f}")
    print("-" * 96)
    print("By batch (intersection):")
    bkeys = sorted(set(before["by_batch"]).intersection(set(after["by_batch"])))
    if not bkeys:
        print("No shared batch keys.")
    else:
        print(f"{'batch':>8} {'before_w':>10} {'after_w':>10} {'delta':>10}")
        for k in bkeys:
            b = before["by_batch"][k]["weighted_ratio"]
            a = after["by_batch"][k]["weighted_ratio"]
            print(f"{k:>8} {b:>10.4f} {a:>10.4f} {a-b:>10.4f}")
    print()


def _run_logs(label, client_log, server1_log, server2_log):
    logs = [
        (f"{label}.client", client_log),
        (f"{label}.server1", server1_log),
        (f"{label}.server2", server2_log),
    ]
    rows = parse_comp_rows(logs)
    return summarize_rows(rows)


def main():
    parser = argparse.ArgumentParser(description="Analyze compression-ratio factors (layer/batch/nnz) from [COMP_RATIO] logs.")
    parser.add_argument("--client-log", type=Path, help="Single-run client log")
    parser.add_argument("--server1-log", type=Path, help="Single-run server1 log")
    parser.add_argument("--server2-log", type=Path, help="Single-run server2 log")
    parser.add_argument("--before-client-log", type=Path, help="Before-run client log")
    parser.add_argument("--before-server1-log", type=Path, help="Before-run server1 log")
    parser.add_argument("--before-server2-log", type=Path, help="Before-run server2 log")
    parser.add_argument("--after-client-log", type=Path, help="After-run client log")
    parser.add_argument("--after-server1-log", type=Path, help="After-run server1 log")
    parser.add_argument("--after-server2-log", type=Path, help="After-run server2 log")
    args = parser.parse_args()

    single_mode = args.client_log and args.server1_log and args.server2_log
    compare_mode = (
        args.before_client_log
        and args.before_server1_log
        and args.before_server2_log
        and args.after_client_log
        and args.after_server1_log
        and args.after_server2_log
    )
    if not single_mode and not compare_mode:
        parser.error(
            "Use either single-run logs (--client-log/--server1-log/--server2-log) "
            "or before/after logs (--before-*/--after-*)."
        )

    if single_mode:
        summary = _run_logs("run", args.client_log, args.server1_log, args.server2_log)
        print_summary("Compression Factor Summary (single run)", summary)
        return

    before = _run_logs("before", args.before_client_log, args.before_server1_log, args.before_server2_log)
    after = _run_logs("after", args.after_client_log, args.after_server1_log, args.after_server2_log)
    print_summary("Compression Factor Summary (before)", before)
    print_summary("Compression Factor Summary (after)", after)
    print_delta(before, after)


if __name__ == "__main__":
    main()
