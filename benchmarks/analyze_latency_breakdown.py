#!/usr/bin/env python3
import argparse
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


NETWORK_SUMMARY_RE = re.compile(
    r"\[NETWORK_TX\] SUMMARY \| step_id=(?P<step_id>[a-f0-9\-]+) \| "
    r"send=(?P<send_kb>[0-9.]+)KB \| recv=(?P<recv_kb>[0-9.]+)KB \| "
    r"serialize=(?P<serialize_ms>[0-9.]+)ms \| network=(?P<network_ms>[0-9.]+)ms \| "
    r"deserialize=(?P<deserialize_ms>[0-9.]+)ms \| total=(?P<total_ms>[0-9.]+)ms"
)
CLIENT_SERVER_END_RE = re.compile(
    r"\[CLIENT_SERVER_END\] ServerIdx=(?P<server_idx>\d+) \| Blocks=(?P<blocks>\d+:\d+) \| Duration=(?P<duration_ms>[0-9.]+)ms"
)
STEP_LATENCY_RE = re.compile(
    r"\[STEP_LATENCY\] Process=(?P<process>\d+) \| Step=(?P<step>\d+) \| Latency=(?P<latency_ms>[0-9.]+)ms"
)
MBPIPE_SUMMARY_RE = re.compile(
    r"\[MBPIPE_SUMMARY\] step=(?P<step>\d+) mb=(?P<mb>\d+) "
    r"compute=(?P<compute_ms>[0-9.]+)ms elapsed=(?P<elapsed_ms>[0-9.]+)ms "
    r"wait=(?P<wait_ms>[0-9.]+)ms\((?P<wait_pct>[0-9.]+)%\) "
    r"launch=(?P<launch_ms>[0-9.]+)ms\((?P<launch_pct>[0-9.]+)%\) "
    r"efficiency=(?P<efficiency_pct>[0-9.]+)%"
)
CROSS_STAGE_SUMMARY_RE = re.compile(
    r"\[CROSS_STAGE_OVERLAP_SUMMARY\] step=(?P<step_id>[a-f0-9\-]+) "
    r"overlap=(?P<overlap_ms>[0-9.]+)ms, Stage2_compute=(?P<stage2_compute_ms>[0-9.]+)ms, "
    r"efficiency=(?P<efficiency_pct>[0-9.]+)%, strict_efficiency=(?P<strict_efficiency_pct>[0-9.]+)%, "
    r"comparable_compute=(?P<comparable_compute_ms>[0-9.]+)ms"
)
STEP_TIMING_BREAKDOWN_RE = re.compile(
    r"\[STEP_TIMING_BREAKDOWN\] step_id=(?P<step_id>\S+) mode=(?P<mode>\S+) "
    r"queue_wait=(?P<queue_wait_ms>-?[0-9.]+)ms queue_source=(?P<queue_source>\S+) "
    r"deserialize=(?P<deserialize_ms>-?[0-9.]+)ms compute=(?P<compute_ms>-?[0-9.]+)ms "
    r"serialize=(?P<serialize_ms>-?[0-9.]+)ms residual=(?P<residual_ms>-?[0-9.]+)ms "
    r"step_total=(?P<step_total_ms>-?[0-9.]+)ms total_with_queue=(?P<total_with_queue_ms>-?[0-9.]+)ms "
    r"cross_gpu_window=(?P<cross_gpu_window_ms>-?[0-9.]+)ms "
    r"batch=(?P<batch>\d+) seq_inc=(?P<seq_inc>-?\d+)"
    r"(?: raw_seq=(?P<raw_seq>-?\d+))? is_spec_dec=(?P<is_spec_dec>\d+)"
    r"(?: "
    r"decompress=(?P<decompress_ms>-?[0-9.]+)ms "
    r"deserialize_unwrap=(?P<deserialize_unwrap_ms>-?[0-9.]+)ms "
    r"deserialize_core=(?P<deserialize_core_ms>-?[0-9.]+)ms "
    r"compress=(?P<compress_ms>-?[0-9.]+)ms "
    r"serialize_wrap=(?P<serialize_wrap_ms>-?[0-9.]+)ms "
    r"serialize_core=(?P<serialize_core_ms>-?[0-9.]+)ms"
    r")?"
)
STEP_TIMING_BREAKDOWN_MB_RE = re.compile(
    r"\[STEP_TIMING_BREAKDOWN_MB\] step_id=(?P<step_id>\S+) mode=micro_batch "
    r"expected_mb=(?P<expected_mb>\d+) recv_mb=(?P<recv_mb>\d+) "
    r"queue_wait_sum=(?P<queue_wait_sum_ms>-?[0-9.]+)ms "
    r"deserialize_sum=(?P<deserialize_sum_ms>-?[0-9.]+)ms "
    r"compute_sum=(?P<compute_sum_ms>-?[0-9.]+)ms "
    r"serialize=(?P<serialize_ms>-?[0-9.]+)ms "
    r"(?:decompress_sum=(?P<decompress_sum_ms>-?[0-9.]+)ms "
    r"deserialize_unwrap_sum=(?P<deserialize_unwrap_sum_ms>-?[0-9.]+)ms "
    r"deserialize_core_sum=(?P<deserialize_core_sum_ms>-?[0-9.]+)ms "
    r"compress=(?P<compress_ms>-?[0-9.]+)ms "
    r"serialize_wrap=(?P<serialize_wrap_ms>-?[0-9.]+)ms "
    r"serialize_core=(?P<serialize_core_ms>-?[0-9.]+)ms )?"
    r"residual=(?P<residual_ms>-?[0-9.]+)ms "
    r"total=(?P<total_ms>-?[0-9.]+)ms "
    r"queue_sources=(?P<queue_sources>\S+)"
    r"(?: queue_wait_pre=(?P<queue_wait_pre_ms>-?[0-9.]+)ms "
    r"queue_wait_inter=(?P<queue_wait_inter_ms>-?[0-9.]+)ms "
    r"total_with_pre_wait=(?P<total_with_pre_wait_ms>-?[0-9.]+)ms)?"
)
HANDLER_STEP_TIMING_RE = re.compile(
    r"\[HANDLER_STEP_TIMING\] step_id=(?P<step_id>\S+) "
    r"queue_wait=(?P<queue_wait_ms>-?[0-9.]+)ms queue_source=(?P<queue_source>\S+) "
    r"push_schedule=(?P<push_schedule_ms>-?[0-9.]+)ms "
    r"response_emit=(?P<response_emit_ms>-?[0-9.]+)ms "
    r"handler_total=(?P<handler_total_ms>-?[0-9.]+)ms "
    r"can_push=(?P<can_push>[01])"
)
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
COMP_TIMING_RE = re.compile(
    r"\[COMP_TIMING\]\s+"
    r"source=(?P<source>\S+)\s+"
    r"channel=(?P<channel>\S+)\s+"
    r"blocks=(?P<blocks>\S+)\s+"
    r"step_id=(?P<step_id>\S+)\s+"
    r"batch=(?P<batch>\d+)\s+"
    r"serialize_core_ms=(?P<serialize_core_ms>-?[0-9.]+)\s+"
    r"serialize_wrap_ms=(?P<serialize_wrap_ms>-?[0-9.]+)\s+"
    r"compress_ms=(?P<compress_ms>-?[0-9.]+)\s+"
    r"deserialize_unwrap_ms=(?P<deserialize_unwrap_ms>-?[0-9.]+)\s+"
    r"decompress_ms=(?P<decompress_ms>-?[0-9.]+)\s+"
    r"deserialize_core_ms=(?P<deserialize_core_ms>-?[0-9.]+)\s+"
    r"serialize_raw_bytes=(?P<serialize_raw_bytes>\d+)\s+"
    r"serialize_wire_bytes=(?P<serialize_wire_bytes>\d+)\s+"
    r"deserialize_wire_bytes=(?P<deserialize_wire_bytes>\d+)\s+"
    r"deserialize_raw_bytes=(?P<deserialize_raw_bytes>\d+)\s+"
    r"serialize_ratio=(?P<serialize_ratio>-?[0-9.]+)\s+"
    r"serialize_savings=(?P<serialize_savings>-?[0-9.]+)\s+"
    r"compress_calls=(?P<compress_calls>\d+)\s+"
    r"decompress_calls=(?P<decompress_calls>\d+)"
)
S2S_WIRE_RE = re.compile(
    r"\[S2S_WIRE\]\s+"
    r"step_id=(?P<step_id>\S+)\s+"
    r"mb_idx=(?P<mb_idx>-?\d+)\s+"
    r"sender_blocks=(?P<sender_blocks>\S+)\s+"
    r"receiver_blocks=(?P<receiver_blocks>\S+)\s+"
    r"batch=(?P<batch>\d+)\s+"
    r"payload_kb=(?P<payload_kb>-?[0-9.]+)\s+"
    r"metadata_b=(?P<metadata_b>\d+)\s+"
    r"raw_transfer_ms=(?P<raw_transfer_ms>-?[0-9.]+)\s+"
    r"sender_serialize_ms=(?P<sender_serialize_ms>-?[0-9.]+)\s+"
    r"sender_sem_wait_ms=(?P<sender_sem_wait_ms>-?[0-9.]+)\s+"
    r"sender_queue_ms=(?P<sender_queue_ms>-?[0-9.]+)\s+"
    r"sender_prep_ms=(?P<sender_prep_ms>-?[0-9.]+)\s+"
    r"wire_ms=(?P<wire_ms>-?[0-9.]+)\s+"
    r"e2e_from_serialize_end_ms=(?P<e2e_from_serialize_end_ms>-?[0-9.]+)\s+"
    r"clock_sync=(?P<clock_sync>[01])\s+"
    r"clock_offset_ms=(?P<clock_offset_ms>-?[0-9.]+)\s+"
    r"clock_rtt_ms=(?P<clock_rtt_ms>-?[0-9.]+)"
)


def _safe_mean(xs):
    return statistics.mean(xs) if xs else 0.0


def _safe_median(xs):
    return statistics.median(xs) if xs else 0.0


def _fmt(v):
    return f"{v:.2f}"


def _weighted_ratio(rows):
    raw_total = sum(float(x.get("raw_bytes", 0.0)) for x in rows)
    wire_total = sum(float(x.get("wire_bytes", 0.0)) for x in rows)
    if raw_total <= 0:
        return 1.0
    return wire_total / raw_total


def _parse_block_key(s: str):
    """
    Parse block-span strings like "0:20" or "0:20->20:40" for ordering.
    """
    m = re.match(r"(?P<a>\d+):(?P<b>\d+)(?:->(?P<c>\d+):(?P<d>\d+))?", s)
    if not m:
        return (10**9, 10**9, s)
    a = int(m.group("a"))
    b = int(m.group("b"))
    return (a, b, s)


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
    if den <= 0:
        return 0.0
    return num / den


def parse_client_log(path: Path):
    step_network = defaultdict(list)
    server_duration = defaultdict(list)
    step_latency = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = NETWORK_SUMMARY_RE.search(line)
            if m:
                step_network[m.group("step_id")].append(
                    {
                        "send_kb": float(m.group("send_kb")),
                        "recv_kb": float(m.group("recv_kb")),
                        "serialize_ms": float(m.group("serialize_ms")),
                        "network_ms": float(m.group("network_ms")),
                        "deserialize_ms": float(m.group("deserialize_ms")),
                        "total_ms": float(m.group("total_ms")),
                    }
                )
                continue

            m = CLIENT_SERVER_END_RE.search(line)
            if m:
                idx = int(m.group("server_idx"))
                server_duration[idx].append(float(m.group("duration_ms")))
                continue

            m = STEP_LATENCY_RE.search(line)
            if m:
                step_latency.append(float(m.group("latency_ms")))

    return step_network, server_duration, step_latency


def parse_server1_log(path: Path):
    vals = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = MBPIPE_SUMMARY_RE.search(line)
            if m:
                vals.append(
                    {
                        "compute_ms": float(m.group("compute_ms")),
                        "elapsed_ms": float(m.group("elapsed_ms")),
                        "wait_ms": float(m.group("wait_ms")),
                        "launch_ms": float(m.group("launch_ms")),
                        "efficiency_pct": float(m.group("efficiency_pct")),
                    }
                )
    return vals


def parse_server2_log(path: Path):
    vals = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = CROSS_STAGE_SUMMARY_RE.search(line)
            if m:
                vals.append(
                    {
                        "overlap_ms": float(m.group("overlap_ms")),
                        "stage2_compute_ms": float(m.group("stage2_compute_ms")),
                        "efficiency_pct": float(m.group("efficiency_pct")),
                        "strict_efficiency_pct": float(m.group("strict_efficiency_pct")),
                        "comparable_compute_ms": float(m.group("comparable_compute_ms")),
                    }
                )
    return vals


def parse_detailed_server_logs(logs):
    step_rows = []
    mb_rows = []
    handler_rows = []
    comp_timing_rows = []
    s2s_wire_rows = []

    for source_name, path in logs:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = STEP_TIMING_BREAKDOWN_RE.search(line)
                if m:
                    step_rows.append(
                        {
                            "source": source_name,
                            "step_id": m.group("step_id"),
                            "mode": m.group("mode"),
                            "queue_source": m.group("queue_source"),
                            "queue_wait_ms": float(m.group("queue_wait_ms")),
                            "deserialize_ms": float(m.group("deserialize_ms")),
                            "compute_ms": float(m.group("compute_ms")),
                            "serialize_ms": float(m.group("serialize_ms")),
                            "decompress_ms": float(m.group("decompress_ms")) if m.group("decompress_ms") is not None else 0.0,
                            "deserialize_unwrap_ms": float(m.group("deserialize_unwrap_ms"))
                            if m.group("deserialize_unwrap_ms") is not None
                            else 0.0,
                            "deserialize_core_ms": float(m.group("deserialize_core_ms"))
                            if m.group("deserialize_core_ms") is not None
                            else 0.0,
                            "compress_ms": float(m.group("compress_ms")) if m.group("compress_ms") is not None else 0.0,
                            "serialize_wrap_ms": float(m.group("serialize_wrap_ms"))
                            if m.group("serialize_wrap_ms") is not None
                            else 0.0,
                            "serialize_core_ms": float(m.group("serialize_core_ms"))
                            if m.group("serialize_core_ms") is not None
                            else 0.0,
                            "residual_ms": float(m.group("residual_ms")),
                            "step_total_ms": float(m.group("step_total_ms")),
                            "total_with_queue_ms": float(m.group("total_with_queue_ms")),
                            "cross_gpu_window_ms": float(m.group("cross_gpu_window_ms")),
                        }
                    )
                    continue

                m = STEP_TIMING_BREAKDOWN_MB_RE.search(line)
                if m:
                    queue_wait_pre_ms = m.group("queue_wait_pre_ms")
                    queue_wait_inter_ms = m.group("queue_wait_inter_ms")
                    total_with_pre_wait_ms = m.group("total_with_pre_wait_ms")
                    mb_rows.append(
                        {
                            "source": source_name,
                            "step_id": m.group("step_id"),
                            "expected_mb": int(m.group("expected_mb")),
                            "recv_mb": int(m.group("recv_mb")),
                            "queue_wait_sum_ms": float(m.group("queue_wait_sum_ms")),
                            "deserialize_sum_ms": float(m.group("deserialize_sum_ms")),
                            "compute_sum_ms": float(m.group("compute_sum_ms")),
                            "serialize_ms": float(m.group("serialize_ms")),
                            "decompress_sum_ms": float(m.group("decompress_sum_ms"))
                            if m.group("decompress_sum_ms") is not None
                            else 0.0,
                            "deserialize_unwrap_sum_ms": float(m.group("deserialize_unwrap_sum_ms"))
                            if m.group("deserialize_unwrap_sum_ms") is not None
                            else 0.0,
                            "deserialize_core_sum_ms": float(m.group("deserialize_core_sum_ms"))
                            if m.group("deserialize_core_sum_ms") is not None
                            else 0.0,
                            "compress_ms": float(m.group("compress_ms")) if m.group("compress_ms") is not None else 0.0,
                            "serialize_wrap_ms": float(m.group("serialize_wrap_ms"))
                            if m.group("serialize_wrap_ms") is not None
                            else 0.0,
                            "serialize_core_ms": float(m.group("serialize_core_ms"))
                            if m.group("serialize_core_ms") is not None
                            else 0.0,
                            "residual_ms": float(m.group("residual_ms")),
                            "total_ms": float(m.group("total_ms")),
                            "queue_sources": m.group("queue_sources"),
                            "queue_wait_pre_ms": float(queue_wait_pre_ms) if queue_wait_pre_ms is not None else None,
                            "queue_wait_inter_ms": float(queue_wait_inter_ms) if queue_wait_inter_ms is not None else None,
                            "total_with_pre_wait_ms": float(total_with_pre_wait_ms)
                            if total_with_pre_wait_ms is not None
                            else None,
                        }
                    )
                    continue

                m = HANDLER_STEP_TIMING_RE.search(line)
                if m:
                    handler_rows.append(
                        {
                            "source": source_name,
                            "step_id": m.group("step_id"),
                            "queue_source": m.group("queue_source"),
                            "queue_wait_ms": float(m.group("queue_wait_ms")),
                            "push_schedule_ms": float(m.group("push_schedule_ms")),
                            "response_emit_ms": float(m.group("response_emit_ms")),
                            "handler_total_ms": float(m.group("handler_total_ms")),
                            "can_push": int(m.group("can_push")),
                        }
                    )
                    continue

                m = COMP_TIMING_RE.search(line)
                if m:
                    comp_timing_rows.append(
                        {
                            "source": source_name,
                            "record_source": m.group("source"),
                            "channel": m.group("channel"),
                            "blocks": m.group("blocks"),
                            "step_id": m.group("step_id"),
                            "batch": int(m.group("batch")),
                            "serialize_core_ms": float(m.group("serialize_core_ms")),
                            "serialize_wrap_ms": float(m.group("serialize_wrap_ms")),
                            "compress_ms": float(m.group("compress_ms")),
                            "deserialize_unwrap_ms": float(m.group("deserialize_unwrap_ms")),
                            "decompress_ms": float(m.group("decompress_ms")),
                            "deserialize_core_ms": float(m.group("deserialize_core_ms")),
                            "serialize_raw_bytes": int(m.group("serialize_raw_bytes")),
                            "serialize_wire_bytes": int(m.group("serialize_wire_bytes")),
                            "deserialize_wire_bytes": int(m.group("deserialize_wire_bytes")),
                            "deserialize_raw_bytes": int(m.group("deserialize_raw_bytes")),
                            "serialize_ratio": float(m.group("serialize_ratio")),
                            "serialize_savings": float(m.group("serialize_savings")),
                            "compress_calls": int(m.group("compress_calls")),
                            "decompress_calls": int(m.group("decompress_calls")),
                        }
                    )
                    continue

                m = S2S_WIRE_RE.search(line)
                if m:
                    s2s_wire_rows.append(
                        {
                            "source": source_name,
                            "step_id": m.group("step_id"),
                            "mb_idx": int(m.group("mb_idx")),
                            "sender_blocks": m.group("sender_blocks"),
                            "receiver_blocks": m.group("receiver_blocks"),
                            "batch": int(m.group("batch")),
                            "payload_kb": float(m.group("payload_kb")),
                            "metadata_b": int(m.group("metadata_b")),
                            "raw_transfer_ms": float(m.group("raw_transfer_ms")),
                            "sender_serialize_ms": float(m.group("sender_serialize_ms")),
                            "sender_sem_wait_ms": float(m.group("sender_sem_wait_ms")),
                            "sender_queue_ms": float(m.group("sender_queue_ms")),
                            "sender_prep_ms": float(m.group("sender_prep_ms")),
                            "wire_ms": float(m.group("wire_ms")),
                            "e2e_from_serialize_end_ms": float(m.group("e2e_from_serialize_end_ms")),
                            "clock_sync": int(m.group("clock_sync")),
                            "clock_offset_ms": float(m.group("clock_offset_ms")),
                            "clock_rtt_ms": float(m.group("clock_rtt_ms")),
                        }
                    )
                    continue

    return step_rows, mb_rows, handler_rows, comp_timing_rows, s2s_wire_rows


def parse_comp_ratio_logs(logs):
    rows = []
    for source_name, path in logs:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = COMP_RATIO_RE.search(line)
                if not m:
                    continue
                rows.append(
                    {
                        "source": source_name,
                        "channel": m.group("channel"),
                        "blocks": m.group("blocks"),
                        "step_id": m.group("step_id"),
                        "batch": int(m.group("batch")),
                        "tensor": m.group("tensor"),
                        "raw_bytes": int(m.group("raw_bytes")),
                        "wire_bytes": int(m.group("wire_bytes")),
                        "ratio": float(m.group("ratio")),
                        "savings": float(m.group("savings")),
                        "nnz": float(m.group("nnz")),
                    }
                )
    return rows


def summarize(client_log: Path, server1_log: Path, server2_log: Path):
    step_network, server_duration, step_latency = parse_client_log(client_log)
    s1_vals = parse_server1_log(server1_log)
    s2_vals = parse_server2_log(server2_log)
    step_breakdown_rows, mb_breakdown_rows, handler_step_rows, comp_timing_rows, s2s_wire_rows = parse_detailed_server_logs(
        [("client", client_log), ("server1", server1_log), ("server2", server2_log)]
    )
    comp_rows = parse_comp_ratio_logs(
        [("client", client_log), ("server1", server1_log), ("server2", server2_log)]
    )

    step_records = []
    for step_id, entries in step_network.items():
        entries = sorted(entries, key=lambda x: x["network_ms"], reverse=True)
        if len(entries) >= 2:
            e0, e1 = entries[0], entries[1]
        elif len(entries) == 1:
            e0, e1 = entries[0], None
        else:
            continue

        rec = {
            "step_id": step_id,
            "send_total_kb": e0["send_kb"] + (e1["send_kb"] if e1 else 0.0),
            "recv_total_kb": e0["recv_kb"] + (e1["recv_kb"] if e1 else 0.0),
            "serialize_total_ms": e0["serialize_ms"] + (e1["serialize_ms"] if e1 else 0.0),
            "network_total_ms": e0["network_ms"] + (e1["network_ms"] if e1 else 0.0),
            "deserialize_total_ms": e0["deserialize_ms"] + (e1["deserialize_ms"] if e1 else 0.0),
            "network_stack_total_ms": e0["total_ms"] + (e1["total_ms"] if e1 else 0.0),
            "stage0_network_ms": e0["network_ms"],
            "stage1_network_ms": e1["network_ms"] if e1 else 0.0,
        }
        step_records.append(rec)

    print("=" * 92)
    print("Latency/Volume Breakdown (from existing logs)")
    print("=" * 92)
    print(f"client_log  : {client_log}")
    print(f"server1_log : {server1_log}")
    print(f"server2_log : {server2_log}")
    print()
    print("Client-side (per generation step):")
    print("-" * 92)
    print(
        f"{'metric':38} {'mean':>12} {'median':>12} {'count':>10}"
    )
    print("-" * 92)
    if step_records:
        for key, label in [
            ("send_total_kb", "send_total_kb"),
            ("recv_total_kb", "recv_total_kb"),
            ("serialize_total_ms", "serialize_total_ms"),
            ("network_total_ms", "network_total_ms"),
            ("deserialize_total_ms", "deserialize_total_ms"),
            ("network_stack_total_ms", "network_stack_total_ms"),
            ("stage0_network_ms", "stage0_network_ms"),
            ("stage1_network_ms", "stage1_network_ms"),
        ]:
            vals = [r[key] for r in step_records]
            print(f"{label:38} {_fmt(_safe_mean(vals)):>12} {_fmt(_safe_median(vals)):>12} {len(vals):>10}")
    else:
        print("No [NETWORK_TX] SUMMARY entries found.")
    print("-" * 92)

    if step_latency:
        print(f"{'step_latency_ms':38} {_fmt(_safe_mean(step_latency)):>12} {_fmt(_safe_median(step_latency)):>12} {len(step_latency):>10}")
        print("-" * 92)

    print()
    print("Client server-span duration (from [CLIENT_SERVER_END]):")
    print("-" * 92)
    print(f"{'server_idx':>10} {'mean_ms':>12} {'median_ms':>12} {'count':>10}")
    print("-" * 92)
    if server_duration:
        for idx in sorted(server_duration):
            vals = server_duration[idx]
            print(f"{idx:>10} {_fmt(_safe_mean(vals)):>12} {_fmt(_safe_median(vals)):>12} {len(vals):>10}")
    else:
        print("No [CLIENT_SERVER_END] entries found.")
    print("-" * 92)

    print()
    print("Server1 micro-batch compute/offload (from [MBPIPE_SUMMARY]):")
    print("-" * 92)
    if s1_vals:
        print(
            f"compute_ms_mean={_fmt(_safe_mean([x['compute_ms'] for x in s1_vals]))}, "
            f"elapsed_ms_mean={_fmt(_safe_mean([x['elapsed_ms'] for x in s1_vals]))}, "
            f"wait_ms_mean={_fmt(_safe_mean([x['wait_ms'] for x in s1_vals]))}, "
            f"launch_ms_mean={_fmt(_safe_mean([x['launch_ms'] for x in s1_vals]))}, "
            f"efficiency_pct_mean={_fmt(_safe_mean([x['efficiency_pct'] for x in s1_vals]))}, "
            f"samples={len(s1_vals)}"
        )
    else:
        print("No [MBPIPE_SUMMARY] entries found in server1 log.")
    print("-" * 92)

    print()
    print("Server2 cross-stage overlap (from [CROSS_STAGE_OVERLAP_SUMMARY]):")
    print("-" * 92)
    if s2_vals:
        print(
            f"stage2_compute_ms_mean={_fmt(_safe_mean([x['stage2_compute_ms'] for x in s2_vals]))}, "
            f"overlap_ms_mean={_fmt(_safe_mean([x['overlap_ms'] for x in s2_vals]))}, "
            f"efficiency_pct_mean={_fmt(_safe_mean([x['efficiency_pct'] for x in s2_vals]))}, "
            f"strict_efficiency_pct_mean={_fmt(_safe_mean([x['strict_efficiency_pct'] for x in s2_vals]))}, "
            f"samples={len(s2_vals)}"
        )
    else:
        print("No [CROSS_STAGE_OVERLAP_SUMMARY] entries found in server2 log.")
    print("-" * 92)

    print()
    print("Server1->Server2 transport timing (from [S2S_WIRE]):")
    print("-" * 92)
    if s2s_wire_rows:
        source_counts = Counter(x["source"] for x in s2s_wire_rows)
        pair_counts = Counter((x["sender_blocks"], x["receiver_blocks"]) for x in s2s_wire_rows)
        valid_wire = [x for x in s2s_wire_rows if x["clock_sync"] == 1 and x["wire_ms"] >= 0.0]
        valid_e2e = [x for x in s2s_wire_rows if x["clock_sync"] == 1 and x["e2e_from_serialize_end_ms"] >= 0.0]
        print(
            "sources="
            + ",".join(f"{k}:{v}" for k, v in sorted(source_counts.items()))
            + f" | samples={len(s2s_wire_rows)} | wire_valid={len(valid_wire)}"
        )
        print(
            f"{'metric':38} {'mean':>12} {'median':>12} {'count':>10}"
        )
        # raw_transfer_ms is same-clock uncorrected; use it as a fallback indicator only.
        for key, label, rows in [
            ("raw_transfer_ms", "raw_transfer_ms(uncorrected)", [x for x in s2s_wire_rows if x["raw_transfer_ms"] >= 0.0]),
            ("wire_ms", "wire_ms(clock_corrected)", valid_wire),
            ("e2e_from_serialize_end_ms", "e2e_from_serialize_end_ms", valid_e2e),
            ("sender_serialize_ms", "sender_serialize_ms", [x for x in s2s_wire_rows if x["sender_serialize_ms"] >= 0.0]),
            ("sender_sem_wait_ms", "sender_sem_wait_ms", [x for x in s2s_wire_rows if x["sender_sem_wait_ms"] >= 0.0]),
            ("sender_queue_ms", "sender_queue_ms", [x for x in s2s_wire_rows if x["sender_queue_ms"] >= 0.0]),
            ("sender_prep_ms", "sender_prep_ms", [x for x in s2s_wire_rows if x["sender_prep_ms"] >= 0.0]),
            ("payload_kb", "payload_kb", s2s_wire_rows),
        ]:
            vals = [r[key] for r in rows]
            print(f"{label:38} {_fmt(_safe_mean(vals)):>12} {_fmt(_safe_median(vals)):>12} {len(vals):>10}")

        print()
        print(
            f"{'sender->receiver':28} {'wire_ms':>12} {'e2e_ser_end_ms':>14} {'count':>8}"
        )
        for pair, count in sorted(pair_counts.items()):
            pair_rows = [x for x in s2s_wire_rows if (x["sender_blocks"], x["receiver_blocks"]) == pair]
            pair_wire = [x["wire_ms"] for x in pair_rows if x["clock_sync"] == 1 and x["wire_ms"] >= 0.0]
            pair_e2e = [
                x["e2e_from_serialize_end_ms"]
                for x in pair_rows
                if x["clock_sync"] == 1 and x["e2e_from_serialize_end_ms"] >= 0.0
            ]
            pair_name = f"{pair[0]}->{pair[1]}"
            print(
                f"{pair_name:28} {_fmt(_safe_mean(pair_wire)):>12} {_fmt(_safe_mean(pair_e2e)):>14} {count:>8}"
            )
    else:
        print("No [S2S_WIRE] entries found.")
        print("Hint: rerun with updated server code on both stages.")
    print("-" * 92)

    print()
    print("Server detailed step breakdown (from [STEP_TIMING_BREAKDOWN]):")
    print("-" * 92)
    if step_breakdown_rows:
        mode_counts = Counter(x["mode"] for x in step_breakdown_rows)
        source_counts = Counter(x["source"] for x in step_breakdown_rows)
        queue_source_counts = Counter(x["queue_source"] for x in step_breakdown_rows)
        print(
            "modes="
            + ",".join(f"{k}:{v}" for k, v in sorted(mode_counts.items()))
            + " | sources="
            + ",".join(f"{k}:{v}" for k, v in sorted(source_counts.items()))
            + " | queue_sources="
            + ",".join(f"{k}:{v}" for k, v in sorted(queue_source_counts.items()))
        )
        print(
            f"{'metric':38} {'mean':>12} {'median':>12} {'count':>10}"
        )
        for key, label in [
            ("queue_wait_ms", "queue_wait_ms"),
            ("deserialize_ms", "deserialize_ms"),
            ("compute_ms", "compute_ms"),
            ("serialize_ms", "serialize_ms"),
            ("decompress_ms", "decompress_ms"),
            ("deserialize_unwrap_ms", "deserialize_unwrap_ms"),
            ("deserialize_core_ms", "deserialize_core_ms"),
            ("compress_ms", "compress_ms"),
            ("serialize_wrap_ms", "serialize_wrap_ms"),
            ("serialize_core_ms", "serialize_core_ms"),
            ("residual_ms", "residual_ms"),
            ("step_total_ms", "step_total_ms"),
            ("total_with_queue_ms", "total_with_queue_ms"),
            ("cross_gpu_window_ms", "cross_gpu_window_ms"),
        ]:
            vals = [r[key] for r in step_breakdown_rows]
            print(f"{label:38} {_fmt(_safe_mean(vals)):>12} {_fmt(_safe_median(vals)):>12} {len(vals):>10}")
    else:
        print("No [STEP_TIMING_BREAKDOWN] entries found.")
    print("-" * 92)

    print()
    print("Server merged micro-batch step breakdown (from [STEP_TIMING_BREAKDOWN_MB]):")
    print("-" * 92)
    if mb_breakdown_rows:
        source_counts = Counter(x["source"] for x in mb_breakdown_rows)
        print("sources=" + ",".join(f"{k}:{v}" for k, v in sorted(source_counts.items())))
        print(
            f"{'metric':38} {'mean':>12} {'median':>12} {'count':>10}"
        )
        for key, label in [
            ("queue_wait_sum_ms", "queue_wait_sum_ms"),
            ("deserialize_sum_ms", "deserialize_sum_ms"),
            ("compute_sum_ms", "compute_sum_ms"),
            ("serialize_ms", "serialize_ms"),
            ("decompress_sum_ms", "decompress_sum_ms"),
            ("deserialize_unwrap_sum_ms", "deserialize_unwrap_sum_ms"),
            ("deserialize_core_sum_ms", "deserialize_core_sum_ms"),
            ("compress_ms", "compress_ms"),
            ("serialize_wrap_ms", "serialize_wrap_ms"),
            ("serialize_core_ms", "serialize_core_ms"),
            ("residual_ms", "residual_ms"),
            ("total_ms", "total_ms"),
        ]:
            vals = [r[key] for r in mb_breakdown_rows]
            print(f"{label:38} {_fmt(_safe_mean(vals)):>12} {_fmt(_safe_median(vals)):>12} {len(vals):>10}")
        pre_vals = [r["queue_wait_pre_ms"] for r in mb_breakdown_rows if r["queue_wait_pre_ms"] is not None]
        inter_vals = [r["queue_wait_inter_ms"] for r in mb_breakdown_rows if r["queue_wait_inter_ms"] is not None]
        twp_vals = [r["total_with_pre_wait_ms"] for r in mb_breakdown_rows if r["total_with_pre_wait_ms"] is not None]
        if pre_vals:
            print(f"{'queue_wait_pre_ms':38} {_fmt(_safe_mean(pre_vals)):>12} {_fmt(_safe_median(pre_vals)):>12} {len(pre_vals):>10}")
        if inter_vals:
            print(f"{'queue_wait_inter_ms':38} {_fmt(_safe_mean(inter_vals)):>12} {_fmt(_safe_median(inter_vals)):>12} {len(inter_vals):>10}")
        if twp_vals:
            print(f"{'total_with_pre_wait_ms':38} {_fmt(_safe_mean(twp_vals)):>12} {_fmt(_safe_median(twp_vals)):>12} {len(twp_vals):>10}")
        print("Note: queue_wait_sum may not be additive with total across micro-batches.")
    else:
        print("No [STEP_TIMING_BREAKDOWN_MB] entries found.")
    print("-" * 92)

    print()
    print("Handler-side step overhead (from [HANDLER_STEP_TIMING]):")
    print("-" * 92)
    if handler_step_rows:
        source_counts = Counter(x["source"] for x in handler_step_rows)
        queue_source_counts = Counter(x["queue_source"] for x in handler_step_rows)
        print(
            "sources="
            + ",".join(f"{k}:{v}" for k, v in sorted(source_counts.items()))
            + " | queue_sources="
            + ",".join(f"{k}:{v}" for k, v in sorted(queue_source_counts.items()))
        )
        print(
            f"{'metric':38} {'mean':>12} {'median':>12} {'count':>10}"
        )
        for key, label in [
            ("queue_wait_ms", "queue_wait_ms"),
            ("push_schedule_ms", "push_schedule_ms"),
            ("response_emit_ms", "response_emit_ms"),
            ("handler_total_ms", "handler_total_ms"),
        ]:
            vals = [r[key] for r in handler_step_rows]
            print(f"{label:38} {_fmt(_safe_mean(vals)):>12} {_fmt(_safe_median(vals)):>12} {len(vals):>10}")
    else:
        print("No [HANDLER_STEP_TIMING] entries found.")
    print("-" * 92)

    print()
    print("Transport compression/decompression timing (from [COMP_TIMING]):")
    print("-" * 92)
    if comp_timing_rows:
        source_counts = Counter(x["source"] for x in comp_timing_rows)
        channel_counts = Counter(x["channel"] for x in comp_timing_rows)
        print(
            "sources="
            + ",".join(f"{k}:{v}" for k, v in sorted(source_counts.items()))
            + " | channels="
            + ",".join(f"{k}:{v}" for k, v in sorted(channel_counts.items()))
        )
        print(f"{'metric':38} {'mean':>12} {'median':>12} {'count':>10}")
        for key, label in [
            ("serialize_core_ms", "serialize_core_ms"),
            ("serialize_wrap_ms", "serialize_wrap_ms"),
            ("compress_ms", "compress_ms"),
            ("deserialize_unwrap_ms", "deserialize_unwrap_ms"),
            ("decompress_ms", "decompress_ms"),
            ("deserialize_core_ms", "deserialize_core_ms"),
            ("serialize_ratio", "serialize_ratio"),
            ("serialize_savings", "serialize_savings"),
        ]:
            vals = [r[key] for r in comp_timing_rows]
            print(f"{label:38} {_fmt(_safe_mean(vals)):>12} {_fmt(_safe_median(vals)):>12} {len(vals):>10}")

        print()
        print(
            f"{'source':10} {'channel':22} {'w_ratio':>10} {'mean_savings':>14} {'count':>8} "
            f"{'raw_kb':>10} {'wire_kb':>10}"
        )
        by_src_channel = defaultdict(list)
        for r in comp_timing_rows:
            by_src_channel[(r["source"], r["channel"])].append(r)
        for (src, channel), rows in sorted(by_src_channel.items()):
            raw_total = sum(x["serialize_raw_bytes"] for x in rows)
            wire_total = sum(x["serialize_wire_bytes"] for x in rows)
            ratio = (wire_total / raw_total) if raw_total > 0 else 1.0
            print(
                f"{src:10} {channel:22} {ratio:>10.4f} {_safe_mean([x['serialize_savings'] for x in rows]):>14.4f} "
                f"{len(rows):>8} {raw_total / 1024.0:>10.1f} {wire_total / 1024.0:>10.1f}"
            )
    else:
        print("No [COMP_TIMING] entries found.")
        print("Hint: ensure BLOOMBEE_COMP_TIMING_PROFILE=1 and rerun inference.")
    print("-" * 92)

    print()
    print("Compression ratio factor analysis (from [COMP_RATIO]):")
    print("-" * 92)
    print("Method / formulas:")
    print("1) ratio = wire_bytes / raw_bytes  (lower is better, savings = 1 - ratio)")
    print("2) layer/span weighted ratio = sum(wire_bytes) / sum(raw_bytes) within each blocks group")
    print("3) batch weighted ratio = sum(wire_bytes) / sum(raw_bytes) within each batch group")
    print("4) nnz relation uses Pearson r between per-record ratio and nnz")
    print("   r = cov(ratio, nnz) / (std(ratio) * std(nnz))")
    print("-" * 92)
    if comp_rows:
        print(
            f"overall: samples={len(comp_rows)}, "
            f"weighted_ratio={_fmt(_weighted_ratio(comp_rows))}, "
            f"mean_ratio={_fmt(_safe_mean([r['ratio'] for r in comp_rows]))}, "
            f"mean_nnz={_fmt(_safe_mean([r['nnz'] for r in comp_rows]))}"
        )

        print()
        print("By source/channel:")
        print(f"{'source':12} {'channel':20} {'w_ratio':>10} {'mean_ratio':>12} {'count':>8}")
        by_src_channel = defaultdict(list)
        for r in comp_rows:
            by_src_channel[(r["source"], r["channel"])].append(r)
        for (src, channel), rows in sorted(by_src_channel.items()):
            print(
                f"{src:12} {channel:20} "
                f"{_fmt(_weighted_ratio(rows)):>10} "
                f"{_fmt(_safe_mean([x['ratio'] for x in rows])):>12} "
                f"{len(rows):>8}"
            )

        print()
        print("By layer/span (blocks):")
        print(f"{'blocks':24} {'w_ratio':>10} {'mean_ratio':>12} {'mean_nnz':>10} {'count':>8}")
        by_blocks = defaultdict(list)
        for r in comp_rows:
            by_blocks[r["blocks"]].append(r)
        ordered_blocks = sorted(by_blocks.items(), key=lambda kv: _parse_block_key(kv[0]))
        for blocks, rows in ordered_blocks:
            print(
                f"{blocks:24} "
                f"{_fmt(_weighted_ratio(rows)):>10} "
                f"{_fmt(_safe_mean([x['ratio'] for x in rows])):>12} "
                f"{_fmt(_safe_mean([x['nnz'] for x in rows])):>10} "
                f"{len(rows):>8}"
            )
        if len(ordered_blocks) >= 2:
            front_blocks, front_rows = ordered_blocks[0]
            back_blocks, back_rows = ordered_blocks[-1]
            front_w = _weighted_ratio(front_rows)
            back_w = _weighted_ratio(back_rows)
            print(
                f"front_vs_back: front={front_blocks}({front_w:.4f}) "
                f"back={back_blocks}({back_w:.4f}) delta(back-front)={(back_w-front_w):.4f}"
            )

        print()
        print("By batch size:")
        print(f"{'batch':>8} {'w_ratio':>10} {'mean_ratio':>12} {'mean_nnz':>10} {'count':>8}")
        by_batch = defaultdict(list)
        for r in comp_rows:
            by_batch[int(r["batch"])].append(r)
        for batch in sorted(by_batch):
            rows = by_batch[batch]
            print(
                f"{batch:>8} "
                f"{_fmt(_weighted_ratio(rows)):>10} "
                f"{_fmt(_safe_mean([x['ratio'] for x in rows])):>12} "
                f"{_fmt(_safe_mean([x['nnz'] for x in rows])):>10} "
                f"{len(rows):>8}"
            )

        ratios = [r["ratio"] for r in comp_rows]
        nnzs = [r["nnz"] for r in comp_rows]
        pearson_r = _pearson(nnzs, ratios)
        median_nnz = _safe_median(nnzs)
        low_rows = [r for r in comp_rows if r["nnz"] <= median_nnz]
        high_rows = [r for r in comp_rows if r["nnz"] > median_nnz]
        print()
        print("nnz vs ratio:")
        print(
            f"pearson_r={pearson_r:.4f} (negative means denser activations tend to compress better)"
        )
        print(
            f"median_split: median_nnz={median_nnz:.4f}, "
            f"low_nnz_w_ratio={_weighted_ratio(low_rows):.4f}, "
            f"high_nnz_w_ratio={_weighted_ratio(high_rows):.4f}"
        )
    else:
        print("No [COMP_RATIO] entries found.")
        print("Hint: ensure BLOOMBEE_COMP_RATIO_PROFILE=1 and rerun inference.")
    print("-" * 92)

    if step_records and step_latency and s1_vals and s2_vals:
        mean_step = _safe_mean(step_latency)
        mean_client_net = _safe_mean([r["network_total_ms"] for r in step_records])
        mean_client_ser = _safe_mean([r["serialize_total_ms"] for r in step_records])
        mean_client_deser = _safe_mean([r["deserialize_total_ms"] for r in step_records])
        mean_s1_elapsed = _safe_mean([x["elapsed_ms"] for x in s1_vals])
        mean_s2_compute = _safe_mean([x["stage2_compute_ms"] for x in s2_vals])
        mean_overlap = _safe_mean([x["overlap_ms"] for x in s2_vals])

        implied_other = mean_step - (mean_client_ser + mean_client_deser + mean_client_net)
        overlap_adjusted_compute = mean_s1_elapsed + mean_s2_compute - mean_overlap

        print()
        print("Derived proxies (approximate, for trend comparison only):")
        print("-" * 92)
        print(f"mean_step_latency_ms              : {_fmt(mean_step)}")
        print(f"mean_client_network_stack_ms     : {_fmt(mean_client_ser + mean_client_net + mean_client_deser)}")
        print(f"implied_non_network_ms           : {_fmt(implied_other)}")
        print(f"overlap_adjusted_s1s2_compute_ms : {_fmt(overlap_adjusted_compute)}")
        print("-" * 92)
        print("Note: exact queue/scheduling split needs additional server-side step-level timestamps.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze BloomBee communication/computation/scheduling breakdown from logs."
    )
    parser.add_argument("--client-log", type=Path, default=Path("/home/cc/inference_batch.log"))
    parser.add_argument("--server1-log", type=Path, default=Path("/home/cc/server1.log"))
    parser.add_argument("--server2-log", type=Path, default=Path("/home/cc/server2.log"))
    args = parser.parse_args()
    summarize(args.client_log, args.server1_log, args.server2_log)


if __name__ == "__main__":
    main()
