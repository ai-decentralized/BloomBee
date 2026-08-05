"""Run-end session timing reports.

emit_session_timing_summary() turns the per-step records collected by the
connection handler into the [TIMING_SUMMARY] / [TIMING_TABLE] /
[PAPER_TIMING_TABLE] / [PIPELINE_COMPONENT_VIEW] / [PIPELINE_EXPOSED_VIEW] /
[PIPELINE_GPU2GPU] lines. Pure reporting, no handler state involved.
"""

from __future__ import annotations

import numpy as np
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)


def _emit_unconditional_summary(message: str) -> None:
    # Intentional print(): timing/paper-table summary lines must always reach
    # stdout regardless of logger configuration so downstream parsers can
    # grep [PAPER_TIMING_TABLE]/[TIMING_TABLE]/[PIPELINE_GPU2GPU] markers.
    print(message, flush=True)


def emit_session_timing_summary(records: list, comm_records: dict, *, blocks_desc: str) -> None:
    """Aggregate one session's per-step timing records into the run-end
    [TIMING_SUMMARY]/[TIMING_TABLE]/[PAPER_TIMING_TABLE]/[PIPELINE_*] lines
    that experiment tooling parses. Pure reporting: no handler state."""
    if not records:
        return

    warmup = 1
    decode_records = records[warmup:] if len(records) > warmup else records
    if not decode_records:
        return

    compute_arr = np.array([r["compute_ms"] for r in decode_records], dtype=np.float64)
    gpu2cpu_arr = np.array([r["t_gpu2cpu_ms"] for r in decode_records], dtype=np.float64)
    step_arr = np.array([r["step_total_ms"] for r in decode_records], dtype=np.float64)
    queue_arr = np.array([r["queue_wait_ms"] for r in decode_records], dtype=np.float64)
    data_arr = np.array([r["data_bytes"] for r in decode_records], dtype=np.float64)
    nic2cpu_arr = np.array([r["t_nic2cpu_ms"] for r in decode_records], dtype=np.float64)
    cpu2gpu_arr = np.array([r["t_cpu2gpu_ms"] for r in decode_records], dtype=np.float64)
    gpu2gpu_arr = np.array([r.get("t_gpu2gpu_ms", 0.0) for r in decode_records], dtype=np.float64)
    gpu2gpu_bytes_arr = np.array([r.get("gpu2gpu_bytes", 0.0) for r in decode_records], dtype=np.float64)
    cpu_serialize_arr = np.array([r.get("cpu_serialize_ms", 0.0) for r in decode_records], dtype=np.float64)
    batch_arr = np.array([r.get("batch_size", 1) for r in decode_records], dtype=np.float64)
    token_arr = np.array([r.get("token_increment", 1) for r in decode_records], dtype=np.float64)

    total_compute = float(compute_arr.sum())
    total_step = float(step_arr.sum())
    total_nic2cpu = float(nic2cpu_arr.sum())
    total_cpu2gpu = float(cpu2gpu_arr.sum())
    total_gpu2cpu = float(gpu2cpu_arr.sum())
    total_host_io = total_nic2cpu + total_cpu2gpu + total_gpu2cpu
    compute_ratio = (total_compute / total_step * 100.0) if total_step > 0 else 0.0
    host_io_ratio = (total_host_io / total_step * 100.0) if total_step > 0 else 0.0
    gpu2cpu_ratio = (total_gpu2cpu / total_step * 100.0) if total_step > 0 else 0.0
    avg_bw = (data_arr.mean() / (gpu2cpu_arr.mean() / 1000.0) / 1e9) if gpu2cpu_arr.mean() > 0 else 0.0
    total_tokens = float(np.sum(batch_arr * token_arr))
    throughput_tok_s = (total_tokens / (total_step / 1000.0)) if total_step > 0 else 0.0
    inference_latency_ms = float(step_arr.mean()) if len(step_arr) > 0 else 0.0

    comm_summary = "\n  s2s_comm : no downstream push samples"
    cpu2nic_mean = 0.0
    nic2nic_mean = 0.0
    push_e2e_mean = 0.0
    avg_nic_bw = 0.0
    avg_nic_bw_gbps = 0.0
    comm_volume_kb = data_arr.mean() / 1024.0 if len(data_arr) > 0 else 0.0
    comm_volume_bytes = data_arr.mean() if len(data_arr) > 0 else 0.0
    total_cpu2nic = 0.0
    total_nic2nic = 0.0
    wire_arr = np.array([], dtype=np.float64)
    matched_comm_records = [comm_records[r["step_id"]] for r in decode_records if r.get("step_id") in comm_records]
    if matched_comm_records:
        cpu2nic_arr = np.array([r["t_cpu2nic_ms"] for r in matched_comm_records], dtype=np.float64)
        nic2nic_arr = np.array([r["t_nic2nic_ms"] for r in matched_comm_records], dtype=np.float64)
        push_e2e_arr = np.array([r["push_e2e_ms"] for r in matched_comm_records], dtype=np.float64)
        receiver_proc_arr = np.array([r["receiver_processing_ms"] for r in matched_comm_records], dtype=np.float64)
        wire_arr = np.array([r["wire_bytes"] for r in matched_comm_records], dtype=np.float64)

        total_cpu2nic = float(cpu2nic_arr.sum())
        total_nic2nic = float(nic2nic_arr.sum())
        total_comm = total_gpu2cpu + total_cpu2nic + total_nic2nic
        gpu2cpu_comm_ratio = (total_gpu2cpu / total_comm * 100.0) if total_comm > 0 else 0.0
        cpu2nic_ratio = (total_cpu2nic / total_comm * 100.0) if total_comm > 0 else 0.0
        nic2nic_ratio = (total_nic2nic / total_comm * 100.0) if total_comm > 0 else 0.0
        cpu2nic_mean = float(cpu2nic_arr.mean())
        nic2nic_mean = float(nic2nic_arr.mean())
        push_e2e_mean = float(push_e2e_arr.mean())
        avg_nic_bw = (wire_arr.mean() / (nic2nic_arr.mean() / 1000.0) / 1e6) if nic2nic_arr.mean() > 0 else 0.0
        avg_nic_bw_gbps = (wire_arr.mean() * 8.0 / (nic2nic_arr.mean() / 1000.0) / 1e9) if nic2nic_arr.mean() > 0 else 0.0
        comm_volume_kb = wire_arr.mean() / 1024.0 if len(wire_arr) > 0 else comm_volume_kb
        comm_volume_bytes = wire_arr.mean() if len(wire_arr) > 0 else comm_volume_bytes

        comm_summary = (
            f"\n  cpu2nic : mean={cpu2nic_arr.mean():.2f}ms  median={np.median(cpu2nic_arr):.2f}ms  "
            f"p95={np.percentile(cpu2nic_arr,95):.2f}ms  max={cpu2nic_arr.max():.2f}ms"
            f"\n  nic2nic : mean={nic2nic_arr.mean():.2f}ms  median={np.median(nic2nic_arr):.2f}ms  "
            f"p95={np.percentile(nic2nic_arr,95):.2f}ms  max={nic2nic_arr.max():.2f}ms"
            f"\n  push_e2e: mean={push_e2e_arr.mean():.2f}ms  median={np.median(push_e2e_arr):.2f}ms  "
            f"p95={np.percentile(push_e2e_arr,95):.2f}ms  max={push_e2e_arr.max():.2f}ms"
            f"\n  recv_proc: mean={receiver_proc_arr.mean():.2f}ms  median={np.median(receiver_proc_arr):.2f}ms  "
            f"p95={np.percentile(receiver_proc_arr,95):.2f}ms  max={receiver_proc_arr.max():.2f}ms"
            f"\n  s2s_ratio: gpu2cpu={gpu2cpu_comm_ratio:.1f}%  cpu2nic={cpu2nic_ratio:.1f}%  "
            f"nic2nic={nic2nic_ratio:.1f}%  avg_bw(nic)={avg_nic_bw:.1f}MB/s  wire_per_push={wire_arr.mean()/1024.0:.1f}KB"
        )

    pipeline_gpu2gpu_samples = []
    pipeline_gpu2gpu_bytes = []
    for rec in decode_records:
        sender_gpu2cpu_ms = float(rec.get("upstream_sender_gpu2cpu_ms", 0.0))
        sender_cpu2nic_ms = float(rec.get("upstream_sender_cpu2nic_ms", 0.0))
        upstream_wire_ms = float(rec.get("upstream_wire_ms", 0.0))
        if sender_gpu2cpu_ms <= 0.0 and sender_cpu2nic_ms <= 0.0 and upstream_wire_ms <= 0.0:
            continue
        pipeline_gpu2gpu_samples.append(
            sender_gpu2cpu_ms
            + sender_cpu2nic_ms
            + upstream_wire_ms
            + float(rec["t_nic2cpu_ms"])
            + float(rec["t_cpu2gpu_ms"])
        )
        pipeline_gpu2gpu_bytes.append(float(rec.get("upstream_payload_bytes", 0)))

    pipeline_gpu2gpu_arr = np.array(pipeline_gpu2gpu_samples, dtype=np.float64)
    pipeline_gpu2gpu_bytes_arr = np.array(pipeline_gpu2gpu_bytes, dtype=np.float64)
    pipeline_gpu2gpu_mean = float(pipeline_gpu2gpu_arr.mean()) if len(pipeline_gpu2gpu_arr) > 0 else 0.0
    pure_gpu2gpu_mean = float(gpu2gpu_arr.mean()) if len(gpu2gpu_arr) > 0 else 0.0
    pure_gpu2gpu_bw_mbps = (
        gpu2gpu_bytes_arr.mean() * 8.0 / (pure_gpu2gpu_mean / 1000.0) / 1e6
        if pure_gpu2gpu_mean > 0 and len(gpu2gpu_bytes_arr) > 0 and gpu2gpu_bytes_arr.mean() > 0
        else 0.0
    )
    local_gpu_staging_mean = float((gpu2cpu_arr + cpu2gpu_arr).mean()) if len(gpu2cpu_arr) > 0 else 0.0
    pure_gpu_compute_mean = float(compute_arr.mean()) if len(compute_arr) > 0 else 0.0
    pipeline_bw_mbps = (
        pipeline_gpu2gpu_bytes_arr.mean() * 8.0 / (pipeline_gpu2gpu_mean / 1000.0) / 1e6
        if pipeline_gpu2gpu_mean > 0 and len(pipeline_gpu2gpu_bytes_arr) > 0 and pipeline_gpu2gpu_bytes_arr.mean() > 0
        else 0.0
    )
    upstream_sender_gpu2cpu_arr = np.array(
        [r.get("upstream_sender_gpu2cpu_ms", 0.0) for r in decode_records if r.get("upstream_sender_gpu2cpu_ms", 0.0) > 0.0],
        dtype=np.float64,
    )
    upstream_sender_cpu2nic_arr = np.array(
        [r.get("upstream_sender_cpu2nic_ms", 0.0) for r in decode_records if r.get("upstream_sender_cpu2nic_ms", 0.0) > 0.0],
        dtype=np.float64,
    )
    upstream_wire_arr = np.array(
        [r.get("upstream_wire_ms", 0.0) for r in decode_records if r.get("upstream_wire_ms", 0.0) > 0.0],
        dtype=np.float64,
    )
    upstream_payload_bytes_arr = np.array(
        [r.get("upstream_payload_bytes", 0.0) for r in decode_records if r.get("upstream_payload_bytes", 0.0) > 0.0],
        dtype=np.float64,
    )
    paper_gpu2cpu_mean = (
        float(upstream_sender_gpu2cpu_arr.mean()) if len(upstream_sender_gpu2cpu_arr) > 0 else float(gpu2cpu_arr.mean())
    )
    paper_cpu2nic_mean = (
        float(upstream_sender_cpu2nic_arr.mean()) if len(upstream_sender_cpu2nic_arr) > 0 else cpu2nic_mean
    )
    paper_nic2nic_mean = float(upstream_wire_arr.mean()) if len(upstream_wire_arr) > 0 else nic2nic_mean
    paper_comm_volume_bytes = (
        float(upstream_payload_bytes_arr.mean()) if len(upstream_payload_bytes_arr) > 0 else float(comm_volume_bytes)
    )
    paper_comm_volume_kb = paper_comm_volume_bytes / 1024.0 if paper_comm_volume_bytes > 0 else 0.0
    paper_net_latency_ms = push_e2e_mean if push_e2e_mean > 0 else paper_nic2nic_mean
    paper_net_bw_mbps = (
        paper_comm_volume_bytes * 8.0 / (paper_nic2nic_mean / 1000.0) / 1e6
        if paper_nic2nic_mean > 0 and paper_comm_volume_bytes > 0
        else 0.0
    )
    exposed_ready_count = sum(int(r.get("pipeline_overlap_breakdown_ready", 0)) for r in decode_records)
    critical_path_exposed_arr = np.array(
        [r.get("critical_path_exposed_ms", 0.0) for r in decode_records if r.get("critical_path_exposed_ms", 0.0) > 0.0],
        dtype=np.float64,
    )
    sender_gpu2cpu_exposed_arr = np.array(
        [r.get("sender_gpu2cpu_exposed_ms", 0.0) for r in decode_records if r.get("sender_gpu2cpu_exposed_ms", 0.0) > 0.0],
        dtype=np.float64,
    )
    sender_cpu2nic_exposed_arr = np.array(
        [r.get("sender_cpu2nic_exposed_ms", 0.0) for r in decode_records if r.get("sender_cpu2nic_exposed_ms", 0.0) > 0.0],
        dtype=np.float64,
    )
    nic2nic_exposed_arr = np.array(
        [r.get("nic2nic_exposed_ms", 0.0) for r in decode_records if r.get("nic2nic_exposed_ms", 0.0) > 0.0],
        dtype=np.float64,
    )
    receiver_nic2cpu_exposed_arr = np.array(
        [r.get("receiver_nic2cpu_exposed_ms", 0.0) for r in decode_records if r.get("receiver_nic2cpu_exposed_ms", 0.0) > 0.0],
        dtype=np.float64,
    )
    receiver_cpu2gpu_exposed_arr = np.array(
        [r.get("receiver_cpu2gpu_exposed_ms", 0.0) for r in decode_records if r.get("receiver_cpu2gpu_exposed_ms", 0.0) > 0.0],
        dtype=np.float64,
    )
    sender_post_compute_exposed_arr = np.array(
        [r.get("sender_post_compute_exposed_ms", 0.0) for r in decode_records if r.get("sender_post_compute_exposed_ms", 0.0) > 0.0],
        dtype=np.float64,
    )
    receiver_dispatch_exposed_arr = np.array(
        [r.get("receiver_dispatch_exposed_ms", 0.0) for r in decode_records if r.get("receiver_dispatch_exposed_ms", 0.0) > 0.0],
        dtype=np.float64,
    )
    critical_path_exposed_mean = (
        float(critical_path_exposed_arr.mean()) if len(critical_path_exposed_arr) > 0 else float(inference_latency_ms)
    )
    sender_gpu2cpu_exposed_mean = (
        float(sender_gpu2cpu_exposed_arr.mean()) if len(sender_gpu2cpu_exposed_arr) > 0 else paper_gpu2cpu_mean
    )
    sender_cpu2nic_exposed_mean = (
        float(sender_cpu2nic_exposed_arr.mean()) if len(sender_cpu2nic_exposed_arr) > 0 else paper_cpu2nic_mean
    )
    nic2nic_exposed_mean = (
        float(nic2nic_exposed_arr.mean()) if len(nic2nic_exposed_arr) > 0 else paper_nic2nic_mean
    )
    receiver_nic2cpu_exposed_mean = (
        float(receiver_nic2cpu_exposed_arr.mean()) if len(receiver_nic2cpu_exposed_arr) > 0 else float(nic2cpu_arr.mean())
    )
    receiver_cpu2gpu_exposed_mean = (
        float(receiver_cpu2gpu_exposed_arr.mean()) if len(receiver_cpu2gpu_exposed_arr) > 0 else float(cpu2gpu_arr.mean())
    )
    sender_post_compute_exposed_mean = (
        float(sender_post_compute_exposed_arr.mean()) if len(sender_post_compute_exposed_arr) > 0 else 0.0
    )
    receiver_dispatch_exposed_mean = (
        float(receiver_dispatch_exposed_arr.mean()) if len(receiver_dispatch_exposed_arr) > 0 else 0.0
    )

    n = len(decode_records)
    summary_message = (
        f"[TIMING_SUMMARY] blocks={blocks_desc} steps={n} (excl {warmup} warmup)\n"
        f"  nic2cpu : mean={nic2cpu_arr.mean():.2f}ms  median={np.median(nic2cpu_arr):.2f}ms  "
        f"p95={np.percentile(nic2cpu_arr,95):.2f}ms  max={nic2cpu_arr.max():.2f}ms\n"
        f"  cpu2gpu : mean={cpu2gpu_arr.mean():.2f}ms  median={np.median(cpu2gpu_arr):.2f}ms  "
        f"p95={np.percentile(cpu2gpu_arr,95):.2f}ms  max={cpu2gpu_arr.max():.2f}ms\n"
        f"  compute : mean={compute_arr.mean():.1f}ms  median={np.median(compute_arr):.1f}ms  "
        f"p95={np.percentile(compute_arr,95):.1f}ms  min={compute_arr.min():.1f}ms  max={compute_arr.max():.1f}ms\n"
        f"  gpu2cpu : mean={gpu2cpu_arr.mean():.2f}ms  median={np.median(gpu2cpu_arr):.2f}ms  "
        f"p95={np.percentile(gpu2cpu_arr,95):.2f}ms  max={gpu2cpu_arr.max():.2f}ms\n"
        f"  step_total: mean={step_arr.mean():.1f}ms  median={np.median(step_arr):.1f}ms  "
        f"p95={np.percentile(step_arr,95):.1f}ms  min={step_arr.min():.1f}ms  max={step_arr.max():.1f}ms\n"
        f"  queue_wait: mean={queue_arr.mean():.1f}ms  median={np.median(queue_arr):.1f}ms  "
        f"p95={np.percentile(queue_arr,95):.1f}ms  max={queue_arr.max():.1f}ms\n"
        f"  summary: inference_latency={inference_latency_ms:.2f}ms  throughput={throughput_tok_s:.2f}tok/s  "
        f"comm_volume={comm_volume_kb:.1f}KB  net_latency={push_e2e_mean:.2f}ms  net_bw={avg_nic_bw:.1f}MB/s\n"
        f"  ratio: compute={compute_ratio:.1f}%  host_io={host_io_ratio:.1f}%  gpu2cpu={gpu2cpu_ratio:.1f}%  "
        f"avg_bw(gpu2cpu)={avg_bw:.2f}GB/s  data_per_step={data_arr.mean()/1024:.1f}KB"
        f"{comm_summary}"
    )
    logger.info(summary_message)

    timing_table_line = (
        f"[TIMING_TABLE] blocks={blocks_desc} steps={n} "
        f"T_GPU->CPU={gpu2cpu_arr.mean():.2f}ms "
        f"T_CPU->NIC={cpu2nic_mean:.2f}ms "
        f"T_NIC->NIC={nic2nic_mean:.2f}ms "
        f"T_NIC->CPU={nic2cpu_arr.mean():.2f}ms "
        f"T_CPU->GPU={cpu2gpu_arr.mean():.2f}ms "
        f"InferenceLatency={inference_latency_ms:.2f}ms "
        f"Throughput={throughput_tok_s:.2f}tok/s "
        f"CommunicateVolume={comm_volume_kb:.1f}KB "
        f"T_GPU_Compute={compute_arr.mean():.2f}ms "
        f"NetLatency={push_e2e_mean:.2f}ms "
        f"NetBandwidth={avg_nic_bw:.2f}MB/s"
    )
    logger.info(timing_table_line)
    _emit_unconditional_summary(timing_table_line)

    paper_timing_table_line = (
        f"[PAPER_TIMING_TABLE] blocks={blocks_desc} steps={n} "
        f"T_GPU->CPU={paper_gpu2cpu_mean:.2f}ms "
        f"T_CPU->NIC={paper_cpu2nic_mean:.2f}ms "
        f"T_NIC->NIC={paper_nic2nic_mean:.2f}ms "
        f"T_NIC->CPU={nic2cpu_arr.mean():.2f}ms "
        f"T_CPU->GPU={cpu2gpu_arr.mean():.2f}ms "
        f"InferenceLatency={inference_latency_ms:.2f}ms "
        f"Throughput={throughput_tok_s:.2f}tok/s "
        f"CommunicationVolume={paper_comm_volume_kb:.1f}KB "
        f"T_GPU_Compute={compute_arr.mean():.2f}ms "
        f"NetworkLatency={paper_net_latency_ms:.2f}ms "
        f"NetworkBandwidth={paper_net_bw_mbps:.2f}Mbps"
    )
    logger.info(paper_timing_table_line)
    _emit_unconditional_summary(paper_timing_table_line)

    component_scope_line = (
        f"[PIPELINE_COMPONENT_VIEW] blocks={blocks_desc} steps={n} "
        f"sender_T_GPU->CPU_RAW={paper_gpu2cpu_mean:.2f}ms "
        f"sender_T_CPU->NIC_RAW={paper_cpu2nic_mean:.2f}ms "
        f"link_T_NIC->NIC_RAW={paper_nic2nic_mean:.2f}ms "
        f"receiver_T_NIC->CPU_RAW={nic2cpu_arr.mean():.2f}ms "
        f"receiver_T_CPU->GPU_RAW={cpu2gpu_arr.mean():.2f}ms "
        f"pipeline_overlap_affects_component_visibility=1"
    )
    logger.info(component_scope_line)
    _emit_unconditional_summary(component_scope_line)

    exposed_component_line = (
        f"[PIPELINE_EXPOSED_VIEW] blocks={blocks_desc} steps={n} "
        f"EndToEndCriticalPathExposed={critical_path_exposed_mean:.2f}ms "
        f"sender_T_GPU->CPU_EXPOSED={sender_gpu2cpu_exposed_mean:.2f}ms "
        f"sender_T_CPU->NIC_EXPOSED={sender_cpu2nic_exposed_mean:.2f}ms "
        f"link_T_NIC->NIC_EXPOSED={nic2nic_exposed_mean:.2f}ms "
        f"receiver_T_NIC->CPU_EXPOSED={receiver_nic2cpu_exposed_mean:.2f}ms "
        f"receiver_T_CPU->GPU_EXPOSED={receiver_cpu2gpu_exposed_mean:.2f}ms "
        f"sender_post_compute_gap_EXPOSED={sender_post_compute_exposed_mean:.2f}ms "
        f"receiver_dispatch_EXPOSED={receiver_dispatch_exposed_mean:.2f}ms "
        f"overlap_breakdown_coverage={exposed_ready_count}/{n}"
    )
    logger.info(exposed_component_line)
    _emit_unconditional_summary(exposed_component_line)

    pipeline_gpu_line = (
        f"[PIPELINE_GPU2GPU] blocks={blocks_desc} steps={n} "
        f"T_GPU->GPU_PIPE={pipeline_gpu2gpu_mean:.2f}ms "
        f"BW_GPU->GPU_PIPE={pipeline_bw_mbps:.2f}Mbps "
        f"T_GPU->GPU_PURE={pure_gpu2gpu_mean:.2f}ms "
        f"BW_GPU->GPU_PURE={pure_gpu2gpu_bw_mbps:.2f}Mbps "
        f"T_GPU_LOCAL_STAGING={local_gpu_staging_mean:.2f}ms "
        f"T_GPU_PURE_COMPUTE={pure_gpu_compute_mean:.2f}ms "
        f"T_CPU_SERIALIZE={cpu_serialize_arr.mean():.2f}ms "
        f"samples={len(pipeline_gpu2gpu_arr)}"
    )
    logger.info(pipeline_gpu_line)
    _emit_unconditional_summary(pipeline_gpu_line)

    timing_note_line = (
        f"[TIMING_NOTE] blocks={blocks_desc} "
        f"T_GPU->GPU_PIPE=sender(T_GPU->CPU+T_CPU->NIC+wire)+receiver(T_NIC->CPU+T_CPU->GPU); "
        f"PAPER_TIMING_TABLE prefers upstream sender+wire fields on downstream stages when available; "
        f"T_GPU->GPU_PURE is same-host cuda->cuda transfer time collected inside task_pool .to(device); "
        f"T_GPU->CPU happens on sender after compute; T_CPU->NIC happens on sender before rpc_push; "
        f"T_NIC->NIC is one-hop wire time; T_NIC->CPU happens on receiver during deserialize/unpack; "
        f"T_CPU->GPU happens on receiver when runtime moves tensors to cuda; "
        f"component fields are raw local segment durations and may be overlapped by pipeline; "
        f"they are not additive to end-to-end latency; "
        f"PIPELINE_EXPOSED_VIEW reports the downstream critical-path-visible portion of each segment; "
        f"with full-batch PP (no micro-batching), EXPOSED is reported from observed per-step segments and coverage should be n/n; "
        f"with micro-batch PP, EXPOSED uses overlap attribution when available and otherwise falls back to RAW means; "
        f"T_NIC->CPU includes deserialize/unpack; "
        f"T_CPU->NIC includes request packing and pre-send prep; "
        f"InferenceLatency/Throughput are stage-local means; "
        f"CommunicationVolume is mean payload per decode step in KB; "
        f"NetworkLatency=push_e2e(send->ack), while T_NIC->NIC subtracts receiver processing; "
        f"NetworkBandwidth=payload_bits/T_NIC->NIC in Mbps."
    )
    logger.info(timing_note_line)
    _emit_unconditional_summary(timing_note_line)

