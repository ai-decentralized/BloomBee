"""Cross-stage overlap attribution for micro-batch pipeline steps.

summarize_cross_stage_overlap() consumes the per-micro-batch timestamp
records gathered while a step's micro-batches were processed and emits the
[CROSS_STAGE_OVERLAP] / [CROSS_STAGE_OVERLAP_SUMMARY] analysis lines used to
quantify how much of this stage's compute was hidden under the previous
stage's work. Pure reporting over collected timestamps; it mutates only the
passed-in overlap_accum dict (adding the queue_wait_pre_* attribution fields
that the step-level [STEP_TIMING_BREAKDOWN_MB] line later reads).
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict

from hivemind.utils.logging import get_logger

logger = get_logger(__name__)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _interval_overlap_ms_from_us(a_start_us: int, a_end_us: int, b_start_us: int, b_end_us: int) -> float:
    if a_start_us <= 0 or a_end_us <= a_start_us or b_start_us <= 0 or b_end_us <= b_start_us:
        return 0.0
    start = max(a_start_us, b_start_us)
    end = min(a_end_us, b_end_us)
    if end <= start:
        return 0.0
    return (end - start) / 1000.0


def summarize_cross_stage_overlap(
    overlap_summary: Dict[int, Dict[str, Any]],
    overlap_accum: Dict[str, Any],
    *,
    step_id,
    step_receive_time: float,
    log_mb_detail: bool,
) -> None:

    total_overlap_ms = 0.0
    total_stage2_compute_ms = 0.0
    total_comparable_stage2_compute_ms = 0.0
    missing_pair_count = 0
    invalid_pair_count = 0
    clock_corrected_pair_count = 0
    clock_uncorrected_pair_count = 0
    clock_offset_abs_sum_ms = 0.0
    clock_rtt_max_ms = 0.0
    total_next_sender_serialize_ms = 0.0
    hidden_next_sender_serialize_ms = 0.0

    # For each pair (MB_n on this stage, MB_{n+1} on previous stage),
    # compute strict overlap as interval intersection:
    # overlap = len([s2_n_start, s2_n_end] ∩ [s1_{n+1}_start, s1_{n+1}_end])

    sorted_mb_indices = sorted(overlap_summary.keys())

    for i, mb_n in enumerate(sorted_mb_indices):
        mb_data = overlap_summary[mb_n]
        this_compute_ms = float(mb_data.get('this_stage_process_time_ms', 0) or 0.0)
        this_compute_start_us = _to_int(mb_data.get('this_stage_compute_start_us'), 0)
        this_compute_end_us = _to_int(mb_data.get('this_stage_compute_end_us'), 0)
        total_stage2_compute_ms += this_compute_ms

        # Check overlap with next MB
        if i + 1 < len(sorted_mb_indices):
            mb_n_plus_1 = sorted_mb_indices[i + 1]
            next_mb_data = overlap_summary.get(mb_n_plus_1, {})

            stage1_next_start_us = _to_int(next_mb_data.get('prev_stage_compute_start_us'), 0)
            stage1_next_end_us = _to_int(next_mb_data.get('prev_stage_compute_end_us'), 0)
            stage1_next_sender_serialize_start_us = _to_int(
                next_mb_data.get('sender_serialize_start_us'), 0
            )
            stage1_next_sender_serialize_end_us = _to_int(
                next_mb_data.get('sender_serialize_end_us'), 0
            )
            stage1_next_clock_offset_us = _to_int(next_mb_data.get('prev_stage_clock_offset_us'), 0)
            stage1_next_clock_rtt_us = max(0, _to_int(next_mb_data.get('prev_stage_clock_rtt_us'), 0))
            stage1_next_clock_samples = _to_int(next_mb_data.get('prev_stage_clock_samples'), 0)

            if (
                this_compute_start_us <= 0
                or this_compute_end_us <= 0
                or stage1_next_start_us <= 0
                or stage1_next_end_us <= 0
            ):
                missing_pair_count += 1
                if log_mb_detail:
                    logger.info(
                        f"[CROSS_STAGE_OVERLAP] step={step_id} MB{mb_n}→MB{mb_n_plus_1}: "
                        f"skipped (missing timestamps for strict overlap)"
                    )
                continue

            # Convert previous-stage timestamps to this host clock domain.
            # local_time ~= remote_time + offset_us, where offset_us = local - remote.
            if stage1_next_clock_samples > 0:
                stage1_next_start_us += stage1_next_clock_offset_us
                stage1_next_end_us += stage1_next_clock_offset_us
                if stage1_next_sender_serialize_start_us > 0:
                    stage1_next_sender_serialize_start_us += stage1_next_clock_offset_us
                if stage1_next_sender_serialize_end_us > 0:
                    stage1_next_sender_serialize_end_us += stage1_next_clock_offset_us
                clock_corrected_pair_count += 1
                clock_offset_abs_sum_ms += abs(stage1_next_clock_offset_us) / 1000.0
                clock_rtt_max_ms = max(clock_rtt_max_ms, stage1_next_clock_rtt_us / 1000.0)
            else:
                clock_uncorrected_pair_count += 1

            if stage1_next_end_us < stage1_next_start_us or this_compute_end_us < this_compute_start_us:
                invalid_pair_count += 1
                if log_mb_detail:
                    logger.info(
                        f"[CROSS_STAGE_OVERLAP] step={step_id} MB{mb_n}→MB{mb_n_plus_1}: "
                        f"skipped (invalid timestamp interval)"
                    )
                continue

            total_comparable_stage2_compute_ms += this_compute_ms
            next_sender_serialize_total_ms = (
                max(0.0, (stage1_next_sender_serialize_end_us - stage1_next_sender_serialize_start_us) / 1000.0)
                if (
                    stage1_next_sender_serialize_start_us > 0
                    and stage1_next_sender_serialize_end_us > stage1_next_sender_serialize_start_us
                )
                else 0.0
            )
            next_sender_serialize_hidden_ms = _interval_overlap_ms_from_us(
                this_compute_start_us,
                this_compute_end_us,
                stage1_next_sender_serialize_start_us,
                stage1_next_sender_serialize_end_us,
            )
            total_next_sender_serialize_ms += next_sender_serialize_total_ms
            hidden_next_sender_serialize_ms += next_sender_serialize_hidden_ms

            overlap_start_us = max(this_compute_start_us, stage1_next_start_us)
            overlap_end_us = min(this_compute_end_us, stage1_next_end_us)

            if overlap_end_us > overlap_start_us:
                actual_overlap_us = overlap_end_us - overlap_start_us
                actual_overlap_ms = actual_overlap_us / 1000.0
                total_overlap_ms += actual_overlap_ms

                mb_overlap_pct = (actual_overlap_ms / this_compute_ms * 100) if this_compute_ms > 0 else 0
                if log_mb_detail:
                    logger.info(
                        f"[CROSS_STAGE_OVERLAP] step={step_id} MB{mb_n}→MB{mb_n_plus_1}: "
                        f"strict_overlap={actual_overlap_ms:.1f}ms ({mb_overlap_pct:.0f}% of MB{mb_n} compute) ✓"
                    )
            else:
                if log_mb_detail:
                    if stage1_next_start_us >= this_compute_end_us:
                        gap_ms = (stage1_next_start_us - this_compute_end_us) / 1000.0
                        reason = f"no_overlap (gap={gap_ms:.1f}ms, Stage1_next_start after Stage2_end)"
                    elif this_compute_start_us >= stage1_next_end_us:
                        gap_ms = (this_compute_start_us - stage1_next_end_us) / 1000.0
                        reason = f"no_overlap (gap={gap_ms:.1f}ms, Stage2_start after Stage1_next_end)"
                    else:
                        reason = "no_overlap (zero-length interval intersection)"
                    logger.info(
                        f"[CROSS_STAGE_OVERLAP] step={step_id} MB{mb_n}→MB{mb_n_plus_1}: {reason}"
                    )

    # Calculate overlap efficiency
    if total_stage2_compute_ms > 0:
        stage2_queue_wait_sum_ms = float(overlap_accum.get('queue_wait_ms_sum', 0.0))
        stage2_queue_wait_pre_ms = float(overlap_accum.get('queue_wait_pre_ms', 0.0))
        stage2_queue_wait_inter_ms = float(overlap_accum.get('queue_wait_inter_ms', 0.0))
        stage2_deserialize_sum_ms = float(overlap_accum.get('deserialize_ms_sum', 0.0))
        stage2_decompress_sum_ms = float(overlap_accum.get('decompress_ms_sum', 0.0))
        stage2_elapsed_to_summary_ms = (
            (perf_counter() - overlap_accum.get('step_start_time', step_receive_time)) * 1000.0
        )
        stage2_critical_path_ms = (
            stage2_queue_wait_sum_ms + stage2_deserialize_sum_ms + total_stage2_compute_ms
        )
        stage2_full_path_ms = stage2_elapsed_to_summary_ms + stage2_queue_wait_pre_ms
        stage2_residual_path_ms = stage2_full_path_ms - stage2_critical_path_ms
        stage2_queue_wait_pre_upstream_compute_ms = 0.0
        stage2_queue_wait_pre_transfer_receive_ms = 0.0
        stage2_queue_wait_pre_precompute_gap_ms = 0.0
        stage2_queue_wait_pre_sender_post_compute_ms = 0.0
        stage2_queue_wait_pre_sender_compute_to_serialize_ms = 0.0
        stage2_queue_wait_pre_sender_serialize_ms = 0.0
        stage2_queue_wait_pre_sender_pre_send_wait_ms = 0.0
        stage2_queue_wait_pre_wire_receive_ms = 0.0
        stage2_queue_wait_pre_receiver_dispatch_ms = 0.0
        stage2_queue_wait_pre_breakdown_ready = 0
        next_sender_serialize_exposed_ms = max(
            0.0, total_next_sender_serialize_ms - hidden_next_sender_serialize_ms
        )
        next_sender_serialize_hidden_pct = (
            (hidden_next_sender_serialize_ms / total_next_sender_serialize_ms) * 100.0
            if total_next_sender_serialize_ms > 0.0
            else 0.0
        )

        mb0_data = overlap_summary.get(0, {})
        mb0_wait_start_us = _to_int(mb0_data.get('this_stage_queue_wait_start_us'), 0)
        mb0_wait_end_us = _to_int(mb0_data.get('this_stage_queue_wait_end_us'), 0)
        mb0_prev_start_us = _to_int(mb0_data.get('prev_stage_compute_start_us'), 0)
        mb0_prev_end_us = _to_int(mb0_data.get('prev_stage_compute_end_us'), 0)
        mb0_clock_offset_us = _to_int(mb0_data.get('prev_stage_clock_offset_us'), 0)
        mb0_clock_samples = _to_int(mb0_data.get('prev_stage_clock_samples'), 0)
        mb0_sender_send_us = _to_int(mb0_data.get('sender_send_us'), 0)
        mb0_sender_ser_start_us = _to_int(mb0_data.get('sender_serialize_start_us'), 0)
        mb0_sender_ser_end_us = _to_int(mb0_data.get('sender_serialize_end_us'), 0)
        mb0_receiver_receive_us = _to_int(mb0_data.get('receiver_receive_us'), 0)
        mb0_receiver_queue_put_us = _to_int(mb0_data.get('receiver_queue_put_us'), 0)

        if (
            stage2_queue_wait_pre_ms > 0.0
            and mb0_wait_end_us <= 0
            and mb0_wait_start_us > 0
        ):
            mb0_wait_end_us = mb0_wait_start_us + int(round(stage2_queue_wait_pre_ms * 1000.0))
        if (
            stage2_queue_wait_pre_ms > 0.0
            and mb0_wait_start_us <= 0
            and mb0_wait_end_us > 0
        ):
            mb0_wait_start_us = mb0_wait_end_us - int(round(stage2_queue_wait_pre_ms * 1000.0))

        def _interval_overlap_ms(
            range_start_us: int,
            range_end_us: int,
            window_start_us: int,
            window_end_us: int,
        ) -> float:
            if (
                range_start_us <= 0
                or range_end_us <= range_start_us
                or window_start_us <= 0
                or window_end_us <= window_start_us
            ):
                return 0.0
            overlap_start_us = max(range_start_us, window_start_us)
            overlap_end_us = min(range_end_us, window_end_us)
            if overlap_end_us <= overlap_start_us:
                return 0.0
            return (overlap_end_us - overlap_start_us) / 1000.0

        if (
            mb0_wait_start_us > 0
            and mb0_wait_end_us > mb0_wait_start_us
            and mb0_prev_start_us > 0
            and mb0_prev_end_us >= mb0_prev_start_us
            and mb0_clock_samples > 0
        ):
            mb0_prev_start_local_us = mb0_prev_start_us + mb0_clock_offset_us
            mb0_prev_end_local_us = mb0_prev_end_us + mb0_clock_offset_us
            mb0_sender_ser_start_local_us = (
                mb0_sender_ser_start_us + mb0_clock_offset_us if mb0_sender_ser_start_us > 0 else 0
            )
            mb0_sender_ser_end_local_us = (
                mb0_sender_ser_end_us + mb0_clock_offset_us if mb0_sender_ser_end_us > 0 else 0
            )
            mb0_sender_send_local_us = (
                mb0_sender_send_us + mb0_clock_offset_us if mb0_sender_send_us > 0 else 0
            )

            stage2_queue_wait_pre_upstream_compute_ms = _interval_overlap_ms(
                mb0_wait_start_us,
                mb0_wait_end_us,
                mb0_prev_start_local_us,
                mb0_prev_end_local_us,
            )
            stage2_queue_wait_pre_sender_compute_to_serialize_ms = _interval_overlap_ms(
                mb0_wait_start_us,
                mb0_wait_end_us,
                mb0_prev_end_local_us,
                mb0_sender_ser_start_local_us,
            )
            stage2_queue_wait_pre_sender_serialize_ms = _interval_overlap_ms(
                mb0_wait_start_us,
                mb0_wait_end_us,
                mb0_sender_ser_start_local_us,
                mb0_sender_ser_end_local_us,
            )
            stage2_queue_wait_pre_sender_pre_send_wait_ms = _interval_overlap_ms(
                mb0_wait_start_us,
                mb0_wait_end_us,
                mb0_sender_ser_end_local_us,
                mb0_sender_send_local_us,
            )
            stage2_queue_wait_pre_sender_post_compute_ms = _interval_overlap_ms(
                mb0_wait_start_us,
                mb0_wait_end_us,
                mb0_prev_end_local_us,
                mb0_sender_send_local_us,
            )
            sender_post_compute_segment_sum_ms = (
                stage2_queue_wait_pre_sender_compute_to_serialize_ms
                + stage2_queue_wait_pre_sender_serialize_ms
                + stage2_queue_wait_pre_sender_pre_send_wait_ms
            )
            if sender_post_compute_segment_sum_ms > 0.0:
                stage2_queue_wait_pre_sender_post_compute_ms = sender_post_compute_segment_sum_ms
            stage2_queue_wait_pre_wire_receive_ms = _interval_overlap_ms(
                mb0_wait_start_us,
                mb0_wait_end_us,
                mb0_sender_send_local_us,
                mb0_receiver_receive_us,
            )
            stage2_queue_wait_pre_receiver_dispatch_ms = _interval_overlap_ms(
                mb0_wait_start_us,
                mb0_wait_end_us,
                mb0_receiver_receive_us,
                mb0_wait_end_us,
            )
            stage2_queue_wait_pre_transfer_receive_ms = (
                stage2_queue_wait_pre_sender_post_compute_ms
                + stage2_queue_wait_pre_wire_receive_ms
                + stage2_queue_wait_pre_receiver_dispatch_ms
            )
            accounted_pre_wait_ms = (
                stage2_queue_wait_pre_upstream_compute_ms
                + stage2_queue_wait_pre_transfer_receive_ms
            )
            stage2_queue_wait_pre_precompute_gap_ms = max(
                0.0, stage2_queue_wait_pre_ms - accounted_pre_wait_ms
            )
            stage2_queue_wait_pre_breakdown_ready = 1

            if (
                mb0_receiver_queue_put_us > 0
                and mb0_receiver_queue_put_us < mb0_receiver_receive_us
            ):
                mb0_receiver_queue_put_us = mb0_receiver_receive_us

            if (
                mb0_receiver_queue_put_us > 0
                and mb0_receiver_receive_us > 0
                and mb0_receiver_queue_put_us < mb0_wait_end_us
            ):
                receiver_handle_ms = _interval_overlap_ms(
                    mb0_wait_start_us,
                    mb0_wait_end_us,
                    mb0_receiver_receive_us,
                    mb0_receiver_queue_put_us,
                )
                receiver_ready_ms = _interval_overlap_ms(
                    mb0_wait_start_us,
                    mb0_wait_end_us,
                    mb0_receiver_queue_put_us,
                    mb0_wait_end_us,
                )
                stage2_queue_wait_pre_receiver_dispatch_ms = receiver_handle_ms + receiver_ready_ms
                stage2_queue_wait_pre_transfer_receive_ms = (
                    stage2_queue_wait_pre_sender_post_compute_ms
                    + stage2_queue_wait_pre_wire_receive_ms
                    + stage2_queue_wait_pre_receiver_dispatch_ms
                )
                accounted_pre_wait_ms = (
                    stage2_queue_wait_pre_upstream_compute_ms
                    + stage2_queue_wait_pre_transfer_receive_ms
                )
                stage2_queue_wait_pre_precompute_gap_ms = max(
                    0.0, stage2_queue_wait_pre_ms - accounted_pre_wait_ms
                )
        overlap_accum['queue_wait_pre_upstream_compute_ms'] = stage2_queue_wait_pre_upstream_compute_ms
        overlap_accum['queue_wait_pre_transfer_receive_ms'] = stage2_queue_wait_pre_transfer_receive_ms
        overlap_accum['queue_wait_pre_precompute_gap_ms'] = stage2_queue_wait_pre_precompute_gap_ms
        overlap_accum['queue_wait_pre_sender_post_compute_ms'] = stage2_queue_wait_pre_sender_post_compute_ms
        overlap_accum['queue_wait_pre_sender_compute_to_serialize_ms'] = (
            stage2_queue_wait_pre_sender_compute_to_serialize_ms
        )
        overlap_accum['queue_wait_pre_sender_serialize_ms'] = (
            stage2_queue_wait_pre_sender_serialize_ms
        )
        overlap_accum['queue_wait_pre_sender_pre_send_wait_ms'] = (
            stage2_queue_wait_pre_sender_pre_send_wait_ms
        )
        overlap_accum['queue_wait_pre_wire_receive_ms'] = stage2_queue_wait_pre_wire_receive_ms
        overlap_accum['queue_wait_pre_receiver_dispatch_ms'] = stage2_queue_wait_pre_receiver_dispatch_ms
        overlap_accum['queue_wait_pre_breakdown_ready'] = stage2_queue_wait_pre_breakdown_ready
        overlap_accum['next_sender_serialize_total_ms'] = total_next_sender_serialize_ms
        overlap_accum['next_sender_serialize_hidden_ms'] = hidden_next_sender_serialize_ms
        overlap_accum['next_sender_serialize_exposed_ms'] = next_sender_serialize_exposed_ms
        overlap_accum['next_sender_serialize_hidden_pct'] = next_sender_serialize_hidden_pct
        overlap_efficiency = (total_overlap_ms / total_stage2_compute_ms) * 100
        strict_efficiency = (
            (total_overlap_ms / total_comparable_stage2_compute_ms) * 100
            if total_comparable_stage2_compute_ms > 0
            else 0.0
        )
        avg_abs_clock_offset_ms = (
            (clock_offset_abs_sum_ms / clock_corrected_pair_count)
            if clock_corrected_pair_count > 0
            else 0.0
        )
        clock_sync_pairs = clock_corrected_pair_count + clock_uncorrected_pair_count
        clock_sync_coverage_pct = (
            (clock_corrected_pair_count / clock_sync_pairs) * 100
            if clock_sync_pairs > 0
            else 0.0
        )
        if log_mb_detail:
            logger.info(
                f"[CROSS_STAGE_OVERLAP_SUMMARY] step={step_id} overlap={total_overlap_ms:.1f}ms, "
            f"Stage2_compute={total_stage2_compute_ms:.1f}ms, "
            f"Stage2_queue_wait={stage2_queue_wait_sum_ms:.1f}ms, "
            f"Stage2_queue_wait_pre={stage2_queue_wait_pre_ms:.1f}ms, "
            f"Stage2_queue_wait_pre_upstream_compute={stage2_queue_wait_pre_upstream_compute_ms:.1f}ms, "
            f"Stage2_queue_wait_pre_transfer_receive={stage2_queue_wait_pre_transfer_receive_ms:.1f}ms, "
            f"Stage2_queue_wait_pre_precompute_gap={stage2_queue_wait_pre_precompute_gap_ms:.1f}ms, "
            f"Stage2_queue_wait_pre_sender_post_compute={stage2_queue_wait_pre_sender_post_compute_ms:.1f}ms, "
            f"Stage2_queue_wait_pre_sender_compute_to_serialize={stage2_queue_wait_pre_sender_compute_to_serialize_ms:.1f}ms, "
            f"Stage2_queue_wait_pre_sender_serialize={stage2_queue_wait_pre_sender_serialize_ms:.1f}ms, "
            f"Stage2_queue_wait_pre_sender_pre_send_wait={stage2_queue_wait_pre_sender_pre_send_wait_ms:.1f}ms, "
            f"Stage2_queue_wait_pre_wire_receive={stage2_queue_wait_pre_wire_receive_ms:.1f}ms, "
            f"Stage2_queue_wait_pre_receiver_dispatch={stage2_queue_wait_pre_receiver_dispatch_ms:.1f}ms, "
            f"Stage2_queue_wait_pre_breakdown_ready={stage2_queue_wait_pre_breakdown_ready}, "
            f"Stage2_queue_wait_inter={stage2_queue_wait_inter_ms:.1f}ms, "
            f"Stage2_deserialize={stage2_deserialize_sum_ms:.1f}ms, "
            f"Stage2_decompress_on_critical_path={stage2_decompress_sum_ms:.1f}ms, "
            f"Stage2_decompress_hidden=0.0ms, "
            f"Stage1_next_sender_serialize_total={total_next_sender_serialize_ms:.1f}ms, "
            f"Stage1_next_sender_serialize_hidden={hidden_next_sender_serialize_ms:.1f}ms, "
            f"Stage1_next_sender_serialize_exposed={next_sender_serialize_exposed_ms:.1f}ms, "
            f"Stage1_next_sender_serialize_hidden_pct={next_sender_serialize_hidden_pct:.1f}%, "
            f"Stage2_critical_path={stage2_critical_path_ms:.1f}ms, "
            f"Stage2_full_path={stage2_full_path_ms:.1f}ms, "
            f"Stage2_residual={stage2_residual_path_ms:.1f}ms, "
            f"efficiency={overlap_efficiency:.1f}%, "
            f"strict_efficiency={strict_efficiency:.1f}%, "
            f"comparable_compute={total_comparable_stage2_compute_ms:.1f}ms, "
            f"missing_pairs={missing_pair_count}, invalid_pairs={invalid_pair_count}, "
            f"clock_corrected_pairs={clock_corrected_pair_count}, "
            f"clock_uncorrected_pairs={clock_uncorrected_pair_count}, "
            f"clock_sync_coverage={clock_sync_coverage_pct:.1f}%, "
            f"avg_abs_clock_offset={avg_abs_clock_offset_ms:.2f}ms, "
            f"max_clock_rtt={clock_rtt_max_ms:.2f}ms"
        )

