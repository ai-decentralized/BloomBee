from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from bloombee.data_structures import Handle, InferenceMetadata
from bloombee.utils.microbatch_config import get_micro_batch_size
from bloombee.utils.misc import is_dummy


@dataclass(frozen=True)
class MicrobatchInputs:
    hidden_states: torch.Tensor
    hypo_ids: Any
    tree_attention_mask: Any
    kv_cache_position_ids: Any
    draft_tokens: Any
    prefill_length: Any
    keep_indices: Any
    batch_offset: int
    full_batch_size: int
    micro_batch_size: int


def slice_batch_aligned(
    value: Any,
    mb_start: int,
    mb_end: int,
    full_batch_size: int,
) -> Any:
    """
    Slice tensor-like request fields only if they are batch-aligned.
    Non-tensor / scalar / already-global fields are returned as-is.
    """
    if value is None or not torch.is_tensor(value):
        return value
    if is_dummy(value):
        return value
    if value.ndim == 0:
        return value
    if value.shape[0] == full_batch_size:
        return value[mb_start:mb_end].contiguous()
    return value


def slice_keep_indices(
    keep_indices: Any,
    mb_start: int,
    mb_end: int,
    full_batch_size: int,
) -> Any:
    if keep_indices is None or not torch.is_tensor(keep_indices):
        return keep_indices
    if is_dummy(keep_indices) or keep_indices.ndim == 0:
        return keep_indices
    if keep_indices.shape[0] == full_batch_size:
        return keep_indices[mb_start:mb_end].contiguous()
    return keep_indices


def slice_microbatch_inputs(
    hidden_states: torch.Tensor,
    hypo_ids: Any,
    tree_attention_mask: Any,
    kv_cache_position_ids: Any,
    draft_tokens: Any,
    prefill_length: Any,
    keep_indices: Any,
    *,
    mb_start: int,
    mb_end: int,
    full_batch_size: int,
) -> MicrobatchInputs:
    mb_size = mb_end - mb_start
    if hypo_ids is None or is_dummy(hypo_ids):
        mb_hypo = hypo_ids
    else:
        mb_hypo = hypo_ids[mb_start:mb_end]

    return MicrobatchInputs(
        hidden_states=hidden_states[mb_start:mb_end].clone(),
        hypo_ids=mb_hypo,
        tree_attention_mask=slice_batch_aligned(tree_attention_mask, mb_start, mb_end, full_batch_size),
        kv_cache_position_ids=kv_cache_position_ids,
        draft_tokens=slice_batch_aligned(draft_tokens, mb_start, mb_end, full_batch_size),
        prefill_length=slice_batch_aligned(prefill_length, mb_start, mb_end, full_batch_size),
        keep_indices=slice_keep_indices(keep_indices, mb_start, mb_end, full_batch_size),
        batch_offset=mb_start,
        full_batch_size=full_batch_size,
        micro_batch_size=mb_size,
    )


def build_inference_metadata(
    uid: Any,
    cache_handles: Sequence[Handle],
    prefix_length: int,
    active_adapter: Optional[str],
    *,
    tree_attention_mask: Any = None,
    kv_cache_position_ids: Any = None,
    draft_tokens: Any = None,
    prefill_length: Any = None,
    keep_indices: Any = None,
    need_pruning: Any = None,
    is_spec_dec: Any = None,
    batch_offset: int = 0,
    full_batch_size: int = 0,
    micro_batch_size: int = 0,
) -> InferenceMetadata:
    return InferenceMetadata(
        uid,
        prefix_length,
        tuple(cache_handles),
        active_adapter,
        tree_attention_mask=tree_attention_mask,
        kv_cache_position_ids=kv_cache_position_ids,
        draft_tokens=draft_tokens,
        prefill_length=prefill_length,
        keep_indices=keep_indices,
        need_pruning=need_pruning,
        is_spec_dec=is_spec_dec,
        batch_offset=batch_offset,
        full_batch_size=full_batch_size,
        micro_batch_size=micro_batch_size,
    )


def build_inference_metadata_batch(
    requested_uids: Sequence[Any],
    cache_handles: Sequence[Sequence[Handle]],
    prefix_length: int,
    active_adapter: Optional[str],
    **metadata_kwargs: Any,
) -> Tuple[InferenceMetadata, ...]:
    return tuple(
        build_inference_metadata(
            uid,
            handles,
            prefix_length,
            active_adapter,
            **metadata_kwargs,
        )
        for uid, handles in zip(requested_uids, cache_handles)
    )


def build_cross_stage_push_metadata(
    step_metadata: Optional[Dict[str, Any]],
    *,
    mb_idx: int,
    mb_offset: int,
    mb_size: int,
    full_batch_size: int,
    sender_blocks: str,
    total_micro_batches: int,
    compute_start_timestamp_us: int,
    compute_end_timestamp_us: int,
    push_timestamp_us: int,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    push_metadata = dict(step_metadata or {})
    push_metadata.update(
        {
            "micro_batch_idx": mb_idx,
            "micro_batch_offset": mb_offset,
            "micro_batch_size": mb_size,
            "full_batch_size": full_batch_size,
            "sender_blocks": sender_blocks,
            "total_micro_batches": total_micro_batches,
            "stage_compute_start_timestamp_us": compute_start_timestamp_us,
            "stage_compute_end_timestamp_us": compute_end_timestamp_us,
            "stage_push_timestamp_us": push_timestamp_us,
        }
    )
    if extra_fields:
        push_metadata.update(extra_fields)
    return push_metadata


def resolve_expected_num_microbatches(
    full_batch_size: int,
    total_micro_batches: Optional[int] = None,
    configured_micro_batch_size: Optional[int] = None,
) -> int:
    if total_micro_batches is not None:
        try:
            return max(1, int(total_micro_batches))
        except Exception:
            pass

    if configured_micro_batch_size is None:
        configured_micro_batch_size = get_micro_batch_size()

    try:
        micro_batch_size = int(configured_micro_batch_size)
    except Exception:
        micro_batch_size = 0

    if micro_batch_size > 0:
        return max(1, (int(full_batch_size) + micro_batch_size - 1) // micro_batch_size)
    return 1
