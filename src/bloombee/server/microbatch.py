from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from bloombee.data_structures import Handle, InferenceMetadata
from bloombee.utils.microbatch_config import get_micro_batch_size
from bloombee.utils.misc import is_dummy, slice_batch_aligned


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
    # Micro-batches are processed sequentially on the current runtime path, so we
    # can pass a view into the original hidden-state batch instead of eagerly
    # cloning each slice. This avoids an extra GPU copy on every micro-batch.
    mb_hidden_states = hidden_states[mb_start:mb_end]
    if not mb_hidden_states.is_contiguous():
        mb_hidden_states = mb_hidden_states.contiguous()

    return MicrobatchInputs(
        hidden_states=mb_hidden_states,
        # Keep global row IDs; the backend normalizes them using batch_offset.
        hypo_ids=hypo_ids,
        tree_attention_mask=slice_batch_aligned(tree_attention_mask, mb_start, mb_end, full_batch_size),
        kv_cache_position_ids=slice_batch_aligned(kv_cache_position_ids, mb_start, mb_end, full_batch_size),
        draft_tokens=slice_batch_aligned(draft_tokens, mb_start, mb_end, full_batch_size),
        prefill_length=slice_batch_aligned(prefill_length, mb_start, mb_end, full_batch_size),
        keep_indices=slice_batch_aligned(keep_indices, mb_start, mb_end, full_batch_size),
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
    push_metadata = {
        key: value
        for key, value in (step_metadata or {}).items()
        if not str(key).startswith("_")
    }
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


S2S_SPEC_TENSOR_NAMES: Tuple[str, ...] = (
    "tree_attention_mask",
    "kv_cache_position_ids",
    "draft_tokens",
    "prefill_length",
)


def _is_s2s_hypo_tensor(value: Any) -> bool:
    return torch.is_tensor(value) and value.ndim == 1 and value.numel() > 0 and not is_dummy(value)


def build_s2s_spec_tensors(
    *,
    is_spec_dec: bool,
    tree_attention_mask: Any = None,
    kv_cache_position_ids: Any = None,
    draft_tokens: Any = None,
    prefill_length: Any = None,
    hypo_ids: Any = None,
) -> Optional[Dict[str, Any]]:
    """Tensors attached to a cross-stage micro-batch push.

    Non-spec extras are optional padding mask then optional hypo_ids.
    Spec extras keep ``S2S_SPEC_TENSOR_NAMES`` first; hypo_ids is appended
    after those four so existing ``len >= 6`` unpacking stays valid.
    """
    tensors: Dict[str, Any] = {}
    if torch.is_tensor(tree_attention_mask) and not is_dummy(tree_attention_mask):
        tensors["tree_attention_mask"] = tree_attention_mask
    if _is_s2s_hypo_tensor(hypo_ids):
        tensors["hypo_ids"] = hypo_ids
    if is_spec_dec:
        tensors["kv_cache_position_ids"] = kv_cache_position_ids
        tensors["draft_tokens"] = draft_tokens
        tensors["prefill_length"] = prefill_length
        return tensors
    return tensors or None


def s2s_extra_tensor_names(
    is_spec_dec: bool,
    spec_tensors: Optional[Dict[str, Any]] = None,
) -> Tuple[str, ...]:
    spec_tensors = spec_tensors or {}
    names: list[str] = []
    if is_spec_dec:
        names.extend(S2S_SPEC_TENSOR_NAMES)
    else:
        mask = spec_tensors.get("tree_attention_mask")
        if torch.is_tensor(mask) and mask.ndim >= 2 and not is_dummy(mask):
            names.append("tree_attention_mask")
    if _is_s2s_hypo_tensor(spec_tensors.get("hypo_ids")):
        names.append("hypo_ids")
    return tuple(names)


def unpack_s2s_extras(
    flat_tensors: Sequence[Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any, Any, Any, Any]:
    """Unpack extras after hidden_states/keep_indices. Scale tensors use metadata index.

    Returns ``(tree_or_mask, kv_pos, draft, prefill, hypo_ids)``.
    Non-spec extras occupy slots after keep_indices in this order:
    optional padding mask, optional hypo_ids. Spec extras occupy ``[2:6]``;
    hypo_ids follows those four when ``s2s_hypo_ids`` is set.
    """
    metadata = metadata or {}
    hypo_ids = None
    if bool(metadata.get("is_spec_dec")) and len(flat_tensors) >= 6:
        tree, kv_pos, draft, prefill = flat_tensors[2:6]
        if metadata.get("s2s_hypo_ids") and len(flat_tensors) >= 7:
            hypo_ids = flat_tensors[6]
        return tree, kv_pos, draft, prefill, hypo_ids
    next_idx = 2
    mask = None
    if metadata.get("s2s_padding_mask") and len(flat_tensors) > next_idx:
        mask = flat_tensors[next_idx]
        next_idx += 1
    if metadata.get("s2s_hypo_ids") and len(flat_tensors) > next_idx:
        hypo_ids = flat_tensors[next_idx]
    return mask, None, None, None, hypo_ids


class MicrobatchStepTracker:
    """Per-(session, step) delivery / abort state for S2S micro-batches.

    Queued micro-batches are treated as committed for replay purposes: a later
    abort or retry must not re-run them against the same KV slab. Aborting a
    step rejects remaining micro-batches instead of clearing the committed set.
    """

    __slots__ = ("ttl_s", "_received", "_received_at", "_aborted_at")

    def __init__(self, ttl_s: float = 120.0) -> None:
        self.ttl_s = float(ttl_s)
        self._received: Dict[tuple, set] = {}
        self._received_at: Dict[tuple, float] = {}
        self._aborted_at: Dict[tuple, float] = {}

    def _now(self) -> float:
        import time

        return time.monotonic()

    def cleanup(self, now: Optional[float] = None) -> None:
        now = self._now() if now is None else now
        stale = [key for key, ts in self._received_at.items() if now - ts > self.ttl_s]
        for key in stale:
            self._received.pop(key, None)
            self._received_at.pop(key, None)
        stale_abort = [key for key, ts in self._aborted_at.items() if now - ts > self.ttl_s]
        for key in stale_abort:
            self._aborted_at.pop(key, None)

    def is_aborted(self, key: tuple) -> bool:
        return key in self._aborted_at

    def received_indices(self, key: tuple) -> set:
        return set(self._received.get(key, ()))

    def decide(self, key: tuple, mb_idx: int) -> str:
        """Return ``accept``, ``duplicate``, or ``aborted``."""
        self.cleanup()
        if key in self._aborted_at:
            return "aborted"
        received = self._received.setdefault(key, set())
        self._received_at.setdefault(key, self._now())
        if mb_idx in received:
            return "duplicate"
        received.add(mb_idx)
        self._received_at[key] = self._now()
        return "accept"

    def abort(self, key: tuple) -> set:
        """Mark the step aborted. Keep received indices so committed MBs are not replayed."""
        self.cleanup()
        self._aborted_at[key] = self._now()
        if key not in self._received:
            self._received[key] = set()
            self._received_at[key] = self._now()
        return set(self._received[key])
