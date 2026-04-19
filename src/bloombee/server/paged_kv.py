"""
Paged KV cache primitive for Phase 2 of the BloomBee arch reform.

Pure data structure: a fixed-size page pool indexing along the sequence dim,
plus a per-sequence page table. No torch kernel code lives here — reads and
writes are done with plain index_select / slice-assign against the underlying
storage tensor. See PHASE2_PAGED_KV_INVARIANTS.md for the state model this
implements.

Design notes
------------
- Storage layout matches FlexGen's internal shape: ``(S_total, BH, D)``. A
  *page* is a contiguous slab of ``BLOCK_SIZE`` tokens along S for all BH
  rows; this lets the existing ``(s, BH, D)`` attention kernels consume a
  gathered view without a transpose.
- Reads are clamped to ``L_acc`` (invariant 3). A rolled-back request that
  has valid bytes past ``L_acc`` in its last page will never expose them.
- Writes are idempotent inside ``[L_acc, L_seq)`` (invariant 4). This is
  what lets spec decoding re-run a verify pass without corrupting the
  committed prefix.
- Free is deterministic: session close or rollback. No mid-step eviction
  (invariant 5).

This module is intentionally self-contained so it can be unit-tested
without spinning up MemoryCache, the server, or CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


BLOCK_SIZE: int = 16


class PagedKVError(RuntimeError):
    """Raised for contract violations (OOM, invalid rollback, etc.)."""


@dataclass
class _SeqState:
    # Ordered physical page ids held by this sequence.
    pages: List[int] = field(default_factory=list)
    # Committed accepted prefix length.
    l_acc: int = 0
    # Logical length (may be >= l_acc when a spec tree is under verification).
    l_seq: int = 0


class PagedKVTable:
    """
    One instance per (peer, block, kv_kind) — i.e. per allocated cache tensor.

    The caller supplies a backing tensor shaped ``(S_total, BH, D)`` at init
    time. ``S_total`` must be an exact multiple of ``BLOCK_SIZE``. The table
    never allocates torch memory on its own; it only hands out page indices
    into the pre-allocated backing tensor.
    """

    def __init__(
        self,
        backing_k: torch.Tensor,
        backing_v: torch.Tensor,
        block_size: int = BLOCK_SIZE,
    ) -> None:
        if backing_k.shape != backing_v.shape:
            raise PagedKVError(
                f"k/v backing tensors must be same shape, got {backing_k.shape} vs {backing_v.shape}"
            )
        if backing_k.ndim != 3:
            raise PagedKVError(
                f"backing tensors must be (S, BH, D); got shape {tuple(backing_k.shape)}"
            )
        s_total = int(backing_k.shape[0])
        if s_total % block_size != 0:
            raise PagedKVError(
                f"S_total={s_total} must be a multiple of block_size={block_size}"
            )

        self.k = backing_k
        self.v = backing_v
        self.block_size = block_size
        self.num_pages = s_total // block_size
        # Free list as a stack (LIFO — keeps recently-freed pages hot).
        self._free: List[int] = list(range(self.num_pages - 1, -1, -1))
        self._seqs: Dict[int, _SeqState] = {}

    # ----- sequence lifecycle --------------------------------------------------

    def register_sequence(self, seq_id: int) -> None:
        if seq_id in self._seqs:
            raise PagedKVError(f"sequence {seq_id} already registered")
        self._seqs[seq_id] = _SeqState()

    def release_sequence(self, seq_id: int) -> None:
        state = self._seqs.pop(seq_id, None)
        if state is None:
            return
        for page in state.pages:
            self._free.append(page)

    def has_sequence(self, seq_id: int) -> bool:
        return seq_id in self._seqs

    # ----- state accessors -----------------------------------------------------

    def l_seq(self, seq_id: int) -> int:
        return self._seqs[seq_id].l_seq

    def l_acc(self, seq_id: int) -> int:
        return self._seqs[seq_id].l_acc

    def num_free_pages(self) -> int:
        return len(self._free)

    def num_used_pages(self, seq_id: int) -> int:
        return len(self._seqs[seq_id].pages)

    # ----- allocation ----------------------------------------------------------

    def _ensure_capacity(self, seq_id: int, target_len: int) -> None:
        """Allocate pages until sequence can address ``target_len`` tokens."""
        state = self._seqs[seq_id]
        pages_needed = (target_len + self.block_size - 1) // self.block_size
        while len(state.pages) < pages_needed:
            if not self._free:
                raise PagedKVError(
                    f"out of pages: need {pages_needed}, have {len(state.pages)} "
                    f"for seq {seq_id} (free={len(self._free)})"
                )
            state.pages.append(self._free.pop())

    # ----- write ---------------------------------------------------------------

    def write(
        self,
        seq_id: int,
        start_position: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        bh_slice: Tuple[int, int],
        commit: bool = True,
    ) -> None:
        """
        Append ``k_new`` / ``v_new`` starting at logical position ``start_position``.

        Shape contract: ``k_new`` and ``v_new`` are ``(s_new, BH_slice, D)`` — i.e.
        already in the internal storage layout (matches ``_write_kvs`` which
        does ``permute(2, 0, 1)`` before the copy in memory_cache_manager.py).

        Args:
            bh_slice: (bh_start, bh_end) along the batch-head dim. Enables
                the micro-batch / GPU-multiplex slicing the existing cache
                manager supports.
            commit: if True, bump ``l_acc`` alongside ``l_seq``. Set to False
                for speculative writes past the accepted prefix; they will
                only raise ``l_seq``, and rollback can drop them without
                touching the committed bytes.
        """
        if k_new.shape != v_new.shape:
            raise PagedKVError(
                f"k_new/v_new shapes differ: {tuple(k_new.shape)} vs {tuple(v_new.shape)}"
            )
        if k_new.ndim != 3:
            raise PagedKVError(
                f"k_new must be (s_new, BH, D); got {tuple(k_new.shape)}"
            )
        s_new = int(k_new.shape[0])
        if s_new == 0:
            return
        end_position = start_position + s_new
        self._ensure_capacity(seq_id, end_position)

        state = self._seqs[seq_id]
        bh_start, bh_end = bh_slice

        # Dispatch the write one page at a time; this keeps the core logic
        # obvious and each page's destination contiguous, which is the best
        # we can do without a custom kernel.
        pos = start_position
        src_offset = 0
        while pos < end_position:
            page_idx = pos // self.block_size
            slot = pos % self.block_size
            tokens_this_page = min(self.block_size - slot, end_position - pos)
            phys_page = state.pages[page_idx]
            s_base = phys_page * self.block_size + slot

            self.k[s_base : s_base + tokens_this_page, bh_start:bh_end, :] = k_new[
                src_offset : src_offset + tokens_this_page
            ]
            self.v[s_base : s_base + tokens_this_page, bh_start:bh_end, :] = v_new[
                src_offset : src_offset + tokens_this_page
            ]

            pos += tokens_this_page
            src_offset += tokens_this_page

        if end_position > state.l_seq:
            state.l_seq = end_position
        if commit and end_position > state.l_acc:
            state.l_acc = end_position

    def track_write(
        self,
        seq_id: int,
        start_position: int,
        s_new: int,
        commit: bool = True,
    ) -> None:
        """
        State-only mirror of ``write``: ensure pages are allocated and advance
        ``l_seq`` / ``l_acc``, but do NOT touch the backing tensors. Use when
        the caller has already landed the bytes via a separate path (e.g.
        legacy ``_write_kvs``) and only needs the shim's state machine to
        reflect that write — this is what makes the shim load-bearing for
        rollback without double-copying every decode step.
        """
        if s_new <= 0:
            return
        if seq_id not in self._seqs:
            self.register_sequence(seq_id)
        end_position = start_position + int(s_new)
        self._ensure_capacity(seq_id, end_position)
        state = self._seqs[seq_id]
        if end_position > state.l_seq:
            state.l_seq = end_position
        if commit and end_position > state.l_acc:
            state.l_acc = end_position

    # ----- commit / rollback ---------------------------------------------------

    def commit(self, seq_id: int, up_to: Optional[int] = None) -> None:
        """Promote speculative writes to accepted. ``up_to`` defaults to ``l_seq``."""
        state = self._seqs[seq_id]
        target = state.l_seq if up_to is None else int(up_to)
        if target > state.l_seq:
            raise PagedKVError(
                f"commit beyond l_seq: target={target}, l_seq={state.l_seq}"
            )
        if target > state.l_acc:
            state.l_acc = target

    def rollback(self, seq_id: int, l_acc_target: int) -> None:
        """Drop speculation past ``l_acc_target``; release newly-orphaned pages."""
        state = self._seqs[seq_id]
        if l_acc_target > state.l_acc:
            raise PagedKVError(
                f"rollback target {l_acc_target} exceeds committed l_acc={state.l_acc}"
            )
        if l_acc_target < 0:
            raise PagedKVError(f"rollback target must be >= 0, got {l_acc_target}")

        state.l_seq = l_acc_target
        state.l_acc = l_acc_target
        keep_pages = (l_acc_target + self.block_size - 1) // self.block_size
        while len(state.pages) > keep_pages:
            released = state.pages.pop()
            self._free.append(released)

    # ----- read ----------------------------------------------------------------

    def gather_prefix(
        self,
        seq_id: int,
        bh_slice: Tuple[int, int],
        length: Optional[int] = None,
        out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return ``(k_prefix, v_prefix)`` of shape ``(length, bh_end - bh_start, D)``.

        Default length is ``l_acc`` — invariant 3 guarantees we never expose
        rolled-back bytes. Caller can pass a smaller length to sub-sample;
        passing a larger value raises.

        If ``out`` is provided, the caller owns a pre-allocated staging buffer
        pair — write into it and return it. Otherwise a fresh tensor is
        allocated on the backing device.
        """
        state = self._seqs[seq_id]
        l_acc = state.l_acc
        length = l_acc if length is None else int(length)
        if length > l_acc:
            raise PagedKVError(
                f"read length {length} exceeds committed l_acc={l_acc} for seq {seq_id}"
            )
        bh_start, bh_end = bh_slice
        bh_width = bh_end - bh_start
        d = self.k.shape[-1]

        if out is None:
            k_out = torch.empty((length, bh_width, d), dtype=self.k.dtype, device=self.k.device)
            v_out = torch.empty_like(k_out)
        else:
            k_out, v_out = out

        pos = 0
        while pos < length:
            page_idx = pos // self.block_size
            slot = pos % self.block_size
            tokens_this_page = min(self.block_size - slot, length - pos)
            phys_page = state.pages[page_idx]
            s_base = phys_page * self.block_size + slot

            k_out[pos : pos + tokens_this_page] = self.k[
                s_base : s_base + tokens_this_page, bh_start:bh_end, :
            ]
            v_out[pos : pos + tokens_this_page] = self.v[
                s_base : s_base + tokens_this_page, bh_start:bh_end, :
            ]
            pos += tokens_this_page

        return k_out, v_out
