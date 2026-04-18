# Phase 2 — Paged KV Cache & Continuous Batching: Invariants

Codex pushed back on implementing paged KV without writing down the state model first: both continuous batching and (later) speculative rollback need **exact** answers for logical length, accepted prefix, page ownership, and reclaim timing. Codifying those now, before code.

All citations reference the `arch-reform` branch state after Phase 0 + Phase 1 land (commits `c04bdee`, `e532764`).

---

## 1. Terminology & state

| Term | Definition |
|------|------------|
| **Request** | One session from a client. Currently pinned to a `cache_handles` tuple (one `Handle` per block). |
| **Logical sequence length** `L_seq` | Number of KV tokens that exist *for this request's attention*. Monotonically non-decreasing within an accepted prefix; can be rolled back on spec reject. |
| **Accepted prefix length** `L_acc` | Number of KV tokens whose correctness is committed. For non-spec decode, `L_acc == L_seq`. For spec, `L_seq >= L_acc` while a tree is being verified. |
| **Page** | Fixed-size KV tile: `block_size = 16` tokens per page, per layer, per (head, dim). |
| **Page table** `P(req)` | Ordered list of physical page ids owned by this request at each block: `P(req, block_idx) = [page_0, page_1, ...]`. Length is `ceil(L_seq / 16)`. |
| **Page id** | Opaque integer indexing the server-local page pool. Never crosses the wire. |
| **Handle** | Server-local integer returned by `MemoryCache._schedule_alloc`. Today it resolves to a contiguous `(k_tensor, v_tensor)` pair in `self._allocated_tensors[handle]`. After Phase 2 it resolves to a page-table-backed view. Handles still never cross the wire. |

## 2. What the wire protocol already allows

From the audit completed in the previous session (see handler.py:2331-2344 and data_structures.py:122-137):

- `InferenceMetadata.prefix_length` is a **logical** scalar. It's the number of KV tokens the server should use for this attention step. It does not encode how they're stored.
- `InferenceMetadata.cache_handles` is **local-only** — constructed per-peer and never serialized (see the metadata whitelist at handler.py:2331-2344; `cache_handles` is absent).
- `start_from_position`, `full_batch_size`, `micro_batch_size`, `prefill_length`, `step_id` are all logical.
- `kv_cache_position_ids` (data_structures.py:128) carries per-token logical positions for the sparse spec-decode path; it's a token index list, not a tensor address.

**Invariant-1 (protocol):** Paging is a *per-peer-local* refactor. No wire format changes required for Phase 2. Each peer can independently switch from contiguous to paged storage as long as it honors logical `prefix_length` semantics.

## 3. Storage invariants

### 3.1 Page pool ownership

- One page pool per peer, per block (matches today's per-block allocation at `memory_cache_manager.py:422`-ish write paths). Sharing the pool across blocks is an optimization; do **not** attempt it in Phase 2.
- Each page carries a reference count `rc`. `rc == 0` means reclaimable. `rc > 0` means at least one request references it.
- Initial implementation: `rc ∈ {0, 1}` (no cross-request KV sharing). Prefix-sharing / copy-on-write is Phase 4+.

### 3.2 Page table lifecycle

For a request with cache handles allocated at `t=t0`:

1. **Allocation:** `schedule_alloc` reserves **metadata only** (a page table struct) — no physical pages. Total-token budget in `MemoryCache._current_tokens` still tracks `max_length * num_layers` as the worst-case pessimistic bound (keeps existing admission control behavior; no surprises).
2. **First prefill step:** page-in N pages where N = `ceil(prefill_length / 16)`. Write KVs into those pages. Update `P(req)` and `L_acc = L_seq = prefill_length`.
3. **Decode step (non-spec):** if `L_seq % 16 == 0`, allocate one new page and append to `P(req)`. Always write new KV at logical position `L_seq`, resolved to `(page=P(req)[L_seq // 16], slot=L_seq % 16)`. Then `L_seq += 1; L_acc = L_seq`.
4. **Decode step (spec, k draft tokens):** append pages as needed for up-to-k speculative writes, same positional math. `L_seq += accepted_k; L_acc += accepted_k`. Any page that only held rejected tail tokens is released (see 3.3).
5. **Session end:** all pages in `P(req, *)` have `rc` decremented; those that hit 0 return to the free list.

### 3.3 Rollback semantics

**Invariant-2 (rollback):** Rollback is a pure operation on the page table + `L_seq`. No KV copies.

Rollback from `L_seq` back to `L_acc = L_acc'`:
- Compute `keep_pages = ceil(L_acc' / 16)`.
- Pages at indices `[keep_pages, len(P(req)))` are released (`rc -= 1`; if 0, returned to free list).
- The last retained page (index `keep_pages - 1`) keeps its stored content but the `[L_acc' % 16, 16)` slots are now **semantically invalid**. They will be overwritten by the next decode/spec write; they MUST NOT be read by any attention op because `prefix_length` is clipped to `L_acc'`.

**Invariant-3 (read clamp):** attention reads ALWAYS respect `prefix_length`. A read never extends beyond `L_acc` even if the last page has trailing written-but-rolled-back bytes. This is already the contract in `select_cache` (memory_cache_manager.py:464-497), which clamps at `prefix_length`. Keep it.

### 3.4 Write atomicity

Phase 1 introduced `_finalize_cache_update` (backend.py:577-584) as the step-level write seam. For paged KV it becomes:

- Given step-level new KVs (shape `(B, H, new_tokens, D)` or FlexGen's `(new_tokens, B*H, D)`) and step-level start position `L_start`:
  - For each logical position `p` in `[L_start, L_start + new_tokens)`:
    - `page = P(req)[p // 16]; slot = p % 16`
    - `write kv[p - L_start] into page[slot]`
  - Write is idempotent under equal `(L_start, new_tokens)` — **important for retry / spec re-verify**.

**Invariant-4 (write idempotence):** writes within `[L_acc, L_seq)` can be re-done without corrupting downstream reads because the committed prefix `[0, L_acc)` is never rewritten.

### 3.5 Free semantics

- Free happens only in three places:
  - Session close → release all pages in `P(req, *)`.
  - Rollback → release trailing pages past `keep_pages`.
  - Admission pressure → **never** reclaim from a live session mid-step. Admission uses logical token budget (section 4).

**Invariant-5 (no mid-step eviction):** a request's pages cannot be reclaimed while `use_cache` is held (i.e., inside an active `inference_step`). This mirrors the current `use_cache` context discipline at memory_cache.py:266-396 and `memory_cache_manager.py:745-755` — keep it.

## 4. Admission / continuous batching invariants

### 4.1 Logical token accounting

Today `MemoryCache.current_size_tokens` is incremented at schedule-alloc time by a worst-case `alloc_tokens_num = max_length` (memory_cache.py:135). After paging:

- **Phase-2a (static):** keep the same pessimistic accounting. Every admitted request still charges `max_length` tokens. This under-utilizes pages in the short-output case but matches existing behavior and guarantees no OOM surprises. Ship this first.
- **Phase-2b (elastic, optional):** admission based on *currently resident* pages plus a reservation headroom. Deferred; do not land until 2a is green on V100.

**Invariant-6 (admission):** `sum_over_live_requests(alloc_tokens_num) <= max_size_tokens` is preserved unchanged. Paging changes how physical memory maps to that budget, not the budget itself.

### 4.2 Continuous batching admission rule

The task pool (`src/bloombee/server/task_pool.py:36, 124-137`) currently pops one task at a time per step. Continuous batching means:

- At step boundary, scan the task pool for tasks whose (a) state permits joining the current step's batch (i.e., their prefill is done or pending) and (b) page-table capacity is OK.
- Join them to the active batch; each joiner's `batch_offset` in the step is `current_batch_size + joiner_index`.
- `full_batch_size` for the step is the new batch size.
- Departing requests (EOS or error) leave gaps. Keep the simple rule: **no re-packing within a step**. Gaps are tolerated until the next step. (This is the approach vLLM uses in its first continuous-batch implementation, and it keeps the mask builder simple.)

**Invariant-7 (joiner alignment):** a newly joined request's position_ids for this step start at its own `L_seq`, NOT at the batch's max `L_seq`. This requires replacing the fixed `cache_len + offset` expression at backend.py:542 with per-sequence logical length.

### 4.3 Mask & position invariants

Today: backend builds `(B, L, L)` causal mask at backend.py:527 and per-sequence `position_ids` at backend.py:537-542 using a single `cache_len` scalar (the batch assumes all sequences share the same cache length).

After Phase 2:

- `cache_len` per request: `L_seq_i` for request `i`.
- Position ids: `position_ids[i, j] = L_seq_i + j` for `j in [0, new_tokens_i)`.
- Mask: block-sparse per-token. For a step where each request contributes `new_tokens_i` tokens, build the combined sequence dim `T = sum(new_tokens_i)` and the cache dim `S_max = max(L_seq_i + new_tokens_i)`. Each query token attends to **only its own request's cache**. This is the standard vLLM `PagedAttention` contract.
- For Phase 2a we can stay on the un-paged attention kernel by materializing per-request KV contiguously into a staging buffer at read time (costs a copy, but avoids writing a new attention kernel). Swap to a true paged-attention kernel in Phase 2c.

**Invariant-8 (no cross-request attention):** under no circumstance does request `i`'s query see request `j`'s KV, including zero-padded slots.

## 5. What Phase 2 does NOT do

To keep blast radius bounded:

- No spec-decode rollback integration (Phase 3).
- No cross-request prefix sharing / copy-on-write.
- No paged attention kernel — we stage to contiguous per-request KV at read time.
- No elastic admission — pessimistic `max_length` charge stays.
- No change to wire format.
- No change to FlexGen offload paths (MIXED device, CPU cache, compress). Those continue to operate on logical `prefix_length` and the page-table→staging shim; if they need rework, that's Phase 2d+.

## 6. Implementation order (in-repo, no branching)

1. **Write invariants doc.** ← this file.
2. **Add `PagedKVTable` struct** (new file `src/bloombee/server/paged_kv.py`). Pure data structure: page table, free list, alloc/free/rollback/write/read_range. Unit-test in isolation.
3. **Shim `MemoryCache._allocated_tensors[handle]`** to a `PagedKVHandle` that, at `use_cache` time, exposes a *view object* with `.data = (k_sbh, v_sbh)` reconstructed by concatenating referenced pages. This keeps `select_cache` / `update_cache` / `_run_block_forward` unchanged — the hot path sees the same tensor shapes.
4. **Replace `update_cache` write** with paged write: call `PagedKVTable.write(req, L_start, new_kvs)` instead of slicing into a contiguous buffer.
5. **Per-sequence `L_seq`:** carry it in `InferenceMetadata`-adjacent state (`L_seq_i`) and remove the shared `cache_len` assumption in backend.py:542.
6. **Task pool continuous admission:** extend `PrioritizedTaskPool.load_batch_to_runtime` (task_pool.py:124-137) to pull N tasks per step until the batch is full.
7. **V100 gate:** static batching under paged storage must match Phase 1 tokens/sec within 5%. Then turn on dynamic admission.

## 7. V100 validation gates (Phase 2)

- **Gate 2a:** paged storage in static batch mode. Decode parity vs. Phase 1 baseline on llama-7b fp16 @ 16GB. No crash, same tokens/sec ± 5%.
- **Gate 2b:** continuous batching under 2–4 concurrent sessions, llama-7b. Aggregate throughput ≥ 1.5× single-stream at matched per-request output length.
- **Gate 2c (optional):** paged attention kernel. Defer unless 2a/2b shipped green.

## 8. Open questions / risks to retire before coding

1. **FlexGen's `(S, BH, D)` storage** — the page tile layout must be compatible. Proposed tile shape: `(16, B*H, D)` per page, one page per sequence block per layer. The `_to_pkv` reshape at memory_cache_manager.py:725-727 expects contiguous-in-S; a paged read staging path must concatenate along dim 0 to reconstruct. Confirm reshape semantics hold.
2. **MIXED-device cache path** (memory_cache_manager.py:620-678) assumes an `(S, BH, D)` tensor with segment points along BH. Paging changes the S dim, not BH; the segment logic can stay but now operates on staged-per-request slices. Audit before coding.
3. **Allocation accounting hysteresis:** `_enqueued_tokens` + `_current_tokens` (memory_cache.py:77-78) dance works for contiguous alloc. Under paging the same counters can be reused since we keep pessimistic accounting in 2a. No change needed.
4. **Spec decode already writes into the same tensor via `update_cache_and_async_reorder`** (memory_cache_manager.py:1833). Paged write path must preserve that branch's reorder semantics. Will be revisited in Phase 3; for 2a, spec-decode path calls unchanged into the shimmed handle, which materializes a contiguous view just like today.

---

Next step after this doc lands: implement `PagedKVTable` struct and the handle shim (step 2 + 3 above). V100 parity gate 2a before wiring continuous admission.
