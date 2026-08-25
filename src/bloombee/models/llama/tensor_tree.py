"""Tensorized EAGLE-2 speculative tree (GPU-tree migration, Stage 1+2).

This module replaces the per-round Python `_CandNode`/`TreeNode` object graph,
the Python global rerank / parent-closure, the recursive DFS linearization, the
per-row attention-mask build, and the Python child-walk in greedy verification
with fixed `[B, max_nodes]` tensors and vectorized GPU ops.

HARD CONTRACT (must stay token-identical to the Python path for greedy decode):

* Node 0 of every row is the ROOT. Draft nodes occupy indices 1..n in **DFS
  pre-order** (visit a node, then its children in child-insertion order) — the
  exact order `spe_dec_tree.linearize_tree_with_positions` produces. So a draft
  node at tensor index ``k`` (1-based among draft nodes, i.e. node index ``k``)
  has ``position_in_sequence == k - 1``.
* ``parent_pos[b, k]`` is the DFS position (``position_in_sequence``) of the
  parent draft node, or ``-1`` when the parent is the root. This matches the
  ``parent_indices`` list consumed by ``build_tree_attention_mask_with_root``.
* Greedy child matching picks the FIRST child in insertion order (lowest node
  index) whose token equals the target argmax — identical to the Python
  ``for child in parent.children: if child.token_id == predicted: break``.

The module is consumed behind ``BLOOMBEE_TENSOR_TREE=1`` with the Python path
kept as the reference/fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from bloombee.models.llama.spe_dec_tree import (
    SpeculativeTree,
    linearize_tree_with_positions,
)


@dataclass
class TensorTreeBatch:
    """Fixed-shape per-row tree tensors. All `[B, max_nodes]` unless noted.

    Index 0 is the root. Draft nodes are 1..n_nodes[b]-1 in DFS pre-order.
    Padding slots (>= n_nodes[b]) are token=pad, parent_idx=-1, alive=False.
    """

    token: torch.Tensor          # [B, N] long  (node 0 = root token)
    parent_idx: torch.Tensor     # [B, N] long  parent NODE index (root's = -1; node0 = -1)
    depth: torch.Tensor          # [B, N] long  (root depth 0)
    n_nodes: torch.Tensor        # [B] long     valid node count incl. root
    device: torch.device
    max_nodes: int               # N (incl. root slot)

    @property
    def batch_size(self) -> int:
        return int(self.token.shape[0])

    def draft_count(self) -> torch.Tensor:
        """Per-row number of DRAFT nodes (excludes root) = n_nodes - 1."""
        return (self.n_nodes - 1).clamp(min=0)

    def position_in_sequence(self) -> torch.Tensor:
        """DFS position of each node: root -> -1, draft node index k -> k-1.

        Padding slots keep -1. Returned shape [B, N] long."""
        ar = torch.arange(self.max_nodes, device=self.device).unsqueeze(0)  # [1, N]
        pos = ar - 1  # node 0 -> -1, node k -> k-1
        valid = ar < self.n_nodes.unsqueeze(1)
        return torch.where(valid, pos, torch.full_like(pos, -1))


def tensor_tree_from_speculative_trees(
    trees: List[SpeculativeTree],
    device: torch.device,
    pad_token_id: int = 0,
) -> TensorTreeBatch:
    """Stage-1 bridge: build a TensorTreeBatch from existing SpeculativeTrees
    using the SAME DFS pre-order as `linearize_tree_with_positions`, so the
    tensor path is byte-identical to the Python path under validation.

    Layout: node 0 = root; draft nodes 1..n in DFS order. parent_idx for a draft
    node = (root's node index 0) if its parent is the root, else the parent draft
    node's node index (= parent position_in_sequence + 1).
    """
    batch_size = len(trees)
    # Per-row linearized draft nodes (DFS pre-order) + parent positions.
    rows_tokens: List[List[int]] = []
    rows_parent_idx: List[List[int]] = []
    rows_depth: List[List[int]] = []
    for tree in trees:
        root_tok = int(tree.root.token_id)
        toks = [root_tok]          # node 0 = root
        par = [-1]                 # root has no parent
        dep = [0]
        if tree.total_nodes > 1:
            linearized_nodes, parent_indices = linearize_tree_with_positions(tree)
            # linearized_nodes[k] has position_in_sequence == k; node index == k + 1.
            for k, node in enumerate(linearized_nodes):
                toks.append(int(node.token_id))
                ppos = int(parent_indices[k])   # parent's position_in_sequence or -1 (root)
                par.append(0 if ppos < 0 else ppos + 1)  # -> node index
                dep.append(int(node.depth))
        rows_tokens.append(toks)
        rows_parent_idx.append(par)
        rows_depth.append(dep)

    max_nodes = max((len(t) for t in rows_tokens), default=1)
    max_nodes = max(max_nodes, 1)
    token = torch.full((batch_size, max_nodes), pad_token_id, dtype=torch.long, device=device)
    parent_idx = torch.full((batch_size, max_nodes), -1, dtype=torch.long, device=device)
    depth = torch.zeros((batch_size, max_nodes), dtype=torch.long, device=device)
    n_nodes = torch.ones(batch_size, dtype=torch.long, device=device)
    for b in range(batch_size):
        n = len(rows_tokens[b])
        token[b, :n] = torch.tensor(rows_tokens[b], dtype=torch.long, device=device)
        parent_idx[b, :n] = torch.tensor(rows_parent_idx[b], dtype=torch.long, device=device)
        depth[b, :n] = torch.tensor(rows_depth[b], dtype=torch.long, device=device)
        n_nodes[b] = n
    return TensorTreeBatch(token=token, parent_idx=parent_idx, depth=depth,
                           n_nodes=n_nodes, device=device, max_nodes=max_nodes)


@torch.no_grad()
def greedy_verify_tensorized(
    *,
    tt: TensorTreeBatch,
    hidden_states: torch.Tensor,     # [B, S, H]
    seq_lengths: torch.Tensor,       # [B] long
    tree_len: int,                   # tree_tokens.shape[1] (max draft nodes across batch)
    is_first_iteration: bool,
    project_rows,                    # callable: [N, H] -> [N, vocab] logits (lm_head)
    logits_processor=None,           # LogitsProcessorList (applied to final bonus token only)
    input_ids: Optional[torch.Tensor] = None,  # [B, *] for logits_processor
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPU greedy tree verification. Token-identical to
    ``_extract_greedy_verified_paths_from_hidden`` but with no per-depth host sync.

    Returns (verified_tokens[B,Lmax] or None, kv_cache_position_ids[B,Lmax+1],
    llm_generated_tokens[B,1], valid_lengths[B], final_positions[B])."""
    device_h = hidden_states.device
    out_device = seq_lengths.device if input_ids is None else input_ids.device
    B = int(hidden_states.shape[0])
    S = int(hidden_states.shape[1])
    N = tt.max_nodes
    token = tt.token
    parent_idx = tt.parent_idx
    n_nodes = tt.n_nodes
    node_ar = torch.arange(N, device=device_h).unsqueeze(0)          # [1, N]
    alive = node_ar < n_nodes.to(device_h).unsqueeze(1)              # [B, N] valid node slots
    # tree_root_positions[b] = seq_lengths[b]-1 (absolute position of the root/last committed tok)
    seq_h = seq_lengths.to(device_h).long()
    tree_root_positions = seq_h - 1

    active_node = torch.zeros(B, dtype=torch.long, device=device_h)  # current parent node index (0 = root)
    # A row is active iff its current parent (active_node) has at least one live child.
    def children_exist(node_b):  # node_b: [B] -> [B] bool
        return ((parent_idx == node_b.unsqueeze(1)) & alive).any(dim=1)
    active = children_exist(active_node) & (n_nodes.to(device_h) > 1)

    max_steps = max(int(tree_len), 0) + 1
    accepted_tokens_steps: List[torch.Tensor] = []   # each [B]
    accepted_nodes_steps: List[torch.Tensor] = []     # each [B] node idx (or -1)
    accept_count = torch.zeros(B, dtype=torch.long, device=device_h)

    for _step in range(max_steps):
        if not bool(active.any()):
            break
        # Parent logits position per row (mirror _tree_parent_logits_position).
        is_root = active_node == 0
        pos_first = torch.where(is_root, seq_h - 1, seq_h + (active_node - 1))
        pos_rest = torch.where(is_root, torch.zeros_like(active_node), active_node)  # non-root: (k-1)+1 = k
        pos = pos_first if is_first_iteration else pos_rest
        in_window = (pos >= 0) & (pos < S)
        cur = active & in_window
        # Rows that are active but out-of-window go inactive (Python warns+drops).
        active = active & in_window
        if not bool(cur.any()):
            break
        pos_clamped = pos.clamp(0, max(S - 1, 0))
        bidx = torch.arange(B, device=device_h)
        parent_hidden = hidden_states[bidx, pos_clamped, :]            # [B, H]
        logits = project_rows(parent_hidden)                          # [B, vocab]
        predicted = logits.argmax(dim=-1)                             # [B]
        # Match: children of active_node whose token == predicted; pick lowest node index.
        is_child = (parent_idx == active_node.unsqueeze(1)) & alive   # [B, N]
        match = is_child & (token == predicted.unsqueeze(1))          # [B, N]
        match = match & cur.unsqueeze(1)
        # lowest matching node index, else -1
        big = N
        idx_or_big = torch.where(match, node_ar.expand(B, N), torch.full((B, N), big, device=device_h, dtype=torch.long))
        matched_idx = idx_or_big.min(dim=1).values                   # [B], == big if none
        matched = matched_idx < big
        matched_idx = torch.where(matched, matched_idx, torch.full_like(matched_idx, -1))
        # Record accepted token/node for matched rows; -1 elsewhere this step.
        step_tok = torch.where(matched, token[bidx, matched_idx.clamp(min=0)], torch.full((B,), -1, device=device_h, dtype=torch.long))
        step_tok = torch.where(matched, step_tok, torch.full_like(step_tok, -1))
        accepted_tokens_steps.append(step_tok)
        accepted_nodes_steps.append(torch.where(matched, matched_idx, torch.full_like(matched_idx, -1)))
        accept_count = accept_count + matched.long()
        # Advance: matched rows move to matched_idx; others go inactive.
        active_node = torch.where(matched, matched_idx.clamp(min=0), active_node)
        active = matched & children_exist(active_node)

    # Assemble per-row accepted token / position lists (now one host sync).
    if accepted_tokens_steps:
        toks_mat = torch.stack(accepted_tokens_steps, dim=1)   # [B, steps]
        nodes_mat = torch.stack(accepted_nodes_steps, dim=1)   # [B, steps]
    else:
        toks_mat = torch.empty(B, 0, dtype=torch.long, device=device_h)
        nodes_mat = torch.empty(B, 0, dtype=torch.long, device=device_h)

    valid_lengths = accept_count.to(out_device)
    Lmax = int(accept_count.max().item()) if accept_count.numel() else 0

    # Compact accepted tokens (drop -1 padding per row, preserving order).
    verified_tokens: Optional[torch.Tensor] = None
    kv_root = tree_root_positions.to(out_device)
    if Lmax > 0:
        verified_tokens = torch.full((B, Lmax), -1, dtype=torch.long, device=out_device)
    # absolute kv positions: root_pos[b] + matched_node_idx (since pos_in_seq = idx-1, +1 -> idx)
    kv_positions_rows: List[torch.Tensor] = []
    toks_cpu = toks_mat.tolist()
    nodes_cpu = nodes_mat.tolist()
    rootpos_cpu = tree_root_positions.tolist()
    for b in range(B):
        row_tok = [t for t in toks_cpu[b] if t >= 0]
        row_nodes = [nd for nd in nodes_cpu[b] if nd >= 0]
        if verified_tokens is not None and row_tok:
            verified_tokens[b, :len(row_tok)] = torch.tensor(row_tok, dtype=torch.long, device=out_device)
        abs_positions = [int(rootpos_cpu[b])] + [int(rootpos_cpu[b]) + int(nd) for nd in row_nodes]
        kv_positions_rows.append(torch.tensor(abs_positions, dtype=torch.long, device=out_device))

    max_pos_len = max((p.shape[0] for p in kv_positions_rows), default=1)
    kv_cache_position_ids = torch.full((B, max_pos_len), -1, dtype=torch.long, device=out_device)
    for b, p in enumerate(kv_positions_rows):
        kv_cache_position_ids[b, :p.shape[0]] = p

    # Final / bonus-token position (mirror Python lines 1024-1046).
    fallback_pos = max(0, S - int(tree_len))
    final_pos_index: List[int] = []
    accept_cpu = accept_count.tolist()
    seq_cpu = seq_h.tolist()
    for b in range(B):
        if accept_cpu[b] > 0:
            # last accepted absolute position, mapped back into the hidden window
            last_node = row_last_node(nodes_cpu[b])
            abs_last = int(rootpos_cpu[b]) + int(last_node)
            pos = abs_last - int(rootpos_cpu[b])   # = last_node (relative)
            if is_first_iteration:
                pos = abs_last
        else:
            real_fallback = int(seq_cpu[b]) if is_first_iteration else fallback_pos
            pos = real_fallback - 1
        pos = min(max(int(pos), 0), max(S - 1, 0))
        final_pos_index.append(pos)

    fpi = torch.tensor(final_pos_index, dtype=torch.long, device=device_h)
    bidx = torch.arange(B, device=device_h)
    final_hidden = hidden_states[bidx, fpi, :]
    final_logits = project_rows(final_hidden)
    if logits_processor and len(logits_processor) > 0 and input_ids is not None:
        rows = []
        for b in range(B):
            processed = final_logits[b:b + 1].clone()
            for proc in logits_processor:
                processed = proc(input_ids[b:b + 1], processed)
            rows.append(torch.argmax(processed[0], dim=-1, keepdim=True).to(out_device))
        llm_generated_tokens = torch.stack(rows, dim=0)
    else:
        llm_generated_tokens = final_logits.argmax(dim=-1, keepdim=True).to(out_device)

    final_positions = fpi.to(out_device)
    return verified_tokens, kv_cache_position_ids, llm_generated_tokens, valid_lengths, final_positions


def row_last_node(node_steps: List[int]) -> int:
    """Last accepted node index in a per-row step list (ignores -1)."""
    last = 0
    for nd in node_steps:
        if nd >= 0:
            last = nd
    return last


@torch.no_grad()
def tensor_tree_from_eagle_candidate_tensors(
    *,
    root_tokens: torch.Tensor,        # [B] long
    cand_token: torch.Tensor,         # [B, C] long
    cand_parent_cidx: torch.Tensor,   # [B, C] long, parent candidate slot (-1 if parent is root)
    cand_depth: torch.Tensor,         # [B, C] long (>=1 for draft candidates)
    cand_path_logp64: torch.Tensor,   # [B, C] float64 cumulative path log-prob
    cand_valid: torch.Tensor,         # [B, C] bool
    total_token: int,                 # tree budget incl. root (m = total_token - 1 kept draft nodes)
    max_candidate_depth: int,
    pad_token_id: int = 0,
) -> TensorTreeBatch:
    """Reconstruct a TensorTreeBatch directly from per-row EAGLE candidate tensors,
    reproducing the Python `_topm_global` + `_close_under_parents` + `_bind` +
    DFS-preorder pipeline EXACTLY (with creation_index == candidate slot index).

    Determinism contract (must match eagle_drafter.py):
      * top-m kept set: largest m valid candidates by (-path_logp64, depth, slot),
        stable on ties (slot ascending) — matches _topm_global stable sort.
      * parent closure: add ancestors (via cand_parent_cidx) — matches _close_under_parents.
      * final DFS pre-order: children visited in (depth, slot) order — matches _bind
        (sorts kept by (depth, creation_index)) + linearize_tree_with_positions.
    """
    device = cand_token.device
    B, C = cand_token.shape
    m = max(0, int(total_token) - 1)
    NEG_INF = torch.tensor(float("-inf"), dtype=torch.float64, device=device)

    # --- 1. top-m selection by (-path_logp64, depth, slot), stable, valid only ---
    slot = torch.arange(C, device=device).unsqueeze(0).expand(B, C)  # [B,C] creation_index
    # Build a single sortable key per candidate that lexicographically orders by
    # (path_logp64 DESC, depth ASC, slot ASC). We sort by composite via successive
    # stable sorts (least-significant key first): slot (already ascending), then
    # depth ascending, then path_logp64 descending. Invalid -> sorted to the end.
    # Use rank assignment rather than float packing to avoid precision loss.
    # Order candidates: primary path_logp64 desc, then depth asc, then slot asc.
    # Implement with a stable argsort chain.
    order = torch.arange(C, device=device).unsqueeze(0).expand(B, C).clone()  # identity (slot asc)
    # stable sort by depth ascending (slot asc already the tie-order)
    dep_keys = torch.gather(cand_depth, 1, order)
    idx = torch.argsort(dep_keys, dim=1, stable=True)
    order = torch.gather(order, 1, idx)
    # stable sort by path_logp64 descending
    lp_keys = torch.gather(cand_path_logp64, 1, order)
    idx = torch.argsort(-lp_keys, dim=1, stable=True)
    order = torch.gather(order, 1, idx)
    # Now `order[b]` lists candidate slots best-first by (-logp, depth, slot)...
    # but invalid candidates must be pushed last. Re-rank with validity as the
    # most-significant key (valid first), stable over the (-logp,depth,slot) order.
    valid_keys = torch.gather(cand_valid.long(), 1, order)  # 1 valid, 0 invalid
    idx = torch.argsort(-valid_keys, dim=1, stable=True)
    order = torch.gather(order, 1, idx)
    # selected = first m slots in `order` that are valid
    rank = torch.arange(C, device=device).unsqueeze(0).expand(B, C)
    n_valid = cand_valid.sum(dim=1, keepdim=True)  # [B,1]
    take = torch.minimum(torch.full_like(n_valid, m), n_valid)  # [B,1]
    sel_in_order = rank < take  # [B,C] over the ordered positions
    selected = torch.zeros(B, C, dtype=torch.bool, device=device)
    selected.scatter_(1, order, sel_in_order)
    selected = selected & cand_valid

    # --- 2. parent closure: add ancestors of selected (depth-bounded) ---
    closed = selected.clone()
    for _ in range(int(max_candidate_depth) + 1):
        # parent slot of each closed candidate (>=0 means parent is a candidate)
        par = cand_parent_cidx.clone()
        has_par = (par >= 0) & closed
        if not bool(has_par.any()):
            break
        par_safe = par.clamp(min=0)
        add = torch.zeros(B, C, dtype=torch.bool, device=device)
        add.scatter_(1, par_safe, has_par)  # mark parents of closed nodes
        new_closed = closed | (add & cand_valid)
        if bool((new_closed == closed).all()):
            break
        closed = new_closed

    # --- 3. final DFS pre-order via path-key lexsort ---
    # path_key[b,c] = [slot at depth1 ancestor, slot at depth2 ancestor, ..., own slot]
    # padded with -1 suffix so a parent sorts before its descendants.
    D = int(max_candidate_depth)
    path_key = torch.full((B, C, D), -1, dtype=torch.long, device=device)
    # walk ancestors: level 0 = own slot at position depth-1; fill from the node up.
    cur = slot.clone()                       # [B,C] current ancestor slot (start: self)
    cur_depth = cand_depth.clamp(min=0)      # [B,C]
    # For each node, its own slot goes at column (depth-1); ancestors fill earlier cols.
    for _level in range(D):
        col = (cur_depth - 1).clamp(min=0, max=D - 1)  # [B,C]
        valid_cur = cur >= 0
        # scatter cur into path_key[:, :, col] per node
        # build via advanced indexing
        bidx = torch.arange(B, device=device).view(B, 1).expand(B, C)
        nidx = torch.arange(C, device=device).view(1, C).expand(B, C)
        pk = path_key.clone()
        pk[bidx[valid_cur], nidx[valid_cur], col[valid_cur]] = cur[valid_cur]
        path_key = pk
        # step up to parent
        par = torch.gather(cand_parent_cidx, 1, slot)  # parent slot of each node's chain head
        # advance cur to its parent
        parent_of_cur = torch.where(cur >= 0, torch.gather(cand_parent_cidx, 1, cur.clamp(min=0)), torch.full_like(cur, -1))
        parent_of_cur = torch.where(cur >= 0, parent_of_cur, torch.full_like(cur, -1))
        cur = parent_of_cur
        cur_depth = (cur_depth - 1).clamp(min=0)
    # closed candidates sort by path_key ascending (lexicographic over columns);
    # unclosed -> large sentinel so they fall to the end.
    BIG = C + 1
    sortable = path_key.clone()
    # replace -1 with -1 (parent-first) is already correct; mask unclosed rows to BIG.
    unclosed = ~closed
    sortable[unclosed] = BIG
    # lexsort over D columns: stable sorts from last column to first.
    order2 = torch.arange(C, device=device).unsqueeze(0).expand(B, C).clone()
    for col in range(D - 1, -1, -1):
        keys = torch.gather(sortable[:, :, col], 1, order2)
        idx = torch.argsort(keys, dim=1, stable=True)
        order2 = torch.gather(order2, 1, idx)
    # order2[b] now lists closed candidates in DFS pre-order, then unclosed.
    draft_count = closed.sum(dim=1)  # [B]
    max_draft = int(draft_count.max().item()) if B > 0 else 0
    max_nodes = max_draft + 1

    token = torch.full((B, max_nodes), pad_token_id, dtype=torch.long, device=device)
    parent_idx = torch.full((B, max_nodes), -1, dtype=torch.long, device=device)
    depth = torch.zeros((B, max_nodes), dtype=torch.long, device=device)
    n_nodes = torch.ones(B, dtype=torch.long, device=device)
    token[:, 0] = root_tokens.to(device)

    # cand slot -> node index map (node 0 = root). Build per row.
    cand_to_node = torch.full((B, C), -1, dtype=torch.long, device=device)
    for b in range(B):
        dc = int(draft_count[b].item())
        if dc == 0:
            n_nodes[b] = 1
            continue
        ordered = order2[b, :dc]  # candidate slots in DFS order
        node_indices = torch.arange(1, dc + 1, device=device)
        cand_to_node[b, ordered] = node_indices
        token[b, 1:dc + 1] = cand_token[b, ordered]
        depth[b, 1:dc + 1] = cand_depth[b, ordered]
        # parent node index: 0 if parent is root, else cand_to_node[parent slot]
        par_slots = cand_parent_cidx[b, ordered]  # [-1 means root]
        par_nodes = torch.where(par_slots < 0, torch.zeros_like(par_slots), cand_to_node[b, par_slots.clamp(min=0)])
        parent_idx[b, 1:dc + 1] = par_nodes
        n_nodes[b] = dc + 1

    return TensorTreeBatch(token=token, parent_idx=parent_idx, depth=depth,
                           n_nodes=n_nodes, device=device, max_nodes=max_nodes)


def parent_pos_list_per_row(tt: TensorTreeBatch) -> List[List[int]]:
    """Return, per row, the `parent_indices` list (DFS-position space, root=-1)
    for the DRAFT nodes only — the exact input `build_tree_attention_mask_with_root`
    and the prefill/generation mask code expect. Used by the tensor prepare path."""
    out: List[List[int]] = []
    n_nodes = tt.n_nodes.tolist()
    parent_idx = tt.parent_idx.tolist()
    for b in range(tt.batch_size):
        n = int(n_nodes[b])
        row: List[int] = []
        for k in range(1, n):  # draft nodes (skip root at index 0)
            pidx = int(parent_idx[b][k])     # parent NODE index (0 == root)
            row.append(-1 if pidx == 0 else pidx - 1)  # -> parent position_in_sequence
        out.append(row)
    return out
