"""Benchmark harness for arch-reform improvements.

Run with:
    python tests/bench_arch_reform.py                # CPU benchmarks only
    BLOOMBEE_BENCH_CUDA=1 python tests/bench_arch_reform.py   # + CUDA benchmarks

Measures wall-clock impact of the three landed optimizations vs. the
pre-refactor implementations. Each benchmark prints "before", "after",
and speedup for a representative shape.
"""

import os
import time
from statistics import mean, median
from typing import Callable, List

import torch


def _bench(fn: Callable[[], None], *, warmup: int = 3, iters: int = 20, sync_cuda: bool = False) -> float:
    """Return median wall-clock ms per call."""
    for _ in range(warmup):
        fn()
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return median(samples)


# -------------------------------------------------------------------- #
# 1) Ancestor matrix: O(n^3) matmul closure vs O(n*depth) parent walk.  #
# -------------------------------------------------------------------- #

def _ancestor_matmul(parent_indices: List[int], device: torch.device) -> torch.Tensor:
    """Pre-refactor implementation (from git history)."""
    n = len(parent_indices)
    if n == 0:
        return torch.empty(0, 0, dtype=torch.bool, device=device)
    A = torch.zeros(n, n, dtype=torch.bool, device=device)
    rows = torch.arange(n, device=device)
    cols = torch.as_tensor(parent_indices, device=device)
    mask = cols >= 0
    if mask.any():
        A[rows[mask], cols[mask]] = True
    ancestor_matrix = A.clone()
    for _ in range(n):
        A_f = A.float()
        anc_f = ancestor_matrix.float()
        power = torch.matmul(anc_f, A_f)
        new_reach = ancestor_matrix | (power > 0)
        if torch.equal(new_reach, ancestor_matrix):
            break
        ancestor_matrix = new_reach
    return ancestor_matrix


def bench_ancestor_matrix():
    from bloombee.models.llama.spe_dec_tree import build_ancestor_matrix_optimized

    print("\n[1] Ancestor matrix (spe_dec_tree) — realistic SD trees")
    print("-" * 72)
    device = torch.device("cpu")

    cases = [
        ("depth=3 width=2 (BloomBee default)", [-1, 0, 0, 1, 1, 2, 2]),
        ("depth=4 width=2 balanced",           [-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]),
        ("width=8 root-only",                  [-1] + [0] * 8),
        ("linear chain n=32",                  [-1] + list(range(31))),
        ("unbalanced n=30",                    [-1] + [i // 2 for i in range(29)]),
    ]

    for name, parents in cases:
        n = len(parents)
        before = _bench(lambda p=parents: _ancestor_matmul(p, device))
        after = _bench(lambda p=parents: build_ancestor_matrix_optimized(p, device))
        speedup = before / after if after > 0 else float("inf")
        print(f"  n={n:>3} {name:<42}  {before:>7.3f} ms → {after:>7.3f} ms  ({speedup:>5.1f}x)")


# -------------------------------------------------------------------- #
# 2) Spec verify: per-node argmax+.item() vs one batched argmax+tolist. #
# -------------------------------------------------------------------- #

def _verify_per_node(logits: torch.Tensor, node_tokens: List[List[int]]) -> int:
    """Old path: argmax + .item() per node position. Simulates the triple-nested
    loop in speculative_model._extract_best_verified_paths_fixed before the hoist."""
    matched = 0
    B = logits.shape[0]
    for b in range(B):
        for path in node_tokens:
            for pos_idx, expected in enumerate(path):
                pos = pos_idx
                if pos >= logits.shape[1]:
                    break
                predicted = torch.argmax(logits[b, pos]).item()
                if predicted == expected:
                    matched += 1
                else:
                    break
    return matched


def _verify_batched(logits: torch.Tensor, node_tokens: List[List[int]]) -> int:
    """New path: one vectorized argmax, move to host, walk in Python."""
    matched = 0
    B = logits.shape[0]
    if logits.numel() > 0 and logits.shape[1] > 0:
        predicted_cpu = logits.argmax(dim=-1).detach().cpu().tolist()
    else:
        predicted_cpu = [[] for _ in range(B)]
    for b in range(B):
        row = predicted_cpu[b]
        for path in node_tokens:
            for pos_idx, expected in enumerate(path):
                pos = pos_idx
                if pos >= len(row):
                    break
                if row[pos] == expected:
                    matched += 1
                else:
                    break
    return matched


def bench_spec_verify(device: torch.device):
    print(f"\n[2] Spec-dec verify argmax hoist — device={device}")
    print("-" * 72)

    vocab = 32000
    tree_len = 15  # depth=3 width=2 tree has ~15 nodes
    node_tokens = [
        [5, 12, 29],
        [5, 12, 40],
        [5, 88, 101],
        [5, 88, 102],
    ]

    for batch in (1, 4, 16):
        torch.manual_seed(0)
        logits = torch.randn(batch, tree_len, vocab, device=device, dtype=torch.float16)

        before = _bench(
            lambda L=logits, t=node_tokens: _verify_per_node(L, t),
            sync_cuda=(device.type == "cuda"),
        )
        after = _bench(
            lambda L=logits, t=node_tokens: _verify_batched(L, t),
            sync_cuda=(device.type == "cuda"),
        )
        speedup = before / after if after > 0 else float("inf")
        print(f"  batch={batch:>2} tree_len={tree_len}  vocab={vocab}  "
              f"{before:>7.3f} ms → {after:>7.3f} ms  ({speedup:>5.1f}x)")


# -------------------------------------------------------------------- #
# 3) FlexGen KV write: torch.cat(history, k_new) vs in-place slab.      #
# -------------------------------------------------------------------- #

def _cat_write(history: torch.Tensor, k_new: torch.Tensor) -> torch.Tensor:
    """Old path: allocate new tensor of size history+k_new and copy both."""
    return torch.cat([history, k_new], dim=0)


def _inplace_write(slab: torch.Tensor, k_new: torch.Tensor, src_s: int, tgt_s: int) -> torch.Tensor:
    """New path: write into preallocated slab, take a view."""
    slab[src_s - tgt_s : src_s] = k_new
    return slab[:src_s]


def bench_flexgen_write(device: torch.device):
    print(f"\n[3] FlexGen decode-path KV write — device={device}")
    print("-" * 72)

    # llama-7b decode step shape
    n_head = 32
    head_dim = 128
    tgt_s = 1
    dtype = torch.float16

    for batch, history_len in [(1, 63), (1, 255), (1, 511), (1, 1023), (4, 511)]:
        src_s = history_len + tgt_s
        BH = batch * n_head
        slab_capacity = max(2048, src_s * 2)

        history = torch.randn(history_len, BH, head_dim, dtype=dtype, device=device)
        k_new = torch.randn(tgt_s, BH, head_dim, dtype=dtype, device=device)
        slab = torch.zeros(slab_capacity, BH, head_dim, dtype=dtype, device=device)
        slab[:history_len] = history

        before = _bench(
            lambda h=history, k=k_new: _cat_write(h, k),
            iters=50,
            sync_cuda=(device.type == "cuda"),
        )
        after = _bench(
            lambda s=slab, k=k_new, src=src_s, tgt=tgt_s: _inplace_write(s, k, src, tgt),
            iters=50,
            sync_cuda=(device.type == "cuda"),
        )
        speedup = before / after if after > 0 else float("inf")
        history_mb = history.numel() * history.element_size() / (1024 ** 2)
        print(f"  batch={batch} history={history_len:>4} (history={history_mb:>5.1f} MB)  "
              f"cat={before:>7.3f} ms → inplace={after:>7.4f} ms  ({speedup:>6.1f}x)")


# -------------------------------------------------------------------- #

def main():
    print("=" * 72)
    print("arch-reform benchmark — measuring landed improvements")
    print("=" * 72)

    # CPU-only benches always run.
    bench_ancestor_matrix()

    use_cuda = os.environ.get("BLOOMBEE_BENCH_CUDA") == "1" and torch.cuda.is_available()
    cpu = torch.device("cpu")
    bench_spec_verify(cpu)
    bench_flexgen_write(cpu)

    if use_cuda:
        cuda = torch.device("cuda:0")
        bench_spec_verify(cuda)
        bench_flexgen_write(cuda)
    else:
        print("\n[cuda benches skipped — set BLOOMBEE_BENCH_CUDA=1 and run on a GPU box]")


if __name__ == "__main__":
    main()
