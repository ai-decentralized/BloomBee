"""Phase 2 MemoryCache → PagedKVTable shim tests.

Exercises ``_register_paged_view`` and ``paged_view`` directly, without
the mp.Pipe / runtime bootstrap that full ``use_cache`` needs. The shim
is additive: ``_allocated_tensors[handle]`` remains the authoritative
(k, v) pair the attention kernel sees; the paged table aliases the same
storage so writes and reads via either channel land in the same bytes.

We verify:

1. Registration is a no-op if the cache shape isn't an (S, BH, D) pair
   (mixed-device segmented caches, compressed caches, etc.). The shim
   must never break a running server.
2. When the shape is compatible, the registered table writes into the
   exact same storage that ``_allocated_tensors`` yields to the
   attention kernel.
3. ``paged_view`` returns None for unregistered handles; the caller
   must fall back to the legacy slab path.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hivemind")  # MemoryCache pulls hivemind via the module init

import torch

from bloombee.flexgen_utils.pytorch_backend import TorchDevice, TorchTensor
from bloombee.server.memory_cache import MemoryCache


class _StubDevice:
    """Just enough of TorchDevice for MemoryCache construction."""
    device_type = "stub"


def _make_cache() -> MemoryCache:
    return MemoryCache(
        max_size_tokens=1024,
        policy=None,
        block_config=None,
        device=_StubDevice(),
    )


def _make_slab_kv(s_total: int, bh: int, d: int, dtype=torch.float32):
    """Return a (k_tt, v_tt) pair mimicking init_cache_one_gpu_batch output.

    We use a raw torch.Tensor wrapped in a tiny shim that has ``.data`` —
    the shim uses ``getattr(..., 'data')`` so this is enough.
    """
    class _TT:
        def __init__(self, t):
            self.data = t
            self.shape = t.shape

    k = torch.zeros((s_total, bh, d), dtype=dtype)
    v = torch.zeros((s_total, bh, d), dtype=dtype)
    return _TT(k), _TT(v)


def test_register_paged_view_succeeds_on_block_multiple_slab():
    from bloombee.server.paged_kv import BLOCK_SIZE

    cache = _make_cache()
    k_tt, v_tt = _make_slab_kv(4 * BLOCK_SIZE, bh=6, d=4)
    cache._register_paged_view(handle=1, allocated_cache=(k_tt, v_tt))

    table = cache.paged_view(1)
    assert table is not None

    # Writing via the paged table must mutate the same storage that
    # _allocated_tensors will hand to the attention kernel.
    table.register_sequence(0)
    s_new = 3
    k_new = torch.arange(s_new * 6 * 4, dtype=torch.float32).reshape(s_new, 6, 4)
    v_new = k_new + 1000.0
    table.write(0, 0, k_new, v_new, bh_slice=(0, 6))

    torch.testing.assert_close(k_tt.data[:s_new], k_new)
    torch.testing.assert_close(v_tt.data[:s_new], v_new)


def test_register_paged_view_skips_non_3d_cache():
    cache = _make_cache()
    # mha (prefill) builds an (S, BH, D)-3d slab; but mixed/compressed
    # caches expose tuple data — shape=None on TorchTensor. Here we just
    # hand a scalar to simulate an incompatible layout.
    class _TT:
        data = torch.zeros(5)  # 1-d — not our slab layout
    cache._register_paged_view(handle=7, allocated_cache=(_TT(), _TT()))
    assert cache.paged_view(7) is None


def test_register_paged_view_skips_non_block_multiple():
    cache = _make_cache()
    # S not a multiple of BLOCK_SIZE: PagedKVTable would raise; shim must
    # swallow and fall back silently.
    k_tt, v_tt = _make_slab_kv(17, bh=2, d=3)
    cache._register_paged_view(handle=2, allocated_cache=(k_tt, v_tt))
    assert cache.paged_view(2) is None


def test_paged_view_returns_none_for_unknown_handle():
    cache = _make_cache()
    assert cache.paged_view(9999) is None


def test_paged_view_decouples_across_handles():
    """Each handle gets its own PagedKVTable; sequences registered under one
    handle must not leak into another's state."""
    from bloombee.server.paged_kv import BLOCK_SIZE

    cache = _make_cache()
    a_k, a_v = _make_slab_kv(2 * BLOCK_SIZE, bh=2, d=2)
    b_k, b_v = _make_slab_kv(2 * BLOCK_SIZE, bh=2, d=2)
    cache._register_paged_view(handle=10, allocated_cache=(a_k, a_v))
    cache._register_paged_view(handle=11, allocated_cache=(b_k, b_v))

    table_a = cache.paged_view(10)
    table_b = cache.paged_view(11)
    assert table_a is not None and table_b is not None
    assert table_a is not table_b

    table_a.register_sequence(0)
    table_a.write(0, 0, torch.ones(3, 2, 2), torch.ones(3, 2, 2) * 2,
                  bh_slice=(0, 2))

    # table_b untouched: no sequence registered, storage untouched.
    assert 0 not in table_b._seqs
    assert torch.all(b_k.data == 0) and torch.all(b_v.data == 0)
