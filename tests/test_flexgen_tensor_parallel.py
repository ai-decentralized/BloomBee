import unittest
from types import SimpleNamespace

import torch

from bloombee.server.flexgen_tensor_parallel import FlexgenLlamaTensorParallel


def _make_tp_stub(local_heads_per_shard):
    tp = object.__new__(FlexgenLlamaTensorParallel)
    tp.output_device = torch.device("cpu")
    tp.tp_shards = [
        SimpleNamespace(layout=SimpleNamespace(local_heads=local_heads))
        for local_heads in local_heads_per_shard
    ]
    return tp


class FlexgenTensorParallelCacheMergeTest(unittest.TestCase):
    def test_merge_cache_parts_preserves_batch_major_head_order_for_keys(self):
        tp = _make_tp_stub([2, 2])

        seq_len = 3
        head_dim = 1
        # Each shard stores cache as (S, B*local_heads, D), where BH is batch-major inside the shard.
        shard0 = torch.tensor(
            [
                [[10.0], [11.0], [20.0], [21.0]],
                [[12.0], [13.0], [22.0], [23.0]],
                [[14.0], [15.0], [24.0], [25.0]],
            ]
        )
        shard1 = torch.tensor(
            [
                [[30.0], [31.0], [40.0], [41.0]],
                [[32.0], [33.0], [42.0], [43.0]],
                [[34.0], [35.0], [44.0], [45.0]],
            ]
        )

        merged = tp._merge_cache_parts([shard0, shard1], is_key=True)

        expected = torch.tensor(
            [
                [[10.0, 12.0, 14.0]],
                [[11.0, 13.0, 15.0]],
                [[30.0, 32.0, 34.0]],
                [[31.0, 33.0, 35.0]],
                [[20.0, 22.0, 24.0]],
                [[21.0, 23.0, 25.0]],
                [[40.0, 42.0, 44.0]],
                [[41.0, 43.0, 45.0]],
            ]
        )
        self.assertEqual(merged.shape, (8, head_dim, seq_len))
        self.assertTrue(torch.equal(merged, expected))

    def test_merge_cache_parts_preserves_batch_major_head_order_for_values(self):
        tp = _make_tp_stub([2, 2])

        seq_len = 2
        head_dim = 1
        shard0 = torch.tensor(
            [
                [[10.0], [11.0], [20.0], [21.0]],
                [[12.0], [13.0], [22.0], [23.0]],
            ]
        )
        shard1 = torch.tensor(
            [
                [[30.0], [31.0], [40.0], [41.0]],
                [[32.0], [33.0], [42.0], [43.0]],
            ]
        )

        merged = tp._merge_cache_parts([shard0, shard1], is_key=False)

        expected = torch.tensor(
            [
                [[10.0], [12.0]],
                [[11.0], [13.0]],
                [[30.0], [32.0]],
                [[31.0], [33.0]],
                [[20.0], [22.0]],
                [[21.0], [23.0]],
                [[40.0], [42.0]],
                [[41.0], [43.0]],
            ]
        )
        self.assertEqual(merged.shape, (8, seq_len, head_dim))
        self.assertTrue(torch.equal(merged, expected))


if __name__ == "__main__":
    unittest.main()
