import torch

from bloombee.models.llama.speculative_model import _compact_prefix_length_from_kv_positions
from bloombee.server.block_functions import _effective_token_increment


def test_compact_prefix_length_ignores_padded_kv_slots():
    kv_cache_position_ids = torch.tensor(
        [
            [255, 256, -1, -1, -1],
            [255, 260, 262, 263, -1],
        ],
        dtype=torch.long,
    )

    assert _compact_prefix_length_from_kv_positions(kv_cache_position_ids) == 259


def test_server_token_increment_uses_current_tree_window_for_spec_decode():
    hidden_states = torch.zeros(2, 6, 16)
    kv_cache_position_ids = torch.tensor(
        [
            [255, -1, -1, -1],
            [255, 256, 260, -1],
        ],
        dtype=torch.long,
    )

    assert _effective_token_increment(hidden_states, kv_cache_position_ids, is_spec_dec=1) == 6
