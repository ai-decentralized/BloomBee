from types import SimpleNamespace

import torch

from bloombee.client.inference_session import InferenceSession


class _FakeSequenceManager:
    def __init__(self):
        self.successful_peer = None

    def __len__(self):
        return 1

    def on_request_success(self, peer_id):
        self.successful_peer = peer_id

    def on_request_failure(self, peer_id):
        raise AssertionError(f"unexpected request failure for {peer_id}")


class _FakeServerSession:
    span = SimpleNamespace(start=0, end=1, peer_id="peer")

    def __init__(self):
        self.calls = []

    def step(
        self,
        inputs,
        prompts,
        hypo_ids,
        tree_attention_mask,
        kv_cache_position_ids,
        draft_tokens,
        prefill_length,
        keep_indices,
        need_pruning,
        is_spec_dec,
        *,
        step_id,
    ):
        self.calls.append(
            {
                "need_pruning": need_pruning,
                "is_spec_dec": is_spec_dec,
                "step_id": step_id,
            }
        )
        return inputs, keep_indices


def _make_session(fake_server):
    session = InferenceSession(_FakeSequenceManager(), max_length=16)
    session._server_sessions = [fake_server]
    return session


def _step_spec(session, *, need_pruning=False):
    return session.step(
        torch.ones(1, 1, 2),
        kv_cache_position_ids=torch.tensor([[0]], dtype=torch.long),
        draft_tokens=torch.tensor([[42]], dtype=torch.long),
        is_spec_decoding=torch.tensor([1], dtype=torch.long),
        need_pruning=need_pruning,
    )


def test_spec_decode_does_not_enable_pruning_by_default():
    fake_server = _FakeServerSession()
    output = _step_spec(_make_session(fake_server))

    assert tuple(output.shape) == (1, 1, 2)
    assert fake_server.calls == [
        {
            "need_pruning": False,
            "is_spec_dec": True,
            "step_id": fake_server.calls[0]["step_id"],
        }
    ]


def test_spec_decode_can_request_pruning_explicitly():
    fake_server = _FakeServerSession()
    output = _step_spec(_make_session(fake_server), need_pruning=True)

    assert tuple(output.shape) == (1, 1, 2)
    assert fake_server.calls[0]["need_pruning"] is True
    assert fake_server.calls[0]["is_spec_dec"] is True
