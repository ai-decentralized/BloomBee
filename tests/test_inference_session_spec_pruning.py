from types import SimpleNamespace
import asyncio

import torch
from hivemind.proto import runtime_pb2

from bloombee.client.config import ClientConfig
import bloombee.client.inference_session as inference_session_module
from bloombee.client.inference_session import InferenceSession
from bloombee.client.inference_session import _ServerInferenceSession
from bloombee.utils.lossless_transport import serialize_torch_tensor


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


def test_spec_decode_position_uses_forward_window_not_padded_kv_ids():
    fake_server = _FakeServerSession()
    session = _make_session(fake_server)
    session._position = 10

    output = session.step(
        torch.ones(2, 3, 2),
        kv_cache_position_ids=torch.tensor(
            [
                [9, -1, -1, -1, -1],
                [9, 10, 12, 14, -1],
            ],
            dtype=torch.long,
        ),
        draft_tokens=torch.tensor(
            [
                [101, 102, 103],
                [201, 202, 203],
            ],
            dtype=torch.long,
        ),
        is_spec_decoding=torch.tensor([1], dtype=torch.long),
    )

    assert tuple(output.shape) == (2, 3, 2)
    assert session.position == 13


def test_spec_downstream_stage_waits_for_server_push(monkeypatch):
    expected = torch.arange(6, dtype=torch.float32).view(1, 3, 2)
    response = runtime_pb2.ExpertResponse(
        tensors=[serialize_torch_tensor(expected, runtime_pb2.CompressionType.NONE)]
    )
    calls = {"awaited_push": 0, "direct_step": 0}

    async def fake_await_pushed_step(self):
        calls["awaited_push"] += 1
        return response

    async def fake_direct_step(self, request):
        calls["direct_step"] += 1
        raise AssertionError("spec downstream stage should not send a direct client relay")

    monkeypatch.setattr(
        inference_session_module.RemoteExpertWorker,
        "run_coroutine",
        staticmethod(lambda coro: asyncio.run(coro)),
    )
    monkeypatch.setattr(_ServerInferenceSession, "_await_pushed_step", fake_await_pushed_step)
    monkeypatch.setattr(_ServerInferenceSession, "_step", fake_direct_step)

    server_session = _ServerInferenceSession(
        ClientConfig(use_server_to_server=True, push_only_downstream_decode=True),
        SimpleNamespace(start=1, end=2, peer_id="peer"),
        "block.1",
        rpc_info={},
        inputs_queue=asyncio.Queue(),
        outputs_aiter=None,
        max_length=16,
    )
    server_session.stepped = True

    output = server_session.step(
        torch.zeros(1, 3, 2),
        prompts=torch.empty(0),
        hypo_ids=torch.empty(0, dtype=torch.long),
        tree_attention_mask=torch.ones(1, 3, 3, dtype=torch.bool),
        kv_cache_position_ids=torch.tensor([[10, 11, 12]], dtype=torch.long),
        draft_tokens=torch.tensor([[101, 102]], dtype=torch.long),
        prefill_length=torch.tensor([10], dtype=torch.long),
        is_spec_dec=True,
        step_id="spec-step",
    )

    assert torch.equal(output[0], expected)
    assert calls == {"awaited_push": 1, "direct_step": 0}
