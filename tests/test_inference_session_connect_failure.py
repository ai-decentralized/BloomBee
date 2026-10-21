"""Exercise pre-session connection failure through the real InferenceSession.step entry."""

import asyncio

import pytest
import torch

import bloombee.client.inference_session as inference_module
import bloombee.client.routing.sequence_manager as sequence_manager_module
from bloombee.client.routing.sequence_manager import MissingBlocksError
from test_client_route_failure_regression import manager, server, snapshot


class ConnectedSession:
    def __init__(self, span, closed):
        self.span = span
        self.closed = closed
        self.history = None
        self.next_session = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed.append(self.span.peer_id)
        return None

    def step(self, inputs, _prompts, _hypo_ids, *_args, **_kwargs):
        return inputs, None


def setup_manager(*, replacement):
    result = manager()
    servers = {"B": server()}
    if replacement:
        servers["R"] = server()
    result.state.sequence_info.update_(snapshot(dict(servers), dict(servers)))
    result.state.p2p = object()
    result.state.rpc_info = {}
    result.config.max_retries = 2
    result.ping_aggregator.to_dict = lambda: {"B": 0.001, "R": 10.0}
    result.get_request_metadata = lambda *_args, **_kwargs: {}
    assert result.make_sequence(0, 2, mode="min_latency", cache_tokens_needed=4)[0].peer_id == "B"
    return result


def install_connect_result(monkeypatch, attempts, closed=None):
    if closed is None:
        closed = []

    async def create(_cls, _config, _p2p, span, _uid, *, rpc_info, **_kwargs):
        attempts.append(span.peer_id)
        if span.peer_id == "B":
            on_connect_failure = _kwargs.get("on_connect_failure")
            if on_connect_failure is not None:
                on_connect_failure()
            raise ConnectionError("failed to dial B")
        return ConnectedSession(span, closed)

    monkeypatch.setattr(inference_module._ServerInferenceSession, "create", classmethod(create))
    monkeypatch.setattr(inference_module.RemoteExpertWorker, "run_coroutine", lambda coro: asyncio.run(coro))


def test_step_attributes_failed_connection_to_selected_peer_and_retries_replacement(monkeypatch, caplog):
    result = setup_manager(replacement=True)
    attempts = []
    install_connect_result(monkeypatch, attempts)
    session = inference_module.InferenceSession(result, max_length=4)
    inputs = torch.zeros(1, 1, 4)

    output = session.step(inputs)

    assert torch.equal(output, inputs)
    assert attempts == ["B", "R"]
    assert result.state.banned_peers.failed == {"B"}
    assert set(result.state.sequence_info.block_infos[0].servers) == {"R"}
    assert "inference via B" in caplog.text


def test_step_reports_missing_blocks_when_failed_peer_has_no_replacement(monkeypatch):
    result = manager()
    stale_snapshot = snapshot(
        {"A": server(start=0, end=1), "R": server(start=0, end=1)},
        {"B": server(start=1, end=2)},
    )
    result.state.sequence_info.update_(stale_snapshot)
    result.state.p2p = object()
    result.state.rpc_info = {}
    result.config.max_retries = 2
    result.ping_aggregator.to_dict = lambda: {"A": 0.001, "R": 10.0, "B": 0.001}
    result.get_request_metadata = lambda *_args, **_kwargs: {}
    assert [span.peer_id for span in result.make_sequence(0, 2, mode="min_latency", cache_tokens_needed=4)] == [
        "A", "B"
    ]
    result.dht = object()
    monkeypatch.setattr(
        sequence_manager_module,
        "get_remote_module_infos",
        lambda *_args, **_kwargs: snapshot(
            {"A": server(start=0, end=1), "R": server(start=0, end=1)},
            {"B": server(start=1, end=2)},
        ),
    )
    refreshes = []

    def refresh(*, wait):
        refreshes.append(wait)
        if wait:
            result._update()

    result.update = refresh
    attempts = []
    install_connect_result(monkeypatch, attempts)
    session = inference_module.InferenceSession(result, max_length=4)

    with pytest.raises(MissingBlocksError, match=r"blocks \[1\]"):
        session.step(torch.zeros(1, 1, 4))

    assert attempts == ["A", "B"]
    assert result.state.banned_peers.failed == {"B"}
    assert refreshes == [False, True, False]
    assert set(result.state.sequence_info.block_infos[0].servers) == {"A", "R"}
    assert not result.state.sequence_info.block_infos[1].servers


def test_local_metadata_failure_does_not_ban_selected_peer(monkeypatch):
    result = setup_manager(replacement=True)
    result.config.max_retries = 1
    result.get_request_metadata = lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("local metadata"))
    attempts = []
    install_connect_result(monkeypatch, attempts)
    session = inference_module.InferenceSession(result, max_length=4)

    with pytest.raises(ValueError, match="local metadata"):
        session.step(torch.zeros(1, 1, 4))

    assert attempts == []
    assert result.state.banned_peers.failed == set()


def test_local_event_loop_startup_failure_does_not_ban_selected_peer(monkeypatch):
    result = setup_manager(replacement=True)
    result.config.max_retries = 1

    def fail_before_running(coroutine):
        coroutine.close()
        raise RuntimeError("local event loop unavailable")

    monkeypatch.setattr(inference_module.RemoteExpertWorker, "run_coroutine", fail_before_running)
    session = inference_module.InferenceSession(result, max_length=4)

    with pytest.raises(RuntimeError, match="local event loop unavailable"):
        session.step(torch.zeros(1, 1, 4))

    assert result.state.banned_peers.failed == set()


def test_second_span_connection_failure_closes_first_and_retries_with_healthy_tail(monkeypatch):
    result = manager()
    result.state.sequence_info.update_(snapshot(
        {"A": server(start=0, end=1)},
        {"B": server(start=1, end=2), "R": server(start=1, end=2)},
    ))
    result.state.p2p = object()
    result.state.rpc_info = {}
    result.config.max_retries = 2
    result.ping_aggregator.to_dict = lambda: {"A": 0.001, "B": 0.001, "R": 10.0}
    result.get_request_metadata = lambda *_args, **_kwargs: {}
    assert [span.peer_id for span in result.make_sequence(0, 2, mode="min_latency", cache_tokens_needed=4)] == [
        "A", "B"
    ]
    attempts, closed = [], []
    install_connect_result(monkeypatch, attempts, closed)
    session = inference_module.InferenceSession(result, max_length=4)
    inputs = torch.zeros(1, 1, 4)

    output = session.step(inputs)

    assert torch.equal(output, inputs)
    assert attempts == ["A", "B", "A", "R"]
    assert closed == ["A"]
    assert result.state.banned_peers.failed == {"B"}
    assert set(result.state.sequence_info.block_infos[0].servers) == {"A"}
    assert set(result.state.sequence_info.block_infos[1].servers) == {"R"}
