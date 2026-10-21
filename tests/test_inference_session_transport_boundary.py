"""Use the real session creator to separate local setup from transport failures."""

import asyncio

import pytest
import torch
from hivemind.p2p.p2p_daemon_bindings.utils import ControlFailure

import bloombee.client.inference_session as inference_module
from test_inference_session_connect_failure import setup_manager


def install_actual_creator(monkeypatch, get_stub):
    monkeypatch.setattr(inference_module.TransformerConnectionHandler, "get_stub", get_stub)
    monkeypatch.setattr(inference_module.RemoteExpertWorker, "run_coroutine", lambda coro: asyncio.run(coro))
    monkeypatch.setattr(
        inference_module._ServerInferenceSession,
        "step",
        lambda _session, inputs, *_args, **_kwargs: (inputs, None),
    )


def test_local_stub_construction_error_does_not_ban_healthy_peer(monkeypatch):
    result = setup_manager(replacement=True)
    result.config.max_retries = 1
    result.config.connect_timeout = 0.1
    attempts = []

    def get_stub(_p2p, peer_id):
        attempts.append(peer_id)
        raise ValueError("local stub construction")

    install_actual_creator(monkeypatch, get_stub)

    with pytest.raises(ValueError, match="local stub construction"):
        inference_module.InferenceSession(result, max_length=4).step(torch.zeros(1, 1, 4))

    assert attempts == ["B"]
    assert result.state.banned_peers.failed == set()


def test_local_stub_rpc_validation_error_does_not_ban_healthy_peer(monkeypatch):
    result = setup_manager(replacement=True)
    result.config.max_retries = 1
    result.config.connect_timeout = 0.1

    class Stub:
        async def rpc_inference(self, _requests):
            raise ValueError("local rpc request validation")

    install_actual_creator(monkeypatch, lambda _p2p, _peer_id: Stub())

    with pytest.raises(ValueError, match="local rpc request validation"):
        inference_module.InferenceSession(result, max_length=4).step(torch.zeros(1, 1, 4))

    assert result.state.banned_peers.failed == set()


def test_control_failure_during_stream_open_marks_b_and_retries_r(monkeypatch):
    result = setup_manager(replacement=True)
    result.config.connect_timeout = 0.1
    attempts = []

    class Stub:
        def __init__(self, peer_id):
            self.peer_id = peer_id

        async def rpc_inference(self, _requests):
            attempts.append(self.peer_id)
            if self.peer_id == "B":
                raise ControlFailure("Connect failed at stream_open")
            return object()

    install_actual_creator(monkeypatch, lambda _p2p, peer_id: Stub(peer_id))
    inputs = torch.zeros(1, 1, 4)
    session = inference_module.InferenceSession(result, max_length=4)

    output = session.step(inputs)

    assert torch.equal(output, inputs)
    assert attempts == ["B", "R"]
    assert result.state.banned_peers.failed == {"B"}


def test_connect_timeout_marks_selected_request_failed_without_changing_exception(monkeypatch):
    result = setup_manager(replacement=True)
    result.config.max_retries = 1
    result.config.connect_timeout = 0.001

    class Stub:
        async def rpc_inference(self, _requests):
            await asyncio.sleep(0.05)
            return object()

    install_actual_creator(monkeypatch, lambda _p2p, _peer_id: Stub())

    with pytest.raises(asyncio.TimeoutError):
        inference_module.InferenceSession(result, max_length=4).step(torch.zeros(1, 1, 4))

    assert result.state.banned_peers.failed == {"B"}


def test_local_session_constructor_error_does_not_ban_peer(monkeypatch):
    result = setup_manager(replacement=True)
    result.config.max_retries = 1
    result.config.connect_timeout = 0.1

    class Stub:
        async def rpc_inference(self, _requests):
            return object()

    install_actual_creator(monkeypatch, lambda _p2p, _peer_id: Stub())

    def fail_constructor(instance, *_args, **_kwargs):
        instance._outputs_stream = None  # keep the pre-existing destructor out of this test
        raise ValueError("local session constructor")

    monkeypatch.setattr(inference_module._ServerInferenceSession, "__init__", fail_constructor)

    with pytest.raises(ValueError, match="local session constructor"):
        inference_module.InferenceSession(result, max_length=4).step(torch.zeros(1, 1, 4))

    assert result.state.banned_peers.failed == set()
