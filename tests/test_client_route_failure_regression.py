"""CPU control-plane regression for peer failure and route refresh.

Deterministic DHT snapshots exercise production routing classes. This does not
exercise a real DHT, worker, model load, or RPC connection.
"""

import threading
from types import SimpleNamespace

import pytest

from bloombee.client.routing.sequence_info import RemoteSequenceInfo
from bloombee.client.routing.sequence_manager import MissingBlocksError, RemoteSequenceManager
from bloombee.data_structures import RemoteModuleInfo, ServerInfo, ServerState


class BannedPeers:
    def __init__(self):
        self.failed = set()

    def __contains__(self, peer):
        return peer in self.failed

    def register_failure(self, peer):
        self.failed.add(peer)

    def register_success(self, peer):
        self.failed.discard(peer)


class Ping:
    def ping(self, peers, **kwargs):
        pass

    def to_dict(self):
        return {}


def server(start=0, end=2):
    return ServerInfo(state=ServerState.ONLINE, throughput=1.0, start_block=start, end_block=end)


def snapshot(*server_maps):
    return [RemoteModuleInfo(f"model.{i}", servers) for i, servers in enumerate(server_maps)]


def manager(block_count=2):
    result = object.__new__(RemoteSequenceManager)
    result.state = SimpleNamespace(
        sequence_info=RemoteSequenceInfo.make_empty(tuple(f"model.{i}" for i in range(block_count))),
        banned_peers=BannedPeers(),
    )
    result.config = SimpleNamespace(active_adapter=None, max_pinged=3, ping_timeout=0.01, show_route=False)
    result.allowed_servers = None
    result.blocked_servers = None
    result.lock_changes = threading.Lock()
    result.ping_aggregator = Ping()
    result._thread = SimpleNamespace(ready=threading.Event(), is_alive=lambda: True)
    result._thread.ready.set()
    result._thread_start_lock = threading.Lock()
    result.update = lambda **kwargs: None
    return result


def routes(result):
    return {
        "max_throughput": result.make_sequence(mode="max_throughput"),
        "min_latency": result.make_sequence(mode="min_latency"),
    }


def assert_only_replacement(result):
    assert all(set(info.servers) == {"replacement"} for info in result.state.sequence_info.block_infos)
    assert all(
        {span.peer_id for span in spans} == {"replacement"}
        for spans in result.state.sequence_info.spans_containing_block
    )
    for route in routes(result).values():
        assert [(span.peer_id, span.start, span.end) for span in route] == [("replacement", 0, 2)]


def test_failed_peer_is_removed_from_both_routes_without_dht_refresh():
    result = manager()
    result.state.sequence_info.update_(snapshot(
        {"failed": server(), "replacement": server()},
        {"failed": server(), "replacement": server()},
    ))
    refreshes = []
    result.update = lambda **kwargs: refreshes.append(kwargs)

    result.on_request_failure("failed")

    assert refreshes == []
    assert_only_replacement(result)


def test_last_replica_failure_reports_exact_missing_block_and_refreshes():
    result = manager()
    result.state.sequence_info.update_(snapshot(
        {"healthy": server(start=0, end=1), "failed": server(start=0, end=1)},
        {"failed": server(start=1, end=2)},
    ))
    refreshes = []
    result.update = lambda **kwargs: refreshes.append(kwargs)

    result.on_request_failure("failed")

    assert refreshes == [{"wait": False}]
    assert {span.peer_id for span in result.state.sequence_info.spans_containing_block[0]} == {"healthy"}
    assert result.state.sequence_info.spans_containing_block[1] == []
    with pytest.raises(MissingBlocksError, match="blocks 1"):
        result._make_sequence_with_max_throughput(0, 2)
    with pytest.raises(MissingBlocksError, match=r"blocks \[1\]"):
        result._make_sequence_with_min_latency(0, 2, cache_tokens_needed=None)


def test_healthy_route_and_repeated_failure_do_not_change_replacement():
    result = manager()
    result.state.sequence_info.update_(snapshot(
        {"failed": server(), "replacement": server()},
        {"failed": server(), "replacement": server()},
    ))
    assert {span.peer_id for span in result.state.sequence_info.spans_containing_block[0]} == {
        "failed", "replacement"
    }
    result.on_request_failure("failed")
    result.on_request_failure("failed")
    assert_only_replacement(result)


def test_healthy_dht_refresh_preserves_both_routes(monkeypatch):
    result = manager()
    result.dht = object()
    monkeypatch.setattr(
        "bloombee.client.routing.sequence_manager.get_remote_module_infos",
        lambda *_args, **_kwargs: snapshot(
            {"replacement": server()}, {"replacement": server()},
        ),
    )

    result._update()

    assert_only_replacement(result)
    assert result.ready.is_set()


def test_refresh_after_failure_uses_new_replacement_snapshot(monkeypatch):
    result = manager()
    result.state.sequence_info.update_(snapshot(
        {"failed": server()}, {"failed": server()},
    ))
    result.on_request_failure("failed")
    result.dht = object()
    monkeypatch.setattr(
        "bloombee.client.routing.sequence_manager.get_remote_module_infos",
        lambda *_args, **_kwargs: snapshot(
            {"failed": server(), "replacement": server()},
            {"failed": server(), "replacement": server()},
        ),
    )

    result._update()

    assert_only_replacement(result)


class PausingLock:
    """Pause a refresh just before it enters the route-state critical section."""

    def __init__(self):
        self.inner = threading.Lock()
        self.at_lock = threading.Event()
        self.resume = threading.Event()

    def __enter__(self):
        if threading.current_thread().name == "refresh":
            self.at_lock.set()
            assert self.resume.wait(5)
        self.inner.acquire()
        return self

    def __exit__(self, *_):
        self.inner.release()


def test_refresh_snapshot_read_before_failure_does_not_restore_failed_peer(monkeypatch):
    result = manager()
    result.state.sequence_info.update_(snapshot(
        {"failed": server(), "replacement": server()},
        {"failed": server(), "replacement": server()},
    ))
    result.dht = object()
    lock = PausingLock()
    result.lock_changes = lock
    monkeypatch.setattr(
        "bloombee.client.routing.sequence_manager.get_remote_module_infos",
        lambda *_args, **_kwargs: snapshot(
            {"failed": server(), "replacement": server()},
            {"failed": server(), "replacement": server()},
        ),
    )
    errors = []

    def refresh():
        try:
            result._update()
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=refresh, name="refresh")
    worker.start()
    assert lock.at_lock.wait(5)
    result.on_request_failure("failed")
    lock.resume.set()
    worker.join(5)

    assert not worker.is_alive()
    assert errors == []
    assert_only_replacement(result)


def test_max_throughput_waits_for_route_state_lock():
    result = manager()
    result.state.sequence_info.update_(snapshot(
        {"replacement": server()}, {"replacement": server()},
    ))
    started = threading.Event()
    finished = threading.Event()
    errors = []

    def route():
        started.set()
        try:
            result._make_sequence_with_max_throughput(0, 2)
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    with result.lock_changes:
        worker = threading.Thread(target=route)
        worker.start()
        assert started.wait(5)
        assert not finished.wait(0.1)
    worker.join(5)
    assert not worker.is_alive()
    assert errors == []
    assert finished.is_set()
