"""Independent CPU counterexamples for the route patch's stated boundary."""

import pytest

from bloombee.client.routing.sequence_manager import SequenceManagerState
from test_client_route_failure_regression import manager, server, snapshot


def test_stale_dht_only_failed_peer_does_not_restore_route(monkeypatch):
    parent = manager()
    parent.state.sequence_info.update_(snapshot(
        {"failed": server()}, {"failed": server()},
    ))
    parent.on_request_failure("failed")
    parent.dht = object()
    monkeypatch.setattr(
        "bloombee.client.routing.sequence_manager.get_remote_module_infos",
        lambda *_args, **_kwargs: snapshot(
            {"failed": server()}, {"failed": server()},
        ),
    )

    parent._update()

    assert all(not info.servers for info in parent.state.sequence_info.block_infos)


def test_parent_failure_invalidates_existing_child_route(monkeypatch):
    parent = manager()
    parent.state.sequence_info.update_(snapshot(
        {"failed": server(), "replacement": server()},
        {"failed": server(), "replacement": server()},
    ))
    parent.state = SequenceManagerState(
        sequence_info=parent.state.sequence_info,
        banned_peers=parent.state.banned_peers,
    )
    child = manager(block_count=1)
    child.state = parent.state[0:1]
    parent.on_request_failure("failed")

    assert "failed" not in parent.state.sequence_info.block_infos[0].servers
    monkeypatch.setattr(
        "bloombee.client.routing.sequence_manager.np.random.choice",
        lambda choices, p: next((span for span in choices if span.peer_id == "failed"), choices[0]),
    )
    assert child.make_sequence(mode="max_throughput")[0].peer_id == "replacement"
    assert {span.peer_id for span in child.state.sequence_info.spans_containing_block[0]} == {"replacement"}
