"""Additional CPU checks for shared views and temporary peer bans."""

import dataclasses
import pickle
from pathlib import Path

from bloombee.client.routing.sequence_manager import MissingBlocksError, SequenceManagerState
from bloombee.client.routing.sequence_info import RemoteSequenceInfo
from test_client_route_failure_regression import manager, server, snapshot


def test_child_failure_evicts_peer_from_parent_and_sibling_views():
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
    sibling = manager(block_count=1)
    sibling.state = parent.state[1:2]

    assert child.state.sequence_info.lock_changes is parent.state.sequence_info.lock_changes
    assert sibling.state.sequence_info.lock_changes is parent.state.sequence_info.lock_changes
    child.lock_changes = child.state.sequence_info.lock_changes
    child.on_request_failure("failed")

    for view in (parent, child, sibling):
        assert all("failed" not in block.servers for block in view.state.sequence_info.block_infos)
        assert all(
            {span.peer_id for span in spans} == {"replacement"}
            for spans in view.state.sequence_info.spans_containing_block
        )


def test_unbanning_only_peer_allows_later_dht_refresh(monkeypatch):
    result = manager()
    result.state.sequence_info.update_(snapshot({"failed": server()}, {"failed": server()}))
    result.on_request_failure("failed")
    result.dht = object()
    monkeypatch.setattr(
        "bloombee.client.routing.sequence_manager.get_remote_module_infos",
        lambda *_args, **_kwargs: snapshot({"failed": server()}, {"failed": server()}),
    )

    result._update()
    assert all(not block.servers for block in result.state.sequence_info.block_infos)
    assert all(not spans for spans in result.state.sequence_info.spans_containing_block)
    try:
        result.make_sequence(mode="max_throughput")
    except MissingBlocksError:
        pass
    else:
        raise AssertionError("A banned sole peer must not be routable")

    result.state.banned_peers.register_success("failed")
    result._update()
    assert [span.peer_id for span in result.make_sequence(mode="max_throughput")] == ["failed"]


def test_sequence_info_pickle_restores_shared_view_tracking():
    parent = RemoteSequenceInfo.make_empty(("model.0", "model.1"))
    parent.update_(snapshot(
        {"failed": server(), "replacement": server()},
        {"failed": server(), "replacement": server()},
    ))
    child = parent[0:1]

    restored_parent, restored_child = pickle.loads(pickle.dumps((parent, child)))

    assert restored_child.lock_changes is restored_parent.lock_changes
    assert restored_child.block_infos[0] is restored_parent.block_infos[0]
    with restored_parent.lock_changes:
        restored_parent.remove_peer_("failed")
    assert {span.peer_id for span in restored_child.spans_containing_block[0]} == {"replacement"}
    assert set(dataclasses.asdict(restored_parent)) == {
        "block_uids", "block_infos", "spans_by_priority", "spans_containing_block", "last_updated_time"
    }


def test_sequence_info_accepts_legacy_pickle_state():
    original = RemoteSequenceInfo.make_empty(("model.0",))
    legacy_state = {
        key: value for key, value in original.__dict__.items() if key != "_shared_views"
    }
    restored = object.__new__(RemoteSequenceInfo)

    restored.__setstate__(legacy_state)

    assert restored.block_uids == ("model.0",)
    with restored.lock_changes:
        restored.rebuild_spans_()
    assert restored.spans_containing_block == ([],)


def test_legacy_parent_child_pickle_rejoins_shared_views():
    fixture = Path(__file__).with_name("legacy_parent_child.pkl")
    parent, child = pickle.loads(fixture.read_bytes())

    assert parent.block_infos[0] is child.block_infos[0]
    assert parent.lock_changes is child.lock_changes
    with parent.lock_changes:
        parent.remove_peer_("failed")
    assert {span.peer_id for span in parent.spans_containing_block[0]} == {"replacement"}
    assert {span.peer_id for span in child.spans_containing_block[0]} == {"replacement"}


def test_separate_legacy_pickle_loads_do_not_share_a_lock():
    fixture = Path(__file__).with_name("legacy_parent_child.pkl").read_bytes()
    first_parent, first_child = pickle.loads(fixture)
    second_parent, second_child = pickle.loads(fixture)

    assert first_parent.lock_changes is first_child.lock_changes
    assert second_parent.lock_changes is second_child.lock_changes
    assert first_parent.lock_changes is not second_parent.lock_changes


def test_legacy_disjoint_children_rejoin_when_parent_loaded_last():
    source_parent = RemoteSequenceInfo.make_empty(("model.0", "model.1"))
    source_parent.update_(snapshot(
        {"failed": server(), "replacement": server()},
        {"failed": server(), "replacement": server()},
    ))
    source_children = (source_parent[0:1], source_parent[1:2])

    def restore_legacy(source):
        legacy_state = {
            key: value for key, value in source.__dict__.items() if key != "_shared_views"
        }
        restored = object.__new__(RemoteSequenceInfo)
        restored.__setstate__(legacy_state)
        return restored

    first_child, second_child = (restore_legacy(source) for source in source_children)
    assert first_child.lock_changes is not second_child.lock_changes
    parent = restore_legacy(source_parent)

    assert first_child.lock_changes is second_child.lock_changes is parent.lock_changes
    with parent.lock_changes:
        parent.remove_peer_("failed")
    assert all(
        {span.peer_id for span in child.spans_containing_block[0]} == {"replacement"}
        for child in (first_child, second_child)
    )
