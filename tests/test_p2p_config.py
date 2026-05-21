from types import SimpleNamespace

from hivemind.p2p.p2p_daemon_bindings.control import DEFAULT_MAX_MSG_SIZE

from bloombee.utils.p2p import apply_p2p_max_msg_size, get_default_p2p_max_msg_size


def _fake_p2p(control_limit=DEFAULT_MAX_MSG_SIZE):
    control = SimpleNamespace(persistent_conn_max_msg_size=control_limit)
    client = SimpleNamespace(control=control)
    return SimpleNamespace(_client=client)


def test_apply_p2p_max_msg_size_updates_replicated_control_client():
    p2p = _fake_p2p()

    apply_p2p_max_msg_size(p2p, 64 * 1024 * 1024)

    assert p2p._client.control.persistent_conn_max_msg_size == 64 * 1024 * 1024


def test_apply_p2p_max_msg_size_never_drops_below_hivemind_default():
    p2p = _fake_p2p(control_limit=64 * 1024 * 1024)

    apply_p2p_max_msg_size(p2p, 1)

    assert p2p._client.control.persistent_conn_max_msg_size == DEFAULT_MAX_MSG_SIZE


def test_default_p2p_max_msg_size_uses_environment(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_P2P_MAX_MSG_SIZE", "123456")

    assert get_default_p2p_max_msg_size() == 123456
