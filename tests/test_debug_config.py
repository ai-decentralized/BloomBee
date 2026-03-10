from bloombee.utils.debug_config import (
    get_env_bool_with_debug_fallback,
    is_debug_group_enabled,
    is_global_debug_enabled,
    is_log_channel_enabled,
)


def test_global_debug_enables_all_groups(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_DEBUG", "1")
    assert is_global_debug_enabled() is True
    assert is_debug_group_enabled("compression") is True
    assert is_debug_group_enabled("kv_cache") is True
    assert is_debug_group_enabled("microbatch") is True
    assert is_debug_group_enabled("inference") is True


def test_group_override_can_disable_under_global_debug(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_DEBUG", "1")
    monkeypatch.setenv("BLOOMBEE_DEBUG_COMPRESSION", "0")
    assert is_debug_group_enabled("compression") is False
    assert is_debug_group_enabled("kv_cache") is True


def test_explicit_feature_env_beats_group_toggle(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_DEBUG_COMPRESSION", "1")
    monkeypatch.setenv("BLOOMBEE_COMP_DETAIL_PROFILE", "0")
    assert (
        get_env_bool_with_debug_fallback(
            "BLOOMBEE_COMP_DETAIL_PROFILE",
            default=False,
            groups=("compression",),
        )
        is False
    )


def test_group_toggle_enables_feature_default(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_DEBUG_KV_CACHE", "1")
    monkeypatch.delenv("BLOOMBEE_VERBOSE_KV_LOGS", raising=False)
    assert (
        get_env_bool_with_debug_fallback(
            "BLOOMBEE_VERBOSE_KV_LOGS",
            default=False,
            groups=("kv_cache",),
        )
        is True
    )


def test_log_channel_defaults_to_group(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_DEBUG_MICROBATCH", "1")
    monkeypatch.delenv("BLOOMBEE_MBPIPE_LOGS", raising=False)
    assert is_log_channel_enabled("microbatch_logs") is True


def test_log_channel_explicit_env_overrides_group(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_DEBUG_MICROBATCH", "1")
    monkeypatch.setenv("BLOOMBEE_MBPIPE_LOGS", "0")
    assert is_log_channel_enabled("microbatch_logs") is False


def test_client_inference_channel_defaults_to_inference_group(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_DEBUG_INFERENCE", "1")
    monkeypatch.delenv("BLOOMBEE_CLIENT_INFERENCE_LOGS", raising=False)
    assert is_log_channel_enabled("client_inference_logs") is True


def test_kv_source_probe_channel_defaults_to_kv_group(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_DEBUG_KV_CACHE", "1")
    monkeypatch.delenv("BLOOMBEE_KV_SOURCE_PROBE_LOGS", raising=False)
    assert is_log_channel_enabled("kv_source_probe_logs") is True
