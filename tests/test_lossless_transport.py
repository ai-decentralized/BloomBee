import zlib

import numpy as np
import pytest
import torch
from hivemind.proto import runtime_pb2

from bloombee.utils import lossless_transport as lt


def _clear_transport_caches() -> None:
    lt._debug_tensor_names.cache_clear()
    lt._wire_truncate_targets.cache_clear()
    lt._wire_truncate_phases.cache_clear()
    lt._lossless_layout_targets.cache_clear()
    lt._get_zipnn_compressor.cache_clear()
    lt._get_zipnn_decompressor.cache_clear()
    lt._zipnn_lossless_dtype_supported.cache_clear()
    lt._get_zstd_dict.cache_clear()
    lt._get_zstd_dict_compressor_cached.cache_clear()
    lt._get_zstd_dict_decompressor_cached.cache_clear()


def _make_split_friendly_fp16(shape=(64, 1, 1024)) -> torch.Tensor:
    numel = int(np.prod(shape))
    rng = np.random.default_rng(1234)
    hi = rng.choice(np.array([0x3C, 0xBC], dtype=np.uint16), size=numel)
    lo = rng.integers(0, 256, size=numel, dtype=np.uint16)
    words = ((hi << 8) | lo).astype(np.uint16, copy=False)
    array = words.view(np.float16).reshape(shape)
    return torch.from_numpy(np.ascontiguousarray(array))


def test_high_byte_lane_helpers_roundtrip():
    raw_fp16 = bytes(range(64))
    extracted_fp16, remaining_fp16 = lt._split_high_byte_lane(raw_fp16, 2)
    reconstructed_fp16 = lt._reconstruct_high_byte_lane(extracted_fp16, remaining_fp16, 2, len(raw_fp16))
    assert reconstructed_fp16 == raw_fp16

    raw_fp32 = bytes(range(96))
    extracted_fp32, remaining_fp32 = lt._split_high_byte_lane(raw_fp32, 4)
    reconstructed_fp32 = lt._reconstruct_high_byte_lane(extracted_fp32, remaining_fp32, 4, len(raw_fp32))
    assert reconstructed_fp32 == raw_fp32


def test_serialize_torch_tensor_byte_split_roundtrip(monkeypatch):
    tensor = _make_split_friendly_fp16().contiguous()
    debug_context = {
        "phase": "prefill",
        "tensor_name": "hidden_states",
        "source": "client",
        "channel": "rpc_inference",
    }

    monkeypatch.setenv("BLOOMBEE_LOSSLESS_WRAPPER", "1")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_ALGO", "zstd")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_MIN_BYTES", "0")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_MIN_GAIN_BYTES", "0")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_LAYOUT_TARGETS", "*:*:hidden_states")

    monkeypatch.setenv("BLOOMBEE_LOSSLESS_LAYOUT", "plain")
    _clear_transport_caches()
    plain_serialized = lt.serialize_torch_tensor(
        tensor,
        runtime_pb2.CompressionType.NONE,
        debug_context=debug_context,
    )
    plain_parsed = lt._parse_wrapper(plain_serialized.buffer)
    assert plain_parsed is not None
    assert plain_parsed[0] == lt._ALGO_ZSTD

    monkeypatch.setenv("BLOOMBEE_LOSSLESS_LAYOUT", "byte_split")
    _clear_transport_caches()
    split_serialized = lt.serialize_torch_tensor(
        tensor,
        runtime_pb2.CompressionType.NONE,
        debug_context=debug_context,
    )
    split_parsed = lt._parse_wrapper(split_serialized.buffer)
    assert split_parsed is not None
    assert split_parsed[0] == lt._ALGO_ZSTD_BYTE_SPLIT
    assert len(split_serialized.buffer) < len(plain_serialized.buffer)

    restored = lt.deserialize_torch_tensor(split_serialized)
    assert torch.equal(restored, tensor)


def test_byte_split_single_path_skips_plain_candidate(monkeypatch):
    tensor = _make_split_friendly_fp16().contiguous()
    debug_context = {
        "phase": "decode",
        "tensor_name": "hidden_states",
        "source": "client",
        "channel": "rpc_inference",
    }

    monkeypatch.setenv("BLOOMBEE_LOSSLESS_WRAPPER", "1")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_ALGO", "zstd")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_LAYOUT", "byte_split")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_SINGLE_PATH", "1")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_MIN_BYTES", "0")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_MIN_GAIN_BYTES", "0")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_LAYOUT_TARGETS", "*:*:hidden_states")
    _clear_transport_caches()

    def fail_plain(_raw):
        raise AssertionError("plain zstd candidate should not run in single-path byte_split mode")

    monkeypatch.setattr(lt, "_build_plain_wrapper", fail_plain)

    serialized = lt.serialize_torch_tensor(
        tensor,
        runtime_pb2.CompressionType.NONE,
        debug_context=debug_context,
    )
    parsed = lt._parse_wrapper(serialized.buffer)
    assert parsed is not None
    assert parsed[0] == lt._ALGO_ZSTD_BYTE_SPLIT
    assert torch.equal(lt.deserialize_torch_tensor(serialized), tensor)


def test_zlib_wrapper_decompression_is_capped_to_declared_size():
    raw = b"x" * 128
    payload = zlib.compress(raw)

    assert lt._decompress_with_algo(lt._ALGO_ZLIB, payload, len(raw)) == raw
    with pytest.raises(ValueError, match="exceeds declared size"):
        lt._decompress_with_algo(lt._ALGO_ZLIB, payload, 8)


def test_zipnn_compare_candidate_fp16(monkeypatch):
    tensor = _make_split_friendly_fp16().contiguous()
    debug_context = {
        "phase": "prefill",
        "tensor_name": "hidden_states",
        "source": "client",
        "channel": "rpc_inference",
    }

    monkeypatch.setenv("BLOOMBEE_COMP_ZIPNN_PROFILE", "1")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_LAYOUT_TARGETS", "*:*:hidden_states")
    _clear_transport_caches()

    raw = memoryview(tensor.numpy()).cast("B").tobytes()
    info = lt._profile_zipnn_candidate(
        tensor=tensor,
        compression_type=runtime_pb2.CompressionType.NONE,
        raw_buffer=raw,
        selected_wire_bytes=len(raw),
        debug_context=debug_context,
    )

    if lt._ZipNN is None:
        assert info["attempted"] == 0
        assert info["available"] == 0
        return

    assert info["supported"] == 1
    assert info["lossless_verified"] == 1
    assert info["attempted"] == 1
    assert int(info["wrapped_bytes"]) > 0
    assert float(info["elapsed_ms"]) >= 0.0


def test_serialize_torch_tensor_zipnn_roundtrip(monkeypatch):
    if lt._ZipNN is None:
        return

    tensor = _make_split_friendly_fp16().contiguous()
    debug_context = {
        "phase": "prefill",
        "tensor_name": "hidden_states",
        "source": "client",
        "channel": "rpc_inference",
    }

    monkeypatch.setenv("BLOOMBEE_LOSSLESS_WRAPPER", "1")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_ALGO", "zipnn")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_LAYOUT", "plain")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_MIN_BYTES", "0")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_MIN_GAIN_BYTES", "0")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_LAYOUT_TARGETS", "*:*:hidden_states")
    _clear_transport_caches()

    serialized = lt.serialize_torch_tensor(
        tensor,
        runtime_pb2.CompressionType.NONE,
        debug_context=debug_context,
    )
    parsed = lt._parse_wrapper(serialized.buffer)
    assert parsed is not None
    assert parsed[0] == lt._ALGO_ZIPNN

    restored = lt.deserialize_torch_tensor(serialized)
    assert torch.equal(restored, tensor)


def test_adaptive_hybrid_routes_dict_only_for_first_prefill_stage(monkeypatch):
    tensor = _make_split_friendly_fp16().contiguous()

    monkeypatch.setenv("BLOOMBEE_LOSSLESS_WRAPPER", "1")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_ALGO", "adaptive_hybrid")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_LAYOUT", "byte_split")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_MIN_BYTES", "0")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_MIN_GAIN_BYTES", "0")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_LAYOUT_TARGETS", "*:*:hidden_states")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_HYBRID_DICT_BLOCKS", "0:20")
    _clear_transport_caches()

    raw_size = tensor.numel() * tensor.element_size()

    def fake_dict_wrapper(_raw, *, elem_size):
        assert elem_size == tensor.element_size()
        return lt._HEADER_STRUCT.pack(lt._MAGIC, lt._VERSION, lt._ALGO_ZSTD_DICT_BYTE_SPLIT, raw_size) + b"d"

    def fake_zipnn_wrapper(_raw, *, tensor):
        return lt._HEADER_STRUCT.pack(lt._MAGIC, lt._VERSION, lt._ALGO_ZIPNN, raw_size) + b"z"

    monkeypatch.setattr(lt, "_build_zstd_dict_byte_split_wrapper", fake_dict_wrapper)
    monkeypatch.setattr(lt, "_build_zipnn_wrapper", fake_zipnn_wrapper)
    monkeypatch.setattr(lt, "_supports_zipnn_transport", lambda *args, **kwargs: True)

    first_stage = {
        "phase": "prefill",
        "tensor_name": "hidden_states",
        "source": "client",
        "channel": "rpc_inference",
        "blocks": "0:20",
    }
    serialized = lt.serialize_torch_tensor(
        tensor,
        runtime_pb2.CompressionType.NONE,
        debug_context=first_stage,
    )
    parsed = lt._parse_wrapper(serialized.buffer)
    assert parsed is not None
    assert parsed[0] == lt._ALGO_ZSTD_DICT_BYTE_SPLIT

    later_stage = dict(first_stage)
    later_stage["blocks"] = "20:40"
    serialized = lt.serialize_torch_tensor(
        tensor,
        runtime_pb2.CompressionType.NONE,
        debug_context=later_stage,
    )
    parsed = lt._parse_wrapper(serialized.buffer)
    assert parsed is not None
    assert parsed[0] == lt._ALGO_ZIPNN


def test_serialize_torch_tensor_byte_split_high_only_roundtrip(monkeypatch):
    tensor = _make_split_friendly_fp16().contiguous()
    debug_context = {
        "phase": "decode",
        "tensor_name": "hidden_states",
        "source": "client",
        "channel": "rpc_inference",
    }
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_WRAPPER", "1")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_ALGO", "zstd")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_LAYOUT", "byte_split_high_only")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_SINGLE_PATH", "1")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_MIN_BYTES", "0")
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_MIN_GAIN_BYTES", "0")
    _clear_transport_caches()

    serialized = lt.serialize_torch_tensor(
        tensor, runtime_pb2.CompressionType.NONE, debug_context=debug_context
    )
    parsed = lt._parse_wrapper(serialized.buffer, strict=True)
    assert parsed is not None, "high_only layout must produce a wrapped buffer"
    algo_id, original_size, _ = parsed
    assert algo_id == lt._ALGO_ZSTD_BYTE_SPLIT_HIGH_ONLY
    assert original_size == tensor.numel() * tensor.element_size()

    restored = lt.deserialize_torch_tensor(serialized)
    assert torch.equal(restored, tensor)


def test_unwrap_rejects_oversized_declared_size(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_LOSSLESS_MAX_DECODED_BYTES", "1024")
    payload = zlib.compress(b"x" * 64)
    fake_buffer = lt._HEADER_STRUCT.pack(lt._MAGIC, lt._VERSION, lt._ALGO_ZLIB, 1 << 40) + payload
    serialized = runtime_pb2.Tensor(buffer=fake_buffer)
    with pytest.raises(ValueError, match="MAX_DECODED_BYTES"):
        lt.unwrap_serialized_tensor(serialized)
