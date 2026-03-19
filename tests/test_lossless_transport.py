import numpy as np
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
