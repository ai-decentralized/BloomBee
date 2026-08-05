import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


def _load_benchmark_module():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "compression" / "benchmark_lossless_codecs.py"
    spec = importlib.util.spec_from_file_location("benchmark_lossless_codecs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_codec_adapters_roundtrip_fp16_and_fp32():
    bench = _load_benchmark_module()
    for dtype in (torch.float16, torch.float32):
        tensor = torch.randn(2, 4, 16, dtype=dtype)
        raw = bench.tensor_to_raw_bytes(tensor)
        for codec in (
            "raw",
            "plain_zstd",
            "current_byte_split_zstd",
            "current_byte_split_zstd_reusectx",
            "bit_group_zstd",
            "byte_split_zstd_high_raw_low",
            "byte_split_selective_zstd",
        ):
            result = bench.CODEC_FUNCS[codec](raw, repeat=1, level=1, dtype=dtype)
            assert result.available == 1
            assert result.roundtrip_ok == 1
            assert result.raw_bytes == len(raw)
            assert result.wire_bytes > 0


@pytest.mark.parametrize("shape", [(1, 1, 5120), (1, 6, 5120), (1, 128, 5120), (1, 512, 5120)])
def test_phase2c_fp16_candidates_roundtrip_dump_shapes(shape):
    bench = _load_benchmark_module()
    tensor = torch.randn(*shape, dtype=torch.float16)
    raw = bench.tensor_to_raw_bytes(tensor)
    for codec in (
        "byte_split_zstd_high_raw_low",
        "byte_split_selective_zstd",
        "token_xor_byte_split_zstd",
        "token_xor_zstd_high_raw_low",
    ):
        result = bench.CODEC_FUNCS[codec](raw, repeat=1, level=1, dtype=torch.float16, shape=shape)
        assert result.available == 1
        assert result.roundtrip_ok == 1
        assert result.raw_bytes == len(raw)
        assert result.wire_bytes > 0


@pytest.mark.parametrize("shape", [(1, 1, 32), (1, 6, 32), (2, 7, 33)])
def test_phase2e_axis_and_selective_candidates_roundtrip(shape):
    bench = _load_benchmark_module()
    tensor = torch.randn(*shape, dtype=torch.float16)
    raw = bench.tensor_to_raw_bytes(tensor)
    for codec in (
        "sample_guided_selective_byte_split_zstd",
        "axis_transpose_byte_split_zstd",
        "blocked_axis_byte_split_zstd_32",
        "blocked_axis_byte_split_zstd_64",
        "blocked_axis_byte_split_zstd_128",
    ):
        result = bench.CODEC_FUNCS[codec](raw, repeat=1, level=1, dtype=torch.float16, shape=shape)
        assert result.available == 1
        assert result.roundtrip_ok == 1
        assert result.raw_bytes == len(raw)
        assert result.wire_bytes > 0


def test_phase2e_zstd_dict_candidate_roundtrip(tmp_path):
    bench = _load_benchmark_module()
    samples = []
    metadata = []
    for idx in range(8):
        tensor = torch.randn(1, 16, 512, dtype=torch.float16)
        path = tmp_path / f"sample_{idx}.pt"
        torch.save(tensor, path)
        samples.append(path)
        metadata.append({"filename": path.name, "phase": "prefill", "direction": "client_to_server"})
    (tmp_path / "metadata.json").write_text(json.dumps({"samples": metadata}))

    rows = bench.run_benchmark(tmp_path, ["zstd_dict_byte_split_zstd"], repeat=1, level=1)

    assert len(rows) == len(samples)
    assert all(int(row["available"]) == 1 for row in rows)
    assert all(int(row["roundtrip_ok"]) == 1 for row in rows)


def test_phase2c_candidates_roundtrip_real_dump_if_available():
    bench = _load_benchmark_module()
    dump_dir = Path("/home/cc/bloombee-runs/phase2b_llama13b_wire_20260502_052021/wire_activations")
    if not dump_dir.exists():
        pytest.skip("Phase 2B real activation dump is not available in this environment")
    tensors = []
    for tensor_path in sorted(dump_dir.glob("*.pt")):
        tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)
        if torch.is_tensor(tensor) and tensor.dtype == torch.float16:
            tensors.append(tensor)
        if len(tensors) >= 3:
            break
    if not tensors:
        pytest.skip("No fp16 tensors found in Phase 2B dump")
    for tensor in tensors:
        raw = bench.tensor_to_raw_bytes(tensor)
        shape = tuple(int(dim) for dim in tensor.shape)
        for codec in (
            "byte_split_zstd_high_raw_low",
            "byte_split_selective_zstd",
            "sample_guided_selective_byte_split_zstd",
            "axis_transpose_byte_split_zstd",
            "blocked_axis_byte_split_zstd_64",
            "token_xor_byte_split_zstd",
            "token_xor_zstd_high_raw_low",
        ):
            result = bench.CODEC_FUNCS[codec](raw, repeat=1, level=1, dtype=tensor.dtype, shape=shape)
            assert result.available == 1
            assert result.roundtrip_ok == 1


def test_bit_group_zstd_roundtrip_bf16():
    bench = _load_benchmark_module()
    tensor = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    raw = bench.tensor_to_raw_bytes(tensor)

    result = bench.codec_bit_group_zstd(raw, repeat=1, level=1, dtype=torch.bfloat16)

    assert result.available == 1
    assert result.roundtrip_ok == 1
    assert result.raw_bytes == len(raw)


def test_zipnn_missing_is_reported_not_raised(monkeypatch):
    bench = _load_benchmark_module()
    monkeypatch.setattr(bench.lt, "_ZipNN", None)
    tensor = torch.randn(2, 4, 16, dtype=torch.float16)
    raw = bench.tensor_to_raw_bytes(tensor)

    result = bench.codec_zipnn(raw, repeat=1, level=1, dtype=torch.float16)

    assert result.available == 0
    assert result.reason == "zipnn_unavailable"


def test_run_benchmark_writes_expected_csv_schema(tmp_path):
    bench = _load_benchmark_module()
    tensor = torch.randn(1, 2, 8, dtype=torch.float16)
    tensor_path = tmp_path / "sample.pt"
    torch.save(tensor, tensor_path)
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "filename": tensor_path.name,
                        "phase": "prefill",
                        "source": "client",
                        "direction": "client_to_server",
                        "channel": "rpc_inference",
                        "blocks": "0:20",
                        "compute_dtype": "float32",
                        "schema_dtype": "float16",
                        "wire_dtype": "float16",
                    }
                ]
            }
        )
    )

    rows = bench.run_benchmark(tmp_path, ["raw", "plain_zstd", "adaptive_selector"], repeat=1, level=1)
    output_csv = tmp_path / "out.csv"
    bench.write_csv(rows, output_csv)

    with output_csv.open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == list(bench.CSV_FIELDS)
        csv_rows = list(reader)

    assert len(csv_rows) == 3
    assert csv_rows[0]["sample"] == tensor_path.name
    assert csv_rows[0]["phase"] == "prefill"
    assert csv_rows[0]["source"] == "client"
    assert csv_rows[0]["direction"] == "client_to_server"
    assert csv_rows[0]["channel"] == "rpc_inference"
    assert csv_rows[0]["wire_dtype"] == "float16"


def test_run_benchmark_filters_metadata_and_shape(tmp_path):
    bench = _load_benchmark_module()
    keep = torch.randn(1, 6, 8, dtype=torch.float16)
    skip = torch.randn(1, 1, 8, dtype=torch.float16)
    keep_path = tmp_path / "keep.pt"
    skip_path = tmp_path / "skip.pt"
    torch.save(keep, keep_path)
    torch.save(skip, skip_path)
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "filename": keep_path.name,
                        "phase": "prefill",
                        "direction": "client_to_server",
                        "channel": "rpc_inference",
                    },
                    {
                        "filename": skip_path.name,
                        "phase": "decode",
                        "direction": "client_to_server",
                        "channel": "rpc_inference",
                    },
                ]
            }
        )
    )

    rows = bench.run_benchmark(
        tmp_path,
        ["raw"],
        repeat=1,
        level=1,
        phases={"prefill"},
        directions={"client_to_server"},
        channels={"rpc_inference"},
        shapes={"1x6x8"},
    )

    assert [row["sample"] for row in rows] == [keep_path.name]
