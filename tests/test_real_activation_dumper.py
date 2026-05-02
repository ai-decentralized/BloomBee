import json

import torch

from bloombee.utils import real_activation_dumper as rad


def _reset_dumper_singleton():
    rad.RealActivationDumper._instance = None
    rad._dumper = None


def test_wire_activation_metadata_merges_existing_file_and_uses_context_batch_size(tmp_path, monkeypatch):
    monkeypatch.setenv("BLOOMBEE_DUMP_WIRE_ACTIVATIONS", "1")
    monkeypatch.setenv("BLOOMBEE_DUMP_ACTIVATIONS", "0")
    monkeypatch.setenv("BLOOMBEE_ACTIVATION_DIR", str(tmp_path))
    monkeypatch.setenv("BLOOMBEE_ACTIVATION_SAMPLES", "10")
    _reset_dumper_singleton()

    existing = {
        "total_samples": 1,
        "samples": [
            {
                "filename": "existing.pt",
                "channel": "rpc_inference",
                "source": "client",
                "batch_size": 1,
            }
        ],
    }
    (tmp_path / "metadata.json").write_text(json.dumps(existing))

    captured = rad.capture_wire_activation(
        torch.randn(2, 1, 8, dtype=torch.float16),
        source="server",
        channel="rpc_push_microbatch",
        direction="server_to_server",
        phase="decode",
        blocks="0:20->20:40",
        compute_dtype="float16",
        schema_dtype="float16",
        wire_dtype="float16",
        batch_size=16,
        prompt_len=1,
    )

    assert captured is not None
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    samples = metadata["samples"]
    assert metadata["total_samples"] == 2
    assert {sample["filename"] for sample in samples} >= {"existing.pt"}

    new_sample = next(sample for sample in samples if sample["filename"] != "existing.pt")
    assert new_sample["batch_size"] == 16
    assert new_sample["channel"] == "rpc_push_microbatch"
    assert new_sample["direction"] == "server_to_server"
    assert new_sample["blocks"] == "0:20->20:40"
