import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "benchmarks" / "compression"


@pytest.mark.parametrize(
    "metadata, layer",
    [
        ({"layer_idx": 3}, "3"),
        ({"block_uid": "model.7", "layer_idx": 0}, "7"),
        ({"block_uid": "0:14->14:28", "layer_idx": 0}, ""),
    ],
)
def test_summary_preserves_layer_and_span_identity(tmp_path, metadata, layer):
    filename = "activation,sample.pt"
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"samples": [{"filename": filename, **metadata}]}))
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            [
                {
                    "name": filename,
                    "direct_zstd_ratio": 0.8,
                    "split_total_with_header_ratio": 0.7,
                    "split_vs_direct_delta_pct": -12.5,
                    "raw_bytes": 100,
                    "direct_zstd_bytes": 80,
                    "split_total_with_header_bytes": 70,
                }
            ]
        )
    )
    output = tmp_path / "summary.csv"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "summarize_layer_compression.py"),
            "--metadata",
            str(metadata_path),
            "--report",
            str(report),
            "--csv",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with output.open(newline="") as f:
        (row,) = csv.DictReader(f)
    assert row["layer"] == layer
    assert row["block_uid"] == metadata.get("block_uid", "")
    assert row["filename"] == filename
    assert float(row["split_ratio"]) == 0.7


def test_latency_report_accepts_full_batch_wire_without_optional_timings(tmp_path):
    client, worker1, worker2 = [tmp_path / name for name in ("client.log", "worker1.log", "worker2.log")]
    client.write_text("")
    worker1.write_text("")
    worker2.write_text(
        "[S2S_WIRE] step_id=a channel=rpc_inference sender_blocks=0:14 receiver_blocks=14:28 "
        "payload_kb=4.0 metadata_b=0 raw_transfer_ms=1.5 wire_ms=1.25 clock_sync=1\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "analyze_latency_breakdown.py"),
            "--client-log",
            str(client),
            "--server1-log",
            str(worker1),
            "--server2-log",
            str(worker2),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    pair = next(line for line in result.stdout.splitlines() if line.startswith("0:14->14:28"))
    assert float(pair.split()[1]) == 1.25
