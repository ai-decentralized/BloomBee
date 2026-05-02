#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class TorchWheel:
    label: str
    torch_spec: str
    index_url: str
    min_driver_cuda: Tuple[int, int]
    min_compute_capability: Tuple[int, int]
    notes: str


WHEEL_CHOICES: Tuple[TorchWheel, ...] = (
    TorchWheel(
        label="cu124-modern",
        torch_spec="torch==2.5.1+cu124",
        index_url="https://download.pytorch.org/whl/cu124",
        min_driver_cuda=(12, 4),
        min_compute_capability=(7, 0),
        notes="Default for Volta/Turing/Ampere/Hopper-class GPUs when the driver supports CUDA 12.4+.",
    ),
    TorchWheel(
        label="cu121-pascal",
        torch_spec="torch==2.4.1+cu121",
        index_url="https://download.pytorch.org/whl/cu121",
        min_driver_cuda=(12, 1),
        min_compute_capability=(6, 0),
        notes="Conservative fallback for Pascal GPUs such as Tesla P100 (SM 6.0).",
    ),
    TorchWheel(
        label="cu118-legacy",
        torch_spec="torch==2.2.2+cu118",
        index_url="https://download.pytorch.org/whl/cu118",
        min_driver_cuda=(11, 8),
        min_compute_capability=(6, 0),
        notes="Legacy CUDA 11.8 fallback for older drivers.",
    ),
)


CU13_PACKAGES = (
    "cuda-bindings",
    "cuda-core",
    "cuda-pathfinder",
    "cuda-python",
    "cuda-toolkit",
    "nvidia-cublas-cu13",
    "nvidia-cuda-cupti-cu13",
    "nvidia-cuda-nvrtc-cu13",
    "nvidia-cuda-runtime-cu13",
    "nvidia-cudnn-cu13",
    "nvidia-cufft-cu13",
    "nvidia-cufile-cu13",
    "nvidia-curand-cu13",
    "nvidia-cusolver-cu13",
    "nvidia-cusparse-cu13",
    "nvidia-cusparselt-cu13",
    "nvidia-nccl-cu13",
    "nvidia-nvjitlink-cu13",
    "nvidia-nvshmem-cu13",
    "nvidia-nvtx-cu13",
)


def run(command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(command, 127, "", str(exc))


def parse_version(value: str) -> Tuple[int, int]:
    match = re.search(r"(\d+)(?:\.(\d+))?", value)
    if match is None:
        raise ValueError(f"Could not parse version from {value!r}")
    return int(match.group(1)), int(match.group(2) or 0)


def query_driver_cuda() -> Optional[Tuple[int, int]]:
    completed = run(["nvidia-smi"], check=False)
    if completed.returncode != 0:
        return None
    match = re.search(r"CUDA Version:\s*([0-9.]+)", completed.stdout)
    return parse_version(match.group(1)) if match else None


def query_gpus() -> Tuple[Tuple[str, Tuple[int, int]], ...]:
    completed = run(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        check=False,
    )
    if completed.returncode != 0:
        return ()

    rows = []
    for name, capability in csv.reader(completed.stdout.splitlines()):
        rows.append((name.strip(), parse_version(capability.strip())))
    return tuple(rows)


def choose_wheel(driver_cuda: Tuple[int, int], gpus: Tuple[Tuple[str, Tuple[int, int]], ...]) -> TorchWheel:
    if not gpus:
        raise SystemExit("No NVIDIA GPUs detected by nvidia-smi. Use a CPU-only PyTorch install instead.")

    min_capability = min(capability for _, capability in gpus)
    for choice in WHEEL_CHOICES:
        if driver_cuda >= choice.min_driver_cuda and min_capability >= choice.min_compute_capability:
            return choice

    gpu_summary = ", ".join(f"{name} sm_{cap[0]}{cap[1]}" for name, cap in gpus)
    raise SystemExit(
        "No compatible BloomBee PyTorch wheel rule matched. "
        f"driver CUDA={driver_cuda[0]}.{driver_cuda[1]}, GPUs={gpu_summary}. "
        "Set BLOOMBEE_TORCH_SPEC and BLOOMBEE_TORCH_INDEX_URL to override."
    )


def build_install_plan(args: argparse.Namespace) -> TorchWheel:
    override_spec = args.torch_spec or os.environ.get("BLOOMBEE_TORCH_SPEC")
    override_index = args.index_url or os.environ.get("BLOOMBEE_TORCH_INDEX_URL")
    if override_spec:
        return TorchWheel(
            label="manual-override",
            torch_spec=override_spec,
            index_url=override_index or "https://download.pytorch.org/whl/cu121",
            min_driver_cuda=(0, 0),
            min_compute_capability=(0, 0),
            notes="Selected by CLI/env override.",
        )

    driver_cuda = query_driver_cuda()
    if driver_cuda is None:
        raise SystemExit("Could not query NVIDIA driver CUDA version with nvidia-smi.")
    return choose_wheel(driver_cuda, query_gpus())


def main() -> int:
    parser = argparse.ArgumentParser(description="Install a PyTorch wheel compatible with the local NVIDIA GPUs.")
    parser.add_argument("--python", default=os.environ.get("PYTHON_BIN", sys.executable), help="Python executable to use")
    parser.add_argument("--torch-spec", help="Manual torch requirement, e.g. torch==2.4.1+cu121")
    parser.add_argument("--index-url", help="Manual PyTorch wheel index URL")
    parser.add_argument("--dry-run", action="store_true", help="Print the selected install command without running it")
    parser.add_argument("--keep-existing", action="store_true", help="Do not uninstall existing torch/CUDA packages first")
    args = parser.parse_args()

    choice = build_install_plan(args)
    print(f"[BloomBee] Selected PyTorch wheel rule: {choice.label}")
    print(f"[BloomBee] {choice.notes}")
    print(f"[BloomBee] Install spec: {choice.torch_spec}")
    print(f"[BloomBee] Wheel index:  {choice.index_url}")

    uninstall_command = [
        args.python,
        "-m",
        "pip",
        "uninstall",
        "-y",
        "torch",
        "torchvision",
        "torchaudio",
        "triton",
        *CU13_PACKAGES,
    ]
    install_command = [
        args.python,
        "-m",
        "pip",
        "install",
        "--extra-index-url",
        choice.index_url,
        choice.torch_spec,
    ]

    if args.dry_run:
        if not args.keep_existing:
            print("[BloomBee] Would run:", " ".join(uninstall_command))
        print("[BloomBee] Would run:", " ".join(install_command))
        return 0

    if not args.keep_existing:
        subprocess.run(uninstall_command, check=False)
    subprocess.run(install_command, check=True)
    subprocess.run(
        [
            args.python,
            "-c",
            (
                "import torch; "
                "print('torch', torch.__version__, 'cuda runtime', torch.version.cuda); "
                "print('cuda available', torch.cuda.is_available(), 'device_count', torch.cuda.device_count()); "
                "assert torch.cuda.is_available()"
            ),
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
