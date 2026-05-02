from __future__ import annotations

import re
import subprocess
from typing import Optional, Tuple

import torch


def _run_nvidia_smi(*args: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ("nvidia-smi", *args),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _parse_version(version: Optional[str]) -> Optional[Tuple[int, ...]]:
    if not version:
        return None
    match = re.search(r"(\d+(?:\.\d+)*)", version)
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _driver_cuda_version() -> Optional[str]:
    output = _run_nvidia_smi()
    if not output:
        return None
    match = re.search(r"CUDA Version:\s*([0-9.]+)", output)
    return match.group(1) if match else None


def _visible_gpu_names() -> Tuple[str, ...]:
    output = _run_nvidia_smi("--query-gpu=name", "--format=csv,noheader")
    if not output:
        return ()
    return tuple(line.strip() for line in output.splitlines() if line.strip())


def get_cuda_unavailable_diagnostic() -> Optional[str]:
    """Explain the common case where nvidia-smi sees GPUs but PyTorch cannot.

    This intentionally avoids raising; callers can append the returned message
    to their own startup warnings or errors.
    """

    if torch.cuda.is_available():
        return None

    gpu_names = _visible_gpu_names()
    if not gpu_names:
        return None

    torch_cuda = torch.version.cuda
    driver_cuda = _driver_cuda_version()
    torch_cuda_version = _parse_version(torch_cuda)
    driver_cuda_version = _parse_version(driver_cuda)

    detail = (
        f"nvidia-smi sees {len(gpu_names)} GPU(s): {', '.join(gpu_names)}; "
        f"PyTorch reports torch={torch.__version__}, torch.version.cuda={torch_cuda}, "
        f"driver CUDA={driver_cuda}."
    )
    if torch_cuda_version and driver_cuda_version and torch_cuda_version > driver_cuda_version:
        detail += " The installed PyTorch CUDA runtime is newer than the NVIDIA driver supports."

    return (
        detail
        + " Reinstall a compatible PyTorch wheel with "
        + "`python scripts/install_compatible_torch.py`, then rerun `pip install -e .`."
    )
