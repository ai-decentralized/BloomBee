#!/usr/bin/env bash
set -euo pipefail

# Install a CUDA wheel that works on Pascal GPUs such as Tesla P100 (SM 6.0)
# and on Chameleon images whose NVIDIA driver supports CUDA 12.x but not
# CUDA 13.x. Run this inside the target Python/conda environment before
# `pip install -e .`.

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCH_VERSION="${TORCH_VERSION:-2.4.1+cu121}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

echo "[BloomBee] Installing PyTorch ${TORCH_VERSION} from ${PYTORCH_INDEX_URL}"

"${PYTHON_BIN}" -m pip uninstall -y \
  torch torchvision torchaudio triton \
  cuda-bindings cuda-core cuda-pathfinder cuda-python cuda-toolkit \
  nvidia-cublas-cu13 nvidia-cuda-cupti-cu13 nvidia-cuda-nvrtc-cu13 \
  nvidia-cuda-runtime-cu13 nvidia-cudnn-cu13 nvidia-cufft-cu13 \
  nvidia-cufile-cu13 nvidia-curand-cu13 nvidia-cusolver-cu13 \
  nvidia-cusparse-cu13 nvidia-cusparselt-cu13 nvidia-nccl-cu13 \
  nvidia-nvjitlink-cu13 nvidia-nvshmem-cu13 nvidia-nvtx-cu13 \
  >/dev/null 2>&1 || true

"${PYTHON_BIN}" -m pip install \
  --extra-index-url "${PYTORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}"

"${PYTHON_BIN}" - <<'PY'
import torch

print("torch", torch.__version__, "cuda runtime", torch.version.cuda)
print("cuda available", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        print(index, torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index))
else:
    raise SystemExit(
        "PyTorch installed, but CUDA is still unavailable. Check the NVIDIA driver and CUDA wheel compatibility."
    )
PY
