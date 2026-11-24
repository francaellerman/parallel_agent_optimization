#!/usr/bin/env bash
# One-time setup on a GHC machine (ghcX.ghc.andrew.cmu.edu, X in 1..86).
# Usage: bash vllm_bench/setup.sh
#
# GHC drivers are 550 (CUDA <= 12.x). vLLM >= 0.33 on PyPI is built for CUDA 13
# (needs driver >= 580) and fails with "libcudart.so.13: cannot open shared
# object", so we install the CUDA 12.9 wheel from GitHub releases instead.
# Override: VLLM_VERSION=0.31.0 bash vllm_bench/setup.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

echo "== GPU"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv
driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if (( driver_major < 525 )); then
  echo "ERROR: NVIDIA driver $driver_major < 525; current vLLM wheels need CUDA 12." >&2
  echo "Ask for/find a machine with a newer driver, or use an old cu118 vLLM build." >&2
  exit 1
elif (( driver_major < 570 )); then
  echo "WARN: driver < 570; relying on CUDA minor-version compatibility (usually fine)."
fi

mkdir -p "$VLLM_WORK"
echo "== Local disk for $VLLM_WORK (need ~15 GB free)"
df -h "$VLLM_WORK"

if ! command -v uv >/dev/null; then
  echo "== Installing uv into $VLLM_WORK/bin"
  mkdir -p "$VLLM_WORK/bin"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$VLLM_WORK/bin" UV_NO_MODIFY_PATH=1 sh
fi

echo "== Creating venv at $VENV"
uv venv --python 3.12 "$VENV"
VLLM_VERSION="${VLLM_VERSION:-0.32.0}"
VLLM_CUDA="${VLLM_CUDA:-cu129}"
WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${VLLM_CUDA}-cp38-abi3-manylinux_2_28_x86_64.whl"
uv pip install --python "$VENV/bin/python" "vllm @ $WHEEL" --torch-backend="$VLLM_CUDA"

"$VENV/bin/python" -c "import torch, vllm; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda ok:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "Done. Next: source vllm_bench/env.sh && bash vllm_bench/serve.sh"
