# Source this on a GHC machine: `source vllm_bench/env.sh`
# Everything big (venv, model weights, compile caches) lives on the machine's
# local disk, NOT in AFS home (small quota). It is per-machine: if you move to a
# different ghcX, re-run setup.sh there.
export VLLM_WORK="${VLLM_WORK:-/tmp/$USER-vllm}"
export VENV="$VLLM_WORK/venv"
export UV_CACHE_DIR="$VLLM_WORK/uv-cache"
export UV_PYTHON_INSTALL_DIR="$VLLM_WORK/python"
export HF_HOME="$VLLM_WORK/hf"
export XDG_CACHE_HOME="$VLLM_WORK/cache"
export VLLM_CACHE_ROOT="$VLLM_WORK/cache/vllm"
export TRITON_CACHE_DIR="$VLLM_WORK/cache/triton"
export PATH="$VLLM_WORK/bin:$PATH"   # uv lives here, not in AFS home

# vLLM's pip wheels bundle their own CUDA 12 runtime. The course's CUDA 11.7
# toolkit (if you added it to ~/.bashrc for asst2) would shadow it / break JIT
# compilation, so strip it from this shell.
_strip() { echo "$1" | tr ':' '\n' | grep -v '/usr/local/cuda' | paste -sd: -; }
export PATH="$(_strip "$PATH")"
export LD_LIBRARY_PATH="$(_strip "${LD_LIBRARY_PATH:-}")"
unset -f _strip

if [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
fi
