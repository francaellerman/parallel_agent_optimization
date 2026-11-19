#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server on this GHC machine's GPU.
# Usage: bash vllm_bench/serve.sh [extra vllm serve args...]
# Env overrides: MODEL, PORT, MAX_NUM_SEQS, MAX_MODEL_LEN, GPU_UTIL
set -euo pipefail
source "$(dirname "$0")/env.sh"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"   # ~3 GB fp16; leaves room for KV cache on 8 GB
PORT="${PORT:-$((8000 + $(id -u) % 1000))}"      # per-user port; machines are shared
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"              # upper bound on the batch size the scheduler will form
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.85}"

echo "== GPU usage (other users' processes show up here; results are noisy if the GPU is shared)"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
echo "== Serving $MODEL on http://127.0.0.1:$PORT  (max_num_seqs=$MAX_NUM_SEQS)"

# --dtype half: RTX 2080 (Turing, sm_75) has no bf16 support.
exec vllm serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --dtype half \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_UTIL" \
  "$@"
