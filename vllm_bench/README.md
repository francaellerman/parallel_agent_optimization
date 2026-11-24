# vLLM batch-size benchmark on GHC GPUs

GHC machines `ghc1`–`ghc86.ghc.andrew.cmu.edu` have an NVIDIA RTX 2080
(Turing, sm_75, 8 GB VRAM). Consequences:

- **fp16, not bf16** (`--dtype half`); Turing has no bf16.
- **Small models only.** Default is `Qwen/Qwen2.5-1.5B-Instruct` (~3 GB weights).
- **Install on local `/tmp`, not AFS home.** The venv, weights and caches are
  ~15 GB, which is over the AFS quota. `/tmp` is per machine, so re-run setup if
  you switch machines, and it may be wiped.
- **Don't use the course's CUDA 11.7 toolkit.** vLLM wheels bundle CUDA 12;
  `env.sh` strips `/usr/local/cuda*` from `PATH`/`LD_LIBRARY_PATH`.
- **Machines are shared.** Check `nvidia-smi` for other users' processes before
  benchmarking, because a shared GPU makes the numbers meaningless.

## Usage

```bash
ssh ghc42.ghc.andrew.cmu.edu        # any X in 1..86
cd ~/parallel_agent_optimization     # repo clone (small; fine in AFS)

bash vllm_bench/setup.sh             # once per machine (~5-10 min)

# terminal 1 (or tmux):
bash vllm_bench/serve.sh             # MODEL=..., MAX_NUM_SEQS=... to override

# terminal 2, once the server logs "Application startup complete":
source vllm_bench/env.sh
python vllm_bench/bench_batch.py --batch-sizes 1,2,4,8,16,32,64,128
```

Results print as a table and are saved to `vllm_bench/results/<timestamp>.json`.

## What "batch size" means here

vLLM does continuous batching, so the batch is whatever requests are in flight
(capped by `--max-num-seqs`). The benchmark fires exactly N identical-length
requests at once, using `ignore_eos` and random-token prompts with no
prefix-cache hits, so decoding runs at batch size N throughout.

- `stream tok/s` is the decode speed one request sees. This is the `--tps`
  that `simulate_sequential.py` currently hardcodes to 100.
- `agg tok/s` is the total throughput across the batch.
- `TTFT` is time to first token, which grows with N because the N prefills
  are queued.

Useful knobs: `--prompt-tokens`, `--output-tokens`, `--rounds`; and on the
server `MAX_NUM_SEQS`, `MAX_MODEL_LEN`, and `--enforce-eager` (no CUDA graphs).
If a big batch doesn't fit in KV cache, vLLM preempts requests and that shows
up as a throughput cliff. The server log reports the KV cache capacity at
startup.
