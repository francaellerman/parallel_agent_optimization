#!/usr/bin/env python3
"""Measure vLLM generation speed as a function of batch size.

For each batch size N, fires exactly N streaming requests at once (all with the
same prompt/output length, `ignore_eos` so each generates exactly
--output-tokens), so the server decodes a batch of N for the whole run. Repeats
--rounds times per N.

Per request we record:
  ttft        time to first token (includes queueing + prefill)
  decode_tps  per-stream decode speed = (out_tokens - 1) / (t_last - t_first)
  e2e         end-to-end latency
Per batch size we report the per-stream speed (what one agent "sees") and the
aggregate throughput (sum over the batch).

Prompts are random token ids (exact length, no tokenizer needed, and no prefix
cache hits between requests).

Run on the GHC machine running serve.sh (server must have max_num_seqs >= the
largest batch size, otherwise extra requests just queue):
  python vllm_bench/bench_batch.py --batch-sizes 1,2,4,8,16,32,64
"""

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from pathlib import Path

import aiohttp


async def one_request(session, url, model, prompt_ids, max_tokens):
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "ignore_eos": True,  # vLLM extension: always generate max_tokens
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    t_first = t_last = None
    out_tokens = None
    async with session.post(f"{url}/v1/completions", json=payload) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("choices"):
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
            if chunk.get("usage"):
                out_tokens = chunk["usage"]["completion_tokens"]
    t_end = time.perf_counter()
    if t_first is None or out_tokens is None:
        raise RuntimeError("stream ended without tokens/usage")
    decode_time = t_last - t_first
    return {
        "ttft": t_first - t0,
        "e2e": t_end - t0,
        "out_tokens": out_tokens,
        "decode_tps": (out_tokens - 1) / decode_time if decode_time > 0 else float("nan"),
    }


def rand_prompt(n, lo, hi):
    return [random.randint(lo, hi) for _ in range(n)]


async def run_batch(session, args, model, n):
    prompts = [rand_prompt(args.prompt_tokens, *args.vocab_range) for _ in range(n)]
    t0 = time.perf_counter()
    reqs = await asyncio.gather(*(
        one_request(session, args.url, model, p, args.output_tokens) for p in prompts))
    wall = time.perf_counter() - t0
    return reqs, wall


def summarize(n, rounds):
    reqs = [r for rs, _ in rounds for r in rs]
    walls = [w for _, w in rounds]
    total_tokens = sum(r["out_tokens"] for rs, _ in rounds for r in rs)
    tps = [r["decode_tps"] for r in reqs]
    ttft = [r["ttft"] for r in reqs]
    return {
        "batch_size": n,
        "requests": len(reqs),
        "per_stream_decode_tps_mean": statistics.mean(tps),
        "per_stream_decode_tps_p50": statistics.median(tps),
        "per_stream_decode_tps_min": min(tps),
        "ttft_mean_s": statistics.mean(ttft),
        "ttft_max_s": max(ttft),
        "e2e_mean_s": statistics.mean(r["e2e"] for r in reqs),
        "aggregate_tps": total_tokens / sum(walls),
    }


async def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=f"http://127.0.0.1:{8000 + os.getuid() % 1000}",
                    help="server base URL (default matches serve.sh's per-user port)")
    ap.add_argument("--model", default=None, help="default: first model the server lists")
    ap.add_argument("--batch-sizes", default="1,2,4,8,16,32,64")
    ap.add_argument("--prompt-tokens", type=int, default=256)
    ap.add_argument("--output-tokens", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=3, help="repetitions per batch size")
    ap.add_argument("--vocab-range", type=int, nargs=2, default=(1000, 30000),
                    metavar=("LO", "HI"), help="token id range for random prompts")
    ap.add_argument("--out", default=None,
                    help="JSON results path (default: vllm_bench/results/<timestamp>.json)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    sizes = [int(x) for x in args.batch_sizes.split(",")]

    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    conn = aiohttp.TCPConnector(limit=0)  # no client-side cap on concurrency
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        async with session.get(f"{args.url}/v1/models") as r:
            r.raise_for_status()
            models = (await r.json())["data"]
        model = args.model or models[0]["id"]
        max_len = models[0].get("max_model_len")
        print(f"server {args.url}  model {model}  max_model_len {max_len}")
        print(f"prompt {args.prompt_tokens} tok, output {args.output_tokens} tok, "
              f"{args.rounds} rounds per batch size\n")

        print("warmup...", flush=True)
        await run_batch(session, args, model, 2)

        hdr = (f"{'batch':>5} {'stream tok/s':>12} {'(p50)':>7} {'(min)':>7} "
               f"{'TTFT s':>7} {'TTFT max':>8} {'agg tok/s':>9}")
        print(hdr)
        print("-" * len(hdr))
        results = []
        for n in sizes:
            rounds = [await run_batch(session, args, model, n) for _ in range(args.rounds)]
            s = summarize(n, rounds)
            results.append(s)
            print(f"{n:5d} {s['per_stream_decode_tps_mean']:12.1f} "
                  f"{s['per_stream_decode_tps_p50']:7.1f} {s['per_stream_decode_tps_min']:7.1f} "
                  f"{s['ttft_mean_s']:7.3f} {s['ttft_max_s']:8.3f} {s['aggregate_tps']:9.1f}",
                  flush=True)

    out = Path(args.out) if args.out else (
        Path(__file__).parent / "results" / time.strftime("%Y%m%d-%H%M%S.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "host": os.uname().nodename, "url": args.url, "model": model,
        "prompt_tokens": args.prompt_tokens, "output_tokens": args.output_tokens,
        "rounds": args.rounds, "results": results,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
