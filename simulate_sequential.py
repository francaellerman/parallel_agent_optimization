#!/usr/bin/env python3
"""Sequential baseline simulator.

Replays agent traces one after another. A trace is a JSON file holding a list
of alternating blocks:

  {"type": "agent", "tokens": <int>,   "note": "..."}  # LLM output
  {"type": "human", "seconds": <float>, "note": "..."} # time spent on human input

Agent output is streamed at a fixed rate (default 100 tokens/s); human blocks
take their recorded wall-clock time. Progress is printed live.

Usage:
  python simulate_sequential.py traces/trace_a.json traces/trace_b.json
  python simulate_sequential.py --speedup 10 traces/*.json   # run 10x faster
"""

import argparse
import json
import sys
import time
from pathlib import Path

TICK = 0.05  # simulated seconds between live progress updates


def load_trace(path):
    data = json.loads(Path(path).read_text())
    for i, b in enumerate(data["blocks"]):
        if b["type"] == "agent":
            assert isinstance(b["tokens"], int) and b["tokens"] > 0, (path, i)
        elif b["type"] == "human":
            assert b["seconds"] > 0, (path, i)
        else:
            raise ValueError(f"{path}: block {i} has unknown type {b['type']!r}")
    return data


class Clock:
    """Simulated clock; sleeps real time scaled down by `speedup`."""

    def __init__(self, speedup):
        self.speedup = speedup
        self.now = 0.0

    def advance(self, dt):
        time.sleep(dt / self.speedup)
        self.now += dt


def bar(frac, width=30):
    filled = int(round(frac * width))
    return "█" * filled + "·" * (width - filled)


def live(clock, msg):
    sys.stdout.write(f"\r\033[K[t={clock.now:7.2f}s] {msg}")
    sys.stdout.flush()


def run_agent_block(clock, label, block, tps):
    total = block["tokens"]
    duration = total / tps
    elapsed = 0.0
    while elapsed < duration:
        dt = min(TICK, duration - elapsed)
        clock.advance(dt)
        elapsed += dt
        done = min(total, int(round(elapsed * tps)))
        live(clock, f"{label} AGENT  {bar(done / total)} {done:5d}/{total} tok")
    print()


def run_human_block(clock, label, block):
    duration = block["seconds"]
    elapsed = 0.0
    while elapsed < duration:
        dt = min(TICK, duration - elapsed)
        clock.advance(dt)
        elapsed += dt
        live(clock, f"{label} HUMAN  {bar(elapsed / duration)} {elapsed:5.1f}/{duration:.1f} s")
    print()


def run_trace(clock, trace, tps):
    name = trace["name"]
    start = clock.now
    agent_time = human_time = 0.0
    print(f"\n=== Trace {name}: {trace.get('description', '')} (start t={start:.2f}s)")
    n = len(trace["blocks"])
    for i, block in enumerate(trace["blocks"], 1):
        label = f"[{name} {i}/{n}]"
        print(f"{label} {block['type']}: {block.get('note', '')}")
        t0 = clock.now
        if block["type"] == "agent":
            run_agent_block(clock, label, block, tps)
            agent_time += clock.now - t0
        else:
            run_human_block(clock, label, block)
            human_time += clock.now - t0
    total = clock.now - start
    print(f"=== Trace {name} done at t={clock.now:.2f}s "
          f"(took {total:.2f}s: agent {agent_time:.2f}s, human {human_time:.2f}s)")
    return {"name": name, "start": start, "end": clock.now,
            "agent": agent_time, "human": human_time}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="trace JSON files, run in order")
    ap.add_argument("--tps", type=float, default=100.0,
                    help="LLM streaming rate in tokens/second (default 100)")
    ap.add_argument("--speedup", type=float, default=1.0,
                    help="run the simulation this many times faster than real time")
    args = ap.parse_args()

    traces = [load_trace(p) for p in args.traces]
    clock = Clock(args.speedup)
    results = [run_trace(clock, t, args.tps) for t in traces]

    print("\n=== Summary (sequential)")
    for r in results:
        print(f"  {r['name']}: t={r['start']:.2f}s -> {r['end']:.2f}s  "
              f"agent {r['agent']:.2f}s, human {r['human']:.2f}s")
    print(f"  Total simulated time: {clock.now:.2f}s")


if __name__ == "__main__":
    main()
