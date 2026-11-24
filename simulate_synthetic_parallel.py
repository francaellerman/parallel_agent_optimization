#!/usr/bin/env python3
"""Parallel-agent simulator: up to 4 agents, one human, a backlog of queries.

Each agent replays traces in a loop (agents start on different traces). Trace
blocks mean:

  agent block -> the agent generates `tokens` tokens at --tps tokens/second
                 (every agent generates independently at the same speed)
  human block -> the agent posts a query and BLOCKS until the human has spent
                 `seconds` on it (reading/approving/answering)

There is one human, who handles queries one at a time, first come first served.
When queries arrive faster than the human can handle them, a backlog builds up
and agents sit idle.

The simulation runs in real time and draws a live timeline: one column per
agent plus one for the human, one row per --row-seconds, time flowing down.

  blue   = agent generating
  yellow = human reviewing this agent's query
  blank  = agent idle, waiting in the human's queue

At the end (or on Ctrl-C) it prints how fast queries arrived vs. how fast the
human approved them, the backlog, and how agents spent their time.

Usage:
  python simulate_synthetic_parallel.py traces/*.json                 # 4 agents, 3 minutes
  python simulate_synthetic_parallel.py traces/*.json --agents 2 --duration 60
  python simulate_synthetic_parallel.py traces/*.json --tps 50 --json run.json
"""

import argparse
import json
import statistics
import sys
import time
from collections import deque
from pathlib import Path

from simulate_sequential import load_trace

TICK = 0.05     # seconds per simulation step
MAX_AGENTS = 4
COL = 18        # width of each timeline column

RESET = "\033[0m"
STYLE = {"gen": "\033[44;97m", "human": "\033[43;30m"}   # blue bg / yellow bg
GLYPH = {"gen": "█", "human": "▒"}                       # used with --no-color


# --------------------------------------------------------------------------- model

class Agent:
    def __init__(self, idx, traces):
        self.idx = idx
        self.traces = traces
        self.trace_i = idx % len(traces)  # stagger which trace each agent starts on
        self.block_i = -1
        self.state = None        # gen | human_wait | human
        self.remaining = 0.0     # tokens left in current agent block
        self.time_in = {"gen": 0.0, "human_wait": 0.0, "human": 0.0}
        self.tasks_done = 0
        self.label = None        # label of a block that started but isn't drawn yet
        self.row_time = {}       # state -> seconds spent in it during the current row

    def trace(self):
        return self.traces[self.trace_i]

    def block(self):
        return self.trace()["blocks"][self.block_i]


class Sim:
    def __init__(self, traces, args):
        self.args = args
        self.now = 0.0
        self.agents = [Agent(i, traces) for i in range(args.agents)]
        self.queue = deque()        # (agent, query) waiting for the human
        self.reviewing = None       # (agent, query) the human is working on
        self.review_left = 0.0
        self.queries = []           # every query ever posted
        self.human_busy = 0.0
        self.backlog_time = 0.0
        self.human_label = None
        self.human_row = {}         # agent idx -> seconds reviewed during current row

    # ---- transitions
    def advance(self, a):
        """Move agent `a` to its next block (looping over traces)."""
        a.block_i += 1
        if a.block_i >= len(a.trace()["blocks"]):
            a.tasks_done += 1
            a.trace_i = (a.trace_i + 1) % len(a.traces)
            a.block_i = 0
        b = a.block()
        name = a.trace()["name"]
        if b["type"] == "agent":
            a.state, a.remaining = "gen", float(b["tokens"])
            a.label = ("gen", f"{name} gen {b['tokens']}t")
        else:
            q = {"agent": a.idx, "trace": name, "block": a.block_i, "note": b.get("note", ""),
                 "seconds": b["seconds"], "arrive": self.now, "start": None, "end": None}
            self.queries.append(q)
            a.state = "human_wait"
            self.queue.append((a, q))

    def start_review(self):
        if self.reviewing is None and self.queue:
            a, q = self.queue.popleft()
            q["start"] = self.now
            a.state = "human"
            a.label = ("human", f"{q['trace']} review {q['seconds']:g}s")
            self.human_label = (a.idx, f"agent {a.idx} ({q['trace']}) {q['seconds']:g}s")
            self.reviewing, self.review_left = (a, q), q["seconds"]

    def step(self, dt):
        self.start_review()
        for a in self.agents:
            a.time_in[a.state] += dt
            a.row_time[a.state] = a.row_time.get(a.state, 0.0) + dt
        self.backlog_time += len(self.queue) * dt
        if self.reviewing:
            self.human_busy += dt
            idx = self.reviewing[0].idx
            self.human_row[idx] = self.human_row.get(idx, 0.0) + dt
        self.now += dt

        for a in self.agents:
            if a.state == "gen":
                a.remaining -= self.args.tps * dt
                if a.remaining <= 1e-9:
                    self.advance(a)
        if self.reviewing:
            self.review_left -= dt
            if self.review_left <= 1e-9:
                a, q = self.reviewing
                q["end"] = self.now
                self.reviewing = None
                self.advance(a)
        self.start_review()

    # ---- results
    def summary(self):
        T = self.now or 1e-9
        approved = [q for q in self.queries if q["end"] is not None]
        waits = [q["start"] - q["arrive"] for q in self.queries if q["start"] is not None]
        traces = self.agents[0].traces
        n_q = sum(b["type"] == "human" for t in traces for b in t["blocks"])
        cycle = sum(b["tokens"] / self.args.tps if b["type"] == "agent" else b["seconds"]
                    for t in traces for b in t["blocks"])
        review_s = [b["seconds"] for t in traces for b in t["blocks"] if b["type"] == "human"]
        capacity = 60 / statistics.mean(review_s)
        # Demand: queries/min if every agent had its own human (never waited).
        demand = len(self.agents) * n_q / cycle * 60
        n = len(self.agents)
        return {
            "config": {"agents": n, "tps": self.args.tps, "elapsed_s": T},
            "queries_arrived": len(self.queries),
            "queries_approved": len(approved),
            "arrival_rate_per_min": len(self.queries) / T * 60,
            "approval_rate_per_min": len(approved) / T * 60,
            "demand_rate_per_min": demand,
            "human_capacity_per_min": capacity,
            "offered_load": demand / capacity,
            "human_utilization": self.human_busy / T,
            "backlog_end": len(self.queue),
            "backlog_mean": self.backlog_time / T,
            "wait_mean_s": statistics.mean(waits) if waits else 0.0,
            "wait_max_s": max(waits) if waits else 0.0,
            "tasks_completed": sum(a.tasks_done for a in self.agents),
            "agent_time_fraction": {k: sum(a.time_in[k] for a in self.agents) / (n * T)
                                    for k in ("gen", "human_wait", "human")},
            "per_agent_idle_fraction": [a.time_in["human_wait"] / T for a in self.agents],
            "queries": self.queries,
        }


# --------------------------------------------------------------------------- live view

class View:
    def __init__(self, sim, color):
        self.sim = sim
        self.color = color

    def cell(self, state, text):
        if state not in STYLE:
            return " " * COL
        if self.color:
            return f"{STYLE[state]}{(' ' + text)[:COL].ljust(COL)}{RESET}"
        fill = GLYPH[state] * COL
        return (text[:COL - 1] + " " + fill)[:COL] if text else fill

    def header(self):
        cols = [f"agent {a.idx}".center(COL) for a in self.sim.agents] + ["human".center(COL)]
        print(f"{'t (s)':>6} │" + "│".join(cols) + f"│{'queue':>6}")
        print("─" * 7 + "┼" + "┼".join("─" * COL for _ in cols) + "┼" + "─" * 6)

    def row(self, t):
        sim, cells = self.sim, []
        for a in sim.agents:
            state = max(a.row_time, key=a.row_time.get) if a.row_time else None
            text = ""
            if a.label and a.label[0] == state:
                text, a.label = a.label[1], None
            cells.append(self.cell(state, text))
            a.row_time = {}
        if sim.human_row:
            idx = max(sim.human_row, key=sim.human_row.get)
            text = ""
            if sim.human_label and sim.human_label[0] == idx:
                text, sim.human_label = sim.human_label[1], None
            cells.append(self.cell("human", text))
        else:
            cells.append(self.cell(None, ""))
        sim.human_row = {}
        self.clear_status()
        print(f"{t:6.0f} │" + "│".join(cells) + f"│{len(sim.queue):6d}")

    def status(self):
        s = self.sim
        approved = sum(q["end"] is not None for q in s.queries)
        sys.stdout.write(f"\r\033[K  t={s.now:6.1f}s  queue={len(s.queue)}  "
                         f"arrived={len(s.queries)}  approved={approved}  (Ctrl-C to stop)")
        sys.stdout.flush()

    def clear_status(self):
        sys.stdout.write("\r\033[K")


def print_summary(s):
    c = s["config"]
    f = s["agent_time_fraction"]
    print(f"\n=== {c['agents']} agents at {c['tps']:g} tok/s, {c['elapsed_s']:.0f}s elapsed")
    print("Rates:")
    print(f"  agent demand          {s['demand_rate_per_min']:6.2f} /min  "
          "(if the human never made agents wait)")
    print(f"  queries arriving      {s['arrival_rate_per_min']:6.2f} /min  ({s['queries_arrived']} total)")
    print(f"  human approving       {s['approval_rate_per_min']:6.2f} /min  ({s['queries_approved']} total)")
    print(f"  human max capacity    {s['human_capacity_per_min']:6.2f} /min  (1 / mean review time)")
    print(f"  offered load          {s['offered_load']:6.2f}       "
          "(demand / capacity; >1 means the human is the bottleneck)")
    print(f"  human utilization     {s['human_utilization']:6.1%}")
    print("Backlog (queries waiting for the human):")
    print(f"  now {s['backlog_end']}, time-averaged {s['backlog_mean']:.2f}")
    print(f"  wait before the human starts a query: mean {s['wait_mean_s']:.1f}s, "
          f"max {s['wait_max_s']:.1f}s")
    print("Agent time (avg over agents):")
    print(f"  generating {f['gen']:.0%}, idle waiting for human {f['human_wait']:.0%}, "
          f"being reviewed {f['human']:.0%}")
    print("  idle per agent: " + ", ".join(f"agent {i} {x:.0%}"
                                           for i, x in enumerate(s["per_agent_idle_fraction"])))
    print(f"  tasks completed: {s['tasks_completed']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="trace JSON files (agents loop over them)")
    ap.add_argument("--agents", type=int, default=MAX_AGENTS, choices=range(1, MAX_AGENTS + 1),
                    help=f"number of parallel agents, 1-{MAX_AGENTS} (default {MAX_AGENTS})")
    ap.add_argument("--tps", type=float, default=100.0,
                    help="tokens/second each agent generates at (default 100)")
    ap.add_argument("--duration", type=float, default=180.0, help="seconds to run (default 180)")
    ap.add_argument("--row-seconds", type=float, default=1.0,
                    help="seconds per timeline row (default 1)")
    ap.add_argument("--no-color", action="store_true", help="draw blocks with characters instead")
    ap.add_argument("--json", help="write the summary and every query here")
    args = ap.parse_args()

    traces = [load_trace(p) for p in args.traces]
    sim = Sim(traces, args)
    view = View(sim, color=not args.no_color)

    legend = ("blue = generating, yellow = human reviewing, blank = idle (waiting for human)"
              if view.color else f"{GLYPH['gen']} = generating, {GLYPH['human']} = human "
                                 "reviewing, blank = idle (waiting for human)")
    print(f"{args.agents} agents, {args.tps:g} tok/s, one human, {args.duration:g}s.  {legend}\n")
    view.header()

    for a in sim.agents:
        sim.advance(a)
    start = time.monotonic()
    ticks_per_row = max(1, round(args.row_seconds / TICK))
    k = 0
    try:
        while sim.now < args.duration - 1e-9:
            sim.step(TICK)
            k += 1
            if k % ticks_per_row == 0:
                view.row(sim.now - args.row_seconds)
            view.status()
            delay = start + k * TICK - time.monotonic()
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    view.clear_status()

    s = sim.summary()
    print_summary(s)
    if args.json:
        Path(args.json).write_text(json.dumps(s, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
