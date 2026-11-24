#!/usr/bin/env python3
"""Live timeline of a real parallel-agent trace (processed.json from parse_swe_chat.py).

One column per agent session, one row per --row-seconds of trace time, time
flowing down, played back at --speedup x real time:

  blue cells   agent working (first row shows the prompt that started the turn)
  blank cells  agent done with its turn, waiting for the human's next prompt
  dim dots     session not open yet / already over
  human col    the human sends a prompt (-> which agent)
  waiting col  how many agents are waiting on the human right now (the backlog)

At the end (or Ctrl-C) it prints, for the window shown: how fast agent turns
finished (queries for the human) vs. how fast the human sent prompts, the
backlog, and how each agent split its time.

Usage:
  python visualize_trace.py                                  # traces/parallel_005, 2026-08-10 11:30-21:20
  python visualize_trace.py traces/parallel_001/processed.json --start 0 --end 600
  python visualize_trace.py --row-seconds 30 --speedup 60    # finer rows, slower playback
  python visualize_trace.py --speedup 0                      # print instantly
"""

import argparse
import json
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "traces/parallel_005/processed.json"
DEFAULT_WINDOW = {"parallel_005": ("2026-08-10T11:30:00Z", "2026-08-10T21:20:00Z")}

RESET, DIM = "\033[0m", "\033[2m"
WORK, HUMAN = "\033[44;97m", "\033[43;30m"


def ts(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def hhmm(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%m-%d %H:%M")


def load(path):
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != 1:
        raise SystemExit(f"{path}: expected schema_version 1 (output of parse_swe_chat.py)")
    sessions = []
    for s in sorted(data["sessions"], key=lambda s: ts(s["start"])):
        blocks = []
        prompt = ""
        for b in s["blocks"]:
            start, end = ts(b["start"]), ts(b["end"])
            if b["role"] == "human":
                prompt = " ".join(b["text"].split())
                blocks.append({"kind": "wait", "start": start, "end": end})
            else:
                blocks.append({"kind": "work", "start": start, "end": end, "prompt": prompt,
                               "tokens": b.get("output_tokens", 0), "boundary": b.get("boundary")})
        sessions.append({"id": s["session_id"], "blocks": blocks,
                         "first": blocks[0]["start"], "last": blocks[-1]["end"]})
    return data, sessions


def parse_time(value, origin):
    """ISO timestamp, or seconds after the trace origin."""
    try:
        return origin + float(value)
    except ValueError:
        return ts(value)


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


class Timeline:
    def __init__(self, sessions, t0, t1, row, color):
        self.sessions, self.t0, self.t1, self.row, self.color = sessions, t0, t1, row, color
        cols = shutil.get_terminal_size((120, 40)).columns
        self.width = max(12, min(24, (cols - 7 - 16 - 9) // len(sessions) - 1))
        self.shown = set()  # (session index, block index) already labelled

    def cell(self, kind, text):
        w = self.width
        if kind == "work":
            return f"{WORK}{(' ' + text)[:w].ljust(w)}{RESET}" if self.color else (text + " " + "█" * w)[:w] if text else "█" * w
        if kind == "wait":
            return f"{DIM}{(' ' + text)[:w].ljust(w)}{RESET}" if (self.color and text) else (" " + text)[:w].ljust(w)
        dots = "·" * w
        return f"{DIM}{dots}{RESET}" if self.color else dots

    def header(self):
        names = [f"agent {i + 1} ({s['id'][:8]})".center(self.width) for i, s in enumerate(self.sessions)]
        print(f"{'UTC':>11} │" + "│".join(names) + f"│{'human':^16}│{'waiting':>8}")
        print("─" * 12 + "┼" + "┼".join("─" * self.width for _ in names) + "┼" + "─" * 16 + "┼" + "─" * 8)

    def render_row(self, r0):
        r1 = r0 + self.row
        cells, prompts = [], []
        for i, s in enumerate(self.sessions):
            if s["last"] <= r0 or s["first"] >= r1:
                cells.append(self.cell("off", ""))
                continue
            time_in = {"work": 0.0, "wait": 0.0}
            label = None
            for j, b in enumerate(s["blocks"]):
                ov = overlap(r0, r1, b["start"], b["end"])
                if ov > 0 or (b["start"] == b["end"] and r0 <= b["start"] < r1):
                    time_in[b["kind"]] += ov
                if b["kind"] == "work" and r0 <= b["start"] < r1:
                    prompts.append(i + 1)
                if label is None and (i, j) not in self.shown and b["start"] < r1 and b["end"] > r0:
                    if b["kind"] == "work":
                        label = (b["kind"], f"▶ {b['prompt'] or '(prompt)'}", j)
                    elif b["end"] - b["start"] >= self.row:
                        label = (b["kind"], f"waits {(b['end'] - b['start']) / 60:.0f}m", j)
            kind = max(time_in, key=time_in.get) if any(time_in.values()) else "wait"
            text = ""
            if label and label[0] == kind:
                text = label[1]
                self.shown.add((i, label[2]))
            cells.append(self.cell(kind, text))
        human = ""
        if prompts:
            human = "→ " + ",".join(str(p) for p in sorted(set(prompts)))
            human = f"{HUMAN}{(' ' + human)[:16].ljust(16)}{RESET}" if self.color else (" " + human)[:16].ljust(16)
        else:
            human = " " * 16
        waiting = sum(1 for s in self.sessions for b in s["blocks"]
                      if b["kind"] == "wait" and b["start"] <= r1 < b["end"])
        print(f"{hhmm(r0):>11} │" + "│".join(cells) + f"│{human}│{waiting:8d}")


def summary(sessions, t0, t1):
    T = t1 - t0
    finished, prompted, waits, backlog = 0, 0, [], 0.0
    print(f"\n=== {len(sessions)} agents, window {hhmm(t0)} → {hhmm(t1)} UTC ({T / 3600:.1f} h)")
    for i, s in enumerate(sessions):
        work = sum(overlap(t0, t1, b["start"], b["end"]) for b in s["blocks"] if b["kind"] == "work")
        wait = sum(overlap(t0, t1, b["start"], b["end"]) for b in s["blocks"] if b["kind"] == "wait")
        open_ = overlap(t0, t1, s["first"], s["last"])
        finished += sum(1 for b in s["blocks"] if b["kind"] == "work" and t0 <= b["end"] < t1
                        and b["boundary"] == "end_turn")
        prompted += sum(1 for b in s["blocks"] if b["kind"] == "work" and t0 <= b["start"] < t1)
        ws = [b["end"] - b["start"] for b in s["blocks"]
              if b["kind"] == "wait" and t0 <= b["end"] < t1 and b["end"] > b["start"]]
        waits += ws
        backlog += wait
        print(f"  agent {i + 1} ({s['id'][:8]}): open {open_ / 60:5.0f} min — working {work / max(open_, 1):4.0%}, "
              f"waiting for human {wait / max(open_, 1):4.0%}"
              + (f", median wait {statistics.median(ws) / 60:.1f} min" if ws else ""))
    print("Human:")
    print(f"  agent turns finished (queries for the human)  {finished / T * 3600:5.1f} /h  ({finished})")
    print(f"  prompts sent by the human (responses)         {prompted / T * 3600:5.1f} /h  ({prompted})")
    print(f"  agents waiting on the human, time-averaged    {backlog / T:5.2f}")
    if waits:
        print(f"  wait for the next prompt: median {statistics.median(waits) / 60:.1f} min, "
              f"mean {statistics.mean(waits) / 60:.1f} min, max {max(waits) / 60:.0f} min")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", default=DEFAULT_INPUT, type=Path, help="processed.json")
    ap.add_argument("--start", help="window start: ISO time or seconds after trace origin")
    ap.add_argument("--end", help="window end: ISO time or seconds after trace origin")
    ap.add_argument("--row-seconds", type=float, default=60, help="trace seconds per row (default 60)")
    ap.add_argument("--speedup", type=float, default=120,
                    help="playback speed vs. real time; 0 = instant (default 120: 1-min row every 0.5 s)")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    data, sessions = load(args.input)
    origin = ts(data["origin"])
    default = DEFAULT_WINDOW.get(args.input.parent.name)
    t0 = parse_time(args.start, origin) if args.start else ts(default[0]) if default else min(s["first"] for s in sessions)
    t1 = parse_time(args.end, origin) if args.end else ts(default[1]) if default else max(s["last"] for s in sessions)
    sessions = [s for s in sessions if s["last"] > t0 and s["first"] < t1]
    if not sessions:
        raise SystemExit("no sessions in that window")

    color = not args.no_color and sys.stdout.isatty()
    tl = Timeline(sessions, t0, t1, args.row_seconds, color)
    legend = ("blue = agent working (▶ prompt), blank = waiting for the human, dots = session not open"
              if color else "█ = agent working (▶ prompt), blank = waiting for the human, · = session not open")
    print(f"{args.input}  ·  {len(sessions)} agents  ·  {args.row_seconds:g}s per row at {args.speedup:g}x\n{legend}\n")
    tl.header()
    wall = time.monotonic()
    r0, k = t0, 0
    try:
        while r0 < t1:
            tl.render_row(r0)
            r0 += args.row_seconds
            k += 1
            if args.speedup > 0:
                delay = wall + k * args.row_seconds / args.speedup - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
    except KeyboardInterrupt:
        t1 = min(t1, r0)
    summary(sessions, t0, min(t1, r0))


if __name__ == "__main__":
    main()
