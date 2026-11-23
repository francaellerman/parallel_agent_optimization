#!/usr/bin/env python3
"""Live timeline of one person's parallel Copilot PRs (traces/aidev/*.json from export_trace.py).

Time flows down, one row per --row-minutes, played back at --speedup x real time.
Each column is a slot that one PR occupies from its first agent run until the
human last responds to it; when a PR is done, a later PR reuses the slot.

  blue    agent running   (first row: ▶ + PR key + the issue title, or ↻ + key +
                           the human's follow-up request on a later run)
  yellow  result waiting for the human to look at it (◆ + key)
  blank   slot free, or PR between a human response and the next run
  human   ↩ comment / ✓ review / ⇡ merge / ✗ close, with the PR key
  (a legend of PR keys, numbers and titles is printed first)
  run     agents running at the end of the row
  wait    results waiting on the human at the end of the row
  idle    minutes in the row where agents were running and nothing waited

Usage:
  python visualize_aidev.py                                   # benbalter, 2025-12-09
  python visualize_aidev.py traces/aidev/danmoseley_2026-03-22.json
  python visualize_aidev.py traces/aidev/jonathanpeppers_2026-03-15.json --row-minutes 5 --speedup 0
"""

import argparse
import re
import shutil
import sys
import time
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "traces/aidev/benbalter_2025-12-09.json"

RESET, DIM, BOLD = "\033[0m", "\033[2m", "\033[1m"
RUN, WAIT, HUMAN, IDLE = "\033[44;97m", "\033[43;30m", "\033[46;30m", "\033[41;97m"


def ts(v):
    return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()


def hhmm(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%H:%M")


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def short(text):
    """The human's request in a comment, without quoted text or the @copilot mention."""
    return re.sub(r"(?i)@copilot\S*\s*", "", text).strip()


def load(path, t0, t1):
    data = json.loads(Path(path).read_text())
    prs = []
    for p in data["prs"]:
        runs = [(ts(s), ts(e)) for s, e in p["runs"]]
        waits = [(ts(s), ts(e)) for s, e in p["waits"]]
        resp = [(ts(r["t"]), r["event"], r["text"]) for r in p["responses"]]
        prs.append({**p, "runs": runs, "waits": waits, "resp": resp})
    day0 = ts(data["day"] + "T00:00:00Z")
    first = min(s for p in prs for s, _ in p["runs"] if day0 <= s < day0 + 86400)
    last_run = max(e for p in prs for s, e in p["runs"] if day0 <= s < day0 + 86400)
    t0 = t0 if t0 is not None else first - 600
    t1 = t1 if t1 is not None else last_run + 3600
    keys = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" + "?" * 1000)
    for p in sorted(prs, key=lambda p: min(s for s, _ in p["runs"])):
        p["key"] = next(keys)
    for p in prs:
        # a PR holds its slot from its first run to its last run/wait/response in the window
        ivs = [iv for iv in p["runs"] + p["waits"] if iv[1] > t0 and iv[0] < t1]
        rs = [r[0] for r in p["resp"] if t0 <= r[0] < t1]
        if not ivs:
            p["span"] = None
            continue
        start = max(t0, min(s for s, _ in ivs))
        end = min(t1, max([e for _, e in ivs] + rs))
        p["span"] = (start, end)
        # label each run with what started it: the issue (first run) or the human's last comment
        p["run_labels"] = []
        for k, (s, _) in enumerate(p["runs"]):
            before = [r for r in p["resp"] if r[0] <= s and
                      (r[1] == "commented" or r[1].startswith("reviewed")) and short(r[2])]
            if k == 0:
                p["run_labels"].append(f"▶{p['key']} {p['title']}")
            else:  # a follow-up run: show the request that (most likely) started it
                p["run_labels"].append(f"↻{p['key']} {short(before[-1][2]) if before else '(follow-up)'}")
    prs = [p for p in prs if p["span"]]
    lanes = []  # greedy slot assignment by start time
    for p in sorted(prs, key=lambda p: p["span"][0]):
        for i, end in enumerate(lanes):
            if end <= p["span"][0]:
                p["lane"], lanes[i] = i, p["span"][1]
                break
        else:
            p["lane"] = len(lanes)
            lanes.append(p["span"][1])
    return data, prs, len(lanes), t0, t1


class View:
    def __init__(self, prs, nlanes, row, color):
        self.prs, self.n, self.row, self.color = prs, nlanes, row, color
        cols = shutil.get_terminal_size((160, 40)).columns
        self.side = 24 + 5 + 5 + 6
        self.w = max(8, min(22, (cols - 7 - self.side) // nlanes - 1))
        self.shown = set()

    def paint(self, style, text, w):
        text = (" " + text)[:w].ljust(w) if text else " " * w
        return f"{style}{text}{RESET}" if self.color and style else text

    def header(self):
        print(f"{'UTC':>5} │" + "│".join(f"slot {i + 1}".center(self.w) for i in range(self.n))
              + f"│{'human':^24}│{'run':>4} │{'wait':>4}│{'idle':>5}")
        print("─" * 6 + "┼" + "┼".join("─" * self.w for _ in range(self.n)) + "┼" + "─" * 24
              + "┼" + "─" * 5 + "┼" + "─" * 4 + "┼" + "─" * 5)

    def counts(self, t):
        r = sum(1 for p in self.prs for s, e in p["runs"] if s <= t < e)
        a = sum(1 for p in self.prs for s, e in p["waits"] if s <= t < e)
        return r, a

    def idle_seconds(self, r0, r1):
        pts = sorted({r0, r1} | {x for p in self.prs for iv in p["runs"] + p["waits"]
                                  for x in iv if r0 < x < r1})
        total = 0.0
        for a, b in zip(pts, pts[1:]):
            r, w = self.counts((a + b) / 2)
            if r >= 1 and w == 0:
                total += b - a
        return total

    def render(self, r0):
        r1 = r0 + self.row
        cells = [self.paint(None, "", self.w)] * self.n
        events = []
        for p in self.prs:
            s0, s1 = p["span"]
            if s1 <= r0 or s0 >= r1:
                continue
            run_t = sum(overlap(r0, r1, s, e) for s, e in p["runs"])
            wait_t = sum(overlap(r0, r1, s, e) for s, e in p["waits"])
            text = ""
            if run_t >= wait_t and run_t > 0:
                style = RUN
                for k, (s, e) in enumerate(p["runs"]):
                    if overlap(r0, r1, s, e) > 0 and (p["pr_id"], "r", k) not in self.shown:
                        self.shown.add((p["pr_id"], "r", k))
                        text = p["run_labels"][k]
                        break
                text = text or ("" if self.color else "█" * self.w)
            elif wait_t > 0:
                style = WAIT
                for k, (s, e) in enumerate(p["waits"]):
                    if overlap(r0, r1, s, e) > 0 and (p["pr_id"], "w", k) not in self.shown:
                        self.shown.add((p["pr_id"], "w", k))
                        text = f"◆{p['key']} ready"
                        break
                text = text or ("" if self.color else "░" * self.w)
            else:
                style, text = (DIM if self.color else None), "·"
            cells[p["lane"]] = self.paint(style, text, self.w)
            for t, ev, _ in p["resp"]:
                sym = {"commented": "↩", "merged": "⇡", "closed": "✗"}.get(
                    ev, "✓" if ev.startswith("reviewed") else None)
                if sym and r0 <= t < r1:
                    events.append((sym, p["key"], ev))
        # merged PRs also emit "closed"; show one symbol per PR, merge winning
        seen = {}
        for sym, num, ev in events:
            if num not in seen or sym == "⇡":
                seen[num] = sym
        human = " ".join(f"{sym}{key}" for key, sym in seen.items())
        human = self.paint(HUMAN if human else None, human, 24)
        r, a = self.counts(r1 - 1e-6)
        idle = self.idle_seconds(r0, r1) / 60
        idle_txt = f"{idle:4.0f}" if idle >= 0.5 else ""
        idle_cell = (f"{IDLE}{idle_txt:>5}{RESET}" if self.color and idle >= self.row / 120
                     else f"{idle_txt:>5}")
        print(f"{hhmm(r0):>5} │" + "│".join(cells) + f"│{human}│{r:4d} │{a:4d}│{idle_cell}")


def summary(view, t0, t1):
    T = t1 - t0
    active = idle = idle2 = 0.0
    pts = sorted({t0, t1} | {x for p in view.prs for iv in p["runs"] + p["waits"] for x in iv if t0 < x < t1})
    for a, b in zip(pts, pts[1:]):
        r, w = view.counts((a + b) / 2)
        active += (b - a) * (r >= 1)
        idle += (b - a) * (r >= 1 and w == 0)
        idle2 += (b - a) * (r >= 2 and w == 0)
    runs = [(s, e) for p in view.prs for s, e in p["runs"] if t0 <= s < t1]
    waits = [e - s for p in view.prs for s, e in p["waits"] if t0 <= s < t1]
    resp = [r for p in view.prs for r in p["resp"] if t0 <= r[0] < t1]
    srt = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0
    print(f"\n=== {len(view.prs)} PRs, {hhmm(t0)} → {hhmm(t1)} UTC ({T / 3600:.1f} h)")
    print(f"  agent runs started        {len(runs):4d}   median {srt([e - s for s, e in runs]) / 60:.0f} min")
    print(f"  driver's responses        "
          f"{sum(r[1] == 'commented' for r in resp)} comments, "
          f"{sum(r[1].startswith('reviewed') for r in resp)} reviews, {sum(r[1] == 'merged' for r in resp)} merges")
    print(f"  result waited for human   median {srt(waits) / 60:.0f} min")
    print(f"  some agent running        {active / 3600:5.1f} h")
    print(f"  ...with nothing to review {idle / 3600:5.1f} h  ({idle / max(active, 1):.0%} of agent-running time)")
    print(f"  >=2 running, nothing to review {idle2 / 3600:.1f} h")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", default=DEFAULT_INPUT, type=Path)
    ap.add_argument("--start", help="window start, ISO UTC (default: 10 min before the day's first run)")
    ap.add_argument("--end", help="window end, ISO UTC (default: 1 h after the day's last run)")
    ap.add_argument("--row-minutes", type=float, default=5)
    ap.add_argument("--speedup", type=float, default=600, help="0 = instant (default 600: 5-min row every 0.5 s)")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    data, prs, nlanes, t0, t1 = load(args.input, args.start and ts(args.start), args.end and ts(args.end))
    color = not args.no_color and sys.stdout.isatty()
    row = args.row_minutes * 60
    view = View(prs, nlanes, row, color)
    print(f"{data['repo']} · Copilot runs started by {data['driver']} · {data['day']} · "
          f"{len(prs)} PRs in {nlanes} slots · {args.row_minutes:g} min per row at {args.speedup:g}x")
    print("blue = agent running, yellow = result waiting for the human, · = PR open between response and "
          "next run;\nhuman: ↩ comment ✓ review ⇡ merge ✗ close (letter = PR key); idle = minutes with agents running and "
          "nothing to review\n")
    width = shutil.get_terminal_size((160, 40)).columns
    print("PRs (key: #number, outcome, title)")
    for p in sorted(prs, key=lambda p: p["key"]):
        outcome = "merged" if p["merged"] else "closed" if p["closed_at"] else "open"
        src = "" if p["from_issue"] else " [no issue]"
        print(f"  {p['key']}  #{p['number']:<7} {outcome:<6} {p['title']}{src}"[:width - 1])
    print()
    view.header()
    wall, r0, k = time.monotonic(), t0, 0
    try:
        while r0 < t1:
            view.render(r0)
            r0 += row
            k += 1
            if args.speedup > 0:
                delay = wall + k * row / args.speedup - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
    except KeyboardInterrupt:
        t1 = min(t1, r0)
    summary(view, t0, min(t1, r0))


if __name__ == "__main__":
    main()
