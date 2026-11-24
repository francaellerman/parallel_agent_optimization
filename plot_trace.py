#!/usr/bin/env python3
"""Draw a whole parallel-agent trace (processed.json) as one image, with idle stretches compressed.

One row per agent session. Time runs left to right:
  solid bar     agent working (from the human's prompt to the end of its turn)
  gray bar      agent done, waiting for the human's next prompt
Any stretch longer than --gap-min minutes in which no agent is working is cut
down to a narrow break labelled with its real length, so a 12-hour trace with an
overnight pause still shows every turn. Everything else is drawn to scale.

Usage:
  python plot_trace.py traces/parallel_001/processed.json
  python plot_trace.py traces/parallel_004/processed.json --gap-min 5
  (default output: analysis/trace_timelines/<trace>.png and .svg)
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4 (dataviz reference palette)
WAIT = "#d6d5cf"
INK, INK2, GRID, SURFACE, BREAK = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb", "#efeee9"


def ts(v):
    return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()


def clock(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%H:%M")


def duration(sec):
    h, m = divmod(round(sec / 60), 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


def load(path):
    data = json.loads(Path(path).read_text())
    sessions = []
    for s in sorted(data["sessions"], key=lambda s: ts(s["start"])):
        work, wait, prompts = [], [], []
        for b in s["blocks"]:
            if b["role"] == "agent":
                work.append((ts(b["start"]), ts(b["end"]), b.get("boundary", "")))
            elif b.get("boundary") != "initial_prompt_unknown_duration" or ts(b["end"]) > ts(b["start"]):
                wait.append((ts(b["start"]), ts(b["end"])))
            if b["role"] == "human":
                prompts.append((ts(b["end"]), " ".join(b["text"].split())))
        sessions.append({"id": s["session_id"], "work": work, "wait": wait, "prompts": prompts})
    return sessions


class Axis:
    """Piecewise-linear time -> x map that squeezes long all-idle stretches."""

    def __init__(self, sessions, gap, gap_width):
        busy = sorted((a, b) for s in sessions for a, b, _ in s["work"])
        busy += [(t, t) for s in sessions for t, _ in s["prompts"]]
        busy.sort()
        merged = [list(busy[0])]
        for a, b in busy[1:]:
            if a - merged[-1][1] <= gap:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        self.t0 = min(a for s in sessions for a, _ in s["wait"] + [(w[0], w[1]) for w in s["work"]])
        self.t1 = max(b for s in sessions for _, b in s["wait"] + [(w[0], w[1]) for w in s["work"]])
        pad = min(gap / 4, 300)
        self.knots, self.breaks = [(self.t0, 0.0)], []   # (time, x) knots; breaks: (x0, x1, real seconds)
        x, t = 0.0, self.t0
        segs = [(max(self.t0, a - pad), min(self.t1, b + pad)) for a, b in merged]
        for a, b in segs:
            if a - t > gap:
                self.breaks.append((x, x + gap_width, a - t))
                x += gap_width
            else:
                x += a - t
            self.knots.append((a, x))
            x += b - a
            self.knots.append((b, x))
            t = b
        if self.t1 > t:
            if self.t1 - t > gap:
                self.breaks.append((x, x + gap_width, self.t1 - t))
                x += gap_width
            else:
                x += self.t1 - t
            self.knots.append((self.t1, x))
        self.width = x

    def __call__(self, t):
        k = self.knots
        if t <= k[0][0]:
            return k[0][1]
        for (ta, xa), (tb, xb) in zip(k, k[1:]):
            if t <= tb:
                return xa if tb == ta else xa + (xb - xa) * (t - ta) / (tb - ta)
        return k[-1][1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", type=Path, default=ROOT / "traces/parallel_001/processed.json")
    ap.add_argument("--gap-min", type=float, default=10, help="compress all-idle stretches longer than this (minutes)")
    ap.add_argument("--gap-width-min", type=float, default=4, help="drawn width of a compressed stretch (minutes)")
    ap.add_argument("--out", type=Path, help="output path without extension")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.edgecolor": GRID, "xtick.color": INK2,
                         "ytick.color": INK, "text.color": INK, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.dpi": 180})

    sessions = load(args.input)
    n = len(sessions)
    ax_map = Axis(sessions, args.gap_min * 60, args.gap_width_min * 60)
    name = args.input.parent.name if args.input.stem == "processed" else args.input.stem
    width_in = min(26, max(12, ax_map.width / 3600 * 3.2))
    fig = plt.figure(figsize=(width_in, 2.2 + 0.62 * n))
    top = 1 - 1.05 / fig.get_figheight()
    chart_h = (0.62 * n + 0.6) / fig.get_figheight()
    ax = fig.add_axes([0.09, top - chart_h, 0.89, chart_h - 0.04])

    rows = {i: n - 1 - i for i in range(n)}      # agent 1 on top
    for x0, x1, real in ax_map.breaks:
        ax.axvspan(x0 / 60, x1 / 60, color=BREAK, linewidth=0, zorder=0)
        for xx in (x0, x1):
            ax.axvline(xx / 60, color=INK2, linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)
        ax.text((x0 + x1) / 120, n - 0.38, f"{duration(real)}\nidle", ha="center", va="bottom", fontsize=8,
                color=INK2)
    for i, s in enumerate(sessions):
        y, col = rows[i], COLORS[i % len(COLORS)]
        for a, b in s["wait"]:
            ax.barh(y, (ax_map(b) - ax_map(a)) / 60, left=ax_map(a) / 60, height=0.34, color=WAIT, linewidth=0, zorder=2)
        for a, b, boundary in s["work"]:
            w = max((ax_map(b) - ax_map(a)) / 60, 0.15)
            ax.barh(y, w, left=ax_map(a) / 60, height=0.56, color=col, edgecolor=SURFACE, linewidth=0.8, zorder=3)
    ax.set_yticks([rows[i] for i in range(n)], [f"Agent {i + 1}\n{s['id'][:8]}" for i, s in enumerate(sessions)])
    ax.set_ylim(-0.55, n - 0.45)
    ax.set_xlim(0, ax_map.width / 60)
    ax.tick_params(axis="y", length=0)

    # clock-time ticks inside the drawn (uncompressed) segments
    ticks, labels = [], []
    step = 15 * 60 if ax_map.width < 3 * 3600 else 30 * 60
    t = (int(ax_map.t0) // step + 1) * step
    in_break = lambda x: any(b0 - 1 <= x <= b1 + 1 for b0, b1, _ in ax_map.breaks)
    while t < ax_map.t1:
        x = ax_map(t)
        if not in_break(x) and all(abs(x / 60 - p) > 4 for p in ticks):
            ticks.append(x / 60)
            labels.append(clock(t))
        t += step
    ax.set_xticks(ticks, labels)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("UTC (shaded stretches are compressed: no agent working)", color=INK2)

    first = datetime.fromtimestamp(ax_map.t0, timezone.utc)
    fig.text(0.09, 1 - 0.22 / fig.get_figheight(),
             f"{name}: {n} parallel agents, {first:%Y-%m-%d %H:%M} → "
             f"{datetime.fromtimestamp(ax_map.t1, timezone.utc):%Y-%m-%d %H:%M} UTC "
             f"({duration(ax_map.t1 - ax_map.t0)}, drawn as {duration(ax_map.width)})", fontsize=13, va="top")
    handles = [Patch(color=COLORS[i % len(COLORS)], label=f"agent {i + 1} working") for i in range(n)]
    handles += [Patch(color=WAIT, label="waiting for the human"), Patch(color=BREAK, label="compressed idle")]
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False, fontsize=9,
               bbox_to_anchor=(0.085, 1 - 0.5 / fig.get_figheight()))
    out = args.out or ROOT / "analysis/trace_timelines" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}")
    print(f"wrote {out}.png  ({len(ax_map.breaks)} compressed stretches, "
          f"{duration(ax_map.t1 - ax_map.t0)} drawn as {duration(ax_map.width)})")


if __name__ == "__main__":
    main()
