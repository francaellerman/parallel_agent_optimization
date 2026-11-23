#!/usr/bin/env python3
"""Draw a trace's replay at 1x next to the replay with one agent sped up.

Uses the partner's simulator (simulate_parallel.replay): every session replays
independently, human windows keep their recorded length, and the sped-up
agent's phases are divided by --speed. One row per agent, time from the start
of the trace; the dashed line is the overall finish (latest session end).

Usage:
  python analysis/speed_by_agents/plot_replay_compare.py data/swechat/episodes/2agents_118.json
  python analysis/speed_by_agents/plot_replay_compare.py <trace.json> --agent 2 --speed 5
  (default --agent: the agent whose speedup shortens the run most)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simulate_parallel import load_sessions, replay  # noqa: E402

AGENT, HUMAN = "#2a78d6", "#d6d5cf"      # series slot 1; neutral gray for fixed human windows
SAVED = "#eda100"                         # series slot 4: time saved
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def run(sessions, speeds):
    r = replay(sessions, inference_speeds=speeds, speedup=0, display=False)
    return r["events"], r["total_time"], {s["session_id"]: s["end"] for s in r["sessions"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--agent", type=int, help="1-based agent to speed up (default: the best one)")
    ap.add_argument("--speed", type=float, default=5.0)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "replays")
    args = ap.parse_args()

    sessions = load_sessions([args.trace])
    n = len(sessions)
    base_events, base_total, base_ends = run(sessions, [1.0] * n)
    if args.agent is None:
        totals = [run(sessions, [args.speed if j == i else 1.0 for j in range(n)])[1] for i in range(n)]
        args.agent = min(range(n), key=lambda i: totals[i]) + 1
    speeds = [args.speed if j == args.agent - 1 else 1.0 for j in range(n)]
    fast_events, fast_total, fast_ends = run(sessions, speeds)
    saved = 100 * (1 - fast_total / base_total)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK, "text.color": INK,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 180})
    ids = [s["session_id"] for s in sessions]
    unit = 60.0
    fig, axes = plt.subplots(2, 1, figsize=(11, 1.4 + 1.25 * n * 2), sharex=True)
    panels = [(axes[0], base_events, base_total, base_ends, "All agents at 1×  (as recorded)"),
              (axes[1], fast_events, fast_total, fast_ends,
               f"Agent {args.agent} at {args.speed:g}×, other agent{'s' if n > 2 else ''} at 1×")]
    for ax, events, total, ends, title in panels:
        for e in events:
            row = n - 1 - ids.index(e["session_id"])
            color = AGENT if e["role"] == "agent" else HUMAN
            width = (e["end"] - e["start"]) / unit
            if width <= 0:
                continue
            ax.barh(row, width, left=e["start"] / unit, height=0.56, color=color,
                    edgecolor=SURFACE, linewidth=1.0)
        for i, sid in enumerate(ids):
            row = n - 1 - i
            ax.plot([ends[sid] / unit], [row], marker="|", markersize=16, color=INK, mew=1.5)
            ax.annotate(f"{ends[sid] / unit:.0f} min", (ends[sid] / unit, row + 0.34),
                        xytext=(3, 0), textcoords="offset points", fontsize=8.5, color=INK2, va="bottom")
        ax.axvline(total / unit, color=INK, linestyle="--", linewidth=1.2)
        if total < base_total * 0.99:
            ax.axvspan(total / unit, base_total / unit, color=SAVED, alpha=0.18, linewidth=0)
            ax.annotate(f"{(base_total - total) / unit:.0f} min saved ({saved:.0f}%)",
                        ((total + base_total) / 2 / unit, n - 0.35), ha="center", fontsize=9, color=INK)
        ax.set_yticks(range(n), [f"Agent {n - k}{' (sped up)' if (n - k) == args.agent and ax is axes[1] else ''}"
                                 f"\n{ids[n - 1 - k][:8]}" for k in range(n)])
        ax.set_ylim(-0.6, n - 0.1)
        ax.set_title(f"{title}  —  run finishes at {total / unit:.0f} min", loc="left", fontsize=11)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[1].set_xlabel("Minutes since the first session started")
    axes[0].set_xlim(0, base_total / unit * 1.04)
    name = args.trace.stem if args.trace.stem != "processed" else args.trace.parent.name
    fig.suptitle(f"{name}: speeding up agent {args.agent} to {args.speed:g}× shortens the run by {saved:.1f}%",
                 x=0.06, ha="left", fontsize=13)
    fig.legend(handles=[Patch(color=AGENT, label="agent working (scales with speed)"),
                        Patch(color=HUMAN, label="human window (fixed length)"),
                        Patch(color=SAVED, alpha=0.35, label="time saved")],
               loc="upper right", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.99, 0.965))
    fig.text(0.06, 0.008, "Replayed with simulate_parallel.replay: sessions are independent, human windows keep their "
             "recorded length,\nthe sped-up agent's phases are divided by its speed.  | = session finish;  "
             "dashed line = overall finish (latest session).", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(args.out / f"{name}_agent{args.agent}_{args.speed:g}x.{ext}")
    plt.close(fig)
    print(f"{name}: agent {args.agent} at {args.speed:g}x -> {base_total / 60:.1f} -> {fast_total / 60:.1f} min ({saved:.1f}% saved)")
    for sid in ids:
        print(f"   {sid[:8]}: finishes {base_ends[sid] / 60:.1f} -> {fast_ends[sid] / 60:.1f} min")


if __name__ == "__main__":
    main()
