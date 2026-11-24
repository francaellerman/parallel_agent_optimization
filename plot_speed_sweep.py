#!/usr/bin/env python3
"""Vary one agent's speed; keep all other agents fixed (default: 1x).

Each input JSON is a separate experiment with any number of sessions.
Format: schema_version=1, time_unit="seconds", timezone-aware origin,
sessions=[{session_id, start_offset, blocks:[{role, text, duration}, ...]}].
Blocks alternate human/agent, starting with human; durations are seconds.
Human windows stay fixed; whole agent phases (including tools) scale.
The default sweep is 0.5x to 5x on a linear axis.
"""

import argparse
import csv
import json
import math
from pathlib import Path

from simulate_parallel import load_sessions, replay

ROOT = Path(__file__).resolve().parent
DEFAULT_SPEEDS = [0.5, 0.75, 0.9, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]


def analyze(path, speeds, align_starts=False, other_speed=1.0):
    sessions = load_sessions([path])
    if align_starts:
        for session in sessions:
            session["start_offset"] = 0
    totals = [{
        "session_id": s["session_id"], "start_offset": s["start_offset"],
        "agent_seconds": sum(b["duration"] for b in s["blocks"] if b["role"] == "agent"),
        "human_seconds": sum(b["duration"] for b in s["blocks"] if b["role"] == "human"),
        "human_windows": sum(b["role"] == "human" for b in s["blocks"]),
        "longest_human_window": max((b["duration"] for b in s["blocks"] if b["role"] == "human"), default=0),
    } for s in sessions]
    individual = []
    fixed_baseline = replay(sessions, inference_speed=other_speed, speedup=0, display=False)["total_time"]
    for varied, session in enumerate(totals):
        curve = []
        limit = max(t["start_offset"] + t["human_seconds"] +
                    (0 if j == varied else t["agent_seconds"] / other_speed)
                    for j, t in enumerate(totals))
        for speed in speeds:
            multipliers = [other_speed] * len(sessions)
            multipliers[varied] = speed
            result = replay(sessions, inference_speeds=multipliers, speedup=0, display=False)
            expected = max(t["start_offset"] + t["human_seconds"] + t["agent_seconds"] / s
                           for t, s in zip(totals, multipliers))
            if not math.isclose(result["total_time"], expected, abs_tol=1e-6):
                raise ValueError("Per-agent replay disagrees with the duration-sum formula")
            curve.append({"inference_speed": speed, "overall_seconds": result["total_time"],
                          "overall_hours": result["total_time"] / 3600,
                          "bottleneck_session": max(result["sessions"], key=lambda s: s["end"])["session_id"]})
        individual.append({"agent": varied + 1, "session_id": session["session_id"],
                           "other_speed": other_speed, "infinite_speed_floor_seconds": limit,
                           "sweep": curve})
    return {
        "datapoint": path.parent.name if path.stem == "processed" else path.stem,
        "input": str(path.resolve()), "align_starts": align_starts,
        "sessions": totals, "per_agent_sweeps": individual,
        "other_speed": other_speed, "fixed_speed_baseline_seconds": fixed_baseline,
    }



def draw(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#475569", "text.color": "#243343",
                         "axes.edgecolor": "#cbd5e1", "grid.color": "#e2e8f0", "savefig.dpi": 180})
    colors = ["#147d92", "#cc6d2a", "#735ab2", "#39945e"]
    columns = min(2, len(results))
    rows = math.ceil(len(results) / columns)
    starts = "All sessions start at zero." if results[0]["align_starts"] else "Observed start offsets retained."
    fig, axes = plt.subplots(rows, columns, figsize=(11, 4.2 * rows), squeeze=False)
    for ax, result in zip(axes.flat, results):
        for i, curve in enumerate(result["per_agent_sweeps"]):
            xs = [r["inference_speed"] for r in curve["sweep"]]
            ys = [r["overall_hours"] for r in curve["sweep"]]
            ax.plot(xs, ys, marker=["o", "s", "^", "D"][i % 4],
                    linestyle=["-", "--", "-.", ":"][i % 4], color=colors[i % len(colors)],
                    linewidth=2, markersize=4,
                    label=f"Vary Agent {curve['agent']} ({curve['session_id'][:8]})")
        ax.axvline(result["other_speed"], color="#94a3b8", linestyle=":", linewidth=1)
        ax.set_xscale("linear")
        if len(xs) > 1:
            ax.set_xlim(min(xs), max(xs))
        ticks = [x for x in [0.5, 1, 2, 3, 4, 5] if min(xs) <= x <= max(xs)]
        ax.set_xticks(ticks, [f"{x:g}×" for x in ticks])
        ax.set_ylim(bottom=0)
        ax.set_title(f"{result['datapoint']} · other agents held at {result['other_speed']:g}×", loc="left", fontsize=11)
        ax.set_xlabel("Speed of the varied agent")
        ax.set_ylabel("Overall completion time (hours)")
        ax.grid(axis="y", alpha=.7)
        ax.legend(frameon=False, fontsize=9)
    for ax in list(axes.flat)[len(results):]:
        ax.set_visible(False)
    fig.suptitle("Speed up one agent at a time", x=.07, ha="left", fontsize=16)
    fig.text(.07, .025, f"Each curve is the whole run's latest finish, not an individual agent's finish.\nHuman windows fixed. {starts} Whole agent phases scale, including tools.", fontsize=9, color="#64748b")
    fig.tight_layout(rect=(0, .09, 1, .95))
    for extension in ("png", "svg"):
        fig.savefig(output / f"per_agent_speed_sweep.{extension}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h", action="help", help="Show this help message and exit")
    parser.add_argument("inputs", nargs="*", type=Path, help="Formatted JSON files; one experiment per file (default: traces/parallel_*/processed.json)")
    parser.add_argument("--speeds", nargs="+", type=float, default=DEFAULT_SPEEDS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/speed_sweep")
    parser.add_argument("--align-starts", action="store_true", help="Start every session at zero instead of preserving observed offsets")
    parser.add_argument("--other-speed", type=float, default=1.0, help="Fixed speed of other agents in per-agent sweeps (default: 1)")
    args = parser.parse_args()
    if any(not math.isfinite(s) or s <= 0 for s in args.speeds):
        parser.error("Speeds must be finite and positive")
    if not math.isfinite(args.other_speed) or args.other_speed <= 0:
        parser.error("--other-speed must be finite and positive")
    speeds = sorted(set(args.speeds + [1.0, args.other_speed]))
    paths = args.inputs or sorted((ROOT / "traces").glob("parallel_*/processed.json"))
    if not paths:
        parser.error("No processed traces found; run parse_swe_chat.py --all")
    try:
        results = [analyze(path, speeds, args.align_starts, args.other_speed) for path in paths]
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    individual_rows = [{"datapoint": r["datapoint"], "varied_agent": c["agent"],
                        "session_id": c["session_id"], "other_speed": c["other_speed"], **row}
                       for r in results for c in r["per_agent_sweeps"] for row in c["sweep"]]
    with (args.output_dir / "per_agent_speed_sweep.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(individual_rows[0]))
        writer.writeheader()
        writer.writerows(individual_rows)
    draw(results, args.output_dir)
    print(f"Plots and numerical results: {args.output_dir}")


if __name__ == "__main__":
    main()
