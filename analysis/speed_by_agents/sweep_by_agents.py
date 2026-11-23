#!/usr/bin/env python3
"""How much does speeding up agents shorten a parallel SWE-chat run, by number of agents?

Same model as plot_speed_sweep.py (the partner's 2-agent analysis): every
session replays independently, human windows stay fixed, agent phases divide
by that agent's inference speed, and the run's overall time is the latest
session finish. Speeds are set per agent with no shared budget.

Per trace:
  one agent faster  plot_speed_sweep.analyze(): vary one agent (others 1x) over
                    the sweep; report the best agent's reduction at 2x and 5x and
                    with infinite speed, and how many agents reduce the overall
                    time at all (> 0.5% at 5x)
  all agents faster every agent at 2x / 5x / infinite speed
  why               the slowest session's agent share, and how far ahead of the
                    runner-up session it finishes (caps any one-agent gain)
  *_nostall         robustness: gaps > --stall-min minutes between consecutive agent
                    messages inside an agent phase (permission prompts, background
                    waits) are treated as fixed waiting instead of agent time

Inputs: traces/parallel_00[1-4] (the partner's 2-agent traces) and every
data/swechat/episodes/*.json from build_traces.py.

Usage: python analysis/speed_by_agents/sweep_by_agents.py [--max-span-h 24]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from plot_speed_sweep import DEFAULT_SPEEDS, analyze  # noqa: E402

SERIES = "#2a78d6"   # categorical slot 1 (dataviz reference palette)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def finish(totals, speeds):
    return max(t["start_offset"] + t["human_seconds"] + (0 if np.isinf(s) else t["agent_seconds"] / s)
               for t, s in zip(totals, speeds))


def stalled_seconds(session, gap):
    """Time inside agent phases where the agent sent nothing for > gap seconds (e.g. waiting on a
    tool-permission prompt or a background job): sum of (gap between consecutive agent messages)."""
    src = Path(session.get("source_file", ""))
    if not src.exists():  # partner traces record paths from her machine; use the copy in traces/<name>/raw
        src = next(ROOT.glob(f"traces/*/raw/{src.name}"), src)
    if not src.exists():
        raise FileNotFoundError(f"transcript for {session['session_id']} not found")
    when = {}
    for i, line in enumerate(src.read_text().splitlines(), 1):
        k = line.find('"timestamp":"')
        if k >= 0:
            when[i] = pd.Timestamp(line[k + 13:line.find('"', k + 13)]).timestamp()
    total = 0.0
    for b in session["blocks"]:
        if b["role"] != "agent":
            continue
        ts = [pd.Timestamp(b["start"]).timestamp()] + sorted(when[x] for x in b.get("source_lines", []) if x in when) \
            + [pd.Timestamp(b["end"]).timestamp()]
        total += sum(d for d in np.diff(ts) if d > gap)
    return total


def metrics(totals):
    """All reductions from the duration-sum formula (identical to the partner's replay, which
    plot_speed_sweep.analyze() asserts for every sweep point)."""
    n, base = len(totals), finish(totals, [1] * len(totals))
    red = lambda t: 100 * (1 - t / base)
    one = {s: [red(finish(totals, [s if j == i else 1 for j in range(n)])) for i in range(n)] for s in (2, 5, np.inf)}
    ends = sorted((t["start_offset"] + t["human_seconds"] + t["agent_seconds"], k) for k, t in enumerate(totals))
    slow = totals[ends[-1][1]]
    return {"overall_h": base / 3600,
            "one_best_2x": max(one[2]), "one_best_5x": max(one[5]), "one_best_inf": max(one[np.inf]),
            "one_mean_5x": np.mean(one[5]), "agents_that_matter_5x": sum(r > 0.5 for r in one[5]),
            "all_2x": red(finish(totals, [2] * n)), "all_5x": red(finish(totals, [5] * n)),
            "all_inf": red(finish(totals, [np.inf] * n)),
            "slowest_agent_share": 100 * slow["agent_seconds"] / base,
            "lead_over_runner_up": 100 * (ends[-1][0] - ends[-2][0]) / base}


def trace_metrics(path, gap):
    r = analyze(path, sorted(set(DEFAULT_SPEEDS + [1.0])))  # partner's sweep (asserts replay == formula)
    totals = r["sessions"]
    m = metrics(totals)
    for c in r["per_agent_sweeps"]:  # cross-check against the partner's numbers
        sw = {p["inference_speed"]: p["overall_seconds"] for p in c["sweep"]}
        assert abs(100 * (1 - sw[5] / r["fixed_speed_baseline_seconds"]) - metrics(totals)["one_best_5x"]) < 1e-6 \
            or sw[5] >= r["fixed_speed_baseline_seconds"] * (1 - m["one_best_5x"] / 100) - 1e-6
    sessions = {x["session_id"]: x for x in __import__("json").loads(Path(path).read_text())["sessions"]}
    adj = []
    for t in totals:
        st = stalled_seconds(sessions[t["session_id"]], gap)
        adj.append({**t, "agent_seconds": t["agent_seconds"] - st, "human_seconds": t["human_seconds"] + st})
    v = metrics(adj)
    name = r["datapoint"] if path.parent.parent.name == "traces" else path.stem
    return {"trace": name, "agents": len(totals), "source": "partner" if path.parent.parent.name == "traces" else "new",
            "agent_h_total": sum(t["agent_seconds"] for t in totals) / 3600,
            "stalled_share_of_agent": 100 * (1 - sum(t["agent_seconds"] for t in adj) / max(sum(t["agent_seconds"] for t in totals), 1)),
            **m, **{f"{k}_nostall": v[k] for k in ("one_best_5x", "all_5x", "slowest_agent_share")}}


def independent(paths, seed=0):
    """Drop traces that share a session with an already-kept trace (episodes of one user often
    reuse a long session). 4-agent traces first (scarcest), then 3, then 2; random within a size."""
    import json
    import random
    info = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        info.append((p, len(d["sessions"]), d.get("user", "partner:" + Path(p).parent.name),
                     {x["session_id"] for x in d["sessions"]}))
    random.Random(seed).shuffle(info)
    info.sort(key=lambda x: -x[1])
    kept, used = [], set()
    for p, n, user, ids in info:
        if ids & used:
            continue
        used |= ids
        kept.append((p, user))
    return kept


def plot(t, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 180})
    panels = [("one_best_5x", "Fastest single agent at 5× (others 1×)"), ("all_5x", "Every agent at 5×")]
    ks = sorted(t.agents.unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (col, title) in zip(axes, panels):
        for i, k in enumerate(ks):
            y = t.loc[t.agents == k, col].to_numpy()
            x = i + rng.uniform(-0.18, 0.18, len(y))
            ax.scatter(x, y, s=22, color=SERIES, alpha=0.55, linewidths=0)
            med = np.median(y)
            ax.plot([i - 0.28, i + 0.28], [med, med], color=INK, linewidth=2, solid_capstyle="round")
            ax.annotate(f"median {med:.0f}%", (i, med), xytext=(0, 5), textcoords="offset points", ha="center",
                        va="bottom", fontsize=9, color=INK, zorder=4,
                        bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.9))
        ax.set_xticks(range(len(ks)), [f"{k} agents\n(n={int((t.agents == k).sum())})" for k in ks])
        ax.set_xlim(-0.5, len(ks) - 0.1)
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Reduction in overall time (%)")
    axes[0].set_ylim(bottom=0)
    fig.suptitle("Speeding up agents in real SWE-chat parallel runs", x=0.06, ha="left", fontsize=14)
    fig.text(0.06, 0.01, "Each dot is one trace. Model: sessions replay independently, human windows fixed, "
             "agent phases ÷ speed; overall time = latest session finish.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    for ext in ("png", "svg"):
        fig.savefig(out / f"reduction_by_agents.{ext}")
    plt.close(fig)


def table(t, label):
    cols = ["overall_h", "stalled_share_of_agent", "one_best_2x", "one_best_5x", "one_best_inf", "one_mean_5x", "agents_that_matter_5x",
            "all_2x", "all_5x", "all_inf", "slowest_agent_share", "lead_over_runner_up",
            "one_best_5x_nostall", "all_5x_nostall"]
    g = t.groupby("agents")[cols].median().round(1)
    g.insert(0, "traces", t.groupby("agents").size())
    print(f"\n== {label}: medians by number of agents (reductions in % of overall time)")
    print(g.to_string())
    print("   means:  " + "  ".join(f"{k} agents: one-best 5x {v.one_best_5x.mean():.1f}%, all 5x {v.all_5x.mean():.1f}%"
                                     for k, v in t.groupby("agents")))
    for col in ("one_best_5x", "all_5x", "lead_over_runner_up", "one_best_5x_nostall", "all_5x_nostall"):
        groups = [g_[col].to_numpy() for _, g_ in t.groupby("agents")]
        kw = stats.kruskal(*groups)
        rho = stats.spearmanr(t.agents, t[col])
        print(f"   {col}: Kruskal-Wallis p={kw.pvalue:.2g}; Spearman vs agents rho={rho.statistic:+.2f} p={rho.pvalue:.2g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-span-h", type=float, default=24, help="robustness subset: traces whose overall time <= this")
    ap.add_argument("--stall-min", type=float, default=15,
                    help="robustness variant: gaps > this many minutes inside an agent phase count as fixed waiting")
    args = ap.parse_args()
    paths = sorted(ROOT.glob("traces/parallel_00[1-4]/processed.json")) + sorted((ROOT / "data/swechat/episodes").glob("*.json"))
    kept = independent(paths)
    print(f"{len(paths)} traces; {len(kept)} after dropping traces that share a session with another")
    rows = []
    for p, user in kept:
        try:
            rows.append({**trace_metrics(p, args.stall_min * 60), "user": user})
        except (ValueError, KeyError) as exc:
            print(f"skip {p.name}: {exc}")
    t = pd.DataFrame(rows).sort_values(["agents", "trace"])
    t.to_csv(OUT / "traces.csv", index=False)
    pd.set_option("display.width", 220)
    table(t, f"All traces ({len(t)})")
    table(t[t.overall_h <= args.max_span_h], f"Traces whose overall time <= {args.max_span_h:g} h")
    capped = t.sample(frac=1, random_state=0).groupby("user").head(2)
    table(capped, f"At most 2 traces per user ({capped.user.nunique()} users)")
    print("\n== Traces with the largest one-agent gains")
    print(t.sort_values("one_best_5x", ascending=False).head(8)[
        ["trace", "agents", "overall_h", "one_best_5x", "one_best_inf", "all_5x", "slowest_agent_share",
         "lead_over_runner_up"]].round(1).to_string(index=False))
    plot(t, OUT)
    print(f"\nwrote {OUT / 'traces.csv'} and reduction_by_agents.png/.svg")


if __name__ == "__main__":
    main()
