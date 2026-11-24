#!/usr/bin/env python3
"""Time saved vs. extra inference cost when agents run faster, for the SWE-chat parallel traces.

Model of a run (the partner's replay model, plot_speed_sweep.py): sessions replay
independently, human windows keep their recorded length, an agent's scalable time
divides by its speed s, and the run ends at the latest session finish T.

Price of speed. Each agent's token bill is multiplied by m(s):
  * fitted curve   m(s) = r / (1 + r - s)        (1 <= s < 1 + r)
      from a batching model: a decode step takes t0 + k*B for batch B, so per-user
      speed is 1/(t0 + k*B) and cost per token is (t0 + k*B)/B; r = k*B/t0 at the
      standard operating point. Cost rises without bound as speed nears the
      batch-of-one limit 1 + r. r is fitted to advertised speed tiers (TIERS).
  * menus          discrete offers: Anthropic fast mode {1x, 2.5x at 2x};
                   LithosAI DeepSeek V4.1 Flash {1x, 1.4x at 1.67x, 1.8x at 2.33x}
Token prices: current Claude list prices per model (PRICES), from the recorded usage
of each session (main thread and subagents). Speeding an agent multiplies all of
its token types.

Optimum. For a time value V ($/h), choose speeds s_i >= 1 to minimise
  V * T(s) + sum_i (m(s_i) - 1) * C_i
For any target finish T each agent needs s_i = A_i / (T - c_i) (or 1 if it already
finishes by T), so the search is over T only (fixed part c_i, scalable part A_i).

What scales with speed (variant):
  phase      whole agent phases scale (the partner's model)
  nostall    gaps > 15 min between consecutive agent messages inside a phase are fixed
  genonly    only output-token generation scales: output_tokens / BASE_TPS per phase

Usage: python analysis/speed_cost/speed_cost.py [--values 30 60 120 240] [--base-tps 60]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "analysis/speed_by_agents"))

# Advertised (speed multiple, price multiple) offers, relative to the provider's base tier.
TIERS = [
    ("LithosAI DeepSeek V4.1 Flash Fast (350+ vs 250+ tok/s, $1.00 vs $0.60)", 1.4, 1.00 / 0.60),
    ("LithosAI DeepSeek V4.1 Flash Ultra (450+ vs 250+ tok/s, $1.40 vs $0.60)", 1.8, 1.40 / 0.60),
    ("LithosAI GLM-5.3-Flash Ultra (450+ vs 250+ tok/s, $1.00 vs $0.50)", 1.8, 1.00 / 0.50),
    ("Anthropic / OpenAI fast mode (up to 2.5x output tok/s, 2x price)", 2.5, 2.0),
]
MENUS = {
    "Anthropic fast mode": [(1.0, 1.0), (2.5, 2.0)],
    "LithosAI tiers": [(1.0, 1.0), (1.4, 1.00 / 0.60), (1.8, 1.40 / 0.60)],
}
# $/1M tokens: input, output, cache read, cache write (5-min, 1.25x input). Opus 5.5 / Sonnet 5.5
# cache reads from the current price list; older models' cache reads at the standard 0.1x input.
PRICES = {
    "claude-opus-5-5": (4, 20, 0.20, 5.0), "claude-opus-5": (5, 25, 0.50, 6.25),
    "claude-opus-4-8": (5, 25, 0.50, 6.25), "claude-opus-4-7": (5, 25, 0.50, 6.25),
    "claude-opus-4-6": (5, 25, 0.50, 6.25), "claude-fable-5": (10, 50, 1.00, 12.5),
    "claude-fable-5-1": (10, 50, 1.00, 12.5), "claude-fable-5-2": (8, 40, 0.20, 10.0),
    "claude-sonnet-5-5": (2, 10, 0.10, 2.5), "claude-sonnet-5": (2, 10, 0.20, 2.5),
    "claude-sonnet-4-6": (3, 15, 0.30, 3.75), "claude-haiku-4-5": (1, 5, 0.10, 1.25),
}
DEFAULT_PRICE = PRICES["claude-opus-5"]  # unknown / non-Claude model ids


def fit_r(points):
    """Least squares on log price multiple for m(s) = r / (1 + r - s)."""
    def loss(log_r):
        r = np.exp(log_r)
        return sum((np.log(r / (1 + r - s)) - np.log(m)) ** 2 for _, s, m in points if 1 + r > s)
    res = minimize_scalar(loss, bounds=(np.log(0.6), np.log(20)), method="bounded")
    return float(np.exp(res.x))


def curve(r):
    return lambda s: np.where(s <= 1, 1.0, r / np.maximum(1 + r - s, 1e-12)), 1 + r


def resolve(src, trace):
    p = Path(src)
    return p if p.exists() else ROOT / "traces" / trace / "raw" / p.name


def session_cost(path):
    """Recorded token bill of one Claude Code session (main thread + subagents), $."""
    seen = {}
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("type") != "assistant":
            continue
        m = r.get("message", {})
        u = m.get("usage") or {}
        k = m.get("id") or r.get("uuid")
        cur = seen.setdefault(k, {"model": m.get("model"), "in": 0, "out": 0, "cr": 0, "cw": 0})
        for key, field in (("in", "input_tokens"), ("out", "output_tokens"),
                           ("cr", "cache_read_input_tokens"), ("cw", "cache_creation_input_tokens")):
            cur[key] = max(cur[key], u.get(field) or 0)
    total = 0.0
    for v in seen.values():
        pi, po, pr, pw = PRICES.get(v["model"], DEFAULT_PRICE)
        total += (v["in"] * pi + v["out"] * po + v["cr"] * pr + v["cw"] * pw) / 1e6
    return total


def stalled(session, src, gap=900):
    when = {}
    for i, line in enumerate(src.read_text().splitlines(), 1):
        k = line.find('"timestamp":"')
        if k >= 0:
            when[i] = pd.Timestamp(line[k + 13:line.find('"', k + 13)]).timestamp()
    out = []
    for b in session["blocks"]:
        if b["role"] != "agent":
            continue
        ts = [pd.Timestamp(b["start"]).timestamp()] + sorted(when[x] for x in b.get("source_lines", []) if x in when) \
            + [pd.Timestamp(b["end"]).timestamp()]
        out.append(sum(d for d in np.diff(ts) if d > gap))
    return out


def load_trace(path, name, base_tps):
    data = json.loads(Path(path).read_text())
    agents = []
    for x in data["sessions"]:
        src = resolve(x["source_file"], name)
        human = sum(b["duration"] for b in x["blocks"] if b["role"] == "human")
        phases = [b for b in x["blocks"] if b["role"] == "agent"]
        dur = np.array([b["duration"] for b in phases])
        st = np.array(stalled(x, src)) if len(phases) else np.zeros(0)
        gen = np.minimum(dur, np.array([b.get("output_tokens", 0) for b in phases]) / base_tps)
        agents.append({"session_id": x["session_id"], "start": x["start_offset"], "human": human,
                       "A": {"phase": dur.sum(), "nostall": (dur - st).sum(), "genonly": gen.sum()},
                       "agent_total": dur.sum(), "cost": session_cost(src)})
    return agents


def optimise(agents, variant, V, price, menu=None, s_max=np.inf):
    """Return (T1, T*, extra $, speeds) minimising V*T + extra inference cost."""
    c = np.array([a["start"] + a["human"] + a["agent_total"] - a["A"][variant] for a in agents])
    A = np.array([a["A"][variant] for a in agents])
    C = np.array([a["cost"] for a in agents])
    T1 = float(np.max(c + A))

    def plan(T):
        need = np.where(c + A > T, A / np.maximum(T - c, 1e-9), 1.0)
        if menu:
            speeds = np.array([min((s for s, _ in menu if s >= n - 1e-9), default=np.nan) for n in need])
            if np.isnan(speeds).any():
                return None
            mult = np.array([dict(menu)[s] for s in speeds])
        else:
            late = c + A > T
            if (T <= c[late]).any() or (need[late] >= s_max).any():
                return None  # an agent would need infinite speed or more than the batch-of-one limit
            speeds, mult = need, price(need)
        extra = float(np.sum((mult - 1) * C))
        return V * T / 3600 + extra, extra, speeds

    if not menu:  # vectorised search over target finish times
        lo = float(np.max(np.where(A > 0, c + A / (s_max * 0.9999), c)))
        Ts = np.r_[np.linspace(min(lo, T1), T1, 4000), T1]
        late = (c + A)[None, :] > Ts[:, None]
        gap = Ts[:, None] - c[None, :]
        need = np.where(late, A[None, :] / np.maximum(gap, 1e-9), 1.0)
        ok = ~((late & (gap <= 0)) | (late & (need >= s_max))).any(axis=1)
        extra = ((price(need) - 1) * C[None, :]).sum(axis=1)
        obj = np.where(ok, V * Ts / 3600 + extra, np.inf)
        k = int(np.argmin(obj))
        return T1, float(Ts[k]), float(extra[k]), need[k]
    cands = {T1} | {float(ci + Ai / s) for ci, Ai in zip(c, A) for s, _ in menu if Ai > 0}
    cands = [T for T in cands if T <= T1 + 1e-9]
    best = None
    for T in cands:
        p = plan(T)
        if p and (best is None or p[0] < best[0] - 1e-12):
            best = (p[0], T, p[1], p[2])
    _, T, extra, speeds = best
    return T1, T, extra, speeds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--values", nargs="+", type=float, default=[30, 60, 120, 240], help="$ per hour of time")
    ap.add_argument("--base-tps", type=float, default=60, help="output tokens/s at 1x for the genonly variant")
    args = ap.parse_args()
    pd.set_option("display.width", 230)

    r_fit = fit_r(TIERS)
    print("== Price of speed: m(s) = r / (1 + r - s)")
    for name, s, m in TIERS:
        print(f"   offer  {s:.1f}x speed at {m:.2f}x price   implied r={m * (s - 1) / (m - 1):.2f}   ({name})")
    curves = {"fitted": r_fit, "steep (r=1)": 1.0, "gentle (r=3)": 3.0}
    for k, r in curves.items():
        m, smax = curve(r)
        print(f"   {k:<13} r={r:.2f}: 1.5x -> {float(m(1.5)):.2f}x price, 2x -> {float(m(2.0)):.2f}x, "
              f"2.5x -> {float(m(2.5)) if 2.5 < smax else float('inf'):.2f}x; speed limit {smax:.2f}x")

    traces = pd.read_csv(ROOT / "analysis/speed_by_agents/traces.csv")
    rows = []
    for tr, n in zip(traces.trace, traces.agents):
        path = ROOT / ("traces/" + tr + "/processed.json" if tr.startswith("parallel") else "data/swechat/episodes/" + tr + ".json")
        agents = load_trace(path, tr, args.base_tps)
        base_cost = sum(a["cost"] for a in agents)
        for variant in ("phase", "nostall", "genonly"):
            pricings = [(f"curve {k}", None, curve(r)) for k, r in curves.items()] + \
                       [(k, menu, (None, np.inf)) for k, menu in MENUS.items()]
            for pname, menu, (mfun, smax) in pricings:
                for V in args.values:
                    T1, T, extra, speeds = optimise(agents, variant, V, mfun, menu, smax)
                    saved_h = (T1 - T) / 3600
                    rows.append({"trace": tr, "agents": n, "variant": variant, "pricing": pname, "value_per_h": V,
                                 "overall_h": T1 / 3600, "token_cost_1x": base_cost,
                                 "time_saved_h": saved_h, "time_saved_pct": 100 * (T1 - T) / T1,
                                 "extra_inference_usd": extra, "time_value_saved_usd": V * saved_h,
                                 "net_saving_usd": V * saved_h - extra,
                                 "net_saving_pct_of_total": 100 * (V * saved_h - extra) / (V * T1 / 3600 + base_cost),
                                 "agents_sped_up": int(np.sum(np.asarray(speeds) > 1.0001)),
                                 "max_speed": float(np.max(speeds))})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results.csv", index=False)

    base = res[(res.variant == "phase")].drop_duplicates("trace")
    print(f"\n== Traces: {base.trace.nunique()}  (2 agents: {(base.agents == 2).sum()}, 3: {(base.agents == 3).sum()}, "
          f"4: {(base.agents == 4).sum()})")
    print(f"   recorded token cost per trace at 1x: median ${base.token_cost_1x.median():.0f}, "
          f"mean ${base.token_cost_1x.mean():.0f}; "
          f"median ${(base.token_cost_1x / (base.overall_h)).median():.0f} per hour of overall run time")

    def summary(sub, label):
        g = sub.groupby(["value_per_h", "agents"])
        t = pd.DataFrame({
            "traces": g.size(),
            "worth_speeding": g.apply(lambda d: (d.agents_sped_up > 0).mean()),
            "med_time_saved_pct": g.time_saved_pct.median(),
            "med_time_saved_min": g.time_saved_h.median() * 60,
            "med_extra_usd": g.extra_inference_usd.median(),
            "med_net_usd": g.net_saving_usd.median(),
            "mean_net_usd": g.net_saving_usd.mean(),
            "total_net_usd": g.net_saving_usd.sum(),
            "med_net_pct_total": g.net_saving_pct_of_total.median(),
            "med_max_speed": g.apply(lambda d: d.loc[d.agents_sped_up > 0, "max_speed"].median()),
        }).round(2)
        print(f"\n== {label}")
        print(t.to_string())

    summary(res[(res.variant == "phase") & (res.pricing == "curve fitted")],
            f"Fitted price curve (r={r_fit:.2f}), whole agent phases scale (partner's model)")
    summary(res[(res.variant == "phase") & (res.pricing == "Anthropic fast mode")],
            "Anthropic fast mode menu {1x, 2.5x at 2x}, whole agent phases scale")
    print("\n== Sensitivity: total net saving across all traces ($), by pricing and what scales")
    piv = res.pivot_table(index=["variant", "pricing"], columns="value_per_h", values="net_saving_usd", aggfunc="sum").round(0)
    print(piv.to_string())
    print("\n== Sensitivity: share of traces where speeding any agent pays off")
    piv2 = res.assign(sped=res.agents_sped_up > 0).pivot_table(index=["variant", "pricing"], columns="value_per_h",
                                                                 values="sped", aggfunc="mean").round(2)
    print(piv2.to_string())


if __name__ == "__main__":
    main()
