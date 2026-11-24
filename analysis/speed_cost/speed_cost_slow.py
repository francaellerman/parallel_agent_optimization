#!/usr/bin/env python3
"""Like speed_cost.py, but agents may also run SLOWER (and cheaper) than standard.

Same run model and token costs as speed_cost.py. Speeds now range over [--s-min, limit]:
  * fitted curve   m(s) = r / (1 + r - s) for every s >= s_min. Below 1x this is the
                   same batching model run the other way (bigger batches: slower per
                   user, cheaper per token); the discount is capped at r / (1 + r).
  * menus          speed_cost.py menus plus a flex-style slow tier: 0.5x speed at 0.5x
                   price (OpenAI flex / Batch-API price; the speed is an assumption,
                   providers only say "slower").
For a target finish T every agent gets exactly the speed that finishes it at T
(clipped to s_min), so agents with slack run slow and cheap and can pay for the
bottleneck's speedup. By default the run may not finish later than recorded
(T <= T1); --allow-later also lets it trade time for money.

Usage: python analysis/speed_cost/speed_cost_slow.py [--s-min 0.25] [--allow-later]
"""

import argparse

import numpy as np
import pandas as pd

from speed_cost import MENUS, OUT, ROOT, TIERS, fit_r, load_trace

SLOW_TIER = (0.5, 0.5)


def optimise(agents, variant, V, r=None, menu=None, s_min=0.25, allow_later=False, slow=True):
    c = np.array([a["start"] + a["human"] + a["agent_total"] - a["A"][variant] for a in agents])
    A = np.array([a["A"][variant] for a in agents])
    C = np.array([a["cost"] for a in agents])
    T1 = float(np.max(c + A))
    lo_speed = s_min if slow else 1.0
    if menu is None:
        s_max = 1 + r
        T_lo = float(np.max(np.where(A > 0, c + A / (s_max * 0.9999), c)))
        T_hi = float(np.max(c + A / lo_speed)) if allow_later else T1
        Ts = np.unique(np.r_[np.linspace(min(T_lo, T1), T_hi, 4000), T1])
        gap = Ts[:, None] - c[None, :]
        need = np.where(A[None, :] > 0, A[None, :] / np.maximum(gap, 1e-9), 1.0)
        bad = ((A[None, :] > 0) & (gap <= 0)) | (need >= s_max)
        speeds = np.clip(need, lo_speed, None)
        mult = r / (1 + r - np.minimum(speeds, s_max * 0.9999))
        extra = ((mult - 1) * C[None, :]).sum(axis=1)
        obj = np.where(bad.any(axis=1), np.inf, V * Ts / 3600 + extra)
        k = int(np.argmin(obj))
        return T1, float(Ts[k]), float(extra[k]), speeds[k]
    tiers = sorted(menu + ([SLOW_TIER] if slow else []))
    price = dict(tiers)
    cands = {T1} | {float(ci + Ai / s) for ci, Ai in zip(c, A) for s, _ in tiers if Ai > 0}
    cands = [T for T in cands if allow_later or T <= T1 + 1e-9]
    best = None
    for T in cands:
        need = np.where(A > 0, A / np.maximum(T - c, 1e-9), 1.0)
        if ((A > 0) & (T <= c)).any():
            continue
        sp = [min((s for s, _ in tiers if s >= n - 1e-9), default=None) for n in need]
        if None in sp:
            continue
        sp = np.array(sp)
        extra = float(np.sum((np.array([price[s] for s in sp]) - 1) * C))
        obj = V * T / 3600 + extra
        if best is None or obj < best[0] - 1e-12:
            best = (obj, T, extra, sp)
    return T1, best[1], best[2], best[3]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--values", nargs="+", type=float, default=[30, 60, 120, 240])
    ap.add_argument("--s-min", type=float, default=0.25)
    ap.add_argument("--allow-later", action="store_true")
    ap.add_argument("--base-tps", type=float, default=60)
    args = ap.parse_args()
    pd.set_option("display.width", 230)
    r = fit_r(TIERS)
    print(f"fitted r={r:.2f}: 0.5x speed -> {r / (1 + r - 0.5):.2f}x price, 0.25x -> {r / (1 + r - 0.25):.2f}x, "
          f"floor {r / (1 + r):.2f}x; flex tier assumed {SLOW_TIER[0]}x speed at {SLOW_TIER[1]}x price")

    traces = pd.read_csv(ROOT / "analysis/speed_by_agents/traces.csv")
    rows = []
    for tr, n in zip(traces.trace, traces.agents):
        path = ROOT / ("traces/" + tr + "/processed.json" if tr.startswith("parallel") else "data/swechat/episodes/" + tr + ".json")
        agents = load_trace(path, tr, args.base_tps)
        bill = sum(a["cost"] for a in agents)
        for variant in ("phase", "nostall", "genonly"):
            for pricing, menu in [("curve fitted", None)] + [(k, m) for k, m in MENUS.items()]:
                for slow in (False, True):
                    for V in args.values:
                        T1, T, extra, sp = optimise(agents, variant, V, r=r if menu is None else None, menu=menu,
                                                    s_min=args.s_min, allow_later=args.allow_later, slow=slow)
                        rows.append({"trace": tr, "agents": n, "variant": variant, "pricing": pricing,
                                     "slowdown_allowed": slow, "value_per_h": V, "overall_h": T1 / 3600,
                                     "token_cost_1x": bill, "time_saved_h": (T1 - T) / 3600,
                                     "time_saved_pct": 100 * (T1 - T) / T1, "extra_inference_usd": extra,
                                     "net_saving_usd": V * (T1 - T) / 3600 - extra,
                                     "net_saving_pct_of_total": 100 * (V * (T1 - T) / 3600 - extra) / (V * T1 / 3600 + bill),
                                     "agents_faster": int(np.sum(sp > 1.0001)), "agents_slower": int(np.sum(sp < 0.9999)),
                                     "min_speed": float(np.min(sp)), "max_speed": float(np.max(sp))})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results_slow.csv", index=False)

    for label, sub in (("runs <= 24 h", res[res.overall_h <= 24]), ("all runs", res)):
        print(f"\n######## {label}: {sub.trace.nunique()} traces")
        for variant in ("phase", "nostall", "genonly"):
            for pricing in ("curve fitted", "Anthropic fast mode"):
                d = sub[(sub.variant == variant) & (sub.pricing == pricing)]
                k = ["trace", "value_per_h"]
                a = d[~d.slowdown_allowed].set_index(k)
                b = d[d.slowdown_allowed].set_index(k)
                j = b.join(a[["net_saving_usd", "time_saved_pct", "extra_inference_usd"]], rsuffix="_up")
                j["gain_usd"] = j.net_saving_usd - j.net_saving_usd_up
                j["gain_pct_bill"] = 100 * j.gain_usd / j.token_cost_1x
                g = j.reset_index().groupby(["value_per_h", "agents"])
                t = pd.DataFrame({
                    "n": g.size(),
                    "net_up_only": g.net_saving_usd_up.median(), "net_with_slow": g.net_saving_usd.median(),
                    "gain_usd": g.gain_usd.median(), "gain_%bill": g.gain_pct_bill.median(),
                    "time_saved%_up": g.time_saved_pct_up.median(), "time_saved%_slow": g.time_saved_pct.median(),
                    "extra_up": g.extra_inference_usd_up.median(), "extra_slow": g.extra_inference_usd.median(),
                    "agents_slowed": g.agents_slower.mean(), "min_speed": g.min_speed.median()}).round(1)
                print(f"\n== {variant} | {pricing}{' + flex 0.5x@0.5x' if pricing != 'curve fitted' else ''}"
                      f"  (medians per trace; gain = extra net saving from allowing slowdowns)")
                print(t.to_string())


if __name__ == "__main__":
    main()
