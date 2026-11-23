#!/usr/bin/env python3
"""Figure for speed_cost.py: the price-of-speed curve vs. offered tiers, and the optimal net saving.

Usage: python analysis/speed_cost/plot_speed_cost.py   (after speed_cost.py)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from speed_cost import TIERS, fit_r

OUT = Path(__file__).resolve().parent
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3 (dataviz reference palette)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 180})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    r = fit_r(TIERS)
    for rr, style, lab in ((1.0, ":", "r = 1"), (3.0, ":", "r = 3"), (r, "-", f"fitted r = {r:.2f}")):
        s = np.linspace(1, 1 + rr - 0.02, 300)
        m = rr / (1 + rr - s)
        a.plot(s, m, style, color=S1 if style == "-" else INK2, linewidth=2 if style == "-" else 1.2)
        a.annotate(lab, (s[-1], min(m[-1], 6)), xytext=(4, 0), textcoords="offset points", fontsize=8.5,
                   color=S1 if style == "-" else INK2, va="center")
    for name, s, m in TIERS:
        a.scatter([s], [m], s=46, color=S2, zorder=3, edgecolor=SURFACE, linewidth=2)
        if "fast mode" in name:
            short, off, ha = "Anthropic/OpenAI fast mode", (6, 8), "left"
        elif "GLM" in name:
            short, off, ha = "Lithos Ultra (GLM-5.3-Flash)", (8, -14), "left"
        elif "Ultra" in name:
            short, off, ha = "Lithos Ultra (DeepSeek V4.1 Flash)", (8, 4), "left"
        else:
            short, off, ha = "Lithos Fast (DeepSeek)", (8, -14), "left"
        a.annotate(short, (s, m), xytext=off, textcoords="offset points", fontsize=8, color=INK2, ha=ha)
    a.set_xlim(1, 4.4)
    a.set_ylim(0.8, 6)
    a.set_xlabel("Speed vs. standard (×)")
    a.set_ylabel("Price per token vs. standard (×)")
    a.set_title("Price of speed: m(s) = r / (1 + r − s)", loc="left", fontsize=11)
    a.grid(color=GRID, linewidth=0.8)
    a.set_axisbelow(True)

    res = pd.read_csv(OUT / "results.csv")
    d = res[(res.overall_h <= 24) & (res.pricing == "curve fitted")]
    for variant, color, label in (("phase", S1, "whole agent phases speed up"),
                                  ("nostall", S2, "…minus gaps > 15 min"),
                                  ("genonly", S3, "only token generation speeds up")):
        g = d[d.variant == variant].groupby("value_per_h").net_saving_usd.median()
        b.plot(g.index, g.values, marker="o", markersize=7, linewidth=2, color=color, label=label,
               markeredgecolor=SURFACE, markeredgewidth=2)
        b.annotate(f"${g.values[-1]:.0f}", (g.index[-1], g.values[-1]), xytext=(6, 0), textcoords="offset points",
                   fontsize=9, color=INK, va="center")
    b.set_xticks([30, 60, 120, 240], ["$30", "$60", "$120", "$240"])
    b.set_xlabel("Value of an hour of the run")
    b.set_ylabel("Net saving per trace, median ($)")
    b.set_title("Optimal speedups: time value saved − extra inference", loc="left", fontsize=11)
    b.legend(frameon=False, fontsize=9, loc="upper left")
    b.grid(axis="y", color=GRID, linewidth=0.8)
    b.set_axisbelow(True)
    b.set_ylim(bottom=0)
    n = d.trace.nunique()
    fig.text(0.06, 0.01, f"Right: {n} SWE-chat parallel traces with overall time ≤ 24 h, fitted price curve, recorded "
             "token usage at Claude list prices; any agent may be sped up independently.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"speed_cost.{ext}")


if __name__ == "__main__":
    main()
