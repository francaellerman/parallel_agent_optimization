#!/usr/bin/env python3
"""How the share of issue-started Copilot coding-agent PRs changed over time (AIDev-pop).

Origin of each Copilot PR, from its description:
  issue    "Fixes/Closes/Resolves [owner/repo]#N" or an <issue_title> block in the
           "Original prompt" section (the issue was assigned to Copilot)
  prompt   an "Original prompt" section without <issue_title>: started from a typed
           prompt (agents panel / Copilot Chat / VS Code / Copilot CLI / API);
           sub-labelled vscode / cli when the description says "Created from ..."
  sub-PR   a follow-up PR Copilot opened from a comment on another PR
           ("Original PR: #N", "Triggering comment", "feedback on #N", "merges into #N")
  unknown  none of these. Before Oct 2025 Copilot did not record the prompt, so a
           non-issue PR has no marker. From Feb 2026 Copilot dropped the prompt
           section for most prompt-started PRs, and during March 2026 a new PR
           template rolled out that drops every marker, so March 2026 is only
           partly identifiable. Trend tests therefore use May 2025 - Feb 2026.
The driver is the account on the PR's first copilot_work_started event; bot
drivers (automation assigning issues) are reported separately and excluded
from the per-person tests.

Usage: python analysis/aidev_issue_trend/issue_trend.py [--data data/aidev]
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def load(data):
    c = duckdb.connect()
    c.sql("set enable_progress_bar=false")
    c.sql(f"""create temp table pr as
      select p.id pr_id, p.created_at::timestamptz created, p.merged_at is not null merged,
             p.state, r.full_name repo, r.stars,
             regexp_matches(body, '(?i)(fixes|closes|resolves)\\s+([\\w.-]+/[\\w.-]+)?#\\d+') fixes,
             body ilike '%<issue_title>%' issue_title,
             body ilike '%<summary>Original prompt</summary>%' orig_prompt,
             body ilike '%Created from%VS Code%' or body ilike '%Created from <a href="https://code%' vscode,
             body ilike '%Created from Copilot CLI%' cli,
             regexp_matches(body, '(?i)(Original PR: #|Triggering comment|feedback (from|on) (PR )?#\\d+|merges into #\\d+)') subpr,
             body ilike '%START COPILOT CODING AGENT%' old_template
      from '{data}/pull_request.parquet' p left join '{data}/repository.parquet' r on r.id = p.repo_id
      where agent = 'Copilot'""")
    c.sql(f"""create temp table runs as
      select pr_id, count(*) filter (where event = 'copilot_work_started') runs,
             arg_min(actor, created_at) filter (where event = 'copilot_work_started') driver
      from '{data}/pr_timeline.parquet' where pr_id in (select pr_id from pr) group by 1""")
    d = c.sql("select * from pr left join runs using (pr_id)").df()
    for col in ("fixes", "issue_title", "orig_prompt", "vscode", "cli", "subpr", "old_template"):
        d[col] = d[col].fillna(False).astype(bool)
    d["origin"] = np.select([d.fixes | d.issue_title, d.orig_prompt, d.subpr], ["issue", "prompt", "sub-PR"], "unknown")  # noqa
    d["via"] = np.select([d.vscode, d.cli], ["vscode", "cli"], "")
    d["month"] = d.created.dt.strftime("%Y-%m")
    d["bot"] = d.driver.fillna("").str.contains(r"\[bot\]$|^github-actions", regex=True)
    d["owner"] = d.repo.str.split("/").str[0]
    d["personal"] = d.owner == d.driver
    d["is_issue"] = (d.origin == "issue").astype(int)
    return d


def pct(x):
    return f"{100 * x:5.1f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data/aidev")
    args = ap.parse_args()
    d = load(args.data)
    d = d[d.month >= "2025-05"]  # one PR in 2025-03
    pd.set_option("display.width", 220)

    # --- 1. monthly origin mix -------------------------------------------------------------
    m = d.groupby("month").agg(prs=("pr_id", "size"), drivers=("driver", "nunique"), repos=("repo", "nunique"),
                               bot_prs=("bot", "mean"))
    mix = pd.crosstab(d.month, d.origin, normalize="index")
    via = pd.crosstab(d.month, d.via, normalize="index").drop(columns="", errors="ignore")
    h = d[~d.bot & d.driver.notna()]
    m["issue_all"] = mix["issue"]
    m["prompt"] = mix.get("prompt", 0)
    m["sub_pr"] = mix.get("sub-PR", 0)
    m["unknown"] = mix.get("unknown", 0)
    m["vscode"], m["cli"] = via.get("vscode", 0), via.get("cli", 0)
    m["issue_human_prs"] = h.groupby("month").is_issue.mean()
    per_driver = h.groupby(["month", "driver"]).is_issue.mean()
    m["issue_mean_driver"] = per_driver.groupby("month").mean()
    m["drivers_mostly_issue"] = (per_driver >= 0.5).groupby("month").mean()
    m["issue_old_template"] = d[d.old_template].groupby("month").is_issue.mean()
    print("== 1. Copilot PRs by month: origin mix (share of PRs) and issue share among human drivers")
    print(m.assign(**{k: m[k].map(pct) for k in m.columns if k not in ("prs", "drivers", "repos")}).to_string())
    m.to_csv(OUT / "monthly.csv")

    # --- 2. new vs returning drivers --------------------------------------------------------
    first = h.groupby("driver").month.min().rename("cohort")
    h = h.join(first, on="driver")
    h["cq"] = pd.PeriodIndex(h.cohort, freq="M").asfreq("Q").astype(str)
    h["q"] = pd.PeriodIndex(h.month, freq="M").asfreq("Q").astype(str)
    print("\n== 2. Issue share by first-use cohort (rows) and calendar quarter (cols), human drivers"
          " [PR-weighted; n PRs in brackets]")
    tab = h.groupby(["cq", "q"]).is_issue.agg(["mean", "size"])
    print(tab["mean"].unstack().map(lambda x: "" if pd.isna(x) else pct(x)).to_string())
    print(tab["size"].unstack().fillna(0).astype(int).to_string())
    new = h.month == h.cohort
    print("\n   issue share of PRs in a driver's first month vs. later months, by calendar month")
    nv = h.groupby(["month", new]).is_issue.mean().unstack()
    nv.columns = ["returning drivers", "drivers in first month"]
    print(nv.map(lambda x: "" if pd.isna(x) else pct(x)).to_string())

    # --- 3. tests -----------------------------------------------------------------------------
    h = h.copy()
    h["t"] = (h.created - pd.Timestamp("2025-05-01", tz="UTC")).dt.days / 30.44
    full = h
    h = h[h.month <= "2026-02"]  # reliable window (see docstring)
    print("\n== 3. Trend tests (human drivers, May 2025 - Feb 2026; t in months)")
    codes = h.driver.astype("category").cat.codes
    X = sm.add_constant(h[["t"]])
    fit = sm.Logit(h.is_issue, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": codes})
    ci = np.exp(fit.conf_int().loc["t"])
    print(f"   pooled logit, SE clustered by driver:   OR/month {np.exp(fit.params.t):.3f}  "
          f"95% CI {ci[0]:.3f}-{ci[1]:.3f}  p={fit.pvalues.t:.2g}")
    # within-driver: linear probability model with driver fixed effects (demeaned), clustered SE
    multi = h.groupby("driver").month.transform("nunique") >= 2
    hw = h[multi]
    y = hw.is_issue - hw.groupby("driver").is_issue.transform("mean")
    x = hw.t - hw.groupby("driver").t.transform("mean")
    fe = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": hw.driver.astype("category").cat.codes})
    print(f"   within-driver (driver fixed effects, linear prob.): {100 * fe.params.iloc[0]:+.2f} pts/month  "
          f"95% CI {100 * fe.conf_int().iloc[0, 0]:+.2f} to {100 * fe.conf_int().iloc[0, 1]:+.2f}  p={fe.pvalues.iloc[0]:.2g}  "
          f"({hw.driver.nunique():,} drivers active in >=2 months, {len(hw):,} PRs)")
    pooled_lpm = sm.OLS(h.is_issue, sm.add_constant(h.t)).fit(cov_type="cluster", cov_kwds={"groups": codes})
    print(f"   pooled linear prob. for comparison:                {100 * pooled_lpm.params.t:+.2f} pts/month")
    slopes = []
    for dv, g in hw.groupby("driver"):
        if len(g) >= 10 and g.t.std() > 0.5:
            slopes.append(np.polyfit(g.t, g.is_issue, 1)[0])
    slopes = np.array(slopes)
    print(f"   per-driver slopes (>=10 PRs over >=2 months, n={len(slopes)}): {np.mean(slopes < 0):.0%} down, "
          f"{np.mean(slopes > 0):.0%} up, {np.mean(slopes == 0):.0%} flat; sign test "
          f"p={stats.binomtest(int((slopes < 0).sum()), int((slopes != 0).sum())).pvalue:.2g}")
    early, late = h[h.month <= "2025-09"], h[(h.month >= "2025-12") & (h.month <= "2026-02")]
    a, b = early.groupby("driver").is_issue.agg(["mean", "size"]), late.groupby("driver").is_issue.agg(["mean", "size"])
    both = a[a["size"] >= 3].add_suffix("_e").join(b[b["size"] >= 3].add_suffix("_l"), how="inner")
    if len(both) > 5:
        w = stats.wilcoxon(both.mean_l, both.mean_e)
        print(f"   paired, same drivers (>=3 PRs in May-Sep 2025 and in Dec 2025-Feb 2026): n={len(both)}  "
              f"mean issue share {both.mean_e.mean():.0%} -> {both.mean_l.mean():.0%}  "
              f"(median change {np.median(both.mean_l - both.mean_e):+.0%}; {np.mean(both.mean_l < both.mean_e):.0%} decreased, "
              f"{np.mean(both.mean_l > both.mean_e):.0%} increased)  Wilcoxon p={w.pvalue:.2g}")
    cut = h.month <= "2025-09"
    ct = pd.crosstab(cut, h.is_issue)
    print(f"   chi-square May-Sep 2025 vs Oct 2025-Feb 2026 (PRs as independent, for reference): "
          f"chi2={stats.chi2_contingency(ct)[0]:.0f} p={stats.chi2_contingency(ct)[1]:.2g}")

    # decomposition: change in the PR-weighted share = within-driver change + mix change
    print("\n   shift-share decomposition, May-Sep 2025 vs Dec 2025-Feb 2026 (PR-weighted):")
    e_share, l_share = early.is_issue.mean(), late.is_issue.mean()
    stay = set(both.index)
    e_stay, l_stay = early[early.driver.isin(stay)], late[late.driver.isin(stay)]
    print(f"     overall {e_share:.0%} -> {l_share:.0%}; drivers active in both periods {e_stay.is_issue.mean():.0%} -> "
          f"{l_stay.is_issue.mean():.0%}; early-only drivers {early[~early.driver.isin(late.driver)].is_issue.mean():.0%}; "
          f"new drivers in late period {late[~late.driver.isin(early.driver)].is_issue.mean():.0%} "
          f"({(~late.driver.isin(early.driver)).mean():.0%} of late PRs)")

    # --- 4. who uses issues -----------------------------------------------------------------
    print("\n== 4. Issue share by context (human drivers), early (May-Sep 2025) vs late (Dec 2025-Feb 2026)")
    h = h.copy()
    h["period"] = np.select([h.month <= "2025-09", h.month >= "2025-12"], ["early", "late"], "mid")
    h["repo_kind"] = np.where(h.personal, "own repo (driver = owner)", "someone else's / org repo")
    h["stars_b"] = pd.cut(h.stars, [0, 500, 2000, 10000, np.inf], labels=["100-500", "500-2k", "2k-10k", "10k+"])
    vol = h.groupby("driver").size()
    h["driver_volume"] = pd.cut(h.driver.map(vol), [0, 4, 19, 99, np.inf], labels=["1-4 PRs", "5-19", "20-99", "100+"])
    for col in ["repo_kind", "stars_b", "driver_volume"]:
        t = h[h.period != "mid"].groupby([col, "period"], observed=True).is_issue.agg(["mean", "size"]).unstack()
        print(f"   by {col}:")
        t.columns = [f"{a_}_{b_}" for a_, b_ in t.columns]
        for p_ in ("early", "late"):
            t[f"mean_{p_}"] = t[f"mean_{p_}"].map(pct)
        print("     " + t[["mean_early", "mean_late", "size_early", "size_late"]]
              .to_string().replace("\n", "\n     "))

    # --- 5. does the workflow differ? -------------------------------------------------------
    print("\n== 5. Issue vs prompt vs sub-PR (Oct 2025-Jan 2026, when all are identifiable; human drivers)")
    late2 = full[(full.month >= "2025-10") & (full.month <= "2026-01") & (full.origin != "unknown")]
    closed = late2[late2.state == "closed"]
    g = late2.groupby("origin").agg(prs=("pr_id", "size"), runs_per_pr=("runs", "mean"),
                                    multi_run=("runs", lambda r: (r > 1).mean()))
    g["merged_of_closed"] = closed.groupby("origin").merged.mean()
    print(g.round(2).to_string())
    u = OUT.parent / "aidev_idle/units_driver.csv"
    if u.exists():
        ud = pd.read_csv(u)
        ud["issue_b"] = pd.cut(ud.issue_share, [-0.01, 0.2, 0.8, 1.0], labels=["mostly prompt (<20%)", "mixed", "mostly issue (>80%)"])
        print("\n   per driver (all months, >=5 PRs): parallelism and idle by how they start Copilot")
        print("   " + ud.groupby("issue_b", observed=True).agg(
            drivers=("prs", "size"), median_prs=("prs", "median"), median_peak=("peak", "median"),
            share_peak_ge3=("peak", lambda p: (p >= 3).mean()), median_idle_frac=("idle_frac", "median"))
            .round(2).to_string().replace("\n", "\n   "))
    d.drop(columns=["created"]).to_csv(OUT / "copilot_prs_origin.csv.gz", index=False)


if __name__ == "__main__":
    main()
