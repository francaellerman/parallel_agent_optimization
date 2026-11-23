#!/usr/bin/env python3
"""When parallel Copilot agents leave the human with nothing to review (AIDev-pop).

For every Copilot coding-agent PR, the PR timeline is turned into two kinds of
intervals:

  running   copilot_work_started -> copilot_work_finished[_failure] (or close),
            capped at --run-cap minutes
  awaiting  copilot_work_finished -> the driver's next response on that PR
            (a comment or review by the driver -- the human who started the
            PR's first Copilot run -- or the next copilot_work_started, merge or
            close), capped at --await-cap hours (--responders anyone: any human's
            comment/review counts)

Per unit (a repository by default, or a repo + the human who drives the agent),
the intervals are swept to get R(t) = agents running and A(t) = agent results
waiting for the human. Then:

  peak      max R(t)                     (max concurrent agents)
  active    time with R >= 1             (some agent is working)
  idle      time with R >= 1 and A == 0  (agents are working, nothing to review)
  idle2     time with R >= 2 and A == 0  (several agents working, nothing to review)

idle / active is the share of agent-working time in which the human had
nothing to review; idle per PR normalises by volume.

Usage: python analysis/aidev_idle/idle.py [--data data/aidev] [--unit repo|driver]
       [--await-cap 168] [--issue-share 0.5] [--out analysis/aidev_idle/units_repo.csv]
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ISSUE_RE = r"(?i)(fixes|closes|resolves)\s+([\w.-]+/[\w.-]+)?#\d+"
BOTS = "('Copilot', 'CLAassistant', 'copilot-swe-agent')"
HUMAN = f"(actor not like '%[bot]' and actor not in {BOTS} and actor not ilike '%-bot')"


def load(data, responders="driver"):
    """PRs and timeline events. responders="driver": only the PR's driver (the human who started
    its first Copilot run) can end a wait by commenting/reviewing; "anyone": any human can."""
    c = duckdb.connect()
    c.sql("set enable_progress_bar=false")
    c.sql(f"""create temp table pr as
        select id pr_id, repo_id, repo_url, state, created_at::timestamptz created,
               coalesce(merged_at, closed_at)::timestamptz ended, merged_at is not null merged,
               (regexp_matches(body, '{ISSUE_RE}') or body ilike '%<issue_title>%') from_issue
        from '{data}/pull_request.parquet' where agent = 'Copilot'""")
    ev = c.sql(f"""
        select t.pr_id, t.created_at::timestamptz t,
          case when event = 'copilot_work_started' then 'start'
               when event like 'copilot_work_finished%' then 'finish'
               when event in ('closed', 'merged') then 'end'
               else 'human' end kind,
          case when event = 'copilot_work_started' then actor end driver, actor
        from '{data}/pr_timeline.parquet' t join pr using (pr_id)
        where t.created_at is not null and (
          event in ('copilot_work_started', 'copilot_work_finished',
                    'copilot_work_finished_failure', 'closed', 'merged')
          or (event in ('commented', 'ready_for_review', 'review_requested',
                        'head_ref_force_pushed', 'convert_to_draft') and {HUMAN}))
        union all
        select r.pr_id, submitted_at::timestamptz, 'human', null, r.user
        from '{data}/pr_reviews.parquet' r join pr using (pr_id)
        where user_type = 'User' and submitted_at is not null and r.user not in {BOTS}
        """).df()
    if responders == "driver":
        first = (ev[ev.kind == "start"].dropna(subset=["driver"]).sort_values("t")
                 .groupby("pr_id").driver.first())
        ev = ev[(ev.kind != "human") | (ev.actor == ev.pr_id.map(first))]
    return c.sql("select * from pr").df(), ev


ORDER = {"finish": 0, "start": 1, "human": 2, "end": 3}  # tie-break at equal timestamps


def intervals(ev, cap, run_cap=3600.0):
    """Per PR state machine -> running and awaiting intervals (epoch seconds)."""
    ev = ev.assign(o=ev.kind.map(ORDER), t=(ev.t - pd.Timestamp(0, tz="UTC")).dt.total_seconds())
    ev = ev.sort_values(["pr_id", "t", "o"])
    run, wait = [], []
    pid = ev.pr_id.to_numpy(); ts = ev.t.to_numpy(); kinds = ev.kind.to_numpy()
    cuts = np.flatnonzero(np.diff(pid)) + 1
    for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, len(pid)]):
        p = pid[lo]
        run_start = wait_start = None
        last = ts[hi - 1]
        for t, k in zip(ts[lo:hi], kinds[lo:hi]):
            if k == "start":
                if wait_start is not None:
                    wait.append((p, wait_start, min(t, wait_start + cap))); wait_start = None
                if run_start is None:
                    run_start = t
            elif k == "finish":
                if run_start is not None:
                    run.append((p, run_start, min(t, run_start + run_cap))); run_start = None
                    wait_start = t
            elif k in ("human", "end"):
                if wait_start is not None:
                    wait.append((p, wait_start, min(t, wait_start + cap))); wait_start = None
                if k == "end" and run_start is not None:
                    run.append((p, run_start, min(t, run_start + run_cap))); run_start = None
        if wait_start is not None:  # no response ever recorded: cap
            wait.append((p, wait_start, wait_start + cap))
        # a run with no finish and no close is dropped (unknown length)
    cols = ["pr_id", "s", "e"]
    return pd.DataFrame(run, columns=cols), pd.DataFrame(wait, columns=cols)


def sweep(run, wait):
    """Return peak R, time R>=1, R>=1&A=0, R>=2, R>=2&A=0 for one unit."""
    pts = np.concatenate([run.s, run.e, wait.s, wait.e])
    dr = np.concatenate([np.ones(len(run)), -np.ones(len(run)), np.zeros(2 * len(wait))])
    da = np.concatenate([np.zeros(2 * len(run)), np.ones(len(wait)), -np.ones(len(wait))])
    o = np.lexsort(((dr + da), pts))  # at equal times, ends (-1) before starts (+1)
    pts, dr, da = pts[o], dr[o], da[o]
    R, A = np.cumsum(dr), np.cumsum(da)
    dt = np.diff(pts)
    R, A = R[:-1], A[:-1]
    return {"peak": int(R.max(initial=0)),
            "active_h": dt[R >= 1].sum() / 3600, "idle_h": dt[(R >= 1) & (A == 0)].sum() / 3600,
            "par_h": dt[R >= 2].sum() / 3600, "idle2_h": dt[(R >= 2) & (A == 0)].sum() / 3600,
            "await_h": dt[A >= 1].sum() / 3600}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data/aidev")
    ap.add_argument("--unit", choices=["repo", "driver"], default="repo")
    ap.add_argument("--await-cap", type=float, default=168, help="hours; cap on an unanswered wait")
    ap.add_argument("--run-cap", type=float, default=60,
                    help="minutes; cap on one agent run (Copilot sessions time out at ~1 h; longer = missing finish)")
    ap.add_argument("--issue-share", type=float, default=0.5,
                    help="keep units where at least this share of Copilot PRs came from an issue")
    ap.add_argument("--responders", choices=["driver", "anyone"], default="driver")
    ap.add_argument("--min-prs", type=int, default=5)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    pr, ev = load(args.data, args.responders)
    drivers = (ev[ev.kind == "start"].dropna(subset=["driver"]).sort_values("t")
               .groupby("pr_id").driver.first())
    pr = pr.join(drivers, on="pr_id")
    run, wait = intervals(ev, args.await_cap * 3600, args.run_cap * 60)
    key = ["repo_url"] if args.unit == "repo" else ["repo_url", "driver"]
    pr = pr.dropna(subset=key)
    run, wait = (x.merge(pr[["pr_id"] + key], on="pr_id") for x in (run, wait))

    print(f"Copilot PRs {len(pr):,}; from an issue {pr.from_issue.mean():.0%}; "
          f"runs {len(run):,} (median {np.median(run.e - run.s) / 60:.0f} min); "
          f"waits {len(wait):,} (median {np.median(wait.e - wait.s) / 3600:.1f} h)")

    tup = lambda k: k if isinstance(k, tuple) else (k,)
    g_run = {tup(k): g for k, g in run.groupby(key)}
    g_wait = {tup(k): g for k, g in wait.groupby(key)}
    empty = run.iloc[:0]
    rows = []
    for k, prs in pr.groupby(key):
        k = tup(k)
        if len(prs) < args.min_prs or k not in g_run:
            continue
        r, w = g_run[k], g_wait.get(k, empty)
        rows.append({**dict(zip(key, k)), "prs": len(prs),
                     "runs": len(r), "issue_share": prs.from_issue.mean(), "merged": prs.merged.mean(),
                     **sweep(r, w)})
    u = pd.DataFrame(rows)
    u["idle_frac"] = u.idle_h / u.active_h
    u["idle_per_pr_min"] = u.idle_h * 60 / u.prs
    u["idle2_frac_of_par"] = u.idle2_h / u.par_h.replace(0, np.nan)
    if args.out:
        u.to_csv(args.out, index=False)

    for label, sub in [("all units", u), (f"issue-driven units (>= {args.issue_share:.0%} of PRs from issues)",
                                          u[u.issue_share >= args.issue_share])]:
        b = pd.cut(sub.peak, [0, 1, 2, 3, 4, 6, 10, np.inf], labels=["1", "2", "3", "4", "5-6", "7-10", "11+"])
        t = sub.groupby(b, observed=True).agg(
            units=("prs", "size"), prs=("prs", "sum"), runs=("runs", "sum"),
            active_h=("active_h", "sum"), idle_h=("idle_h", "sum"), par_h=("par_h", "sum"),
            idle2_h=("idle2_h", "sum"), med_idle_frac=("idle_frac", "median"),
            med_idle_per_pr=("idle_per_pr_min", "median"))
        t["pooled_idle_frac"] = t.idle_h / t.active_h
        t["idle2_share_of_par"] = t.idle2_h / t.par_h
        t["idle_min_per_pr"] = t.idle_h * 60 / t.prs
        print(f"\n== {label}: {len(sub):,} {args.unit} units with >= {args.min_prs} PRs, by peak concurrent agents")
        print(t[["units", "prs", "runs", "active_h", "idle_h", "pooled_idle_frac", "med_idle_frac",
                 "idle_min_per_pr", "med_idle_per_pr", "par_h", "idle2_share_of_par"]]
              .round(2).to_string())


if __name__ == "__main__":
    main()
