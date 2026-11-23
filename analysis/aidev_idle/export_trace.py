#!/usr/bin/env python3
"""Export one person's Copilot PRs for one day of AIDev into a trace for visualize_aidev.py.

Each PR gets its agent runs, the times its finished results waited for the
human, and the human's responses (comments, reviews, ready-for-review,
merge/close), using the same definitions as idle.py.

Usage:
  python analysis/aidev_idle/export_trace.py --repo benbalter/benbalter.github.com \
      --driver benbalter --day 2025-12-09 --out traces/aidev/benbalter_2025-12-09.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

import idle


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat().replace("+00:00", "Z")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--driver", required=True, help="GitHub login that started the Copilot runs")
    ap.add_argument("--day", required=True, help="YYYY-MM-DD (UTC); PRs with a run starting that day")
    ap.add_argument("--data", type=Path, default=idle.ROOT / "data/aidev")
    ap.add_argument("--await-cap", type=float, default=168, help="hours")
    ap.add_argument("--run-cap", type=float, default=60, help="minutes")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pr, ev = idle.load(args.data)
    pr = pr[pr.repo_url.str.endswith("/repos/" + args.repo)]
    ev = ev[ev.pr_id.isin(pr.pr_id)]
    drivers = ev[ev.kind == "start"].dropna(subset=["driver"]).sort_values("t").groupby("pr_id").driver.first()
    mine = drivers[drivers == args.driver].index
    run, wait = idle.intervals(ev[ev.pr_id.isin(mine)], args.await_cap * 3600, args.run_cap * 60)
    d0 = pd.Timestamp(args.day, tz="UTC").timestamp()
    ids = sorted(run[(run.s >= d0) & (run.s < d0 + 86400)].pr_id.unique())
    if not ids:
        raise SystemExit("no runs by that person on that day")

    c = duckdb.connect()
    c.sql("set enable_progress_bar=false")
    idlist = ",".join(map(str, ids))
    meta = c.sql(f"""select id pr_id, number, title, html_url, merged_at, closed_at
                     from '{args.data}/pull_request.parquet' where id in ({idlist})""").df().set_index("pr_id")
    resp = c.sql(f"""
        select pr_id, created_at::timestamptz t, event, actor, coalesce(message, '') text
        from '{args.data}/pr_timeline.parquet'
        where pr_id in ({idlist}) and created_at is not null and (
          event in ('closed', 'merged') or
          (event in ('commented', 'ready_for_review', 'review_requested', 'head_ref_force_pushed',
                     'convert_to_draft') and actor = '{args.driver}'))
        union all
        select pr_id, submitted_at::timestamptz, 'reviewed:' || lower(state), "user", coalesce(body, '')
        from '{args.data}/pr_reviews.parquet'
        where pr_id in ({idlist}) and "user" = '{args.driver}' and submitted_at is not null
        order by 1, 2""").df()
    resp["t"] = (resp.t - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    flags = pr.set_index("pr_id").from_issue

    prs = []
    for i in ids:
        m = meta.loc[i]
        r = resp[resp.pr_id == i]
        prs.append({
            "pr_id": int(i), "number": int(m.number), "title": m.title, "url": m.html_url,
            "from_issue": bool(flags.get(i, False)), "merged": m.merged_at is not None,
            "closed_at": m.closed_at,
            "runs": [[iso(s), iso(e)] for s, e in run[run.pr_id == i][["s", "e"]].itertuples(index=False)],
            "waits": [[iso(s), iso(e)] for s, e in wait[wait.pr_id == i][["s", "e"]].itertuples(index=False)],
            "responses": [{"t": iso(t), "event": e, "actor": a, "text": " ".join(x.split())[:300]}
                          for t, e, a, x in r[["t", "event", "actor", "text"]].itertuples(index=False)],
        })
    out = {"schema": "aidev-trace-1", "dataset": "hao-li/AIDev v5 (AIDev-pop)", "repo": args.repo,
           "driver": args.driver, "day": args.day,
           "definitions": {"run": "copilot_work_started -> copilot_work_finished[_failure] or close, "
                                  f"capped at {args.run_cap:g} min",
                           "wait": "finish -> the driver's next response on the PR (their comment or review, "
                                   "the next run, merge/close), capped at "
                                   f"{args.await_cap:g} h"},
           "prs": prs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"{args.out}: {len(prs)} PRs, {sum(len(p['runs']) for p in prs)} runs")


if __name__ == "__main__":
    main()
