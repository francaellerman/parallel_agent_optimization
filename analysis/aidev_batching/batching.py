#!/usr/bin/env python3
"""Do people answer their parallel Copilot agents in batches ("sweeps")? (AIDev-pop)

A *result* is a copilot_work_finished[_failure] event; its *response* is the
driver's first action on that PR afterwards (their comment, review,
ready-for-review, merge or close), within 7 days. The driver is the account on
the PR's first copilot_work_started; responses are pooled across all of a
driver's repos (one person).

A *session* is a run of the driver's responses with gaps <= --gap minutes.
Batching measures:
  share_b3     share of a driver's responses made in sessions that touch >= 3 PRs
  null_b3      the same, if each result were answered independently: response
               delays are shuffled among the driver's results (finish times kept,
               so agents that finish together still count as together)
  excess       share_b3 - null_b3  (> 0: the person sweeps through results)

Also: for each session, the time since the person's previous session (away
time), the number of results waiting at the start, and how many it cleared; and
where an automated Copilot code review (copilot-pull-request-reviewer[bot])
landed between the result and the response.

Usage: python analysis/aidev_batching/batching.py [--gap 10] [--perms 20]
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
REVIEW_BOT = "copilot-pull-request-reviewer[bot]"


def load(data):
    c = duckdb.connect()
    c.sql("set enable_progress_bar=false")
    c.sql(f"""create temp table pr as select id pr_id, created_at::timestamptz created
              from '{data}/pull_request.parquet' where agent = 'Copilot'""")
    c.sql(f"""create temp table tl as select t.pr_id, t.created_at::timestamptz t, event, actor
              from '{data}/pr_timeline.parquet' t where pr_id in (select pr_id from pr) and t.created_at is not null""")
    c.sql("""create temp table drv as select pr_id, arg_min(actor, t) driver from tl
             where event = 'copilot_work_started' group by 1""")
    c.sql("delete from drv where driver like '%[bot]' or driver is null")
    fin = c.sql("""select pr_id, driver, t from tl join drv using (pr_id)
                   where event like 'copilot_work_finished%'""").df()
    act = c.sql(f"""
        select pr_id, driver, t, event from tl join drv using (pr_id)
        where actor = driver and event in ('commented', 'merged', 'closed', 'ready_for_review')
        union all
        select r.pr_id, driver, submitted_at::timestamptz, 'reviewed'
        from '{data}/pr_reviews.parquet' r join drv using (pr_id)
        where r."user" = driver and submitted_at is not null""").df()
    bot = c.sql(f"""select pr_id, submitted_at::timestamptz t from '{data}/pr_reviews.parquet'
                    where "user" = '{REVIEW_BOT}' and submitted_at is not null
                      and pr_id in (select pr_id from drv)""").df()
    for x in (fin, act, bot):
        x["t"] = (x.t - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    return fin, act, bot


def pair(fin, act, bot, max_wait=7 * 86400):
    """Each result -> the driver's first action on that PR after it."""
    fin = fin.sort_values("t")
    act = act.sort_values("t").rename(columns={"t": "r"})
    m = pd.merge_asof(fin, act[["pr_id", "r", "event"]], left_on="t", right_on="r", by="pr_id",
                      direction="forward", allow_exact_matches=True)
    m = m.dropna(subset=["r"])
    m = m[m.r - m.t <= max_wait]
    # several results answered by the same action (e.g. finish, restart, finish) -> keep the last
    m = m.sort_values("t").drop_duplicates(["pr_id", "r"], keep="last")
    b = bot.sort_values("t").rename(columns={"t": "bt"})
    m = pd.merge_asof(m.sort_values("t"), b, left_on="t", right_on="bt", by="pr_id", direction="forward")
    m["bot_between"] = m.bt.notna() & (m.bt <= m.r)
    m["delay"] = m.r - m.t
    return m.reset_index(drop=True)


def sessions(r, prs, gap):
    """Session id per response (r sorted)."""
    return np.r_[0, np.cumsum(np.diff(r) > gap)]


def share_b3(r, prs, gap):
    o = np.argsort(r)
    r, prs = r[o], prs[o]
    sid = sessions(r, prs, gap)
    df = pd.DataFrame({"s": sid, "p": prs})
    n_prs = df.groupby("s").p.transform("nunique").to_numpy()
    return (n_prs >= 3).mean(), (n_prs >= 2).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data/aidev")
    ap.add_argument("--gap", type=float, default=10, help="minutes between responses within one session")
    ap.add_argument("--perms", type=int, default=20)
    ap.add_argument("--min-responses", type=int, default=20)
    args = ap.parse_args()
    gap = args.gap * 60
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 220)

    fin, act, bot = load(args.data)
    m = pair(fin, act, bot)
    print(f"results answered by their driver: {len(m):,} (of {len(fin):,} results); "
          f"median delay {m.delay.median() / 60:.0f} min; drivers {m.driver.nunique():,}")

    # --- 1. per-driver batching vs. independent-response null --------------------------------
    rows = []
    for dv, g in m.groupby("driver"):
        if len(g) < args.min_responses or g.pr_id.nunique() < 5:
            continue
        t, d, p = g.t.to_numpy(), g.delay.to_numpy(), g.pr_id.to_numpy()
        obs3, obs2 = share_b3(t + d, p, gap)
        null = np.array([share_b3(t + rng.permutation(d), p, gap) for _ in range(args.perms)])
        rows.append(dict(driver=dv, responses=len(g), prs=g.pr_id.nunique(), share_b3=obs3, null_b3=null[:, 0].mean(),
                         share_b2=obs2, null_b2=null[:, 1].mean(), p_perm=(null[:, 0] >= obs3).mean(),
                         median_delay_min=np.median(d) / 60, month=pd.to_datetime(np.median(t), unit="s").strftime("%Y-%m")))
    u = pd.DataFrame(rows)
    u["excess_b3"] = u.share_b3 - u.null_b3
    pk = OUT.parent / "aidev_idle/units_driver.csv"
    if pk.exists():
        peaks = pd.read_csv(pk).groupby("driver").agg(peak=("peak", "max"), idle_frac=("idle_frac", "median"),
                                                      issue_share=("issue_share", "mean"))
        u = u.join(peaks, on="driver")
    u.to_csv(OUT / "drivers.csv", index=False)
    w = stats.wilcoxon(u.share_b3, u.null_b3)
    print(f"\n== 1. Batching: drivers with >= {args.min_responses} answered results (n={len(u):,}); session gap {args.gap:g} min")
    print(f"   share of responses in sessions touching >=3 PRs: observed median {u.share_b3.median():.0%} (mean {u.share_b3.mean():.0%}), "
          f"independent-response null median {u.null_b3.median():.0%} (mean {u.null_b3.mean():.0%})")
    print(f"   >=2 PRs: observed mean {u.share_b2.mean():.0%} vs null {u.null_b2.mean():.0%}")
    print(f"   drivers batching more than chance: {np.mean(u.excess_b3 > 0):.0%}  (significantly, permutation p<0.05: "
          f"{np.mean(u.p_perm < 0.05):.0%}); Wilcoxon observed vs null p={w.pvalue:.2g}")
    print("   excess by peak concurrent agents (median share_b3 / null / excess, drivers):")
    if "peak" in u:
        b = pd.cut(u.peak, [0, 1, 2, 3, 5, 10, np.inf], labels=["1", "2", "3", "4-5", "6-10", "11+"])
        print("   " + u.groupby(b, observed=True).agg(drivers=("driver", "size"), share_b3=("share_b3", "median"),
              null_b3=("null_b3", "median"), excess=("excess_b3", "median"),
              batching_sig=("p_perm", lambda p: (p < 0.05).mean())).round(2).to_string().replace("\n", "\n   "))

    # --- 2. sessions: away time, backlog, clearance -----------------------------------------
    srows = []
    for dv, g in m[m.driver.isin(u.driver)].groupby("driver"):
        g = g.sort_values("r")
        r = g.r.to_numpy()
        sid = sessions(r, None, gap)
        g = g.assign(s=sid)
        st = g.groupby("s").agg(start=("r", "min"), end=("r", "max"), prs=("pr_id", "nunique"))
        st["away_h"] = (st.start - st.end.shift()) / 3600
        # results finished but not yet answered at the session start
        st["waiting"] = [int(((g.t < s0) & (g.r >= s0)).sum()) for s0 in st.start]
        st["oldest_wait_h"] = [((s0 - g.t[(g.t < s0) & (g.r >= s0)]).max() / 3600) if ((g.t < s0) & (g.r >= s0)).any()
                               else 0 for s0 in st.start]
        st["driver"] = dv
        srows.append(st.dropna(subset=["away_h"]))
    s = pd.concat(srows)
    s = s[s.waiting >= 1]
    s["cleared"] = (s.prs / s.waiting).clip(upper=1)
    s.to_csv(OUT / "sessions.csv.gz", index=False)
    print(f"\n== 2. Response sessions with >= 1 result waiting (n={len(s):,}, {s.driver.nunique():,} drivers)")
    s["away_b"] = pd.cut(s.away_h, [0, 0.5, 1, 2, 4, 8, 16, np.inf],
                         labels=["10-30m", "30-60m", "1-2h", "2-4h", "4-8h", "8-16h", "16h+"])
    print("   " + s.groupby("away_b", observed=True).agg(
        sessions=("prs", "size"), mean_waiting=("waiting", "mean"), mean_prs_answered=("prs", "mean"),
        share_3plus=("prs", lambda x: (x >= 3).mean()), mean_cleared=("cleared", "mean"),
        median_oldest_wait_h=("oldest_wait_h", "median")).round(2).to_string().replace("\n", "\n   "))
    X = pd.DataFrame({"log_away": np.log(s.away_h), "log_waiting": np.log(s.waiting)})
    fit = sm.OLS(np.log(s.prs), sm.add_constant(X)).fit(cov_type="cluster",
                                                         cov_kwds={"groups": s.driver.astype("category").cat.codes})
    print(f"   log(PRs answered) ~ log(away) + log(waiting), SE clustered by driver: "
          f"away elasticity {fit.params.log_away:.3f} (p={fit.pvalues.log_away:.2g}), "
          f"waiting elasticity {fit.params.log_waiting:.3f} (p={fit.pvalues.log_waiting:.2g})")

    # --- 3. automated review between result and response -------------------------------------
    mm = m[m.driver.isin(u.driver)]
    print(f"\n== 3. Automated Copilot code review between result and response: {mm.bot_between.mean():.0%} of answered results")
    br = mm[mm.bot_between]
    print(f"   when present: review lands median {((br.bt - br.t) / 60).median():.0f} min after the result, "
          f"person responds median {((br.r - br.bt) / 60).median():.0f} min after the review "
          f"(total {(br.delay / 60).median():.0f} min vs {(mm[~mm.bot_between].delay / 60).median():.0f} min without)")
    by_m = mm.assign(month=pd.to_datetime(mm.t, unit="s").dt.strftime("%Y-%m")).query("month <= '2026-03'").groupby("month").bot_between.mean()
    print("   share with an automated review, by month: " + "  ".join(f"{k} {v:.0%}" for k, v in by_m.items()))

    # --- 4. trend and cost --------------------------------------------------------------------
    mm = mm.assign(month=pd.to_datetime(mm.r, unit="s").dt.strftime("%Y-%m"))
    mm = mm[mm.month <= "2026-03"]  # dataset cutoff; later responses come from re-fetched threads
    print("\n== 4. By month (responses of the drivers above)")
    rows = []
    for mo, g in mm.groupby("month"):
        obs, nul = [], []
        for dv, gd in g.groupby("driver"):
            if len(gd) < 5:
                continue
            t, d, p = gd.t.to_numpy(), gd.delay.to_numpy(), gd.pr_id.to_numpy()
            o3, _ = share_b3(t + d, p, gap)
            n3 = np.mean([share_b3(t + rng.permutation(d), p, gap)[0] for _ in range(5)])
            obs.append((o3, len(gd)))
            nul.append(n3)
        if obs:
            wts = np.array([n for _, n in obs])
            rows.append(dict(month=mo, drivers=len(obs), share_b3=np.average([o for o, _ in obs], weights=wts),
                             null_b3=np.average(nul, weights=wts), median_delay_min=(g.delay / 60).median()))
    tm = pd.DataFrame(rows).set_index("month")
    tm["excess"] = tm.share_b3 - tm.null_b3
    print("   " + tm.round(2).to_string().replace("\n", "\n   "))
    q = pd.qcut(u.excess_b3.rank(method="first"), 4, labels=["least batching", "q2", "q3", "most batching"])
    print("\n   cost: drivers by batching-excess quartile")
    print("   " + u.groupby(q, observed=True).agg(drivers=("driver", "size"), excess=("excess_b3", "median"),
          median_delay_min=("median_delay_min", "median"), median_peak=("peak", "median"),
          median_idle_frac=("idle_frac", "median"), issue_share=("issue_share", "median"))
          .round(2).to_string().replace("\n", "\n   "))
    rho = stats.spearmanr(u.excess_b3, u.median_delay_min)
    print(f"   Spearman excess vs median response delay: rho={rho.statistic:.2f} p={rho.pvalue:.2g}")


if __name__ == "__main__":
    main()
