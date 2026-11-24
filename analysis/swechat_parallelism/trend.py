import sys, warnings
import numpy as np, pandas as pd
from collections import defaultdict
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

sess = pd.read_parquet(sys.argv[1], columns=["session_id", "user_id", "kind", "agent"])
sess = sess[(sess.kind == "") & sess.user_id.notna()]
conv = pd.read_parquet(sys.argv[2])
conv = conv[conv.session_id.isin(sess.session_id) & conv.timestamp.notna()]
conv["t"] = conv.timestamp.astype("int64") / 1e6
conv = conv.sort_values(["session_id", "t"])
user_of = dict(zip(sess.session_id, sess.user_id))

# drop forks: sessions of the same user with the identical first timestamp (keep one)
first = conv.groupby("session_id").t.min().rename("first").reset_index()
first["user"] = first.session_id.map(user_of)
keep = set(first.sort_values("session_id").drop_duplicates(["user", "first"]).session_id)
conv = conv[conv.session_id.isin(keep)]
print(f"sessions after dropping {len(first) - len(keep)} forks: {len(keep)}")

GAP = 15 * 60
iv_by_user = defaultdict(list)          # user -> [(start, end, sid)]
for sid, g in conv.groupby("session_id", sort=False):
    t = g.t.to_numpy(); s = t[0]
    for a, b in zip(t[:-1], t[1:]):
        if b - a > GAP:
            iv_by_user[user_of[sid]].append((s, a, sid)); s = b
    iv_by_user[user_of[sid]].append((s, t[-1], sid))

# sweep each user's intervals: per-session peak concurrency, and time at each level per month
peak, time_lvl = {}, defaultdict(float)  # time_lvl[(month, level)]
for u, iv in iv_by_user.items():
    ev = sorted([(a, 1, s) for a, b, s in iv] + [(b, -1, s) for a, b, s in iv], key=lambda x: (x[0], x[1]))
    open_, prev = {}, None
    for t, d, s in ev:
        n = len(open_)
        if prev is not None and n and t > prev:
            m = pd.Timestamp(prev, unit="s").strftime("%Y-%m")
            time_lvl[(m, min(n, 4))] += t - prev
        open_[s] = open_.get(s, 0) + d
        if open_[s] == 0: del open_[s]
        for x in open_: peak[x] = max(peak.get(x, 1), len(open_))
        peak.setdefault(s, 1)
        prev = t

df = first[first.session_id.isin(keep)].copy()
df["peak"] = df.session_id.map(peak).fillna(1).astype(int)
df["month"] = pd.to_datetime(df["first"], unit="s").dt.strftime("%Y-%m")
user_max = df.groupby("user").peak.max()
auto = set(user_max[user_max > 10].index)
print(f"excluding {len(auto)} likely-automation users (peak > 10): {sorted(auto)} ({df.user.isin(auto).sum()} sessions)")
df = df[~df.user.isin(auto)]
df["cat"] = df.peak.clip(upper=4)
df["par"] = (df.peak >= 2).astype(int)
df["par3"] = (df.peak >= 3).astype(int)
months = sorted(df.month.unique())
df["mi"] = df.month.map({m: i for i, m in enumerate(months)})

print("\n== Per month: sessions by peak # of the user's sessions engaged at once")
tab = pd.crosstab(df.month, df.cat)
tab.columns = ["1", "2", "3", "4+"]
pct = tab.div(tab.sum(1), axis=0)
out = pd.DataFrame({"sessions": tab.sum(1), "users": df.groupby("month").user.nunique()})
for c in tab.columns: out[c] = (pct[c] * 100).round(1).astype(str) + "%"
out["parallel (>=2)"] = (df.groupby("month").par.mean() * 100).round(1).astype(str) + "%"
tl = pd.Series(time_lvl).unstack(fill_value=0)
tl = tl[tl.index.isin(months)]
out["active time in parallel"] = ((1 - tl[1] / tl.sum(1)) * 100).round(1).astype(str) + "%"
print(out.to_string())

print("\n== Tests")
small = tab.sum(1) < 100
t2 = tab[~small]
chi2, p, dof, _ = stats.chi2_contingency(t2)
print(f"1) chi-square month x {{1,2,3,4+}} (months with >=100 sessions; treats sessions as independent): "
      f"chi2={chi2:.1f}, dof={dof}, p={p:.2g}")

for y, name in [("par", ">=2 (parallel)"), ("par3", ">=3")]:
    m = smf.logit(f"{y} ~ mi", df).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(df.user)[0]})
    b, se, pv = m.params["mi"], m.bse["mi"], m.pvalues["mi"]
    print(f"2) trend in P({name}) per month, SE clustered by user: odds x{np.exp(b):.3f}/month "
          f"(95% CI {np.exp(b - 1.96 * se):.3f}-{np.exp(b + 1.96 * se):.3f}), p={pv:.3g}")

# within-user: conditional logit with user fixed effects (users with variation in outcome)
from statsmodels.discrete.conditional_models import ConditionalLogit
for y, name in [("par", ">=2"), ("par3", ">=3")]:
    v = df.groupby("user")[y].transform(lambda s: s.nunique() > 1)
    d = df[v]
    cm = ConditionalLogit(d[y], d[["mi"]], groups=d.user).fit(disp=0)
    b, se = cm.params.iloc[0], cm.bse.iloc[0]
    print(f"3) within-user trend in P({name}) (user fixed effects, {d.user.nunique()} users, {len(d)} sessions): "
          f"odds x{np.exp(b):.3f}/month (95% CI {np.exp(b - 1.96 * se):.3f}-{np.exp(b + 1.96 * se):.3f}), p={cm.pvalues.iloc[0]:.3g}")

mid = months[len(months) // 2]
h = df.assign(half=np.where(df.month < mid, "early", "late")).groupby(["user", "half"]).par.agg(["mean", "size"]).unstack()
h = h[(h["size"]["early"] >= 5) & (h["size"]["late"] >= 5)]
diff = h["mean"]["late"] - h["mean"]["early"]
w = stats.wilcoxon(h["mean"]["late"], h["mean"]["early"], zero_method="zsplit")
print(f"4) paired, users with >=5 sessions both before and from {mid} (n={len(h)}): parallel-session share "
      f"median change {diff.median() * 100:+.1f} pts, mean {diff.mean() * 100:+.1f} pts, "
      f"{(diff > 0).sum()} up / {(diff < 0).sum()} down; Wilcoxon p={w.pvalue:.3g}")

# composition: when did users join, and do later joiners parallelize more?
fm = df.groupby("user").month.min().rename("cohort")
d2 = df.merge(fm, left_on="user", right_index=True)
print("\n== Parallel-session share by user cohort (month of user's first session) x session month")
ct = d2.pivot_table(index="cohort", columns="month", values="par", aggfunc="mean")
n = d2.pivot_table(index="cohort", columns="month", values="par", aggfunc="size")
print((ct.where(n >= 30) * 100).round(0).fillna("").astype(str).replace("", "·").to_string())
df.to_csv(sys.argv[3], index=False)
