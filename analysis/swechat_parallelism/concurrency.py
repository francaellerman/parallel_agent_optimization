import sys
import pandas as pd
from collections import Counter

df = pd.read_parquet(sys.argv[1], columns=["session_id","user_id","owner_id","agent","kind","created_at",
                                           "duration_seconds","prompt_count","transcript_path"])
n0 = len(df)
df = df[(df.kind == "") & df.user_id.notna() & df.duration_seconds.notna() & (df.duration_seconds > 0)].copy()
print(f"sessions: {n0} total -> {len(df)} normal, with user_id and duration, from {df.user_id.nunique()} users")
df["start"] = df.created_at.astype("int64") / 1e9
df["end"] = df.start + df.duration_seconds

def groups(sub):
    """Connected components of overlapping intervals; returns list of (members, peak, overlap_secs_by_level)."""
    ev = sorted([(s, 1, i) for i, s in zip(sub.index, sub.start)] + [(e, -1, i) for i, e in zip(sub.index, sub.end)],
                key=lambda x: (x[0], x[1]))  # ends before starts at equal time
    out, cur, members, peak, t_prev, level_time = [], 0, [], 0, None, Counter()
    for t, d, i in ev:
        if cur > 0 and t_prev is not None:
            level_time[cur] += t - t_prev
        cur += d
        if d == 1:
            members.append(i); peak = max(peak, cur)
        if cur == 0:
            out.append((members, peak, level_time)); members, peak, level_time = [], 0, Counter()
        t_prev = t
    return out

def report(df, cap=None):
    d = df.copy()
    if cap:
        d["end"] = d.start + d.duration_seconds.clip(upper=cap)
    peaks, sizes, user_peak, time_at = Counter(), Counter(), {}, Counter()
    for uid, sub in d.groupby("user_id"):
        for members, peak, lt in groups(sub):
            peaks[peak] += 1
            user_peak[uid] = max(user_peak.get(uid, 0), peak)
            time_at.update(lt)
    label = f"durations capped at {cap/3600:g}h" if cap else "raw durations"
    print(f"\n== {label}")
    tot = sum(peaks.values())
    print("  groups (sets of overlapping sessions) by peak # concurrent sessions:")
    for k in sorted(peaks): print(f"    {k:3d}: {peaks[k]:6d} groups ({peaks[k]/tot:6.1%})")
    up = Counter(user_peak.values()); tu = sum(up.values())
    print("  users by their max concurrency ever:")
    for k in sorted(up): print(f"    {k:3d}: {up[k]:5d} users ({up[k]/tu:6.1%})")
    tt = sum(time_at.values())
    print("  share of session-time spent at each concurrency level:")
    for k in sorted(time_at)[:10]: print(f"    {k:3d}: {time_at[k]/tt:6.1%}")
    return d

report(df)
report(df, cap=4*3600)
report(df, cap=3600)
