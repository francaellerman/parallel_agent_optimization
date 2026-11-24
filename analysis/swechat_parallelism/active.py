import sys
import numpy as np, pandas as pd
from collections import Counter, defaultdict

sess = pd.read_parquet(sys.argv[1], columns=["session_id","user_id","agent","kind","prompt_count","transcript_path"])
sess = sess[(sess.kind == "") & sess.user_id.notna()]
conv = pd.read_parquet(sys.argv[2])
conv = conv[conv.session_id.isin(sess.session_id) & conv.timestamp.notna()]
conv["t"] = conv.timestamp.astype("int64") / 1e6
conv = conv.sort_values(["session_id", "t"])
user_of = dict(zip(sess.session_id, sess.user_id))
agent_of = dict(zip(sess.session_id, sess.agent))
AGENT_TYPES = {"assistant_response", "assistant_thinking", "tool_use", "tool_result"}
GAP = 15 * 60

busy, engaged = defaultdict(list), defaultdict(list)   # user -> [(start, end, session)]
for sid, g in conv.groupby("session_id", sort=False):
    t, tt = g.t.to_numpy(), g.turn_type.to_numpy()
    u = user_of[sid]
    # engaged: all activity, split at gaps > GAP
    s = t[0]
    for a, b in zip(t[:-1], t[1:]):
        if b - a > GAP:
            engaged[u].append((s, a, sid)); s = b
    engaged[u].append((s, t[-1], sid))
    # busy: prompt -> last agent activity before next prompt
    start, last = None, None
    for ti, ty in zip(t, tt):
        if ty == "user_prompt":
            if start is not None and last is not None and last > start:
                busy[u].append((start, last, sid))
            start, last = ti, None
        elif ty in AGENT_TYPES and start is not None:
            last = ti
    if start is not None and last is not None and last > start:
        busy[u].append((start, last, sid))

def analyze(intervals, name, min_ep=300):
    level_time, ep_peaks, ep_peaks_long, user_max = Counter(), Counter(), Counter(), {}
    episodes = []
    for u, iv in intervals.items():
        ev = sorted([(a, 1, s) for a, b, s in iv] + [(b, -1, s) for a, b, s in iv], key=lambda x: (x[0], x[1]))
        open_ = Counter(); prev = None; ep = None
        for t, d, s in ev:
            n = len(open_)
            if prev is not None and n:
                level_time[n] += t - prev
                if ep: ep["time_at"][n] += t - prev
            open_[s] += d
            if open_[s] == 0: del open_[s]
            n2 = len(open_)
            if n2 >= 2 and ep is None:
                ep = {"user": u, "start": t, "sessions": set(open_), "peak": n2, "time_at": Counter()}
            elif ep is not None:
                ep["sessions"] |= set(open_); ep["peak"] = max(ep["peak"], n2)
                if n2 < 2:
                    ep["end"] = t; episodes.append(ep); ep = None
            prev = t
    for ep in episodes:
        dur = ep["end"] - ep["start"]
        ep_peaks[ep["peak"]] += 1
        if dur >= min_ep: ep_peaks_long[ep["peak"]] += 1
        user_max[ep["user"]] = max(user_max.get(ep["user"], 1), ep["peak"])
    users_all = {u for u in intervals}
    print(f"\n===== {name}")
    tot = sum(level_time.values())
    print("share of active time at each concurrency level: " +
          ", ".join(f"{k}: {v/tot:.1%}" for k, v in sorted(level_time.items())[:8]))
    print(f"parallel episodes (>=2 at once): {sum(ep_peaks.values())}; lasting >= {min_ep//60} min: {sum(ep_peaks_long.values())}")
    print(f"{'peak':>6} {'episodes':>9} {'>=5min':>8}")
    for k in sorted(ep_peaks):
        if k <= 10: print(f"{k:6d} {ep_peaks[k]:9d} {ep_peaks_long[k]:8d}")
    big = sum(v for k, v in ep_peaks.items() if k > 10); bigl = sum(v for k, v in ep_peaks_long.items() if k > 10)
    print(f"{'>10':>6} {big:9d} {bigl:8d}")
    um = Counter(user_max.get(u, 1) for u in users_all)
    print(f"users ({len(users_all)}) by max concurrency: " + ", ".join(f"{k}: {v}" for k, v in sorted(um.items()) if k <= 8)
          + f", >8: {sum(v for k, v in um.items() if k > 8)}")
    return episodes

ep_busy = analyze(busy, "AGENTS BUSY AT ONCE (prompt -> agent's last action)")
ep_eng = analyze(engaged, "SESSIONS ENGAGED AT ONCE (activity incl. human turns, split at 15-min gaps)")

# candidate datapoints for the backlog simulator: engaged episodes with peak 3-4,
# >= 20 min long, all sessions Claude Code (parse_swe_chat.py handles that format)
rows = []
for ep in ep_eng:
    if 3 <= ep["peak"] <= 4 and ep["end"] - ep["start"] >= 1200:
        agents = {agent_of[s] for s in ep["sessions"]}
        rows.append({"user": ep["user"], "peak": ep["peak"], "n_sessions": len(ep["sessions"]),
                     "minutes": round((ep["end"] - ep["start"]) / 60),
                     "min_at_peak": round(ep["time_at"][ep["peak"]] / 60, 1),
                     "all_claude_code": agents == {"Claude Code"}, "agents": ",".join(sorted(agents)),
                     "start": pd.Timestamp(ep["start"], unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M"),
                     "sessions": ",".join(sorted(ep["sessions"]))})
cand = pd.DataFrame(rows).sort_values(["peak", "min_at_peak"], ascending=False)
cand.to_csv(sys.argv[3], index=False)
print(f"\ncandidate 3-4 agent episodes (engaged, >=20 min): {len(cand)}, "
      f"of which all-Claude-Code: {cand.all_claude_code.sum()}, from {cand.user.nunique()} users")
print(cand.drop(columns=["sessions"]).head(15).to_string(index=False))
