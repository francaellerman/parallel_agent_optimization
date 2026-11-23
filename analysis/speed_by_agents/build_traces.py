#!/usr/bin/env python3
"""Build SWE-chat traces with 2, 3 and 4 parallel Claude Code agents for speed sweeps.

Episodes come from the earlier concurrency scan (analysis/swechat_parallelism):
  3-4 agents  candidates.csv: one user's sessions engaged at once (15-min gap
              rule), peak 3 or 4, all Claude Code, exactly `peak` sessions
  2 agents    session_peaks.csv: two peak-2 sessions of one user whose
              [first event, first event + duration] spans overlap and that
              overlap no other session of that user
Each episode's transcripts are downloaded (gated dataset: token read from
~/.cache/huggingface/token, never printed), parsed with parse_swe_chat.py into
the same format as traces/parallel_00* (an episode is rejected if any session
lacks origin.kind=human prompt markers, which Claude Code records only from
late June 2026, so episodes are taken from --since onwards), forks (sessions that share their first
prompt timestamp) are dropped, and the episode is kept only if all of its
sessions are open at one common moment, so "N agents" is unambiguous.

Usage: python analysis/speed_by_agents/build_traces.py [--n2 60 --n3 0 --n4 0 (0 = all)] [--since 2026-06-22] [--max-mb 800]
"""

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from parse_swe_chat import parse_session, timestamp  # noqa: E402

DATA = ROOT / "data/swechat"
URL = "https://huggingface.co/datasets/SALT-NLP/SWE-chat/resolve/main/"


def pick_episodes(n2, n3, n4, seed, since):
    rng = random.Random(seed)
    s = pd.read_parquet(DATA / "sessions.parquet", columns=["session_id", "agent", "duration_seconds", "transcript_path"])
    cc = set(s.session_id[s.agent == "Claude Code"])
    cand = pd.read_csv(ROOT / "analysis/swechat_parallelism/candidates.csv")
    cand = cand[cand.all_claude_code & (cand.peak == cand.n_sessions) & (pd.to_datetime(cand.start) >= since)]
    eps = []
    for k, n in ((3, n3), (4, n4)):
        rows = cand[cand.peak == k].to_dict("records")
        rng.shuffle(rows)
        for r in rows[:n] if n else rows:
            eps.append({"agents": k, "user": r["user"], "sessions": r["sessions"].split(",")})
    # 2-agent pairs
    p = pd.read_csv(ROOT / "analysis/swechat_parallelism/session_peaks.csv").merge(
        s[["session_id", "duration_seconds"]], on="session_id")
    p["end"] = p["first"] + p.duration_seconds.fillna(0)
    p = p[p["first"] >= pd.Timestamp(since, tz="UTC").timestamp()]
    pairs = []
    for user, g in p.groupby("user"):
        g = g.sort_values("first").to_dict("records")
        for i, a in enumerate(g):
            if a["peak"] != 2 or a["session_id"] not in cc:
                continue
            over = [b for b in g if b is not a and b["first"] < a["end"] and b["end"] > a["first"]]
            if len(over) == 1 and over[0]["peak"] == 2 and over[0]["session_id"] in cc:
                b = over[0]
                others = [c for c in g if c is not a and c is not b and c["first"] < b["end"] and c["end"] > b["first"]]
                if not others and a["first"] < b["first"]:
                    pairs.append({"agents": 2, "user": user, "sessions": [a["session_id"], b["session_id"]]})
    rng.shuffle(pairs)
    eps += pairs[:n2]
    paths = dict(zip(s.session_id, s.transcript_path))
    return eps, paths


def download(sid, paths, token, budget):
    out = DATA / "transcripts" / f"{sid}.jsonl"
    if out.exists() and out.stat().st_size:
        return out
    rel = paths.get(sid) or f"transcripts/{sid}.jsonl"
    req = urllib.request.Request(URL + rel, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read()
    budget["bytes"] += len(body)
    if budget["bytes"] > budget["max"]:
        raise SystemExit(f"download budget exceeded ({budget['max'] / 1e6:.0f} MB); rerun with a larger --max-mb")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(body)
    return out


def build(ep, paths, token, budget):
    sessions = []
    for sid in ep["sessions"]:
        try:
            sessions.append(parse_session(download(sid, paths, token, budget)))
        except (ValueError, KeyError) as exc:
            return None, f"parse failed: {str(exc).split('jsonl:')[-1].strip()}"[:120]
    # drop forks: same first prompt timestamp -> keep the session with more blocks
    sessions.sort(key=lambda x: -len(x["blocks"]))
    kept, firsts = [], set()
    for x in sessions:
        key = x["blocks"][0]["end"]
        if key in firsts:
            continue
        firsts.add(key)
        kept.append(x)
    starts = [timestamp({"timestamp": x["start"]}) for x in kept]
    ends = [timestamp({"timestamp": x["end"]}) for x in kept]
    if len(kept) < 2 or max(starts) >= min(ends):
        return None, f"{len(kept)} sessions after fork removal; all-open overlap {max(starts) < min(ends)}"
    origin = min(starts)
    for x, st in zip(kept, starts):
        x["start_offset"] = (st - origin).total_seconds()
    return {"schema_version": 1, "time_unit": "seconds", "origin": origin.isoformat(),
            "agents": len(kept), "user": ep["user"], "sessions": kept}, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n2", type=int, default=60)
    ap.add_argument("--n3", type=int, default=0, help="0 = all")
    ap.add_argument("--n4", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--since", default="2026-06-22", help="first day with origin.kind prompt markers")
    ap.add_argument("--max-mb", type=float, default=800)
    args = ap.parse_args()
    token = (Path.home() / ".cache/huggingface/token").read_text().strip()
    eps, paths = pick_episodes(args.n2, args.n3, args.n4, args.seed, args.since)
    budget = {"bytes": 0, "max": args.max_mb * 1e6}
    out_dir = DATA / "episodes"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for i, ep in enumerate(eps):
        data, why = build(ep, paths, token, budget)
        name = f"{ep['agents']}sel_{i:03d}"
        if data:
            name = f"{data['agents']}agents_{i:03d}"
            (out_dir / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False) + "\n")
        log.append({"episode": name, "selected_as": ep["agents"], "agents": data["agents"] if data else None,
                    "user": ep["user"], "sessions": ",".join(ep["sessions"]), "skipped": why or ""})
        print(f"{name}: {'ok' if data else why}  ({budget['bytes'] / 1e6:.0f} MB downloaded)", flush=True)
    pd.DataFrame(log).to_csv(Path(__file__).resolve().parent / "episodes.csv", index=False)


if __name__ == "__main__":
    main()
