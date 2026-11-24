#!/usr/bin/env python3
"""Replace SWE-chat user ids with stable codes (u0001, u0002, ...) in CSVs before committing.

SWE-chat is a gated dataset and some user ids are personal emails or machine
hostnames, so committed CSVs carry codes instead. The id -> code mapping lives
in the git-ignored data/swechat/user_codes.json, so codes stay the same across
re-runs and new ids get new codes. Values that are not SWE-chat user ids
(e.g. "partner:parallel_002") are left alone.

Usage: python analysis/swechat_parallelism/pseudonymize.py [csv ...]
       (default: the committed CSVs that have a `user` column)
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "data/swechat/user_codes.json"
DEFAULT = ["analysis/swechat_parallelism/session_peaks.csv", "analysis/swechat_parallelism/candidates.csv",
           "analysis/speed_by_agents/episodes.csv", "analysis/speed_by_agents/traces.csv"]


def is_code(v):
    return isinstance(v, str) and len(v) == 5 and v[0] == "u" and v[1:].isdigit()


def main():
    files = [ROOT / f for f in (sys.argv[1:] or DEFAULT)]
    codes = json.loads(MAP.read_text()) if MAP.exists() else {}
    frames = {f: pd.read_csv(f) for f in files if f.exists()}
    ids = sorted({v for df in frames.values() if "user" in df for v in df["user"].dropna().astype(str)
                  if not is_code(v) and not v.startswith("partner:")})
    for v in ids:
        codes.setdefault(v, f"u{len(codes) + 1:04d}")
    MAP.parent.mkdir(parents=True, exist_ok=True)
    MAP.write_text(json.dumps(codes, indent=1, sort_keys=True) + "\n")
    for f, df in frames.items():
        if "user" not in df:
            continue
        df["user"] = df["user"].map(lambda v: codes.get(str(v), v) if pd.notna(v) else v)
        df.to_csv(f, index=False)
        print(f"{f.relative_to(ROOT)}: {df['user'].nunique()} users coded")


if __name__ == "__main__":
    main()
