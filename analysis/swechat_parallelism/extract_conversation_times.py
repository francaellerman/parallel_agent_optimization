#!/usr/bin/env python3
"""Reduce SWE-chat's conversations.parquet (6.4 GB) to the columns the concurrency scans need.

Reads only session_id, timestamp and turn_type, which active.py and trend.py use
to find when each session was active.

Usage: python analysis/swechat_parallelism/extract_conversation_times.py \
           data/swechat/conversations.parquet data/swechat/conv_min.parquet
"""

import sys

import pyarrow.parquet as pq


def main():
    src, dst = sys.argv[1], sys.argv[2]
    table = pq.read_table(src, columns=["session_id", "timestamp", "turn_type"])
    pq.write_table(table, dst)
    print(f"{dst}: {table.num_rows:,} rows")


if __name__ == "__main__":
    main()
