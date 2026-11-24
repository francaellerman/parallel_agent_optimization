# How many agents do people run in parallel? (SWE-chat)

Scans of [SALT-NLP/SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat) (gated; needs a
Hugging Face token) for how many of one user's coding-agent sessions run at the same time.

A session is *engaged* while its events are less than 15 minutes apart. For each user, the
engaged intervals are swept to find how many sessions are engaged at once. Forks (sessions
with the same user and first timestamp) are dropped, and one account whose concurrency
exceeds 10 is excluded as likely automation.

| Script | What it does | Output |
|---|---|---|
| `concurrency.py` | First pass using whole-session intervals | printed summary |
| `active.py` | Activity-based concurrency distribution, and candidate 3–4-agent episodes | `candidates.csv` |
| `trend.py` | Peak concurrency per session by month, with significance tests (pooled, clustered by user, within-user, paired) | `session_peaks.csv` |
| `extract_conversation_times.py` | Reduces `conversations.parquet` to the three columns the scans use | `conv_min.parquet` |
| `pseudonymize.py` | Replaces user ids with codes in committed CSVs | in place |

## Reproduce

```bash
# download sessions.parquet and conversations.parquet into data/swechat/ (git-ignored)
python analysis/swechat_parallelism/extract_conversation_times.py \
    data/swechat/conversations.parquet data/swechat/conv_min.parquet
python analysis/swechat_parallelism/concurrency.py data/swechat/sessions.parquet
python analysis/swechat_parallelism/active.py data/swechat/sessions.parquet data/swechat/conv_min.parquet \
    analysis/swechat_parallelism/candidates.csv
python analysis/swechat_parallelism/trend.py data/swechat/sessions.parquet data/swechat/conv_min.parquet \
    analysis/swechat_parallelism/session_peaks.csv
python analysis/swechat_parallelism/pseudonymize.py   # before committing
```

User ids in the committed CSVs are replaced with codes (`u0001`, …). The mapping is kept in the
git-ignored `data/swechat/user_codes.json`.
