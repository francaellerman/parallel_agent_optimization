#!/usr/bin/env python3
"""Convert Claude Code SWE-chat JSONL into timed human/agent blocks."""

import argparse
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "traces/parallel_001/raw"
DEFAULT_OUTPUT = ROOT / "traces/parallel_001/processed.json"


def timestamp(record):
    value = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Timestamps must include a timezone")
    return value


def text_content(record):
    content = record.get("message", {}).get("content", [])
    if isinstance(content, str):
        return content
    return "\n\n".join(
        part["text"] for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


def is_human(record):
    return (
        record.get("type") == "user"
        and not record.get("isMeta")
        and not record.get("isCompactSummary")
        and (record.get("origin", {}).get("kind") == "human" or (
            not record.get("origin")
            and "<command-name>/simplify</command-name>" in text_content(record)
        ))
    )


def is_tool_result(record):
    content = record.get("message", {}).get("content", [])
    return record.get("type") == "user" and isinstance(content, list) and any(
        isinstance(part, dict) and part.get("type") == "tool_result"
        for part in content
    )


def is_interruption(record):
    return record.get("type") == "user" and (
        bool(record.get("interruptedMessageId"))
        or text_content(record).startswith("[Request interrupted by user")
    )


def parse_session(path):
    records, seen = [], set()
    for line, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        record = json.loads(raw)
        if record.get("isSidechain") or not record.get("timestamp"):
            continue
        key = record.get("uuid")
        if key and key in seen:
            continue
        seen.add(key)
        record["source_line"] = line
        timestamp(record)
        records.append(record)
    records.sort(key=timestamp)
    prompts = [i for i, record in enumerate(records) if is_human(record)]
    if not prompts:
        raise ValueError(f"{path}: no origin.kind=human prompts; unsupported format")
    session_ids = {r["sessionId"] for r in records if r.get("sessionId")}
    if len(session_ids) > 1:
        raise ValueError(f"{path}: contains multiple session IDs")
    blocks, warnings = [], []
    previous_end = records[prompts[0]]["timestamp"]

    for number, index in enumerate(prompts):
        prompt = records[index]
        next_index = prompts[number + 1] if number + 1 < len(prompts) else len(records)
        segment = records[index + 1:next_index]
        assistant = [r for r in segment if r.get("type") == "assistant"]
        activity = [r for r in segment if r.get("type") == "assistant" or is_tool_result(r)]
        endings = [r for r in segment if is_interruption(r) or (
            r.get("type") == "assistant"
            and r.get("message", {}).get("stop_reason") == "end_turn"
        )]
        if endings and (not activity or timestamp(endings[-1]) >= timestamp(activity[-1])):
            endpoint = endings[-1]
            boundary = "interrupted" if is_interruption(endpoint) else "end_turn"
        elif next_index < len(records):
            endpoint = records[next_index]
            boundary = "next_prompt_before_end_turn"
        else:
            endpoint = activity[-1] if activity else prompt
            boundary = "trace_end_incomplete"

        human_duration = (timestamp(prompt) - timestamp({"timestamp": previous_end})).total_seconds()
        blocks.append({
            "role": "human", "text": text_content(prompt), "duration": human_duration,
            "start": previous_end, "end": prompt["timestamp"],
            "boundary": "initial_prompt_unknown_duration" if number == 0 else "prompt_sent",
            "source_lines": [prompt["source_line"]],
        })
        usage, missing_usage, texts = {}, set(), []
        for record in assistant:
            message = record.get("message", {})
            key = message.get("id") or record.get("uuid") or str(record["source_line"])
            count = (message.get("usage") or {}).get("output_tokens")
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                usage[key] = max(usage.get(key, 0), count)
            else:
                missing_usage.add(key)
            text = text_content(record)
            if text:
                texts.append(text)
        blocks.append({
            "role": "agent", "text": "\n\n".join(texts),
            "duration": (timestamp(endpoint) - timestamp(prompt)).total_seconds(),
            "start": prompt["timestamp"], "end": endpoint["timestamp"],
            "boundary": boundary,
            "output_tokens": sum(usage.values()),
            "token_usage_complete": not bool(missing_usage - usage.keys()),
            "source_lines": [r["source_line"] for r in assistant],
        })
        if boundary != "end_turn":
            warnings.append(f"Agent block {len(blocks) - 1}: {boundary}")
        previous_end = endpoint["timestamp"]

    return {
        "session_id": next(iter(session_ids), path.stem),
        "source_file": str(path.resolve()),
        "start": records[prompts[0]]["timestamp"],
        "end": previous_end,
        "warnings": warnings,
        "blocks": blocks,
    }


def convert(paths):
    sessions = [parse_session(Path(path)) for path in paths]
    ids = [s["session_id"] for s in sessions]
    if len(set(ids)) != len(ids):
        raise ValueError("Input sessions must have distinct IDs")
    origin = min(timestamp({"timestamp": s["start"]}) for s in sessions)
    for session in sessions:
        session["start_offset"] = (timestamp({"timestamp": session["start"]}) - origin).total_seconds()
    return {"schema_version": 1, "time_unit": "seconds", "origin": origin.isoformat(), "sessions": sessions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h", action="help", help="Show this help message and exit")
    parser.add_argument("inputs", nargs="*", type=Path, help="Raw JSONL files; defaults to parallel_001")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--all", action="store_true", help="Parse every traces/parallel_*/raw folder")
    args = parser.parse_args()
    if args.all and (args.inputs or args.output):
        parser.error("--all cannot be combined with inputs or --output")
    jobs = [(sorted(folder.glob('*.jsonl')), folder.parent / 'processed.json')
            for folder in sorted((ROOT / 'traces').glob('parallel_*/raw'))] if args.all else [
                (args.inputs or sorted(RAW.glob('*.jsonl')), args.output or DEFAULT_OUTPUT)]
    if not jobs or any(not paths for paths, _ in jobs):
        parser.error("No input traces found for a requested datapoint")
    for paths, output in jobs:
        try:
            data = convert(paths)
        except (ValueError, KeyError, OSError) as exc:
            parser.error(str(exc))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        print(f"Wrote {output}")
        for session in data["sessions"]:
            print(f"  {session['session_id']}: {len(session['blocks'])} blocks, start offset {session['start_offset']:.3f}s")
            for warning in session["warnings"]:
                print(f"    {warning}")


if __name__ == "__main__":
    main()
