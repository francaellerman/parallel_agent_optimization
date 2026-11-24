#!/usr/bin/env python3
"""Replay independent sessions on one clock, with overlapping human windows."""

import argparse
import heapq
import json
import math
import re
import shutil
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

DEFAULT_INPUT = Path(__file__).resolve().parent / "traces/parallel_001/processed.json"


def format_message(text, width):
    lines = []
    fence = None
    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            if fence is None:
                fence = marker.group(1)[0]
            elif marker.group(1)[0] == fence:
                fence = None
            lines.append(line)
        elif fence or line.startswith("    ") or line.lstrip().startswith("|"):
            lines.append(line)
        elif not line.strip():
            lines.append("")
        else:
            prefix = re.match(r"^(\s*(?:[-*+]\s+|\d+[.)]\s+|>\s*))", line)
            indent = " " * len(prefix.group(1)) if prefix else ""
            lines.extend(textwrap.wrap(line, width=width, subsequent_indent=indent,
                                       break_long_words=False, break_on_hyphens=False))
    return "\n".join("    " + line if line else "" for line in lines)


def load_sessions(paths):
    sessions = []
    origins = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        if data.get("schema_version") != 1 or data.get("time_unit") != "seconds":
            raise ValueError(f"{path}: expected schema_version=1, time_unit=seconds")
        origin = datetime.fromisoformat(data["origin"].replace("Z", "+00:00"))
        if origin.tzinfo is None:
            raise ValueError("Origin must include a timezone")
        origins.append(origin.timestamp())
        for session in data["sessions"]:
            session = dict(session)
            offset = session["start_offset"]
            if not isinstance(offset, (int, float)) or not math.isfinite(offset) or offset < 0:
                raise ValueError("start_offset must be finite and nonnegative")
            session["absolute_start"] = origin.timestamp() + offset
            blocks = session["blocks"]
            if not blocks:
                raise ValueError("Each session needs at least one block")
            for i, block in enumerate(blocks):
                duration = block["duration"]
                if block["role"] != ("human" if i % 2 == 0 else "agent"):
                    raise ValueError("Blocks must alternate human/agent, starting with human")
                if not isinstance(block["text"], str):
                    raise ValueError("Block text must be a string")
                if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
                    raise ValueError("Block duration must be finite and nonnegative")
            sessions.append(session)
    if not sessions:
        raise ValueError("No sessions found")
    if len({s["session_id"] for s in sessions}) != len(sessions):
        raise ValueError("Duplicate session IDs")
    origin = min(origins)
    for session in sessions:
        session["start_offset"] = round(session.pop("absolute_start") - origin, 6)
    return sessions


def replay(sessions, inference_speed=1.0, speedup=100.0, display=True, inference_speeds=None):
    if not math.isfinite(inference_speed) or inference_speed <= 0:
        raise ValueError("inference_speed must be finite and positive")
    if not math.isfinite(speedup) or speedup < 0:
        raise ValueError("speedup must be finite and nonnegative (0 means instant)")
    speeds = list(inference_speeds) if inference_speeds is not None else [inference_speed] * len(sessions)
    if len(speeds) != len(sessions) or any(not math.isfinite(s) or s <= 0 for s in speeds):
        raise ValueError("inference_speeds must contain one finite positive speed per session")
    states = [{"index": -1, "start": s["start_offset"], "end": s["start_offset"], "done": False} for s in sessions]
    queue = [(s["start_offset"], i) for i, s in enumerate(sessions)]
    heapq.heapify(queue)
    events = []
    wall_start = time.monotonic()
    interactive = display and sys.stdout.isatty()
    drawn_lines = 0
    labels = [f"Agent {i + 1}" for i in range(len(sessions))]

    if display:
        print("Parallel chat replay")
        for label, session, speed in zip(labels, sessions, speeds):
            print(f"  {label}: {session['session_id']} · inference speed {speed:g}x")
        print()

    def clear_progress():
        nonlocal drawn_lines
        if drawn_lines:
            sys.stdout.write(f"\033[{drawn_lines}A\r\033[J")
            drawn_lines = 0

    def show_message(i, block, start, end):
        if not display:
            return
        clear_progress()
        human = block["role"] == "human"
        speaker = f"HUMAN → {labels[i].upper()}" if human else labels[i].upper()
        action = "human window" if block["role"] == "human" else "agent work"
        boundary = block.get("boundary", "")
        status = f" | {boundary}" if boundary in {
            "interrupted", "next_prompt_before_end_turn", "trace_end_incomplete"
        } else ""
        width = max(24, min(88, shutil.get_terminal_size().columns - 6))
        color = "\033[1;36m" if human else "\033[1;32m" if i % 2 == 0 else "\033[1;35m"
        color, reset = (color, "\033[0m") if interactive else ("", "")
        print("\n")
        print(f"{color}  {'═' * width}")
        print(f"  {speaker}{reset}")
        print(f"  t={end:.3f}s · {action}: {end - start:.3f}s{status}")
        print(f"{color}  {'─' * width}{reset}\n")
        print(format_message(block["text"] or "(No recorded text.)", width - 2))
        print("\n", flush=True)

    def render(now):
        nonlocal drawn_lines
        if not interactive:
            return
        lines = [f"  LIVE · t={now:.1f}s · playback {speedup:g}x"]
        for label, session, state, speed in zip(labels, sessions, states, speeds):
            label = f"{label} ({speed:g}x)"
            if state["done"]:
                lines.append(f"{label}: DONE at {state['end']:.3f}s")
            elif state["index"] < 0:
                lines.append(f"{label}: starts at {state['start']:.3f}s")
            else:
                block = session["blocks"][state["index"]]
                duration = state["end"] - state["start"]
                elapsed = max(0.0, min(duration, now - state["start"]))
                fraction = elapsed / duration if duration else 1.0
                bar = "#" * int(fraction * 20) + "." * (20 - int(fraction * 20))
                activity = "HUMAN WINDOW" if block["role"] == "human" else "AGENT WORK"
                lines.append(f"  {label} · {activity}: [{bar}] {fraction:3.0%} · {elapsed:.0f}/{duration:.0f}s")
        clear_progress()
        for line in lines:
            sys.stdout.write("\r\033[2K" + line[:max(1, shutil.get_terminal_size().columns - 1)] + "\n")
        sys.stdout.flush()
        drawn_lines = len(lines)

    def wait_until(target):
        if speedup == 0:
            return
        deadline = wall_start + target / speedup
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            render(min(target, (time.monotonic() - wall_start) * speedup))
            time.sleep(min(0.05, remaining))

    while queue:
        now, i = heapq.heappop(queue)
        wait_until(now)
        session, state = sessions[i], states[i]
        if state["index"] >= 0:
            block = session["blocks"][state["index"]]
            events.append({
                "order": len(events), "session_id": session["session_id"],
                "block_index": state["index"], "role": block["role"],
                "start": state["start"], "end": now,
                "duration": now - state["start"], "baseline_duration": block["duration"],
            })
            show_message(i, block, state["start"], now)
        state["index"] += 1
        if state["index"] == len(session["blocks"]):
            state["done"] = True
        else:
            block = session["blocks"][state["index"]]
            duration = block["duration"] / speeds[i] if block["role"] == "agent" else block["duration"]
            state.update(start=now, end=now + duration)
            heapq.heappush(queue, (state["end"], i))
        render(now)

    clear_progress()
    summary = []
    for session, state, speed in zip(sessions, states, speeds):
        summary.append({
            "session_id": session["session_id"], "start": session["start_offset"], "end": state["end"],
            "inference_speed": speed,
            "agent_seconds": sum(b["duration"] for b in session["blocks"] if b["role"] == "agent") / speed,
            "human_seconds": sum(b["duration"] for b in session["blocks"] if b["role"] == "human"),
        })
    return {"inference_speed": speeds[0] if len(set(speeds)) == 1 else None,
            "inference_speeds": dict(zip((s["session_id"] for s in sessions), speeds)),
            "total_time": max(s["end"] for s in states), "sessions": summary, "events": events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h", action="help", help="Show this help message and exit")
    parser.add_argument("inputs", nargs="*", type=Path, help="One or more replay JSON files; each may contain any number of sessions")
    parser.add_argument("--inference-speed", type=float, default=1.0, help="Agent duration divisor (default: recorded speed)")
    parser.add_argument("--inference-speeds", nargs="+", type=float,
                        help="One speed per session in input order; overrides --inference-speed")
    parser.add_argument("--speedup", type=float, default=1000.0, help="Playback multiplier; 0 runs instantly (default: 1000)")
    parser.add_argument("--output", type=Path, help="Optional JSON event log and summary")
    args = parser.parse_args()
    try:
        sessions = load_sessions(args.inputs or [DEFAULT_INPUT])
        result = replay(sessions, args.inference_speed, args.speedup, inference_speeds=args.inference_speeds)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("\nParallel replay summary")
    for session in result["sessions"]:
        print(f"  {session['session_id'][:8]}: {session['start']:.3f}s -> {session['end']:.3f}s; agent {session['agent_seconds']:.3f}s, human windows {session['human_seconds']:.3f}s")
    print(f"  All {len(result['sessions'])} recorded sequences finished at t={result['total_time']:.3f}s")


if __name__ == "__main__":
    main()
