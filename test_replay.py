import json
import tempfile
import unittest
from pathlib import Path

from parse_swe_chat import convert, timestamp
from simulate_parallel import load_sessions, replay


def record(second, kind, content, **extra):
    return {
        "timestamp": f"2026-09-03T12:00:{second:06.3f}Z",
        "type": kind, "sessionId": "test-session",
        "message": {"content": content}, **extra,
    }


def prompt(second, text):
    return record(second, "user", text, origin={"kind": "human"})


def response(second, text, stop="tool_use", message_id="message-1", tokens=20):
    result = record(second, "assistant", [{"type": "text", "text": text}])
    result["message"].update(id=message_id, stop_reason=stop, usage={"output_tokens": tokens})
    return result


class ParserTests(unittest.TestCase):
    def convert_records(self, records):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in records))
            return convert([path])["sessions"][0]

    def test_grouping_windows_and_token_deduplication(self):
        session = self.convert_records([
            prompt(0, "Do work"),
            response(2, "First part"),
            record(3, "user", [{"type": "tool_result", "content": "tool output"}]),
            record(4, "user", "injected context", isMeta=True),
            response(5, "Second part"),
            response(10, "Done", "end_turn", "message-2", 30),
            record(11, "system", [], subtype="turn_duration", durationMs=11000),
            prompt(25, "Follow up"),
            response(30, "Final", "end_turn", "message-3", 10),
        ])
        self.assertEqual([b["duration"] for b in session["blocks"]], [0, 10, 15, 5])
        self.assertEqual(session["blocks"][1]["text"], "First part\n\nSecond part\n\nDone")
        self.assertEqual(session["blocks"][1]["output_tokens"], 50)
        self.assertFalse(session["warnings"])

    def test_interrupted_and_unfinished_blocks(self):
        session = self.convert_records([
            prompt(0, "Work"), response(2, "Working"),
            record(4, "user", "[Request interrupted by user]", interruptedMessageId="message-1"),
            prompt(10, "New request"), response(12, "Still working"),
        ])
        self.assertEqual([b["duration"] for b in session["blocks"]], [0, 4, 6, 2])
        self.assertEqual(session["blocks"][1]["boundary"], "interrupted")
        self.assertEqual(session["blocks"][3]["boundary"], "trace_end_incomplete")

    def test_prompt_during_agent_is_flagged(self):
        session = self.convert_records([
            prompt(0, "Work"), response(2, "Working"),
            prompt(3, "Change course"), response(5, "Done", "end_turn"),
        ])
        self.assertEqual(session["blocks"][1]["boundary"], "next_prompt_before_end_turn")
        self.assertEqual(session["blocks"][2]["duration"], 0)

    def test_unknown_format_is_rejected(self):
        with self.assertRaises(ValueError):
            self.convert_records([record(0, "user", "Unmarked user")])

    def test_simplify_command_splits_but_injected_instructions_do_not(self):
        session = self.convert_records([
            prompt(0, "Work"), response(5, "Done", "end_turn"),
            record(10, "user", "<command-name>/simplify</command-name>"),
            record(10, "user", "Injected simplify instructions", isMeta=True),
            response(15, "Cleaned up", "end_turn", "message-2"),
        ])
        self.assertEqual([b["duration"] for b in session["blocks"]], [0, 5, 5, 5])


def session(name, offset, durations):
    return {
        "session_id": name, "start_offset": offset,
        "blocks": [{"role": "human" if i % 2 == 0 else "agent", "duration": d, "text": name}
                   for i, d in enumerate(durations)],
    }


class SimulatorTests(unittest.TestCase):
    def test_parallel_windows_and_agent_speed(self):
        sessions = [session("a", 0, [0, 10, 20, 10]), session("b", 5, [0, 10, 20, 10])]
        baseline = replay(sessions, speedup=0, display=False)
        faster = replay(sessions, inference_speed=2, speedup=0, display=False)
        self.assertEqual(baseline["total_time"], 45)
        self.assertEqual(faster["total_time"], 35)
        windows = [e for e in baseline["events"] if e["role"] == "human" and e["block_index"] == 2]
        self.assertEqual([(e["start"], e["end"]) for e in windows], [(10, 30), (15, 35)])
        self.assertEqual([s["human_seconds"] for s in faster["sessions"]], [20, 20])

    def test_simultaneous_sends_are_ordered_without_added_effort(self):
        result = replay([session("a", 0, [0, 1, 1, 0]), session("b", 0, [0, 1, 1, 0])], speedup=0, display=False)
        sends = [e for e in result["events"] if e["role"] == "human" and e["block_index"] == 2]
        self.assertEqual([e["session_id"] for e in sends], ["a", "b"])
        self.assertEqual([e["end"] for e in sends], [2, 2])
        self.assertLess(sends[0]["order"], sends[1]["order"])

    def test_invalid_speed(self):
        for speed in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                replay([session("a", 0, [0, 1])], inference_speed=speed, speedup=0, display=False)

    def test_distinct_agent_speeds_preserve_human_windows(self):
        sessions = [session("a", 0, [0, 100, 10, 0]), session("b", 5, [0, 10, 60, 0])]
        result = replay(sessions, inference_speeds=[2, 4], speedup=0, display=False)
        self.assertEqual(result["total_time"], 67.5)
        self.assertEqual([s["agent_seconds"] for s in result["sessions"]], [50, 2.5])
        self.assertEqual([s["human_seconds"] for s in result["sessions"]], [10, 60])
        self.assertEqual(result["inference_speeds"], {"a": 2, "b": 4})
        for speeds in [[2], [2, 0], [2, float("nan")], [1, 2, 3]]:
            with self.assertRaises(ValueError):
                replay(sessions, inference_speeds=speeds, speedup=0, display=False)

    def test_multiple_files_preserve_absolute_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, origin in [("a", "2026-09-03T12:00:00Z"), ("b", "2026-09-03T12:00:05Z")]:
                path = Path(directory) / f"{name}.json"
                path.write_text(json.dumps({"schema_version": 1, "time_unit": "seconds", "origin": origin,
                                            "sessions": [session(name, 0, [0, 10])]}))
                paths.append(path)
            self.assertEqual([s["start_offset"] for s in load_sessions(paths)], [0, 5])

    def test_actual_traces_reproduce_every_block_timestamp(self):
        paths = sorted((Path(__file__).resolve().parent / "traces").glob("parallel_*/processed.json"))
        if not paths:
            self.skipTest("Run parse_swe_chat.py to generate the real fixture")
        for path in paths:
            with self.subTest(datapoint=path.parent.name):
                self.check_actual_trace(path)

    def check_actual_trace(self, path):
        data = json.loads(path.read_text())
        sessions = load_sessions([path])
        result = replay(sessions, speedup=0, display=False)
        by_id = {s["session_id"]: s for s in sessions}
        origin = timestamp({"timestamp": data["origin"]})
        self.assertEqual(len(result["events"]), sum(len(s["blocks"]) for s in sessions))
        for event in result["events"]:
            block = by_id[event["session_id"]]["blocks"][event["block_index"]]
            for edge in ["start", "end"]:
                expected = (timestamp({"timestamp": block[edge]}) - origin).total_seconds()
                self.assertAlmostEqual(event[edge], expected, places=5)


class SpeedSweepTests(unittest.TestCase):
    def test_bottleneck_switch_and_human_floor(self):
        from plot_speed_sweep import analyze
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "processed.json"
            path.write_text(json.dumps({"schema_version": 1, "time_unit": "seconds",
                                       "origin": "2026-09-03T12:00:00Z", "sessions": [
                session("compute", 0, [0, 100, 10, 0]),
                session("human", 5, [0, 10, 60, 0]),
            ]}))
            result = analyze(path, [1, 2, 10, 100])
            a, b = result["per_agent_sweeps"]
            self.assertEqual([r["overall_seconds"] for r in a["sweep"]], [110, 75, 75, 75])
            self.assertEqual([r["overall_seconds"] for r in b["sweep"]], [110, 110, 110, 110])
            self.assertEqual([a["infinite_speed_floor_seconds"], b["infinite_speed_floor_seconds"]], [75, 110])
            self.assertEqual(analyze(path, [1], align_starts=True)["per_agent_sweeps"][0]["infinite_speed_floor_seconds"], 70)
            custom = analyze(path, [1, 2], other_speed=2)
            self.assertEqual([r["overall_seconds"] for r in custom["per_agent_sweeps"][0]["sweep"]], [110, 70])
            self.assertEqual([r["overall_seconds"] for r in custom["per_agent_sweeps"][1]["sweep"]], [75, 70])


if __name__ == "__main__":
    unittest.main()
