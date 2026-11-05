import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("analyze", Path(__file__).with_name("analyze.py"))
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    def row(self, harness="keel", exercise="one", **extra):
        return {"harness": harness, "exercise": exercise, "passed": False,
                "secs": 300, "agent_exit": "timeout", "pytest": "1 failed, 1 passed",
                **extra}

    def test_unmatched_attempts_do_not_bias_comparison(self):
        self.write("results.jsonl", [self.row(), self.row("aider"),
                                    self.row(exercise="two", passed=True, pytest="20 passed")])
        data, markdown = analyze.report(self.root, ["keel", "aider"], [])
        self.assertEqual(data["recorded_attempts"], 3)
        self.assertEqual(data["matched_blocks"], 1)
        self.assertIn("| keel | 0 | 1 | 50.0% | 1 | 1 |", markdown)

    def test_repetitions_are_distinct_and_legacy_rows_default_to_one(self):
        self.write("results.jsonl", [self.row(), self.row(rep=2)])
        self.assertEqual(len(analyze.load_results(self.root / "results.jsonl", [])), 2)
        self.write("results.jsonl", [self.row(), self.row(rep=1)])
        with self.assertRaisesRegex(ValueError, "duplicate attempt"):
            analyze.load_results(self.root / "results.jsonl", [])

    def test_live_incomplete_tail_is_warned_but_earlier_corruption_fails(self):
        path = self.write("events", [self.row()])
        with path.open("a") as f:
            f.write('{"harness":')
        warnings = []
        self.assertEqual(len(analyze.read_jsonl(path, warnings)), 1)
        self.assertEqual(len(warnings), 1)
        path.write_text('{broken}\n{"valid": true}\n')
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            analyze.read_jsonl(path, [])

    def test_score_errors_and_missing_counts_stay_unknown(self):
        for summary in ["1 passed, 1 error", "2 errors", "pytest timeout", "8 skipped", ""]:
            self.assertIsNone(analyze.fraction({"pytest": summary}), summary)
        self.assertEqual(analyze.fraction({"pytest": "2 failed, 3 passed, 4 skipped"}), 0.6)
        self.assertEqual(analyze.fraction({"pytest": "4 failed"}), 0)

    def test_context_evidence_comes_from_log_not_exit_status(self):
        self.write("results.jsonl", [self.row(agent_exit=0)])
        logs = self.root / "keel"
        logs.mkdir()
        (logs / "one.log").write_text(
            "request (18909 tokens) exceeds the \n\x1b[31mavailable context size\x1b[0m")
        data, _ = analyze.report(self.root, ["keel"], [])
        self.assertTrue(data["attempts"][0]["context_error_text_seen"])
        self.assertEqual(data["attempts"][0]["agent_exit"], 0)
        self.assertFalse(data["attempts"][0]["passed"])

    def test_session_tracks_real_receipts_and_unverified_late_edit(self):
        def call(stamp, ident, name, args):
            return {"kind": "assistant", "ts": stamp, "data": {"tool_calls": [
                {"id": ident, "name": name, "args": args}]}}
        def receipt(stamp, ident, content):
            return {"kind": "tool", "ts": stamp, "data": {"id": ident, "content": content}}
        path = self.write("session.jsonl", [
            {"kind": "user", "ts": 1000, "data": {"text": "solve"}},
            call(31000, "a", "write_file", {"path": "solution.py", "content": "wrong"}),
            receipt(31020, "a", "wrote 5 bytes"),
            call(41000, "b", "shell", {"command": "python3 -m pytest -q"}),
            receipt(42000, "b", "exit: 1\n1 failed, 2 passed in 0.2s\n"),
            {"kind": "compaction", "ts": 43000, "data": {}},
            call(51000, "c", "patch_file", {"path": "solution.py", "patch": "better"}),
            receipt(51030, "c", "wrote 6 bytes"),
            call(61000, "d", "read_file", {"path": "solution.py"}),
        ])
        data = analyze.session_metrics(path, [])
        self.assertEqual(data["first_explicit_edit_seconds"], 30)
        self.assertEqual(data["explicit_edit_calls"], 2)
        self.assertEqual(data["pytest_calls"], 1)
        self.assertEqual(data["tool_receipts"], 3)
        self.assertEqual(data["pending_calls"], 1)
        self.assertAlmostEqual(data["tool_receipt_seconds"], 1.05)
        self.assertTrue(data["edit_after_last_pytest"])
        self.assertEqual(data["compactions_logged"], 1)
        self.assertEqual(data["test_feedback"][0]["summary"], "1 failed, 2 passed in 0.2s")

    def test_invalid_results_are_rejected(self):
        for extra in [{"secs": -1}, {"secs": float("nan")}, {"secs": True}, {"passed": "false"}]:
            self.write("results.jsonl", [self.row(**extra)])
            with self.assertRaisesRegex(ValueError, "invalid result"):
                analyze.load_results(self.root / "results.jsonl", [])

    def test_invalid_attempt_identity_and_harness_selection_fail_closed(self):
        for extra in [{"exercise": "../outside"}, {"rep": 0}, {"rep": "1"}]:
            self.write("results.jsonl", [self.row(**extra)])
            with self.assertRaisesRegex(ValueError, "invalid attempt identity"):
                analyze.load_results(self.root / "results.jsonl", [])
        for harnesses in [[], [""], ["keel", "keel"]]:
            with self.assertRaisesRegex(ValueError, "distinct, nonempty"):
                analyze.report(self.root, harnesses, [])


if __name__ == "__main__":
    unittest.main()
