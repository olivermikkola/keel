import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import compare_loop as compare


class ComparisonTests(unittest.TestCase):
    def test_pair_order_rotates_and_each_task_has_both_variants(self):
        plan = list(compare.schedule(3))
        self.assertEqual(len(plan), 18)
        for offset in range(0, len(plan), 2):
            first, second = plan[offset:offset + 2]
            self.assertEqual(first[1:], second[1:])
            self.assertEqual({first[0], second[0]}, set(compare.PROMPTS))
        self.assertNotEqual(plan[0][0], plan[6][0])

    def test_real_schedule_keeps_prompts_separate_resumes_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            calls = []

            def attempt(name, source, out, timeout, rep=1):
                calls.append((source.name, rep, compare.run.PROMPT))
                out.mkdir(parents=True, exist_ok=True)
                row = {"harness": name, "exercise": source.name, "rep": rep,
                       "outcome": "completed", "passed": True, "secs": 1, "pytest": "1 passed"}
                with (out / "results.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                return row

            argv = ["compare_loop.py", "--exercises", str(root / "data"), "--out", str(root / "out"),
                    "--keel-bin", "/usr/bin/true", "--repeat", "1"]
            with patch.object(sys, "argv", argv), patch.object(compare, "snapshot", return_value={"version": 1}) as snapshot, \
                    patch.object(compare, "slots_idle", return_value=True), \
                    patch.object(compare, "time_available", return_value=True), \
                    patch.object(compare.run, "cool_down"), patch.object(compare.run, "run_one", side_effect=attempt), \
                    patch.object(compare.run, "PROMPT", compare.PROMPTS["default"]), contextlib.redirect_stdout(io.StringIO()):
                compare.main()
                compare.main()
                self.assertEqual(len(calls), 7)  # One smoke, six attempts; resume runs nothing.
                for (variant, task, rep), recorded in zip(compare.schedule(1), calls[1:]):
                    self.assertEqual(recorded, (task, rep, compare.PROMPTS[variant]))
                snapshot.return_value = {"version": 2}
                with self.assertRaisesRegex(SystemExit, "settings changed"):
                    compare.main()
            report = json.loads((root / "out/report.json").read_text())
            self.assertEqual(len(report), 6)
            self.assertEqual({r["variant"] for r in report}, set(compare.PROMPTS))


if __name__ == "__main__":
    unittest.main()
