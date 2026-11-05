import importlib.util
import http.server
import io
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("runner", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.KEEL_BIN = os.environ.get("KEEL_TEST_BIN", runner.KEEL_BIN)


class RunnerTests(unittest.TestCase):
    def test_manifest_and_repeat_resume_refuse_changed_binary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            src, out, binary = root / "exercises", root / "out", root / "keel"
            (src / "one").mkdir(parents=True)
            binary.write_bytes(b"version one")
            calls = []
            def run(name, source, dest, timeout, rep):
                calls.append(rep)
                row = {"harness": name, "exercise": source.name, "rep": rep, "outcome": "completed"}
                with (dest / "results.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                return row
            argv = ["run.py", "--exercises", str(src), "--out", str(out), "--harness", "keel",
                    "--keel-bin", str(binary), "--repeat", "2"]
            with patch.object(sys, "argv", argv), patch.object(runner, "run_one", side_effect=run), \
                    patch.object(runner, "KEEL_BIN", str(binary)), patch.object(runner, "cool_down"), \
                    patch.object(runner.subprocess, "check_output", return_value="revision"), \
                    patch.object(runner.urllib.request, "urlopen", side_effect=lambda *a, **k: io.BytesIO(b"{}")):
                runner.main()
                runner.main()
                self.assertEqual(calls, [1, 2])
                binary.write_bytes(b"version two")
                with self.assertRaisesRegex(SystemExit, "settings changed"):
                    runner.main()

    def test_temperature_missing_or_invalid_fails_closed(self):
        for output in ["", "Tctl: +nan°C", "Tctl: +0.0°C"]:
            with patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, output)):
                with self.assertRaises(RuntimeError):
                    runner.tctl()

    def test_quiet_hours_are_checked_after_cooling(self):
        with patch.object(runner, "tctl", side_effect=[70, 55]), \
                patch.object(runner.time, "strftime", side_effect=["19:29", "19:30"]), \
                patch.object(runner.time, "sleep"):
            with self.assertRaises(TimeoutError):
                runner.cool_down(stop_at="19:30")

    def test_setup_error_is_archived_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            out, source = Path(folder) / "out", Path(folder) / "exercise"
            def broken(name, src, timeout, artifact, ws, home, row):
                ws.mkdir()
                (ws / "solution.py").write_text("preserve this")
                raise OSError("missing executable")
            with patch.object(runner, "attempt", side_effect=broken):
                row = runner.run_one("keel", source, out, 10)
                with self.assertRaises(FileExistsError):
                    runner.run_one("keel", source, out, 10)
            self.assertEqual(row["outcome"], "setup_error")
            self.assertEqual((out / row["artifact"] / "workspace/solution.py").read_text(), "preserve this")
            self.assertEqual(len((out / "results.jsonl").read_text().splitlines()), 1)


@unittest.skipUnless(os.environ.get("KEEL_BENCH_SANDBOX_TESTS") == "1", "requires working bubblewrap namespaces")
class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pg-integrity-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.src, self.out = self.root / "exercise", self.root / "out"
        (self.src / ".docs").mkdir(parents=True)
        (self.src / ".meta").mkdir()
        (self.src / ".meta/reference.py").write_text("PRIVATE_REFERENCE_SENTINEL")
        (self.src / ".docs/instructions.md").write_text("Implement answer() returning 42.")
        (self.src / "solution.py").write_text("def answer():\n    return 0\n")
        (self.src / "solution_test.py").write_text("from solution import answer\n\ndef test_answer():\n    assert answer() == 42\n")

    def run_agent(self, code, rep=1, timeout=10):
        with patch.object(runner, "harness_cmd", return_value=(["/usr/bin/python3", "-c", code], ())), \
                patch.object(runner, "tctl", return_value=50):
            return runner.run_one("keel", self.src, self.out, timeout, rep)

    def test_reference_hidden_but_workspace_usable_and_background_child_dies(self):
        ws, home = self.root / "workspace", self.root / "home"
        runner.prepare(self.src, ws)
        home.mkdir()
        late = str(ws / "leaked")
        code = ("from pathlib import Path; import subprocess,sys; "
                f"assert not Path({str(self.src)!r}).exists(); "
                f"assert not Path({str(runner.REAL_HOME / '.ssh')!r}).exists(); "
                "assert Path('solution.py').is_file(); Path('marker').write_text('ok'); "
                f"subprocess.Popen([sys.executable,'-c',\"import time; from pathlib import Path; time.sleep(.4); Path({late!r}).touch()\"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)")
        subprocess.run([*runner.sandbox(ws, home), "python3", "-c", code], check=True, timeout=10)
        time.sleep(.6)
        self.assertEqual((ws / "marker").read_text(), "ok")
        self.assertFalse((ws / "leaked").exists())

    def test_test_tampering_and_pytest_hook_cannot_fake_a_pass(self):
        row = self.run_agent("from pathlib import Path; Path('solution_test.py').write_text('def test_fake(): pass'); "
                             "Path('conftest.py').write_text('import os; os._exit(0)')")
        self.assertEqual(row["outcome"], "completed")
        self.assertFalse(row["passed"])
        self.assertTrue(row["tampered_test"])
        self.assertIn("1 failed", (self.out / row["artifact"] / "score.stdout").read_text())

    def test_pass_is_preserved_separately_from_another_repeat(self):
        first = self.run_agent("from pathlib import Path; Path('solution.py').write_text('def answer(): return 42')")
        second = self.run_agent("pass", rep=2)
        self.assertTrue(first["passed"])
        self.assertFalse(second["passed"])
        self.assertNotEqual(first["artifact"], second["artifact"])
        self.assertIn("return 42", (self.out / first["artifact"] / "workspace/solution.py").read_text())

    def test_syntax_error_is_a_coding_failure_not_infrastructure(self):
        row = self.run_agent("from pathlib import Path; Path('solution.py').write_text('def broken(')")
        self.assertEqual(row["outcome"], "completed")
        self.assertFalse(row["passed"])
        self.assertEqual(row["score_exit"], 2)

    def test_solution_symlink_is_not_followed_by_host_scorer(self):
        row = self.run_agent(f"from pathlib import Path; p=Path('solution.py'); p.unlink(); p.symlink_to({str(self.src / '.meta/reference.py')!r})")
        self.assertEqual(row["outcome"], "completed")
        self.assertFalse(row["passed"])
        self.assertIn("unsafe", row["pytest"])
        self.assertTrue((self.out / row["artifact"] / "workspace/solution.py").is_symlink())

    def test_candidate_cannot_rewrite_original_tests_during_scoring(self):
        source = "from pathlib import Path\nPath('solution_test.py').write_text('def test_fake(): pass')\ndef answer(): return 42\n"
        row = self.run_agent(f"from pathlib import Path; Path('solution.py').write_text({source!r})")
        self.assertFalse(row["passed"])
        self.assertIn("assert answer() == 42", (self.out / row["artifact"] / "score/solution_test.py").read_text())

    def test_timeout_still_preserves_and_scores_candidate(self):
        row = self.run_agent("import time; time.sleep(30)", timeout=1)
        self.assertEqual(row["agent_exit"], "timeout")
        self.assertEqual(row["outcome"], "completed")
        self.assertFalse(row["passed"])

    def test_scoring_infrastructure_failure_stays_separate(self):
        real_run = subprocess.run
        def fail_score(cmd, **kwargs):
            if "pytest" in cmd:
                return subprocess.CompletedProcess(cmd, 4, "", "scorer setup failed")
            return real_run(cmd, **kwargs)
        with patch.object(runner.subprocess, "run", side_effect=fail_score):
            row = self.run_agent("pass")
        self.assertEqual(row["outcome"], "scoring_error")
        self.assertEqual((self.out / row["artifact"] / "score.stderr").read_text(), "scorer setup failed")

    def test_missing_harness_is_setup_failure_not_a_zero_score(self):
        with patch.object(runner, "harness_cmd", return_value=(["/not-installed/keel"], ())):
            row = runner.run_one("keel", self.src, self.out, 10)
        self.assertEqual(row["outcome"], "setup_error")
        self.assertIsNone(row["agent_exit"])

    def test_installed_harness_runtimes_start_without_host_home(self):
        if not (runner.REAL_HOME / ".npm-global").exists():
            self.skipTest("optional local OpenCode and Aider installations unavailable")
        ws, home = self.root / "workspace", self.root / "home"
        runner.prepare(self.src, ws)
        home.mkdir()
        for name in ["keel", "opencode", "aider"]:
            with self.subTest(harness=name):
                cmd, extra = runner.harness_cmd(name, ws, "smoke", ["solution.py"], "solution_test.py")
                result = subprocess.run([*runner.sandbox(ws, home, extra), cmd[0], "--version"],
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_keel_pipeline_archives_session_and_scores_original_tests(self):
        class Provider(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.reply("application/json", json.dumps({"data": [
                    {"id": runner.MODEL, "context_length": 16384}]}).encode())

            def do_POST(self):
                self.rfile.read(int(self.headers.get("content-length", 0)))
                self.server.step += 1
                step = self.server.step
                name = "write_file" if step == 1 else "shell"
                args = {"path": "solution.py", "content": "def answer(): return 42"} if step == 1 else {
                    "command": "python3 -m pytest -q solution_test.py"}
                delta = {"tool_calls": [{"index": 0, "id": str(step), "function": {
                    "name": name, "arguments": json.dumps(args)}}]} if step < 3 else {"content": "Done."}
                self.reply("text/event-stream", ("data: " + json.dumps({"choices": [{"delta": delta}]})
                           + "\n\ndata: [DONE]\n\n").encode())

            def reply(self, kind, body):
                self.send_response(200)
                self.send_header("content-type", kind)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Provider)
        server.step = 0
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with patch.object(runner, "BASE_URL", f"http://127.0.0.1:{server.server_port}/v1"), \
                    patch.object(runner, "tctl", return_value=50):
                row = runner.run_one("keel", self.src, self.out, 15)
        finally:
            server.shutdown()
            server.server_close()
            worker.join()
        self.assertEqual(row["outcome"], "completed", row)
        self.assertTrue(row["passed"], row)
        self.assertTrue((self.out / row["artifact"] / "input/solution_test.py").is_file())
        self.assertTrue((self.out / row["artifact"] / "score/solution_test.py").is_file())
        session = [json.loads(line) for line in (self.out / row["session"]).read_text().splitlines()]
        receipts = [e["data"]["content"] for e in session if e["kind"] == "tool"]
        self.assertEqual(len(receipts), 2)
        self.assertTrue(any("1 passed" in output for output in receipts))
        print("REAL KEEL COMMAND", shlex.join(json.loads((self.out / row["artifact"] / "command.json").read_text())))
        print("REAL KEEL RESULT", json.dumps(row))


if __name__ == "__main__":
    unittest.main()
