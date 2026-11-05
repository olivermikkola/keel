#!/usr/bin/env python3
"""Run the Aider polyglot benchmark's Python exercises through several
coding-agent harnesses with the same local model, and score them.

Every harness gets the same prompt, the same time limit, the same
bubblewrap sandbox (writes only in its task folder and a throwaway HOME)
and stock settings. A task passes only if the ORIGINAL test file passes.

usage: run.py --exercises DIR --out DIR [--harness keel,opencode,aider]
              [--only name,name] [--timeout 300] [--keel-bin PATH]

Needs: bwrap, lm-sensors, a llama-server with the model at BASE_URL,
opencode, and `uv tool run --from aider-chat aider` cached once.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1:8080/v1"
MODEL = "bonsai-9b"
CONTEXT = 16384
REAL_HOME = Path.home()
KEEL_BIN = str(REAL_HOME / ".local/bin/keel")

PROMPT = (
    "Solve this coding exercise. The instructions are in .docs/instructions.md"
    "{append}. Write the solution in {stubs}, keeping the existing function and "
    "class names. Do not change {test}. Check your work with: "
    "python3 -m pytest -q {test}"
)

OPENCODE_CONFIG = {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
        "llama": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "llama.cpp",
            "options": {"baseURL": BASE_URL},
            "models": {MODEL: {"name": MODEL, "limit": {"context": CONTEXT, "output": 4096}}},
        }
    },
    "model": f"llama/{MODEL}",
    "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny"},
}


def sandbox(ws, home, readonly=()):
    """Expose system/tool runtimes, never the host home or source dataset."""
    args = ["bwrap", "--unshare-pid", "--ro-bind", "/usr", "/usr", "--tmpfs", "/tmp"]
    for name in ("bin", "lib", "lib64"):
        path = Path("/") / name
        if path.is_symlink():
            args += ["--symlink", os.readlink(path), str(path)]
        elif path.exists():
            args += ["--ro-bind", str(path), str(path)]
    paths = [Path(KEEL_BIN), REAL_HOME / ".npm-global", REAL_HOME / ".cache/uv",
             REAL_HOME / ".local/share/uv/python"]
    paths += [Path("/etc") / p for p in ("ssl", "ld.so.cache", "resolv.conf", "hosts",
                                        "nsswitch.conf", "passwd", "group", "localtime")]
    for path in paths:
        if path.exists():
            args += ["--ro-bind", str(path.resolve()), str(path)]
    args += ["--dev", "/dev", "--proc", "/proc",
            "--bind", str(ws), str(ws), "--bind", str(home), str(home),
            # Clean environment: no API keys or base URLs leak in from Oliver's shell.
            "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
            "--setenv", "LANG", "C.UTF-8", "--setenv", "TERM", "dumb", "--setenv", "BROWSER", "true"]
    # Tool caches are read-only. Aider uses its cached environment directly.
    for path in readonly:
        args += ["--ro-bind", str(path), str(path)]
    return args + ["--setenv", "HOME", str(home),
                   "--setenv", "UV_CACHE_DIR", str(REAL_HOME / ".cache/uv"),
                   "--setenv", "UV_PYTHON_INSTALL_DIR", str(REAL_HOME / ".local/share/uv/python"),
                   "--chdir", str(ws), "--remount-ro", "/", "--die-with-parent"]


def harness_cmd(name, ws, prompt, stubs, test):
    if name == "keel":
        return [KEEL_BIN, "run", "--yolo", "--model", MODEL,
                "--base-url", BASE_URL, "--root", str(ws),
                "--log", str(ws.parent / "home/session.jsonl"), "--", prompt], ()
    if name == "opencode":
        (ws / "opencode.json").write_text(json.dumps(OPENCODE_CONFIG))
        # A fresh service port prevents connecting to a different attempt.
        svc = ws.parent / "home/.config/opencode/service.json"
        if not svc.exists():
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = str(sock.getsockname()[1])
            subprocess.run([*sandbox(ws, svc.parents[2]), str(REAL_HOME / ".npm-global/bin/opencode"),
                            "service", "set", "port", port], capture_output=True, timeout=60, check=True)
        return [str(REAL_HOME / ".npm-global/bin/opencode"), "run", "-m", f"llama/{MODEL}", prompt], ()
    if name == "aider":
        environments = sorted((REAL_HOME / ".cache/uv/environments-v2").glob("*/*/bin/aider"))
        if len(environments) != 1:
            raise RuntimeError("cache exactly one Aider environment with uv tool run first")
        return [str(environments[0].resolve()),
                "--model", f"openai/{MODEL}", "--openai-api-base", BASE_URL,
                "--openai-api-key", "local", "--yes-always", "--no-auto-commits",
                "--no-show-model-warnings", "--no-check-update", "--analytics-disable",
                "--no-gitignore", "--no-show-release-notes", "--no-browser", "--auto-test", "--test-cmd", f"/usr/bin/python3 -m pytest -q {test}",
                "--read", test, "--read", ".docs/instructions.md",
                "--message", prompt, *stubs], ()
    raise SystemExit(f"unknown harness {name}")


def tctl():
    out = subprocess.run(["sensors"], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.startswith("Tctl:"):
            value = float(line.split()[1].strip("+°C"))
            if math.isfinite(value) and value > 0:
                return value
    raise RuntimeError("CPU temperature unavailable; refusing to start inference")


def cool_down(limit=60.0, stop_at=""):
    """Start a task only on a cool machine (the Beelink gets loud above ~70)."""
    while (t := tctl()) >= limit:
        if stop_at and time.strftime("%H:%M") >= stop_at:
            raise TimeoutError("quiet-hours cutoff reached while cooling")
        print(f"cooling: Tctl {t} >= {limit}", flush=True)
        time.sleep(15)
    if stop_at and time.strftime("%H:%M") >= stop_at:
        raise TimeoutError("quiet-hours cutoff reached")


def prepare(src, ws):
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    shutil.copytree(src / ".docs", ws / ".docs")  # never .meta: it holds the solution
    tests = sorted(p.name for p in src.glob("*_test.py"))
    stubs = sorted(p.name for p in src.glob("*.py") if p.name not in tests)
    for f in tests + stubs:
        shutil.copy2(src / f, ws / f)
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
    subprocess.run(["git", "-c", "user.name=bench", "-c", "user.email=bench@local",
                    "commit", "-qm", "start"], cwd=ws, check=True)
    return stubs, tests


def run_one(name, src, out, timeout, rep=1):
    artifact = out / "attempts" / name / f"{src.name}.{rep}"
    artifact.mkdir(parents=True)  # Never overwrite evidence, even after a crash.
    with tempfile.TemporaryDirectory(prefix="pg-attempt-") as temporary:
        return execute(name, src, out, timeout, rep, artifact, Path(temporary))


def execute(name, src, out, timeout, rep, artifact, root):
    ws, home = root / "workspace", root / "home"
    home.mkdir()
    started = time.time()
    row = {"harness": name, "exercise": src.name, "rep": rep, "passed": False,
           "outcome": "setup_error", "agent_exit": None, "pytest": "", "secs": 0,
           "started_at": started, "artifact": str(artifact.relative_to(out)),
           "agent_log": str((artifact / "agent.log").relative_to(out))}
    try:
        attempt(name, src, timeout, artifact, ws, home, row)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        row["error"] = str(error)
        (artifact / "error.txt").write_text(str(error) + "\n" + str(getattr(error, "stderr", "")))
    finally:
        if ws.exists():
            shutil.copytree(ws, artifact / "workspace", symlinks=True,
                            ignore=lambda directory, names: [n for n in names if not (
                                (Path(directory) / n).is_symlink() or (Path(directory) / n).is_file()
                                or (Path(directory) / n).is_dir())])
        if (root / "score").exists():
            shutil.copytree(root / "score", artifact / "score", symlinks=True)
        session = home / "session.jsonl"
        if session.is_file() and not session.is_symlink():
            shutil.copy2(session, artifact / "session.jsonl")
            row["session"] = str((artifact / "session.jsonl").relative_to(out))
        row["finished_at"] = time.time()
        (artifact / "result.json").write_text(json.dumps(row, indent=2) + "\n")
        with (out / "results.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
    return row


def attempt(name, src, timeout, artifact, ws, home, row):
    stubs, tests = prepare(src, ws)
    shutil.copytree(ws, artifact / "input", ignore=shutil.ignore_patterns(".git"))
    row["input_sha256"] = {str(p.relative_to(artifact / "input")): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (artifact / "input").rglob("*") if p.is_file()}
    test = tests[0]
    append = " and .docs/instructions.append.md" if (ws / ".docs/instructions.append.md").exists() else ""
    prompt = PROMPT.format(append=append, stubs=", ".join(stubs), test=test)
    cmd, extra = harness_cmd(name, ws, prompt, stubs, test)

    subprocess.run([*sandbox(ws, home), "python3", "-c", "import pytest"],
                   capture_output=True, text=True, check=True, timeout=20)
    (artifact / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    with Path(cmd[0]).open("rb") as binary:
        row["agent_executable_sha256"] = hashlib.file_digest(binary, "sha256").hexdigest()
    startup = subprocess.run([*sandbox(ws, home), cmd[0], "--version"],
                             capture_output=True, text=True, timeout=30)
    (artifact / "startup.stdout").write_text(startup.stdout)
    (artifact / "startup.stderr").write_text(startup.stderr)
    startup.check_returncode()
    t0 = time.time()
    peak = tctl()
    row["outcome"] = "agent_error"
    with open(artifact / "agent.log", "w") as log:
        agent = subprocess.Popen(["nice", "-n", "19", *sandbox(ws, home, extra), *cmd],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        agent_exit = None
        try:
            while agent_exit is None:
                try:
                    agent_exit = agent.wait(timeout=min(10, timeout))
                except subprocess.TimeoutExpired:
                    peak = max(peak, tctl())
                    if peak > 95 or time.time() - t0 > timeout:
                        agent_exit = "too_hot" if peak > 95 else "timeout"
        finally:
            if agent.poll() is None:
                os.killpg(agent.pid, 9)
            agent.wait()
            row.update(secs=round(time.time() - t0, 1), agent_exit=agent_exit, peak_tctl=peak)
    secs = round(time.time() - t0, 1)
    row.update(secs=secs, agent_exit=agent_exit, peak_tctl=peak, outcome="scoring_error")
    row["tampered_test"] = any((ws / t).is_symlink() or not (ws / t).is_file()
                               or (ws / t).read_bytes() != (src / t).read_bytes() for t in tests)
    score_ws = ws.parent / "score"
    score_ws.mkdir()
    for f in stubs:
        if (ws / f).is_symlink() or not (ws / f).is_file():
            row.update(outcome="completed", pytest=f"unsafe or missing solution: {f}")
            return
        shutil.copy2(ws / f, score_ws / f)
    for t in tests:
        shutil.copy2(src / t, score_ws / t)
    score_home = ws.parent / "score-home"
    score_home.mkdir()
    try:
        score = subprocess.run(["nice", "-n", "19", *sandbox(score_ws, score_home, [score_ws / t for t in tests]),
                                "--setenv", "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1", "python3", "-m", "pytest",
                                "-q", "-p", "no:cacheprovider", *tests],
                               stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
        (artifact / "score.stdout").write_text(score.stdout)
        (artifact / "score.stderr").write_text(score.stderr)
        tail = score.stdout.strip().splitlines()[-1:] or [""]
        row.update(score_exit=score.returncode, pytest=tail[0][:200])
        if score.returncode in (0, 1, 2) and re.search(r"\b\d+ (passed|failed|errors?)\b", tail[0]):
            row.update(passed=score.returncode == 0, outcome="completed")
    except subprocess.TimeoutExpired as error:
        row["pytest"] = "pytest timeout"
        for filename, value in (("score.stdout", error.stdout), ("score.stderr", error.stderr)):
            (artifact / filename).write_bytes(value.encode() if isinstance(value, str) else value or b"")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exercises", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--harness", default="keel,opencode,aider")
    ap.add_argument("--only", default="")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--keel-bin", default=KEEL_BIN)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--stop-at", default="")
    a = ap.parse_args()
    if a.timeout <= 0 or a.repeat <= 0 or set(a.harness.split(",")) - {"keel", "opencode", "aider"}:
        ap.error("positive timeout/repeat and known harness names are required")
    if a.stop_at:
        time.strptime(a.stop_at, "%H:%M")
    a.out, a.exercises = a.out.resolve(), a.exercises.resolve()
    names = sorted(p.name for p in a.exercises.iterdir() if p.is_dir())
    if a.only:
        if set(a.only.split(",")) - set(names):
            ap.error("unknown exercise name")
        names = [n for n in names if n in a.only.split(",")]
    if not names:
        ap.error("no exercises selected")
    globals()["KEEL_BIN"] = str(Path(a.keel_bin).resolve())
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": "isolated-attempt-v1", "model": MODEL, "base_url": BASE_URL,
                "context": CONTEXT, "sampling": "stock harness defaults; not request-verified",
                "timeout": a.timeout, "repeat": a.repeat, "exercises": names, "harness": a.harness,
                "keel_sha256": hashlib.sha256(Path(KEEL_BIN).read_bytes()).hexdigest(),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "dataset_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=a.exercises, text=True).strip()}
    with urllib.request.urlopen(BASE_URL.removesuffix("/v1") + "/props", timeout=5) as response:
        props = json.load(response)
    manifest["server_defaults"] = {k: props[k] for k in ("model_alias", "model_path", "default_generation_settings") if k in props}
    metadata = a.out / "manifest.json"
    if metadata.exists():
        if json.loads(metadata.read_text()) != manifest:
            raise SystemExit("run settings changed; use a new output directory")
    elif (a.out / "results.jsonl").exists():
        raise SystemExit("legacy results: use a new output directory")
    else:
        metadata.write_text(json.dumps(manifest, indent=2) + "\n")
    done = set()
    if (a.out / "results.jsonl").exists():  # resume
        for line in (a.out / "results.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("outcome") != "completed":
                raise SystemExit("previous infrastructure failure: use a new output directory")
            done.add((r["harness"], r["exercise"], r["rep"]))
    for rep in range(1, a.repeat + 1):
        for ex in names:
            for h in a.harness.split(","):
                if (h, ex, rep) in done:
                    continue
                try:
                    cool_down(stop_at=a.stop_at)
                except TimeoutError as error:
                    print(error)
                    return
                r = run_one(h, a.exercises / ex, a.out, a.timeout, rep)
                print(json.dumps(r), flush=True)
                if r.get("peak_tctl", 0) > 95 or r["outcome"] != "completed":
                    raise SystemExit("benchmark stopped; inspect the archived attempt before resuming")


if __name__ == "__main__":
    sys.exit(main())
