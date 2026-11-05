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
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE_URL = "http://127.0.0.1:8080/v1"
MODEL = "bonsai-9b"
CONTEXT = 16384
REAL_HOME = Path.home()
KEEL_BIN = str(REAL_HOME / ".local/bin/keel")  # --keel-bin; must not live under /tmp

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


def sandbox(ws, home, extra_rw=()):
    """bwrap prefix: whole system read-only, task folder + HOME writable."""
    args = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
            "--tmpfs", "/tmp",
            "--bind", str(ws), str(ws), "--bind", str(home), str(home),
            # Clean environment: no API keys or base URLs leak in from Oliver's shell.
            "--clearenv", "--setenv", "PATH", f"{REAL_HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin",
            "--setenv", "LANG", "C.UTF-8", "--setenv", "TERM", "dumb", "--setenv", "BROWSER", "true"]
    for p in extra_rw:
        args += ["--bind", str(p), str(p)]
    return args + ["--setenv", "HOME", str(home),
                   "--setenv", "UV_CACHE_DIR", str(REAL_HOME / ".cache/uv"),
                   "--setenv", "UV_PYTHON_INSTALL_DIR", str(REAL_HOME / ".local/share/uv/python"),
                   "--chdir", str(ws), "--die-with-parent"]


def harness_cmd(name, ws, prompt, stubs, test):
    if name == "keel":
        return [KEEL_BIN, "run", "--yolo", "--model", MODEL,
                "--base-url", BASE_URL, "--root", str(ws), "--", prompt], ()
    if name == "opencode":
        (ws / "opencode.json").write_text(json.dumps(OPENCODE_CONFIG))
        # Own service port, so it never collides with Oliver's running OpenCode.
        svc = Path("/tmp/pg-home/opencode/.config/opencode/service.json")
        if not svc.exists():
            subprocess.run([*sandbox(ws, svc.parents[2]), str(REAL_HOME / ".npm-global/bin/opencode"),
                            "service", "set", "port", "49391"], capture_output=True, timeout=60)
        return [str(REAL_HOME / ".npm-global/bin/opencode"), "run", "-m", f"llama/{MODEL}", prompt], ()
    if name == "aider":
        return [str(REAL_HOME / ".local/bin/uv"), "tool", "run", "--offline", "--python", "3.12",
                "--from", "aider-chat", "aider",
                "--model", f"openai/{MODEL}", "--openai-api-base", BASE_URL,
                "--openai-api-key", "local", "--yes-always", "--no-auto-commits",
                "--no-show-model-warnings", "--no-check-update", "--analytics-disable",
                "--no-gitignore", "--no-show-release-notes", "--no-browser", "--auto-test", "--test-cmd", f"/usr/bin/python3 -m pytest -q {test}",
                "--read", test, "--read", ".docs/instructions.md",
                "--message", prompt, *stubs], (REAL_HOME / ".cache/uv",)
    raise SystemExit(f"unknown harness {name}")


def tctl():
    out = subprocess.run(["sensors"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("Tctl:"):
            return float(line.split()[1].strip("+°C"))
    return 0.0


def cool_down(limit=60.0):
    """Start a task only on a cool machine (the Beelink gets loud above ~70)."""
    while (t := tctl()) >= limit:
        print(f"cooling: Tctl {t} >= {limit}", flush=True)
        time.sleep(15)


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


def run_one(name, src, out, timeout):
    ws = Path("/tmp/pg-run") / name / src.name
    home = Path("/tmp/pg-home") / name
    home.mkdir(parents=True, exist_ok=True)
    stubs, tests = prepare(src, ws)
    test = tests[0]
    append = " and .docs/instructions.append.md" if (ws / ".docs/instructions.append.md").exists() else ""
    prompt = PROMPT.format(append=append, stubs=", ".join(stubs), test=test)
    cmd, extra = harness_cmd(name, ws, prompt, stubs, test)

    logdir = out / name
    logdir.mkdir(parents=True, exist_ok=True)
    cool_down()
    t0 = time.time()
    peak = tctl()
    with open(logdir / f"{src.name}.log", "w") as log:
        agent = subprocess.Popen(["nice", "-n", "19", *sandbox(ws, home, extra), *cmd],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        agent_exit = None
        while agent_exit is None:
            try:
                agent_exit = agent.wait(timeout=10)
            except subprocess.TimeoutExpired:
                peak = max(peak, tctl())
                if peak > 95 or time.time() - t0 > timeout:
                    os.killpg(agent.pid, 9)
                    agent.wait()
                    agent_exit = "too_hot" if peak > 95 else "timeout"
    secs = round(time.time() - t0, 1)
    if agent_exit == "too_hot":
        raise SystemExit(f"stopping benchmark: Tctl {peak} > 95")

    tampered = (ws / test).read_bytes() != (src / test).read_bytes()
    for t in tests:  # score against the original tests only
        shutil.copy2(src / t, ws / t)
    try:
        score = subprocess.run(["nice", "-n", "19", *sandbox(ws, home), "python3", "-m", "pytest",
                                "-q", "-p", "no:cacheprovider", *tests],
                               stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
        passed = score.returncode == 0
        tail = score.stdout.strip().splitlines()[-1:] or [""]
    except subprocess.TimeoutExpired:
        passed, tail = False, ["pytest timeout"]
    row = {"harness": name, "exercise": src.name, "passed": passed, "secs": secs,
           "agent_exit": agent_exit, "peak_tctl": peak, "tampered_test": tampered, "pytest": tail[0][:200]}
    with open(out / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exercises", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--harness", default="keel,opencode,aider")
    ap.add_argument("--only", default="")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--keel-bin", default=KEEL_BIN)
    a = ap.parse_args()
    globals()["KEEL_BIN"] = str(Path(a.keel_bin).resolve())
    a.out.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in a.exercises.iterdir() if p.is_dir())
    if a.only:
        names = [n for n in names if n in a.only.split(",")]
    done = set()
    if (a.out / "results.jsonl").exists():  # resume
        for line in open(a.out / "results.jsonl"):
            r = json.loads(line)
            done.add((r["harness"], r["exercise"]))
    for ex in names:  # interleave harnesses so a stop leaves comparable data
        for h in a.harness.split(","):
            if (h, ex) in done:
                continue
            r = run_one(h, a.exercises / ex, a.out, a.timeout)
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    sys.exit(main())
