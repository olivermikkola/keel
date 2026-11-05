#!/usr/bin/env python3
"""Compare Keel's stock prompt with explicit edit/test/fix instructions."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

import analyze
import run


TASKS = ("affine-cipher", "bowling", "food-chain")
GUIDANCE = (
    " After each substantive solution edit, run the test command above before "
    "editing again. If tests fail, identify a specific reported failure and fix "
    "its cause; avoid broad rewrites unrelated to the failures. Then run the "
    "tests again. Finish only after checking the latest solution."
)
PROMPTS = {"default": run.PROMPT, "test-after-edit": run.PROMPT + GUIDANCE}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schedule(repeat):
    for rep in range(1, repeat + 1):
        for index, task in enumerate(TASKS):
            variants = list(PROMPTS)
            if (rep + index) % 2 == 0:
                variants.reverse()
            for variant in variants:
                yield variant, task, rep


def slots_idle():
    with urllib.request.urlopen(run.BASE_URL.removesuffix("/v1") + "/slots", timeout=5) as response:
        return not any(slot["is_processing"] for slot in json.load(response))


def pilot_complete(path):
    if not path.exists():
        return False
    rows = analyze.load_results(path, [])
    expected = {(h, task, rep) for h in ("keel", "opencode", "aider")
                for task in (*TASKS, "grep", "phone-number", "proverb", "scale-generator",
                             "tree-building", "zebra-puzzle") for rep in (1, 2)}
    return expected <= {(r["harness"], r["exercise"], r.get("rep", 1)) for r in rows}


def time_available(stop_at, timeout):
    now = time.localtime()
    hour, minute = map(int, stop_at.split(":"))
    remaining = hour * 3600 + minute * 60 - (now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec)
    return remaining > timeout + 30


def snapshot(args):
    with urllib.request.urlopen(run.BASE_URL.removesuffix("/v1") + "/props", timeout=5) as response:
        props = json.load(response)
    inputs = {}
    for task in TASKS:
        source = args.exercises / task
        paths = list((source / ".docs").rglob("*")) + list(source.glob("*.py"))
        inputs[task] = {str(p.relative_to(source)): digest(p) for p in sorted(paths) if p.is_file()}
    return {"protocol": "keel-test-after-edit-v1", "prompts": PROMPTS,
            "schedule": list(schedule(args.repeat)), "timeout": args.timeout,
            "keel_sha256": digest(Path(run.KEEL_BIN)), "model": run.MODEL,
            "base_url": run.BASE_URL, "context": run.CONTEXT,
            "sampling": "same Keel defaults; actual request parameters not captured",
            "dataset_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=args.exercises, text=True).strip(),
            "input_sha256": inputs, "runner_sha256": digest(Path(run.__file__)),
            "comparison_sha256": digest(Path(__file__)),
            "server_defaults": {k: props[k] for k in
                                ("model_alias", "model_path", "default_generation_settings") if k in props}}


def write_report(out):
    records = []
    lines = ["# Keel edit/test/fix comparison", "",
             "Only the user prompt differs. Final scores use protected original tests.",
             "A prompt experiment does not enforce testing in Keel's production loop.", "",
             "| Task | Rep | Prompt | Solved | Original tests | First pytest (s) | Edits | Pytests |",
             "|---|---:|---|---|---|---:|---:|---:|"]
    for variant in PROMPTS:
        root = out / variant
        if not (root / "results.jsonl").exists():
            continue
        for row in analyze.load_results(root / "results.jsonl", []):
            metrics = analyze.session_metrics(root / row["session"], []) if row.get("session") else {}
            first = next(iter(metrics.get("test_feedback", [])), {}).get("after_seconds")
            record = {**row, "variant": variant, "metrics": metrics}
            records.append(record)
            values = [row["exercise"], row["rep"], variant, row["passed"], row["pytest"],
                      first, metrics.get("explicit_edit_calls"), metrics.get("pytest_calls")]
            lines.append("| " + " | ".join(map(analyze.cell, values)) + " |")
    lines += ["", "Compare completed pairs, not unequal running totals. Repeats are not independent tasks.",
              "Session metrics omit shell edits; more tests alone is not success.",
              "Inspect edits and test output to establish whether a reported failure was corrected."]
    (out / "report.json").write_text(json.dumps(records, indent=2) + "\n")
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exercises", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--keel-bin", required=True, type=Path)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--stop-at", default="19:30")
    ap.add_argument("--wait-for-pilot", type=Path)
    args = ap.parse_args()
    if args.repeat <= 0 or args.timeout <= 0:
        ap.error("positive repeat and timeout required")
    time.strptime(args.stop_at, "%H:%M")
    args.out, args.exercises = args.out.resolve(), args.exercises.resolve()
    run.KEEL_BIN = str(args.keel_bin.resolve())
    while args.wait_for_pilot and not pilot_complete(args.wait_for_pilot):
        if not time_available(args.stop_at, args.timeout):
            raise SystemExit("quiet-hours cutoff; pilot left unchanged, no inference started")
        print("Waiting for all 54 pilot attempts; no competing inference started.", flush=True)
        time.sleep(45)
    if not slots_idle():
        raise SystemExit("model server busy; refusing overlapping inference")
    args.out.mkdir(parents=True, exist_ok=True)
    settings = snapshot(args)
    metadata = args.out / "manifest.json"
    # JSON normalizes the schedule's tuples to lists.
    settings = json.loads(json.dumps(settings))
    if metadata.exists():
        if json.loads(metadata.read_text()) != settings:
            raise SystemExit("experiment settings changed; use a new output directory")
    else:
        if any(args.out.iterdir()):
            raise SystemExit("output already contains untracked evidence; use a new directory")
        metadata.write_text(json.dumps(settings, indent=2) + "\n")
    done = set()
    for variant in PROMPTS:
        ledger = args.out / variant / "results.jsonl"
        if ledger.exists():
            for row in analyze.load_results(ledger, []):
                if row.get("outcome") != "completed":
                    raise SystemExit("infrastructure failure; inspect evidence before a new run")
                done.add((variant, row["exercise"], row["rep"]))
    # A real model-backed smoke precedes the timed comparison.
    smoke = args.out / "smoke"
    if not (smoke / "results.jsonl").exists():
        if not time_available(args.stop_at, min(args.timeout, 120)):
            raise SystemExit("quiet-hours cutoff; no smoke started")
        source = args.out / "smoke-source"
        (source / ".docs").mkdir(parents=True)
        (source / ".docs/instructions.md").write_text("Implement answer() returning the integer 42.")
        (source / "solution.py").write_text("def answer():\n    pass\n")
        (source / "solution_test.py").write_text("from solution import answer\ndef test_answer():\n    assert answer() == 42\n")
        run.cool_down(stop_at=args.stop_at)
        row = run.run_one("keel", source, smoke, min(args.timeout, 120))
        print("SMOKE", json.dumps(row), flush=True)
    smoke_rows = analyze.load_results(smoke / "results.jsonl", [])
    if (len(smoke_rows) != 1 or smoke_rows[0].get("outcome") != "completed"
            or not smoke_rows[0]["passed"] or smoke_rows[0].get("peak_tctl", 0) > 95):
        raise SystemExit("model-backed smoke failed; no comparison started")
    for variant, task, rep in schedule(args.repeat):
        if (variant, task, rep) in done:
            continue
        if not time_available(args.stop_at, args.timeout):
            print("Quiet-hours cutoff; resume the same command for remaining attempts.", flush=True)
            break
        run.cool_down(stop_at=args.stop_at)
        if not time_available(args.stop_at, args.timeout) or not slots_idle():
            break
        if json.loads(json.dumps(snapshot(args))) != settings:
            raise SystemExit("binary, input, code or server settings changed; stopping")
        run.PROMPT = PROMPTS[variant]
        row = run.run_one("keel", args.exercises / task, args.out / variant, args.timeout, rep)
        print(variant, json.dumps(row), flush=True)
        write_report(args.out)
        if row.get("outcome") != "completed" or row.get("peak_tctl", 0) > 95:
            raise SystemExit("attempt stopped; inspect archived evidence")
    write_report(args.out)


if __name__ == "__main__":
    main()
