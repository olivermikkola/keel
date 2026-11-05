#!/usr/bin/env python3
"""Read a live polyglot run without changing it or contacting a model."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re


def read_jsonl(path, warnings):
    """A writer may still be appending the last line. Never hide earlier damage."""
    raw = path.read_text()
    lines = raw.splitlines()
    rows = []
    for number, line in enumerate(lines, 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            if number == len(lines) and not raw.endswith("\n"):
                warnings.append(f"{path.name}: incomplete final record ignored")
            else:
                raise ValueError(f"{path}:{number}: invalid JSON") from error
    return rows


def test_counts(summary):
    return {kind: int(n) for n, kind in re.findall(
        r"\b(\d+) (passed|failed|errors?|skipped|xfailed|xpassed)\b", summary
    )}


def fraction(row):
    counts = test_counts(row.get("pytest", ""))
    total = counts.get("passed", 0) + counts.get("failed", 0)
    if counts.get("error") or counts.get("errors") or not total:
        return None
    return counts.get("passed", 0) / total


def load_results(path, warnings):
    rows = read_jsonl(path, warnings)
    seen = set()
    for row in rows:
        key = (row["harness"], row["exercise"], row.get("rep", 1))
        if (any(not isinstance(v, str) or not re.fullmatch(r"[\w-]+", v)
                for v in key[:2]) or type(key[2]) is not int or key[2] < 1):
            raise ValueError(f"invalid attempt identity: {key}")
        if key in seen:
            raise ValueError(f"duplicate attempt: {key}")
        seen.add(key)
        seconds = row["secs"]
        if (not isinstance(row["passed"], bool)
                or type(seconds) not in (int, float)
                or not math.isfinite(seconds) or seconds < 0):
            raise ValueError(f"invalid result: {key}")
    return rows


def session_metrics(path, warnings):
    """Count persisted events; do not infer tokens or unseen final model output."""
    events = read_jsonl(path, warnings)
    start = next((e["ts"] for e in events if e.get("kind") == "user"), None)
    calls, pending, receipts = [], {}, []
    compactions = 0
    for event in events:
        data = event.get("data", {})
        if event.get("kind") == "assistant":
            for call in data.get("tool_calls", []):
                item = {**call, "ts": event["ts"]}
                pending[call["id"]] = item
                calls.append(item)
        elif event.get("kind") == "tool":
            call = pending.pop(data["id"], None)
            receipts.append((call, event))
        elif event.get("kind") == "compaction":
            compactions += 1
    edits = [c for c in calls if c["name"] in ("write_file", "edit_file", "patch_file", "apply_patch")]
    # Shell edits cannot be reliably identified from arbitrary shell syntax.
    tests = [c for c in calls if c["name"] == "shell"
             and re.search(r"\bpytest\b", c.get("args", {}).get("command", ""))]
    outputs = [e.get("data", {}).get("content", "") for _, e in receipts]
    elapsed = sum(max(0, e["ts"] - c["ts"]) for c, e in receipts if c) / 1000
    test_results = []
    for call, event in receipts:
        if call and call in tests:
            content = event["data"].get("content", "")
            summaries = re.findall(r"[^\n]*(?:\d+ passed|\d+ failed|\d+ errors?)[^\n]*", content)
            test_results.append({"after_seconds": (event["ts"] - start) / 1000
                                 if start is not None else None,
                                 "summary": summaries[-1] if summaries else None})
    return {
        "file": path.name,
        "tool_calls": dict(Counter(c["name"] for c in calls)),
        "tool_receipts": len(receipts), "pending_calls": len(pending),
        "explicit_edit_calls": len(edits), "pytest_calls": len(tests),
        "first_explicit_edit_seconds": (edits[0]["ts"] - start) / 1000
        if edits and start is not None else None,
        "tool_receipt_seconds": round(elapsed, 3),
        "raw_tool_output_chars": sum(map(len, outputs)),
        "largest_raw_tool_output_chars": max(map(len, outputs), default=0),
        "compactions_logged": compactions, "test_feedback": test_results,
        "edit_after_last_pytest": bool(edits and tests and edits[-1]["ts"] > tests[-1]["ts"]),
    }


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def report(run, harnesses, sessions):
    if not harnesses or any(not h for h in harnesses) or len(set(harnesses)) != len(harnesses):
        raise ValueError("select distinct, nonempty harness names")
    warnings = []
    rows = load_results(run / "results.jsonl", warnings)
    blocks = {}
    for row in rows:
        blocks.setdefault((row["exercise"], row.get("rep", 1)), {})[row["harness"]] = row
    matched = [v for v in blocks.values() if set(harnesses) <= v.keys()]
    diagnostics = [session_metrics(p, warnings) for p in sessions]
    attempts = []
    for row in rows:
        rep = row.get("rep", 1)
        log = run / row["harness"] / f"{row['exercise']}.{rep}.log"
        if not log.exists() and rep == 1:
            log = run / row["harness"] / f"{row['exercise']}.log"
        text = log.read_text(errors="replace") if log.exists() else ""
        text = re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text))
        context_error = bool(re.search(
            r"exceeds the available context|context_length_exceeded|exceed_context_size", text
        ))
        attempts.append({**row, "rep": rep, "test_fraction": fraction(row),
                         "agent_log_present": log.exists(),
                         "context_error_text_seen": context_error})
    result = {"recorded_attempts": len(rows), "matched_blocks": len(matched),
              "harnesses": harnesses, "warnings": warnings,
              "attempts": attempts, "sessions": diagnostics}
    lines = ["# Polyglot evidence", "",
             f"{len(rows)} recorded attempts; {len(matched)} complete matched blocks.", "",
             "Only blocks completed by every selected harness enter this table.", "",
             "| Harness | Solved | Matched runs | Mean tests passing | Scorable runs | Timeouts |",
             "|---|---:|---:|---:|---:|---:|"]
    for harness in harnesses:
        selected = [b[harness] for b in matched]
        fractions = [f for r in selected if (f := fraction(r)) is not None]
        mean = f"{sum(fractions) / len(fractions):.1%}" if fractions else "unknown"
        lines.append(f"| {cell(harness)} | {sum(r['passed'] for r in selected)} | "
                     f"{len(selected)} | {mean} | {len(fractions)} | "
                     f"{sum(r.get('agent_exit') == 'timeout' for r in selected)} |")
    lines += ["", "Partial credit averages each attempt's passing-test fraction; it is not a solve rate.",
              "Collection errors and missing test counts remain unknown, rather than zero.", "",
              "## Recorded attempts", "",
              "| Exercise | Rep | Harness | Solved | Agent exit | Context error text | Original tests |",
              "|---|---:|---|---|---|---|---|"]
    for r in attempts:
        values = [r["exercise"], r["rep"], r["harness"], r["passed"],
                  r.get("agent_exit", "unknown"), r["context_error_text_seen"], r.get("pytest", "")]
        lines.append("| " + " | ".join(map(cell, values)) + " |")
    lines += ["", "## Keel session evidence", "",
              "Sessions are listed separately: legacy results do not identify their session.",
              "No session is automatically attributed to an exercise or repetition.", "",
              "| Session | First edit call (s) | Edit calls | Pytest calls | Raw tool chars | Largest result | Compactions | Edit call after last test |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for d in diagnostics:
        values = [d["file"], d["first_explicit_edit_seconds"], d["explicit_edit_calls"],
                  d["pytest_calls"], d["raw_tool_output_chars"], d["largest_raw_tool_output_chars"],
                  d["compactions_logged"], d["edit_after_last_pytest"]]
        lines.append("| " + " | ".join(map(cell, values)) + " |")
    lines += ["", "## Interpretation limits", "",
              "- A timeout records the budget limit, not its underlying cause.",
              "- Context flags are log text matches; absence is not proof no overflow occurred.",
              "- Session output sizes are persisted characters before pruning, not model input tokens.",
              "- Reasoning streamed after the last saved event may be absent from session metrics.",
              "- Edit counts are calls, not proof of successful writes; shell edits are excluded.",
              "- Stock harness defaults can differ; this report does not verify matched sampling.",
              "- This pilot's read-only root mount leaves reference solutions reachable by absolute path.",
              "- Do not change architecture or declare a winner from partial credit on this small sample."]
    lines += ["", *[f"Warning: {w}" for w in warnings]]
    return result, "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--harnesses", default="keel,opencode,aider")
    ap.add_argument("--sessions", type=Path, nargs="*", default=[])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    try:
        data, markdown = report(args.run, args.harnesses.split(","), args.sessions)
    except (ValueError, KeyError, OSError, TypeError) as error:
        ap.exit(1, f"analysis failed: {error}\n")
    print(json.dumps(data, indent=2) if args.json else markdown, end="\n" if args.json else "")


if __name__ == "__main__":
    main()
