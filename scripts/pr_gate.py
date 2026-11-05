#!/usr/bin/env python3
"""PR gate: the hoops a Keel pull request must pass before review.

Usage:
  scripts/pr_gate.py --base origin/main --body-file body.md \
      [--labels large-pr,frozen-ok] [--open-prs N]

Checks (see AGENTS.md):
  1. The PR body has a filled `## Proof` section with a fenced code block
     showing a real `keel` run.
  2. At most MAX_LINES changed lines, Cargo.lock excluded (label: large-pr).
  3. No new source modules, `pub mod` lines or CLI subcommands
     (label: new-feature-approved).
  4. No changes to frozen modules (label: frozen-ok).
  5. At most MAX_OPEN_PRS open PRs, this one included.

Exits 1 and prints every failure if any check fails.
"""

import argparse
import re
import subprocess
import sys

MAX_LINES = 600
MAX_OPEN_PRS = 3
FROZEN = [
    "crates/keel-agent/src/epistemic.rs",
    "crates/keel-agent/src/swarm.rs",
    "crates/keel-agent/src/tree.rs",
    "crates/keel-agent/src/speculate.rs",
    "crates/keel-agent/src/cascade.rs",
    "crates/keel-agent/src/recorder.rs",
    "crates/keel-agent/src/daemon.rs",
]
ARGS_RS = "crates/keel-cli/src/args.rs"
SOURCE_RE = re.compile(r"^crates/[^/]+/src/.+\.rs$")


def git(*args):
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def proof_failures(body):
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    m = re.search(r"^##\s*Proof\s*$(.*?)(?=^##\s|\Z)", body, flags=re.M | re.S)
    if not m:
        return ["PR body has no `## Proof` section (use the PR template)."]
    blocks = re.findall(r"```[^\n]*\n(.*?)```", m.group(1), flags=re.S)
    real = [b for b in blocks if len([l for l in b.splitlines() if l.strip()]) >= 2]
    if not real:
        return ["`## Proof` needs a fenced code block with a real run (command + output)."]
    if not any(re.search(r"(^|[\s/])keel([ \t]|$)", b, flags=re.M) for b in real):
        return ["`## Proof` must show a run of the `keel` binary, not only tests."]
    return []


def size_failures(base, labels):
    changed = 0
    for line in git("diff", "--numstat", f"{base}...HEAD").splitlines():
        add, rem, path = line.split("\t", 2)
        if path.endswith("Cargo.lock") or add == "-":
            continue
        changed += int(add) + int(rem)
    if changed > MAX_LINES and "large-pr" not in labels:
        return [f"{changed} changed lines (limit {MAX_LINES}). Split the PR, or ask Oliver for `large-pr`."]
    return []


def subcommands(text):
    m = re.search(r"SUBCOMMANDS: &\[&str\] = &\[(.*?)\];", text, flags=re.S)
    return set(re.findall(r'"([^"]+)"', m.group(1))) if m else set()


def feature_failures(base, labels):
    if "new-feature-approved" in labels:
        return []
    out = []
    added = git("diff", "--name-only", "--diff-filter=A", f"{base}...HEAD").split()
    new_src = [p for p in added if SOURCE_RE.match(p)]
    if new_src:
        out.append("New source modules: " + ", ".join(new_src))
    diff = git("diff", "-U0", f"{base}...HEAD", "--", "crates")
    new_mods = re.findall(r"^\+\s*pub mod (\w+)", diff, flags=re.M)
    if new_mods:
        out.append("New `pub mod`: " + ", ".join(new_mods))
    try:
        before = subcommands(git("show", f"{base}:{ARGS_RS}"))
        after = subcommands(open(ARGS_RS).read())
        if after - before:
            out.append("New CLI subcommands: " + ", ".join(sorted(after - before)))
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    if out:
        out.append("New features need Oliver's `new-feature-approved` label.")
    return out


def frozen_failures(base, labels):
    if "frozen-ok" in labels:
        return []
    touched = git("diff", "--name-only", f"{base}...HEAD").split()
    hit = [p for p in touched if p in FROZEN]
    if hit:
        return ["Frozen modules changed: " + ", ".join(hit) + " (needs Oliver's `frozen-ok`)."]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--body-file", required=True)
    ap.add_argument("--labels", default="")
    ap.add_argument("--open-prs", type=int, default=0)
    a = ap.parse_args()
    labels = {l.strip() for l in a.labels.split(",") if l.strip()}
    body = open(a.body_file, encoding="utf-8").read()

    failures = (
        proof_failures(body)
        + size_failures(a.base, labels)
        + feature_failures(a.base, labels)
        + frozen_failures(a.base, labels)
    )
    if a.open_prs > MAX_OPEN_PRS:
        failures.append(
            f"{a.open_prs} open PRs (limit {MAX_OPEN_PRS}). Get one merged or closed first."
        )

    if failures:
        print("PR gate FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PR gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
