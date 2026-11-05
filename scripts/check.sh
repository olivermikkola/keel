#!/usr/bin/env bash
# Run before opening a PR: the same build, lint and test steps as CI.
# Low priority and few jobs so it stays quiet on the Beelink.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

run() { echo "=== $*"; nice -n 19 "$@"; }

run cargo fmt --all -- --check
run cargo clippy --locked -j 2 --all-targets -- -D warnings
run cargo test --locked -j 2 -- --test-threads=2
run cargo build --release --locked -j 2

echo "All checks passed. Now prove it with a real keel run (see AGENTS.md)."
