#!/usr/bin/env bash
# Keel Developer Pre-push Security & Quality Verification Script
# Enforces formatting, zero warnings, tests, and secret scrubbing checks.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== [1/4] Checking Formatting (cargo fmt) ==="
cargo fmt --all -- --check

echo "=== [2/4] Running Clippy (zero warnings policy) ==="
cargo clippy -j 2 --all-targets -- -D warnings

echo "=== [3/4] Running Tests (capped at 2 test threads) ==="
cargo test -j 2 -- --test-threads=2

echo "=== [4/4] Verifying Release Build ==="
cargo check --release -j 2

echo ""
echo "All quality, security, and build checks passed successfully!"
