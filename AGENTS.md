# Keel: rules for agents

Keel is a Rust coding-agent harness. The goal right now is **fewer, better
features**: make the basic agent loop, tools and sandbox reliable before
anything new. Many agents and models work here, including small local ones,
so this file is short on purpose.

## Before you open a PR

1. One topic per PR. If you find a second problem, write it down in the PR
   body under "Follow-ups" and stop.
2. Run `scripts/check.sh`. It must pass.
3. Prove the change with a **real run** of the `keel` binary (not only unit
   tests). Paste the command and its output into the PR's `## Proof` section.
   A test that only checks a constant or a list is not proof.
4. Keep it small: at most 600 changed lines (Cargo.lock excluded).
5. No new modules (`crates/*/src/*.rs`), no new `pub mod`, no new CLI
   subcommands.

CI (`pr-gate` job) checks 3, 4 and 5 and blocks the PR. Only Oliver may add
these override labels: `large-pr`, `new-feature-approved`, `frozen-ok`.
**Never add them yourself.**

## Frozen modules

Do not extend these. Bug fixes need the `frozen-ok` label from Oliver:
`epistemic.rs`, `swarm.rs`, `tree.rs`, `speculate.rs`, `cascade.rs`,
`recorder.rs`, `daemon.rs`.

## PR flow

- Branch from current `origin/main`. Never stack PRs on other open PRs.
- At most 3 open PRs at a time (CI checks this too).
- Open PRs ready for review, not draft. Say "ready to merge" only when every
  check on the latest commit is green. Red CI is yours to fix.

## Where things are

- `crates/keel-kernel`: plugin kernel (small, keep it that way).
- `crates/keel-agent/src/lib.rs`: agent loop. `tools.rs`: tools.
  `sandbox.rs`: Landlock. `compaction.rs`: context handling.
- `crates/keel-cli/src/main.rs`: CLI entry; `tui.rs`: terminal cockpit;
  `bench.rs`: `keel bench` tasks.
- Local model for real runs: `--base-url http://localhost:8080/v1 --model bonsai-9b`
  (llama-server on the Beelink, if running).

## Safety

Never print API keys or tokens. Never run Keel with `~` as the workspace;
use a scratch directory such as `/tmp/keel-proof`.
