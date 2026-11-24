<img src="docs/brand/mark.svg" width="64" height="64" alt="Keel jade and rust sail" />

# Keel Code

[![CI](https://github.com/olivermikkola/keel-code/actions/workflows/ci.yml/badge.svg)](https://github.com/olivermikkola/keel-code/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**A coding agent that runs on small local models — in your terminal.**
Point Keel Code (`keel`) at a model on your own machine (2–27B, e.g. Bonsai or Qwen on
llama.cpp), say what you want changed, and it reads, edits and tests code
while you approve the risky parts. Think Claude Code, except the model is
yours and nothing has to leave your machine.

[Product vision](docs/VISION.md) · [Visual identity & Keel family](docs/brand/README.md) · [moesystems.com/keel](https://moesystems.com/keel/)

## What it does

- **Edits your project.** Reads files, searches code, applies patches, runs
  shell commands — inside the folder you point it at, nowhere else.
- **You approve, the model asks.** Workspace edits are auto-allowed by
  default; shell commands ask first. `--read-only`, `--safety-mode safe` and
  `--yolo` change that. Keel refuses `~` or `/` as a workspace.
- **Remembers the conversation.** Every run appends to a JSONL session log
  you can resume later — from the CLI, the TUI, or another tool.
- **Works with weak models.** Text-format tool-call fallback, repeat-tool
  reminders, and compaction sized to your server's real context window keep
  2–9B models on task.
- **Has a full terminal cockpit.** `keel tui`: past sessions on the left,
  the live run in the middle, model/safety/token status on the right.

## Build and try Keel Code

Install a stable [Rust toolchain](https://rustup.rs/). For local inference, start
an OpenAI-compatible server and load a compatible model first. The example uses
`llama-server` on port 8080 and the server's model alias `bonsai-9b`.

```bash
git clone https://github.com/olivermikkola/keel-code.git
cd keel-code
cargo build --release

# Local model (llama-server, Ollama, LM Studio — any OpenAI-compatible API):
export KEEL_BASE_URL=http://localhost:8080/v1 KEEL_MODEL=bonsai-9b
./target/release/keel "explain this repo"        # run inside a project folder

# Or a hosted compatible API:
export DEEPSEEK_API_KEY=...
./target/release/keel "fix the failing test"

./target/release/keel tui                        # the terminal cockpit
./target/release/keel bench --runs 3             # smoke-test your model + harness
```

## Terminal cockpit (`keel tui`)

`keel tui` is the session-centric terminal cockpit: a **log book** of past
wakes on the left, the live **wake** in the middle, and a **ship** status rail
on the right (model/provider, safety, sandbox, VFS, last-turn tokens, changed
files).

```bash
keel tui --model bonsai-9b --root ~/Projects/my-app
keel tui --model bonsai-9b "explain this repo"   # start with a prompt
```

- **Non-blocking input:** keep typing while a turn runs; Enter queues the next
  prompt and it resumes the same session log when the current turn exits.
- **Streaming NDJSON:** reasoning, tool calls, tool results, `file_edit`
  (unified diffs), sentinel vetoes and turn summaries render as they arrive.
- **Session log is the source of truth:** the log book loads
  `~/.keel/sessions/*.jsonl`; closing the TUI never deletes a wake.

| Key | Action |
|---|---|
| `Tab` | cycle Log book → Wake → Wake line |
| `j` / `k`, `↑` / `↓` | move or scroll |
| `Enter` | send, or queue while running |
| `Esc` | leave composer / interrupt from the wake |
| `n` / `s` / `/` | new wake / focus log book / search log book |
| `d` / `v` | details / ship rail |
| `D` | review the workspace git diff |
| `y` / `a` / `n` | approval: allow once / always / deny |
| `m` | model picker (Bonsai 2B/4B/9B/27B, Qwen, Kat, DeepSeek) |
| `Ctrl-P`, `:` | command palette |
| `x` | interrupt |
| `?` / `q` | help / quit (`Ctrl-C` also quits) |

## Does it actually work? (`keel bench`)

9 small tasks (create a file, rename a function, fix an obvious bug), 2 runs
each. A smoke test for a model + harness pair — **not** comparable to public
benchmarks, and it does not show Keel beats anything. Full reproducible
evaluations live in [Keel Arena](https://github.com/olivermikkola/keel-arena).

Tested locally on Vulkan (`llama-server`) across **Huihui Qwen3.8 27B (`Q4_K_M`, 21 GB)** and the full **PrismML Ternary Bonsai 2 (`Q2_0`)** family (**2B, 4B, 9B, 27B**):

| Model | Size | Pass Rate | `create_file` | `find_value` | `fix_bug` | `rename_symbol` | `locate_def` | `misleading_error` |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ternary-Bonsai-2-2B (`Q2_0`)** | 522 MB | **4/12 (33%)** | 2/2 (1.2s) | 0/2 (5.7s) | 0/2 (6.1s) | 0/2 (23.1s) | 2/2 (2.2s) | 0/2 (25.4s) |
| **Ternary-Bonsai-2-4B (`Q2_0`)** | 1.2 GB | **10/12 (83%)** | 2/2 (3.3s) | 2/2 (2.3s) | 2/2 (6.9s) | 1/2 (18.0s) | 2/2 (2.1s) | 1/2 (25.1s) |
| **Ternary-Bonsai-2-9B (`Q2_0`)** | 2.4 GB | **12/12 (100%)** | 2/2 (4.2s) | 2/2 (10.2s) | 2/2 (9.7s) | 2/2 (22.8s) | 2/2 (3.4s) | 2/2 (59.3s) |
| **Ternary-Bonsai-2-27B (`Q2_0`)** | 7.2 GB | **12/12 (100%)** | 2/2 (14.8s) | 2/2 (10.8s) | 2/2 (33.1s) | 2/2 (62.9s) | 2/2 (9.8s) | 2/2 (131.4s) |
| **Qwen3.8-27B-abliterated (`Q4_K_M`)** | 21.0 GB | **6/6 (100%)** | 1/1 (55.5s) | 1/1 (23.6s) | 1/1 (67.1s) | 1/1 (144.5s) | 1/1 (24.0s) | 1/1 (300.4s) |

## Under the hood

- **Plugin kernel.** A small `keel-kernel` hosts services, typed events and
  reversible effects; the model adapter, tools, session log, sandbox,
  approvals, compaction and agent loop are plugins mounted on it. The plugin
  set is **hardcoded** in `keel-cli` (all of them mount on every run; see
  `keel --dump-config`). There is no config file or profile system yet.
- **One provider: OpenAI-compatible chat completions** (llama.cpp, Lemonade,
  DeepSeek, OmniRoute, ...). No native Anthropic or Gemini adapters.
- **Safety:** Linux Landlock confines `shell` (writes only in the workspace
  and `/tmp`); file tools are confined to the workspace. No seccomp.
- **Append-only JSONL session log**, which [MoE Deck](https://github.com/olivermikkola/moedeck-community)
  can tail as one of its harnesses.

```
crates/keel-kernel   plugin kernel (Context, Plugin, Scope, effects)
crates/keel-agent    agent loop, tools, Landlock sandbox, OpenAI/DeepSeek provider
crates/keel-cli      `keel` binary, `keel tui` cockpit, `keel bench` suite
docs/REFERENCES.md   what we study and borrow, with licenses
```

### Resuming a conversation

`keel run --resume session.jsonl "continue"` loads the complete saved conversation
before mounting runtime plugins or calling the model. Missing, empty, malformed,
or oversized session logs fail explicitly; Keel does not silently start fresh or
use a truncated prefix. In `--json` mode, this returns a `done` event with
`status: "error"` and exits unsuccessfully. The source log is left unchanged.
Use a valid log, or omit `--resume` to deliberately start a new conversation.

### JSON approval requests (for GUIs and parent tools)

One-shot `keel run --json` advertises `approval_protocol: "stdio-line-v1"` in a
`capabilities` event. A `tool_call` event is progress, not a permission request.
When the active policy actually needs a decision, Keel emits and flushes:

```json
{"type":"approval_request","protocol":"stdio-line-v1","id":"approval-123-1","tool_call_id":"call-1","tool":"shell","arguments":{"command":"cargo test"},"choices":["allow","deny","always"]}
```

The parent responds with one line on stdin: `allow`, `deny`, or `always` (existing
`y`, `n`, and `a` responses remain supported). Exchanges are serialized; the
command waits for the decision. EOF or a failed request write denies the call.
Reads, policy-authorized calls, read-only rejections, and previously authorized
calls do not emit an approval request. Parents should scope every response to
its current subprocess/run and never infer a request from ordinary tool progress.

### JSON `file_edit` events (Diff / `--shadow`)

When `write_file`, `edit_file`, or `patch_file` successfully apply — including
under `--shadow` (in-memory VFS overlay) — `--json` emits a dedicated event with
a repo-relative path and unified diff. Parents (e.g. Keel Desktop Diff) must not
invent Diff rows from `tool_call` alone; wait for this event:

```json
{"type":"file_edit","path":"src/main.rs","diff":"--- a/src/main.rs\n+++ b/src/main.rs\n@@ -1,1 +1,1 @@\n-old\n+new\n","tool_call_id":"call-1"}
```

`tool_call_id` is included when available. Failed or denied edits do not emit
`file_edit`. There is no parallel `patch` event type.

## Status & roadmap

Early, and honest about it. The basics come before features (see
[AGENTS.md](AGENTS.md)); many modules are **frozen** until the core is solid:
epistemic memory, swarm/handoff, exploration tree, speculate, cascade,
replay/recorder, and the daemon.

1. **Reliable basics:** agent loop, tools, sandbox, approvals, resume and
   compaction work in real runs with small local models.
2. **Honest benchmark:** Keel vs. other harnesses, same small model, across
   the 7-suite gauntlet in [Keel Arena](https://github.com/olivermikkola/keel-arena).
3. Only then: config-driven plugin profiles, native providers, unfreezing
   modules that the benchmark shows are worth it.

## The Keel family

One engine, focused surfaces. The canonical positioning is in
[docs/VISION.md](docs/VISION.md): **Keel Code is the terminal coding agent (the
Claude Code role); Keel Desktop is the everyday app (the Claude Desktop role)
whose Code mode is this CLI.**

| Product | What it is |
| --- | --- |
| **Keel Code** (this repo) | CLI + TUI coding-agent harness (`keel`) for small local models |
| [Keel Desktop](https://github.com/olivermikkola/keel-desktop) | Everyday local-AI app for macOS/Linux/Windows: Chat, Work, Images, and Code powered by Keel Code |
| [Keel Mobile](https://moesystems.com/keel-mobile/) | Standalone on-device iPhone agent — the model runs on the phone |
| [Keel Mobile for Android](https://github.com/olivermikkola/keel-mobile-android) | On-device Android agent, source-build pre-alpha |
| [Keel Console](https://github.com/olivermikkola/keel-xbox) | Living-room app for Xbox / TV |
| [Keel Arena](https://github.com/olivermikkola/keel-arena) | Public benchmarks and scorecards — the proof layer |

## Contributing

Agents and humans: read [AGENTS.md](AGENTS.md) first. Every PR needs a real-run
`## Proof`, stays under 600 changed lines and adds no new modules without
approval; the `PR gate` CI job enforces this.

## License

MIT — see [LICENSE](LICENSE).
