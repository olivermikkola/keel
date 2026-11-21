<img src="docs/brand/mark.svg" width="64" height="64" alt="Keel jade and rust sail" />

# Keel

A **Rust CLI coding-agent harness** aimed at making **small local models**
(2–27B, e.g. Bonsai and Qwen on llama.cpp) do useful coding work. Private, early.

[Product vision](docs/VISION.md) · [Visual identity, palette & Keel family](docs/brand/README.md) · [moesystems.com/keel](https://moesystems.com/keel/).

## First 60 seconds

```bash
cargo build --release
export DEEPSEEK_API_KEY=...        # or KEEL_BASE_URL / KEEL_API_KEY / KEEL_MODEL for any compatible API
./target/release/keel "explain this repo"           # run inside a project folder; edits auto-allowed, shell asks first
./target/release/keel tui --model bonsai-9b         # session-centric terminal cockpit
```

## What is true today

- **Plugin kernel.** A small `keel-kernel` hosts services, typed events and
  reversible effects; the model adapter, tools, session log, sandbox,
  approvals, compaction and agent loop are plugins mounted on it. The plugin
  set is **hardcoded** in `keel-cli` (all of them mount on every run; see
  `keel --dump-config`). There is no config file or profile system yet.
- **One provider: OpenAI-compatible chat completions** (llama.cpp, Lemonade,
  DeepSeek, OmniRoute, ...). No native Anthropic or Gemini adapters.
- **Append-only JSONL session log**, which [MoE-Deck](https://github.com/olivermikkola/moedeck)
  can tail as one of its harnesses.
- **Safety:** Linux Landlock confines `shell` (writes only in the workspace
  and `/tmp`), file tools are confined to the workspace. Default policy
  `standard`: workspace edits are auto-allowed, `shell` asks first (`--read-only`,
  `--safety-mode safe`, `--yolo` change this). There is
  a line-based approval protocol on stdin, and Keel refuses `~` or `/` as the
  workspace. No seccomp.
- **Weak-model help:** `<tool_call>` text fallback, repeat-tool reminders,
  compaction sized to the server's real context window.

## Direction

Fix the basics before adding features (see [AGENTS.md](AGENTS.md)). Many
modules were added quickly and are **frozen** until the core is solid:
epistemic memory, swarm/handoff, exploration tree, speculate, cascade,
replay/recorder, and the daemon. The open question that decides whether Keel
is worth more work: **does it beat other harnesses (OpenCode, Aider, Codex
CLI) with the same small local model on a real benchmark?** `keel bench`
today is 9 toy tasks; full reproducible evaluations live in
[Keel Arena](https://github.com/olivermikkola/keel-arena).

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
- **Streaming NDJSON:** reasoning, tool calls, tool results, sentinel vetoes
  and turn summaries render as they arrive.
- **Session log is the source of truth:** the log book loads
  `~/.keel/sessions/*.jsonl`; closing the TUI never deletes a wake.
- **Command palette:** `Ctrl-P` or `:`.
- **Model picker:** `m` (Bonsai 2B/4B/9B/27B, Qwen, Kat, DeepSeek).
- **Details:** `d` expands reasoning/tool output; `v` toggles the ship rail.
- **Interrupt:** `x`, or `Esc` from the wake pane.

| Key | Action |
|---|---|
| `Tab` | cycle Log book → Wake → Wake line |
| `j` / `k`, `↑` / `↓` | move or scroll |
| `Enter` | send, or queue while running |
| `Esc` | leave composer / interrupt from the wake |
| `n` | new wake (new session log) |
| `s` | focus log book |
| `/` | search log book |
| `d` / `v` | details / ship rail |
| `D` | review the workspace git diff |
| `y` / `a` / `n` | approval: allow once / always / deny |
| `m` | model picker |
| `Ctrl-P`, `:` | command palette |
| `x` | interrupt |
| `?` | help |
| `q`, `Ctrl-C` | quit |

## Quick start

```bash
export DEEPSEEK_API_KEY=...        # or KEEL_BASE_URL / KEEL_API_KEY / KEEL_MODEL for any compatible API
keel "explain this repo"           # run inside a project folder; edits auto-allowed there, shell asks first
keel --yolo --log s.jsonl "fix the failing test"
keel bench --runs 3                # multi-run coding benchmark with thinking-token meter
```

## Benchmark (`keel bench`)

9 small toy tasks (create a file, rename a function, fix an obvious bug), 2 runs
each. Useful as a smoke test for a model + harness pair; it is **not**
comparable to public benchmarks and does not show Keel beats anything.

Tested locally on Vulkan (`llama-server`) across **Huihui Qwen3.8 27B (`Q4_K_M`, 21 GB)** and the full **PrismML Ternary Bonsai 2 (`Q2_0`)** family (**2B, 4B, 9B, 27B**):

| Model | Size | Pass Rate | `create_file` | `find_value` | `fix_bug` | `rename_symbol` | `locate_def` | `misleading_error` |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ternary-Bonsai-2-2B (`Q2_0`)** | 522 MB | **4/12 (33%)** | 2/2 (1.2s) | 0/2 (5.7s) | 0/2 (6.1s) | 0/2 (23.1s) | 2/2 (2.2s) | 0/2 (25.4s) |
| **Ternary-Bonsai-2-4B (`Q2_0`)** | 1.2 GB | **10/12 (83%)** | 2/2 (3.3s) | 2/2 (2.3s) | 2/2 (6.9s) | 1/2 (18.0s) | 2/2 (2.1s) | 1/2 (25.1s) |
| **Ternary-Bonsai-2-9B (`Q2_0`)** | 2.4 GB | **12/12 (100%)** | 2/2 (4.2s) | 2/2 (10.2s) | 2/2 (9.7s) | 2/2 (22.8s) | 2/2 (3.4s) | 2/2 (59.3s) |
| **Ternary-Bonsai-2-27B (`Q2_0`)** | 7.2 GB | **12/12 (100%)** | 2/2 (14.8s) | 2/2 (10.8s) | 2/2 (33.1s) | 2/2 (62.9s) | 2/2 (9.8s) | 2/2 (131.4s) |
| **Qwen3.8-27B-abliterated (`Q4_K_M`)** | 21.0 GB | **6/6 (100%)** | 1/1 (55.5s) | 1/1 (23.6s) | 1/1 (67.1s) | 1/1 (144.5s) | 1/1 (24.0s) | 1/1 (300.4s) |

## Roadmap

1. **Reliable basics:** agent loop, tools, sandbox, approvals, resume and
   compaction work in real runs with small local models.
2. **Honest benchmark:** evaluate Keel vs. other harnesses and small local models
   across the 7-suite gauntlet in [Keel Arena](https://github.com/olivermikkola/keel-arena)
   and Harbor-isolated tasks.
3. Only then: config-driven plugin profiles, native providers, unfreezing
   modules that the benchmark shows are worth it.

## Ecosystem

- [**Keel Arena**](https://github.com/olivermikkola/keel-arena) — Reproducible Edge AI & coding-agent benchmarks (7-suite gauntlet, frontier LLM judge, hardware profiling).
- [**Keel Desktop**](https://github.com/olivermikkola/keel-desktop) — Tauri 2 local AI desktop workspace built on Keel's small-model harness.
- [**Keel Mobile**](https://github.com/olivermikkola/keel-mobile) — On-device iOS local coding & assistant app.
- [**MoE Deck**](https://github.com/olivermikkola/moedeck-community) — Multi-harness cockpit and PWA that tails `~/.keel/sessions/*.jsonl`.

## Contributing

Agents and humans: read [AGENTS.md](AGENTS.md) first. Every PR needs a real-run
`## Proof`, stays under 600 changed lines and adds no new modules without
approval; the `PR gate` CI job enforces this.

## Layout

```
crates/keel-kernel   plugin kernel (Context, Plugin, Scope, effects)
crates/keel-agent    agent loop, tools, Landlock sandbox, OpenAI/DeepSeek provider
crates/keel-cli      `keel` binary + `keel bench` harness benchmark suite
docs/REFERENCES.md   what we study and borrow, with licenses
```

## Resuming a conversation

`keel run --resume session.jsonl "continue"` loads the complete saved conversation
before mounting runtime plugins or calling the model. Missing, empty, malformed,
or oversized session logs fail explicitly; Keel does not silently start fresh or
use a truncated prefix. In `--json` mode, this returns a `done` event with
`status: "error"` and exits unsuccessfully. The source log is left unchanged.
Use a valid log, or omit `--resume` to deliberately start a new conversation.

## JSON approval requests

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
