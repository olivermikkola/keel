# keel (placeholder name)

A **model-agnostic, Rust, everything-is-a-plugin CLI coding agent harness.** Private, early.

## The gist

- **Kernel, not a monolith.** A tiny `keel-kernel` hosts services, typed events and *reversible effects*. The model adapter, tools, session log, sandbox, approvals, compaction and the agent loop are all plugins mounted on it. Any of them can be swapped from config. No privileged core.
- **Model-agnostic.** Provider plugins (OpenAI-compatible, Anthropic, DeepSeek, Gemini, local llama.cpp, routers like OmniRoute) all implement one `llm` service. DeepSeek is just the first.
- **Profiles.** A profile is an ordered list of plugin bundles plus a user patch file: `tui`, `headless`, `acp`, `sdk`, `daemon`.
- **Append-only session log** (JSONL). Durable facts are events; the log is the source of truth. [Helm](https://github.com/olivermikkola/helm-gateway) can tail it natively as a fifth harness.
- **Approvals and sandboxing are first-class**: Landlock/seccomp on Linux, an approval protocol that a phone UI can answer, safe-by-default permissions.
- **Reverse-engineer the best ideas** from open source and re-implement them as plugins (see [docs/REFERENCES.md](docs/REFERENCES.md)). Copy designs and permissively licensed code (with attribution), not private leaks.

## Status

**M1 & M2 core operational** (12 tests, clippy clean):
- **Plugin-mounted architecture:** `LlmPlugin`, `ApproverPlugin`, `SessionLogPlugin`, `LandlockPlugin`, `BuiltinToolsPlugin`, `Printer`, and `ReasoningMeter` all mount onto `keel_kernel::Context`.
- **Linux Landlock sandbox (`ABI::V3`):** `shell` child processes are confined in `pre_exec` so writes outside the workspace + `/tmp` are blocked by the kernel. Canonical symlink escape checks protect all file tools.
- **Streaming + weak-model resilience:** Live SSE streaming for both content and reasoning tokens, `KEEL_THINKING=0|1` control, surgical `edit_file`, `grep`, `list_dir`, and `<tool_call>` text fallback recovery.

```bash
export DEEPSEEK_API_KEY=...        # or KEEL_BASE_URL / KEEL_API_KEY / KEEL_MODEL for any compatible API
keel "explain this repo"           # read-only by default
keel --yolo --log s.jsonl "fix the failing test"
keel bench --runs 3                # multi-run coding benchmark with thinking-token meter
```

## Benchmark (`keel bench`)

Tested locally on Vulkan (`llama-server`) across **Huihui Qwen3.8 27B (`Q4_K_M`, 21 GB)** and the full **PrismML Ternary Bonsai 2 (`Q2_0`)** family (**2B, 4B, 9B, 27B**):

| Model | Size | Pass Rate | `create_file` | `find_value` | `fix_bug` | `rename_symbol` | `locate_def` | `misleading_error` |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ternary-Bonsai-2-2B (`Q2_0`)** | 522 MB | **4/12 (33%)** | 2/2 (1.2s) | 0/2 (5.7s) | 0/2 (6.1s) | 0/2 (23.1s) | 2/2 (2.2s) | 0/2 (25.4s) |
| **Ternary-Bonsai-2-4B (`Q2_0`)** | 1.2 GB | **10/12 (83%)** | 2/2 (3.3s) | 2/2 (2.3s) | 2/2 (6.9s) | 1/2 (18.0s) | 2/2 (2.1s) | 1/2 (25.1s) |
| **Ternary-Bonsai-2-9B (`Q2_0`)** | 2.4 GB | **12/12 (100%)** | 2/2 (4.2s) | 2/2 (10.2s) | 2/2 (9.7s) | 2/2 (22.8s) | 2/2 (3.4s) | 2/2 (59.3s) |
| **Ternary-Bonsai-2-27B (`Q2_0`)** | 7.2 GB | **12/12 (100%)** | 2/2 (14.8s) | 2/2 (10.8s) | 2/2 (33.1s) | 2/2 (62.9s) | 2/2 (9.8s) | 2/2 (131.4s) |
| **Qwen3.8-27B-abliterated (`Q4_K_M`)** | 21.0 GB | **6/6 (100%)** | 1/1 (55.5s) | 1/1 (23.6s) | 1/1 (67.1s) | 1/1 (144.5s) | 1/1 (24.0s) | 1/1 (300.4s) |

## Roadmap

1. **M0, kernel:** services, events, effects, config-driven plugin tree, `--dump-config`.
2. **M1, minimal agent (done):** `llm` service + OpenAI-compatible and DeepSeek providers, `agent-loop` (turn/step), tools (`read_file`, `list_dir`, `grep`, `edit_file`, `write_file`, `shell`), JSONL session log.
3. **M2, safety (Landlock done):** sandbox plugin (Landlock `ABI::V3`), approval policy, guard plugins (timeouts, repeat-tool reminder).
4. **M3, context:** compaction, skills (`SKILL.md`), `AGENTS.md`, MCP client, subagents, todo/plan.
5. **M4, frontends:** TUI, ACP server, SDK JSON-RPC, Helm integration.
6. **M5, extras:** LSP, worktree isolation per thread, scheduling/jobs, goal mode, cost meter.

## Layout

```
crates/keel-kernel   plugin kernel (Context, Plugin, Scope, effects)
crates/keel-agent    agent loop, tools, Landlock sandbox, OpenAI/DeepSeek provider
crates/keel-cli      `keel` binary + `keel bench` harness benchmark suite
docs/REFERENCES.md   what we study and borrow, with licenses
```
