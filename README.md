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

**M1 first cut works** (8 tests, clippy clean): turn/step agent loop, OpenAI-compatible provider (DeepSeek default), `read_file`/`write_file`/`shell` tools, approval policy (read-only by default, `--yolo` to allow mutation), JSONL session log.

```
export DEEPSEEK_API_KEY=...        # or KEEL_BASE_URL / KEEL_API_KEY / KEEL_MODEL for any compatible API
keel "explain this repo"           # read-only
keel --yolo --log s.jsonl "fix the failing test"
```

**Known gaps (honest list):** not yet exercised against a live model; no streaming; the loop takes its seams as struct fields rather than kernel services (the next refactor); path confinement is lexical, so symlinks inside the workspace can escape (needs `canonicalize` + Landlock); `shell` is unsandboxed; no compaction, no MCP, no TUI.

## Roadmap

1. **M0, kernel:** services, events, effects, config-driven plugin tree, `--dump-config`.
2. **M1, minimal agent:** `llm` service + OpenAI-compatible and DeepSeek providers, `agent-loop` (turn/step), tools (read, edit, grep, shell), JSONL session log.
3. **M2, safety:** sandbox plugin (Landlock), approval policy, guard plugins (timeouts, repeat-tool reminder).
4. **M3, context:** compaction, skills (`SKILL.md`), `AGENTS.md`, MCP client, subagents, todo/plan.
5. **M4, frontends:** TUI, ACP server, SDK JSON-RPC, Helm integration.
6. **M5, extras:** LSP, worktree isolation per thread, scheduling/jobs, goal mode, cost meter.

## Layout

```
crates/keel-kernel   plugin kernel (Context, Plugin, Scope, effects)
crates/keel-cli      `keel` binary
docs/REFERENCES.md   what we study and borrow, with licenses
```
