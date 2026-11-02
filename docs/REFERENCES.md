# References and borrow map

Cloned locally for study at `~/Projects/ref/` (not committed here).

## DeepSeek Harness (`deepseek-ai/deepseek-harness`, MIT, TypeScript, ~270k stars)
Built on Cordis: plugins contribute services, typed events and reversible effects. Profiles (`web`, `headless`, `sdk`, `sdk-minimal`, `acp`) stack bundles plus a user `cordis.patch.yml`. `dsh --profile web --dump-config` prints the booted tree.

| dsh package | What it does | keel plan |
| --- | --- | --- |
| `core/agent-loop`, `agent` | turn/step loop, events (`turn/*`, `step/*`, `tools/pre-execute`...) | `keel-agent-loop` plugin |
| `core/tools`, `tool-executor-native`, `tool-executor-ptc` | tool registry, native and programmatic tool calling | `keel-tools` |
| `session`, `session-query` | append-only session log | `keel-session` (JSONL) |
| `llm/*` (deepseek, retry, token-meter, pi-ai) | provider adapters, retry, token metering | `keel-llm-*` |
| `sandbox/*`, `native/system` | Landlock launcher, flock session lock | `keel-sandbox` (Rust natively, no addon needed) |
| `guard/*` | timeout policy, repeat-tool reminder | `keel-guard-*` |
| `compaction/*` | basic compaction, tool-result pruning, image offload | `keel-compaction-*` |
| `skill`, `mcp`, `acp`, `lsp`, `subagent`, `plan`, `todo`, `goal`, `schedule`, `jobs`, `workflow`, `spill`, `browser-use`, `computer-use` | feature packages | one plugin each, in priority order |
| `boot/*`, `preset`, `bundle/*` | profile and patch composition, plugin manager, HMR | `keel-boot` |

Read first: `docs/architecture.md` (turn flow and event domains), `packages/core/agent-loop/README.md`, `SAFETY.md`.

## Other sources to study (verify licenses before copying code)
- OpenAI Codex `codex-rs` (Apache-2.0): sandbox, exec policy, MCP, app-server protocol.
- OpenCode (MIT): client/server split, SDK.
- Claude Code: hooks, skills conventions (behaviour only).
- Kimi Code, Qwen Code (local clones in `~/Projects`): loop and tool designs.

## Rules
- Copy designs freely; copy code only with a compatible license and attribution in `THIRD_PARTY_NOTICES.md`.
- Note: dsh is in developer preview with breaking changes. Don't track it, learn from it.
