# Keel product vision

Canonical positioning for the Keel family. Other Keel repositories link here
instead of restating it. Visual identity lives in [brand/README.md](brand/README.md).

## The two anchors

**Keel Code (CLI + TUI, binary `keel`) is the coding agent — the role Claude Code plays.**
One engine in the terminal: `keel` and `keel tui` over the same session logs,
tools, sandbox and approvals. It is the core of the family; the other surfaces
front it or carry its techniques to other devices.

**Keel Desktop is the everyday app — the role Claude Desktop plays.**
Chat, Work (Websites, Sheets, Docs, Study, Slides), Images, projects and saved
preferences: the place you go for ease, no terminal required. Its **Code mode
is Keel Code** — the same engine you can run in a terminal, fronted by the
GUI, not a second coding agent.

## What follows from this

1. **One coding engine.** This repo owns the coding loop, tools, sandbox,
   approvals and the JSONL session log. Desktop's Code mode drives the `keel`
   CLI (streaming events, approval requests, resumable sessions) rather than
   growing a parallel coding harness.
2. **One session story.** A wake started in the terminal can continue in
   Desktop, and back; the session log is the interchange.
3. **Desktop owns the non-coding surface.** Chat, Work, Images and the
   built-in apps stay in Desktop; they do not move into the CLI.
4. **Same rules everywhere.** Small local models first, no silent cloud
   fallback, you approve / the model asks.

## The rest of the family

| Product | Role under this vision |
| --- | --- |
| **Keel Mobile** | Standalone on-device iPhone agent. Not a remote control for Desktop; pairing stays an Advanced option. |
| **Keel Console** (`keel-xbox`) | Living-room surface; can borrow Desktop's model over LAN. |
| **Keel Arena** | Public benchmarks and scorecards — the proof that the engine works. |

## Today vs. the vision

Honest status: Desktop's Code mode currently defaults to its own built-in
harness (`keel-core`) and can pass prompts through to an installed `keel` CLI
as a one-shot subprocess. Driving the Keel CLI properly — streaming, approvals
in the GUI, shared resumable sessions — is the direction this document fixes.
It is not finished yet.
