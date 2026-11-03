//! `keel "prompt"` - one-shot headless run. `keel bench [filter]` - run the benchmark tasks.
//! `keel patch <file> [diff]` - apply unified diff or search/replace hunks (supports stdin).
//! `keel edit <file> --old <str> --new <str>` - surgical fuzzy replacement.
//! `keel checkpoint <save|restore|list|diff>` - git shadow snapshots and instant rollback.
//! `keel undo` - revert workspace to most recent snapshot.
//! `keel mcp <list|call|test>` - Model Context Protocol (MCP) client hub.
//! `keel memory <show|add-fact|set-hypothesis|record-failure|clear>` - epistemic working memory.
//!
//! Env: KEEL_THINKING=0 disables model thinking (Qwen templates). KEEL_BASE_URL (default https://api.deepseek.com), KEEL_API_KEY (or DEEPSEEK_API_KEY),
//! KEEL_MODEL (default deepseek-chat). Flags: --yolo (allow mutating tools), --no-sandbox (disable Landlock), --log <file>.
//! Default policy is read-only: mutating tools are denied unless --yolo is given.

use anyhow::Context as _;
use keel_agent::Chunk;
use keel_agent::{
    cascade::CascadeProvider,
    checkpoint::{CheckpointManager, CheckpointPlugin},
    compaction::{ToolResultPrunerPlugin, WorkspaceContextPlugin, WorkspaceInstructions},
    daemon::DaemonConfig,
    epistemic::{EpistemicMemoryPlugin, EpistemicStore},
    guard::RepeatToolReminderPlugin,
    lsp::{LspPlugin, LspService},
    mcp::{McpClient, McpPlugin, McpServerConfig},
    plan::PlanPlugin,
    provider::OpenAiCompat,
    sandbox::LandlockPlugin,
    speculate::{SpeculatePlugin, SpeculationConfig, SpeculationEngine},
    subagent::SubagentPlugin,
    tools::BuiltinToolsPlugin,
    vfs::{TransactionalVfs, VfsMode, VfsPlugin},
    Agent, AllowAll, ApprovalPolicy, Approver, ApproverPlugin, InteractiveApprover, Llm, LlmPlugin,
    Message, ReadOnly, SessionLog, SessionLogPlugin, ToolResult, ToolStart,
};
use keel_kernel::{Context, Plugin, Scope};
use std::io::Write;
use std::sync::Arc;

mod bench;

/// Live terminal output: answer text on stdout, reasoning dimmed on stderr.
struct Printer;
impl Plugin for Printer {
    fn name(&self) -> &str {
        "printer"
    }
    fn apply(&self, s: &mut Scope) {
        s.on::<Chunk, _>(|c| match c {
            Chunk::Text(t) => {
                print!("{t}");
                let _ = std::io::stdout().flush();
            }
            Chunk::Reasoning(t) => {
                eprint!("\x1b[2m{t}\x1b[0m");
                let _ = std::io::stderr().flush();
            }
        });
    }
}

/// Machine-readable NDJSON streaming output for calling agents and automated harnesses.
struct JsonPrinter;
impl Plugin for JsonPrinter {
    fn name(&self) -> &str {
        "json-printer"
    }
    fn apply(&self, s: &mut Scope) {
        s.on::<Chunk, _>(|c| match c {
            Chunk::Text(t) => {
                println!("{}", serde_json::json!({ "type": "text", "content": t }));
                let _ = std::io::stdout().flush();
            }
            Chunk::Reasoning(t) => {
                println!(
                    "{}",
                    serde_json::json!({ "type": "reasoning", "content": t })
                );
                let _ = std::io::stdout().flush();
            }
        });
        s.on::<ToolStart, _>(|ts| {
            println!(
                "{}",
                serde_json::json!({
                    "type": "tool_call",
                    "name": ts.0.name,
                    "arguments": ts.0.args,
                })
            );
            let _ = std::io::stdout().flush();
        });
        s.on::<ToolResult, _>(|tr| {
            println!(
                "{}",
                serde_json::json!({
                    "type": "tool_result",
                    "id": tr.0,
                    "output": tr.1,
                })
            );
            let _ = std::io::stdout().flush();
        });
    }
}

fn print_usage() {
    eprintln!(
        r#"usage:
  keel [options] "prompt"          run one-shot agent turn
  keel run [options] "prompt"      explicit run command
  keel repl [options]              interactive multi-turn REPL
  keel patch <file> [diff]         apply diff or search/replace hunks (supports stdin)
  keel edit <file> --old S --new S surgical fuzzy replacement
  keel checkpoint <cmd>            git shadow snapshots (save, restore, list, diff)
  keel undo                        revert workspace to most recent snapshot
  keel mcp <list|call|test>        Model Context Protocol (MCP) client hub
  keel lsp <check|def|hover>       Language Server & compiler diagnostics hub
  keel vfs <status|diff|commit|rollback> transactional CoW / shadow VFS control
  keel speculate "prompt" [opts]   parallel best-of-N speculative sandbox tournament
  keel daemon [options]            ACP bidirectional JSON-RPC 2.0 daemon (stdio, socket, tcp)
  keel memory <cmd>                epistemic belief state (show, add-fact, set-hypothesis, clear)
  keel handoff <cmd>               sovereign cross-harness context distillation (Helm swarm)
  keel oracle <cmd>                test diagnostic oracle & regression sentinel (test, baseline, diff)
  keel bench [filter] [--runs N]   run benchmark suite

options:
  --sentinel <mode>                regression sentinel mode (warn, rollback, strict)
  --handoff <file>                 bootstrap session directly from handoff envelope
  --listen <addr>                  listen on TCP address (e.g. 127.0.0.1:4040)
  --socket <path>                  listen on Unix domain socket (e.g. ~/.keel/daemon.sock)
  --auth-token <token>             enforce bearer authentication token
  --json                           emit machine-readable JSON output
  --dry-run                        preview patch/edit without writing
  --yolo                           allow mutating tools (write/edit/patch/shell)
  --no-sandbox                     disable Linux Landlock sandbox
  --auto-rollback                  automatically revert workspace if verification fails
  --verify <cmd>                   enforce verification command gate
  --mcp-server <name=cmd...>       attach ad-hoc stdio MCP server
  --root <dir>                     workspace root directory
  --log <file>                     session JSONL log path
  --resume <file>                  resume session from previous JSONL log
  --dump-config                    display composed kernel plugin tree"#
    );
}

fn cli_patch(mut args: Vec<String>) -> anyhow::Result<()> {
    let dry_run = take_flag(&mut args, "--dry-run");
    let json_mode = take_flag(&mut args, "--json");
    if args.is_empty() {
        eprintln!("usage: keel patch <file> [patch_file_or_dash] [--dry-run] [--json]");
        std::process::exit(2);
    }
    let target_path = &args[0];
    let patch_content = if args.len() > 1 && args[1] != "-" {
        std::fs::read_to_string(&args[1])
            .with_context(|| format!("reading patch file {}", args[1]))?
    } else {
        use std::io::Read;
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .context("reading patch from stdin")?;
        buf
    };

    let target_text = std::fs::read_to_string(target_path)
        .with_context(|| format!("reading target file {target_path}"))?;
    let outcome = keel_agent::patch::apply_patch(&target_text, &patch_content)?;

    if !dry_run {
        std::fs::write(target_path, &outcome.modified_text)
            .with_context(|| format!("writing patched file {target_path}"))?;
    }

    if json_mode {
        println!(
            "{}",
            serde_json::json!({
                "status": "ok",
                "file": target_path,
                "dry_run": dry_run,
                "hunks_applied": outcome.hunks_applied,
                "quality": outcome.quality.to_string(),
            })
        );
    } else {
        println!(
            "{} {target_path} ({} hunk(s) applied via {})",
            if dry_run { "would patch" } else { "patched" },
            outcome.hunks_applied,
            outcome.quality
        );
    }
    Ok(())
}

fn cli_edit(mut args: Vec<String>) -> anyhow::Result<()> {
    let dry_run = take_flag(&mut args, "--dry-run");
    let json_mode = take_flag(&mut args, "--json");
    let old_text = take_opt(&mut args, "--old");
    let new_text = take_opt(&mut args, "--new");
    if args.is_empty() || old_text.is_none() || new_text.is_none() {
        eprintln!("usage: keel edit <file> --old <search> --new <replace> [--dry-run] [--json]");
        std::process::exit(2);
    }
    let target_path = &args[0];
    let old = old_text.unwrap();
    let new = new_text.unwrap();

    let target_text = std::fs::read_to_string(target_path)
        .with_context(|| format!("reading target file {target_path}"))?;
    let (modified, quality) = keel_agent::patch::fuzzy_replace(&target_text, &old, &new)?;

    if !dry_run {
        std::fs::write(target_path, &modified)
            .with_context(|| format!("writing edited file {target_path}"))?;
    }

    if json_mode {
        println!(
            "{}",
            serde_json::json!({
                "status": "ok",
                "file": target_path,
                "dry_run": dry_run,
                "quality": quality.to_string(),
            })
        );
    } else {
        println!(
            "{} {target_path} ({quality})",
            if dry_run { "would edit" } else { "edited" },
        );
    }
    Ok(())
}

fn cli_checkpoint(mut args: Vec<String>, root_opt: Option<String>) -> anyhow::Result<()> {
    let json_mode = take_flag(&mut args, "--json");
    let root = match root_opt {
        Some(r) => std::fs::canonicalize(r)?,
        None => std::env::current_dir()?,
    };
    let mgr = CheckpointManager::new(root);

    let subcmd = args.first().map(String::as_str).unwrap_or("list");
    match subcmd {
        "save" => {
            let desc = if args.len() > 1 {
                args[1..].join(" ")
            } else {
                "manual checkpoint".into()
            };
            let ckpt = mgr.save(&desc)?;
            if json_mode {
                println!("{}", serde_json::to_string(&ckpt)?);
            } else {
                println!("Saved checkpoint {} ({})", ckpt.id, ckpt.description);
            }
        }
        "restore" => {
            let target = if args.len() > 1 { &args[1] } else { "latest" };
            let rep = mgr.restore(target)?;
            if json_mode {
                println!("{}", serde_json::to_string(&rep)?);
            } else {
                println!(
                    "Restored checkpoint {} (restored {} files, removed {} newly created files)",
                    rep.checkpoint_id, rep.restored_files, rep.removed_files
                );
            }
        }
        "diff" => {
            let target = args.get(1).map(String::as_str);
            let diff_out = mgr.diff(target)?;
            print!("{diff_out}");
        }
        "list" => {
            let list = mgr.list()?;
            if json_mode {
                println!("{}", serde_json::to_string(&list)?);
            } else if list.is_empty() {
                println!("No checkpoints recorded yet.");
            } else {
                println!("Checkpoints ({}):", list.len());
                for c in list {
                    println!(
                        "  - {} [{}] ({})",
                        c.id,
                        c.commit_hash.chars().take(7).collect::<String>(),
                        c.description
                    );
                }
            }
        }
        other => {
            eprintln!(
                "unknown checkpoint subcommand `{other}`. Available: save, restore, diff, list"
            );
            std::process::exit(2);
        }
    }
    Ok(())
}

fn cli_undo(mut args: Vec<String>, root_opt: Option<String>) -> anyhow::Result<()> {
    let json_mode = take_flag(&mut args, "--json");
    let root = match root_opt {
        Some(r) => std::fs::canonicalize(r)?,
        None => std::env::current_dir()?,
    };
    let mgr = CheckpointManager::new(root);
    let rep = mgr.undo()?;
    if json_mode {
        println!("{}", serde_json::to_string(&rep)?);
    } else {
        println!(
            "Restored checkpoint {} (restored {} files, removed {} newly created files)",
            rep.checkpoint_id, rep.restored_files, rep.removed_files
        );
    }
    Ok(())
}

fn cli_mcp(mut args: Vec<String>, root_opt: Option<String>) -> anyhow::Result<()> {
    let json_mode = take_flag(&mut args, "--json");
    let root = match root_opt {
        Some(r) => std::fs::canonicalize(r)?,
        None => std::env::current_dir()?,
    };

    let subcmd = args.first().map(String::as_str).unwrap_or("list");
    match subcmd {
        "list" => {
            let servers = McpPlugin::load_config(&root);
            if servers.is_empty() {
                if json_mode {
                    println!("[]");
                } else {
                    println!("No MCP servers configured in .keel/mcp.json");
                }
                return Ok(());
            }

            let mut all_tools = Vec::new();
            for (name, cfg) in &servers {
                match McpClient::spawn(name, cfg) {
                    Ok(client) => {
                        for t in client.tools() {
                            all_tools.push(serde_json::json!({
                                "server": name,
                                "name": t.name,
                                "exposed_name": format!("mcp__{}__{}", name, t.name),
                                "description": t.description,
                                "input_schema": t.input_schema,
                            }));
                        }
                    }
                    Err(e) => {
                        eprintln!("warning: failed to connect to MCP server `{name}`: {e:#}");
                    }
                }
            }

            if json_mode {
                println!("{}", serde_json::to_string_pretty(&all_tools)?);
            } else if all_tools.is_empty() {
                println!("No tools exposed by configured MCP servers.");
            } else {
                println!("MCP Tools ({}):", all_tools.len());
                for t in all_tools {
                    println!(
                        "  - {} (server: {}) - {}",
                        t["exposed_name"].as_str().unwrap_or(""),
                        t["server"].as_str().unwrap_or(""),
                        t["description"].as_str().unwrap_or("")
                    );
                }
            }
        }
        "call" => {
            if args.len() < 3 {
                eprintln!("usage: keel mcp call <server> <tool> '<json_arguments>' [--json]");
                std::process::exit(2);
            }
            let server_name = &args[1];
            let tool_name = &args[2];
            let raw_args = if args.len() > 3 { &args[3] } else { "{}" };
            let parsed_args: serde_json::Value = serde_json::from_str(raw_args)
                .with_context(|| format!("parsing JSON arguments `{raw_args}`"))?;

            let servers = McpPlugin::load_config(&root);
            let cfg = servers
                .get(server_name)
                .with_context(|| format!("server `{server_name}` not found in .keel/mcp.json"))?;

            let client = McpClient::spawn(server_name, cfg)?;
            let res = client.call_tool(tool_name, &parsed_args)?;

            if json_mode {
                println!(
                    "{}",
                    serde_json::json!({
                        "status": "ok",
                        "server": server_name,
                        "tool": tool_name,
                        "result": res,
                    })
                );
            } else {
                println!("{res}");
            }
        }
        "test" => {
            if args.len() < 2 {
                eprintln!("usage: keel mcp test <command> [args...]");
                std::process::exit(2);
            }
            let cmd = &args[1];
            let cmd_args = args[2..].to_vec();
            let cfg = McpServerConfig {
                command: cmd.clone(),
                args: cmd_args,
                env: std::collections::HashMap::new(),
            };
            println!("Testing MCP handshake with `{cmd}`...");
            let client = McpClient::spawn("test", &cfg)?;
            println!(
                "Handshake successful! Discovered {} tools:",
                client.tools().len()
            );
            for t in client.tools() {
                println!("  - {}: {}", t.name, t.description.as_deref().unwrap_or(""));
            }
        }
        other => {
            eprintln!("unknown mcp subcommand `{other}`. Available: list, call, test");
            std::process::exit(2);
        }
    }
    Ok(())
}

fn cli_lsp(mut args: Vec<String>, root_opt: Option<String>) -> anyhow::Result<()> {
    let json_mode = take_flag(&mut args, "--json");
    let root = match root_opt {
        Some(r) => std::fs::canonicalize(r)?,
        None => std::env::current_dir()?,
    };
    let configs = LspService::load_config(&root);
    let svc = LspService::new(root, configs, true);

    let subcmd = args.first().map(String::as_str).unwrap_or("check");
    match subcmd {
        "check" | "diagnostics" => {
            if args.len() < 2 {
                eprintln!("usage: keel lsp check <file> [--json]");
                std::process::exit(2);
            }
            let file = &args[1];
            let items = svc.check_file(std::path::Path::new(file));
            if json_mode {
                println!("{}", serde_json::to_string_pretty(&items)?);
            } else if items.is_empty() {
                println!("No diagnostics for {file}");
            } else {
                for item in items {
                    println!("{}", item.format_compact());
                }
            }
        }
        "def" | "definition" => {
            if args.len() < 4 {
                eprintln!("usage: keel lsp def <file> <line> <column> [--json]");
                std::process::exit(2);
            }
            let file = &args[1];
            let line: usize = args[2].parse().unwrap_or(1);
            let col: usize = args[3].parse().unwrap_or(1);
            let res = svc.definition(file, line, col)?;
            println!("{}", serde_json::to_string_pretty(&res)?);
        }
        "hover" => {
            if args.len() < 4 {
                eprintln!("usage: keel lsp hover <file> <line> <column> [--json]");
                std::process::exit(2);
            }
            let file = &args[1];
            let line: usize = args[2].parse().unwrap_or(1);
            let col: usize = args[3].parse().unwrap_or(1);
            let res = svc.hover(file, line, col)?;
            println!("{}", serde_json::to_string_pretty(&res)?);
        }
        other => {
            eprintln!("unknown lsp subcommand `{other}`. Available: check, def, hover");
            std::process::exit(2);
        }
    }
    Ok(())
}

fn cli_memory(mut args: Vec<String>, root_opt: Option<String>) -> anyhow::Result<()> {
    let json_mode = take_flag(&mut args, "--json");
    let root = match root_opt {
        Some(r) => std::fs::canonicalize(r)?,
        None => std::env::current_dir()?,
    };
    let persist = EpistemicMemoryPlugin::persist_file(&root);
    let store = EpistemicStore::new(Some(persist));

    let subcmd = args.first().map(String::as_str).unwrap_or("show");
    match subcmd {
        "show" => {
            let state = store.get();
            if json_mode {
                println!("{}", serde_json::to_string_pretty(&state)?);
            } else if state.is_empty() {
                println!("Epistemic working memory is currently empty.");
            } else {
                print!("{}", state.render_markdown());
            }
        }
        "add-fact" => {
            if args.len() < 2 {
                eprintln!("usage: keel memory add-fact \"<fact>\" [--json]");
                std::process::exit(2);
            }
            let fact = args[1..].join(" ");
            store.add_fact(&fact)?;
            if json_mode {
                println!("{}", serde_json::json!({ "status": "ok", "fact": fact }));
            } else {
                println!("Recorded fact: {fact}");
            }
        }
        "set-hypothesis" => {
            if args.len() < 2 {
                eprintln!("usage: keel memory set-hypothesis \"<hypothesis>\" [--json]");
                std::process::exit(2);
            }
            let hyp = args[1..].join(" ");
            store.set_hypothesis(Some(hyp.clone()))?;
            if json_mode {
                println!(
                    "{}",
                    serde_json::json!({ "status": "ok", "hypothesis": hyp })
                );
            } else {
                println!("Updated hypothesis: {hyp}");
            }
        }
        "record-failure" => {
            if args.len() < 3 {
                eprintln!("usage: keel memory record-failure \"<approach>\" \"<reason>\" [--json]");
                std::process::exit(2);
            }
            let approach = &args[1];
            let reason = args[2..].join(" ");
            store.record_failure(approach, &reason)?;
            if json_mode {
                println!(
                    "{}",
                    serde_json::json!({ "status": "ok", "approach": approach, "reason": reason })
                );
            } else {
                println!("Logged failed attempt: {approach} -> {reason}");
            }
        }
        "clear" => {
            store.clear()?;
            if json_mode {
                println!("{}", serde_json::json!({ "status": "ok", "cleared": true }));
            } else {
                println!("Epistemic memory cleared.");
            }
        }
        "graph" => {
            let mermaid_mode = take_flag(&mut args, "--mermaid");
            if json_mode {
                let state = store.get();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "nodes": state.nodes,
                        "edges": state.edges,
                        "mermaid": store.render_mermaid()
                    }))?
                );
            } else if mermaid_mode {
                println!("{}", store.render_mermaid());
            } else {
                let state = store.get();
                println!(
                    "Epistemic Graph ({} nodes, {} edges):",
                    state.nodes.len(),
                    state.edges.len()
                );
                for n in &state.nodes {
                    println!(
                        "  [{}] ({}) {} (confidence: {}%)",
                        n.id,
                        n.category.as_str(),
                        n.statement,
                        n.confidence
                    );
                }
                for e in &state.edges {
                    println!("  {} --[{}]--> {}", e.from, e.relation.as_str(), e.to);
                }
            }
        }
        "link" => {
            if args.len() < 4 {
                eprintln!("usage: keel memory link <from> <to> <relation> [--json]");
                std::process::exit(2);
            }
            let from = &args[1];
            let to = &args[2];
            let rel_str = &args[3];
            let rel = keel_agent::RelationType::from_str_lossy(rel_str);
            let from_id =
                store.add_fact_node(from, keel_agent::FactCategory::ToolObservation, 80, None)?;
            let to_id =
                store.add_fact_node(to, keel_agent::FactCategory::ToolObservation, 80, None)?;
            store.add_edge(&from_id, &to_id, rel)?;
            if json_mode {
                println!(
                    "{}",
                    serde_json::json!({ "status": "ok", "from": from_id, "to": to_id, "relation": rel.as_str() })
                );
            } else {
                println!(
                    "Linked `{from_id}` -> `{to_id}` with relationship `{}`.",
                    rel.as_str()
                );
            }
        }
        "check" => {
            let alerts = store.detect_contradictions();
            if json_mode {
                println!("{}", serde_json::to_string_pretty(&alerts)?);
            } else if alerts.is_empty() {
                println!("✅ Zero contradictions detected in epistemic knowledge graph.");
            } else {
                println!("⚠️ Detected {} contradiction(s):", alerts.len());
                for a in alerts {
                    println!("  - {}", a.reason);
                }
            }
        }
        other => {
            eprintln!(
                "unknown memory subcommand `{other}`. Available: show, add-fact, set-hypothesis, record-failure, graph, link, check, clear"
            );
            std::process::exit(2);
        }
    }
    Ok(())
}

fn cli_handoff(mut args: Vec<String>, root_opt: Option<String>) -> anyhow::Result<()> {
    let sub = if !args.is_empty() {
        args.remove(0)
    } else {
        "ingest".to_string()
    };
    let root = match root_opt {
        Some(r) => std::fs::canonicalize(r)?,
        None => std::env::current_dir()?,
    };
    match sub.as_str() {
        "ingest" => {
            use std::io::Read;
            let file_arg = args.first().cloned();
            let raw_json = if let Some(path) = file_arg {
                if path == "-" {
                    let mut s = String::new();
                    std::io::stdin().read_to_string(&mut s)?;
                    s
                } else {
                    std::fs::read_to_string(&path)
                        .with_context(|| format!("reading handoff file {path}"))?
                }
            } else {
                let mut s = String::new();
                std::io::stdin().read_to_string(&mut s)?;
                s
            };
            let envelope = keel_agent::swarm::HandoffEnvelope::from_json_str(&raw_json)?;
            println!(
                "Ingesting handoff envelope from {} (session: {})...",
                envelope.source_harness, envelope.source_conversation_id
            );
            println!("  Goal: {}", envelope.goal);
            println!("  Relevant files: {:?}", envelope.relevant_files);
            println!(
                "  Tasks: {} open / {} completed",
                envelope.open_tasks.len(),
                envelope.completed_tasks.len()
            );

            let epistemic = keel_agent::epistemic::EpistemicStore::new(Some(
                keel_agent::epistemic::EpistemicMemoryPlugin::persist_file(&root),
            ));
            let _ = epistemic.set_hypothesis(Some(envelope.goal.clone()));
            for f in &envelope.relevant_files {
                let _ = epistemic.add_fact(&format!("Relevant file: {f}"));
            }
            if !envelope.context_summary.is_empty() {
                let _ =
                    epistemic.add_fact(&format!("Context summary: {}", envelope.context_summary));
            }
            let _ = epistemic.save_to_file();
            println!("  Epistemic belief state updated in {}", root.display());
            println!(
                "\nSynthesized prompt:\n---\n{}\n---",
                envelope.synth_prompt()
            );
            Ok(())
        }
        "export" => {
            let target_harness = take_opt(&mut args, "--target").unwrap_or_else(|| "helm".into());
            let epistemic = keel_agent::epistemic::EpistemicStore::new(Some(
                keel_agent::epistemic::EpistemicMemoryPlugin::persist_file(&root),
            ));
            let state = epistemic.get();
            let goal = state
                .hypothesis
                .clone()
                .unwrap_or_else(|| "Keel workspace".into());
            let mut relevant_files = Vec::new();
            for f in &state.facts {
                if let Some(p) = f.strip_prefix("Relevant file: ") {
                    relevant_files.push(p.to_string());
                }
            }
            relevant_files.sort();
            let envelope = keel_agent::swarm::HandoffEnvelope {
                source_conversation_id: format!("keel-{}", std::process::id()),
                source_harness: "keel".into(),
                target_harness,
                goal,
                relevant_files,
                open_tasks: Vec::new(),
                completed_tasks: Vec::new(),
                context_summary: state.render_markdown(),
            };
            println!("{}", envelope.to_json_pretty()?);
            Ok(())
        }
        other => {
            eprintln!("unknown handoff subcommand `{other}`. Usage: keel handoff <ingest|export>");
            std::process::exit(2);
        }
    }
}

fn cli_oracle(mut args: Vec<String>, root_opt: Option<String>) -> anyhow::Result<()> {
    let sub = if !args.is_empty() {
        args.remove(0)
    } else {
        "test".to_string()
    };
    let json_mode = take_flag(&mut args, "--json");
    let cmd_opt = take_opt(&mut args, "--cmd");
    let root = match root_opt {
        Some(r) => std::fs::canonicalize(r)?,
        None => std::env::current_dir()?,
    };
    let oracle = keel_agent::oracle::DiagnosticOracle::new(
        root,
        cmd_opt,
        keel_agent::oracle::SentinelMode::Warn,
    );
    match sub.as_str() {
        "test" | "run" => {
            let snap = oracle.run_tests()?;
            if json_mode {
                println!("{}", serde_json::to_string_pretty(&snap)?);
            } else {
                println!(
                    "Oracle test run completed in {}ms: {} total, {} passed, {} failed, {} ignored.",
                    snap.elapsed_ms, snap.total, snap.passed, snap.failed, snap.ignored
                );
                for f in &snap.failures {
                    println!("  ❌ {}", f.name);
                    if let Some(loc) = &f.location {
                        println!("     Location: {loc}");
                    }
                    if let Some(ass) = &f.assertion {
                        println!("     Assertion: {ass}");
                    }
                }
            }
            if snap.failed > 0 {
                std::process::exit(1);
            }
            Ok(())
        }
        "baseline" => {
            let snap = oracle.capture_baseline()?;
            if json_mode {
                println!("{}", serde_json::to_string_pretty(&snap)?);
            } else {
                println!(
                    "Captured test baseline: {} passed, {} failed across {} tests.",
                    snap.passed, snap.failed, snap.total
                );
            }
            Ok(())
        }
        "diff" => {
            let delta = oracle.check_delta()?;
            if json_mode {
                println!("{}", serde_json::to_string_pretty(&delta)?);
            } else if delta.is_clean {
                println!("✅ Zero test regressions detected.");
                if !delta.pre_existing_failures.is_empty() {
                    println!(
                        "   (quarantined {} pre-existing failures)",
                        delta.pre_existing_failures.len()
                    );
                }
            } else if let Some(rem) = delta.render_remediation() {
                println!("{rem}");
                std::process::exit(1);
            }
            Ok(())
        }
        other => {
            eprintln!(
                "unknown oracle subcommand `{other}`. Usage: keel oracle <test|baseline|diff> [--cmd \"...\"] [--json]"
            );
            std::process::exit(2);
        }
    }
}

fn cli_symbols(mut args: Vec<String>, root_opt: Option<String>) -> anyhow::Result<()> {
    let sub = if !args.is_empty() && !args[0].starts_with('-') {
        args.remove(0)
    } else {
        "search".to_string()
    };
    let json_mode = take_flag(&mut args, "--json");
    let root = match root_opt {
        Some(r) => std::fs::canonicalize(r)?,
        None => std::env::current_dir()?,
    };
    let index = keel_agent::SymbolIndex::new();

    match sub.as_str() {
        "outline" => {
            let file = args.first().cloned().unwrap_or_default();
            if file.is_empty() {
                eprintln!("Usage: keel symbols outline <file> [--json]");
                std::process::exit(2);
            }
            let abs = keel_agent::tools::confine(&root, &file)?;
            let outline = index.outline_file(&abs, &file)?;
            if json_mode {
                println!("{}", serde_json::to_string_pretty(&outline)?);
            } else {
                print!("{}", keel_agent::render_outline_markdown(&outline));
            }
            Ok(())
        }
        "search" => {
            let query = args.first().cloned().unwrap_or_default();
            if query.is_empty() {
                eprintln!(
                    "Usage: keel symbols search <query> [--path <filter>] [--limit <n>] [--json]"
                );
                std::process::exit(2);
            }
            let path_filter = take_opt(&mut args, "--path");
            let limit = take_opt(&mut args, "--limit")
                .and_then(|v| v.parse().ok())
                .unwrap_or(20);
            let matches = index.search_symbols(&root, &query, path_filter.as_deref(), limit)?;
            if json_mode {
                println!("{}", serde_json::to_string_pretty(&matches)?);
            } else if matches.is_empty() {
                println!("No symbols found matching `{query}`.");
            } else {
                println!("Symbol Matches for `{query}`:");
                for m in matches {
                    println!(
                        "  [{}] {} (score: {})",
                        m.symbol.kind.as_str(),
                        m.symbol.name,
                        m.score
                    );
                    println!(
                        "    File: {}:{}-{}",
                        m.path, m.symbol.start_line, m.symbol.end_line
                    );
                    println!("    Sig:  {}", m.symbol.signature);
                }
            }
            Ok(())
        }
        "get" => {
            if args.len() < 2 {
                eprintln!("Usage: keel symbols get <file> <symbol_name>");
                std::process::exit(2);
            }
            let file = &args[0];
            let name = &args[1];
            let abs = keel_agent::tools::confine(&root, file)?;
            match index.get_symbol_source(&abs, file, name)? {
                Some((sym, code)) => {
                    println!(
                        "Symbol `{}` ({} L{}-L{}):",
                        sym.name,
                        sym.kind.as_str(),
                        sym.start_line,
                        sym.end_line
                    );
                    println!("{code}");
                }
                None => {
                    eprintln!("Symbol `{name}` not found in `{file}`.");
                    std::process::exit(1);
                }
            }
            Ok(())
        }
        other => {
            eprintln!(
                "unknown symbols subcommand `{other}`. Usage: keel symbols <outline|search|get> ..."
            );
            std::process::exit(2);
        }
    }
}

fn cli_tree(mut args: Vec<String>, root_opt: Option<String>) -> anyhow::Result<()> {
    let sub = if !args.is_empty() && !args[0].starts_with('-') {
        args.remove(0)
    } else {
        "status".to_string()
    };
    let json_mode = take_flag(&mut args, "--json");
    let root = match root_opt {
        Some(r) => std::fs::canonicalize(r)?,
        None => std::env::current_dir()?,
    };
    let cp = keel_agent::checkpoint::CheckpointManager::new(root.clone());
    let mut tree = keel_agent::ExplorationTree::load_or_init(&root, &cp)?;

    match sub.as_str() {
        "status" => {
            if json_mode {
                println!("{}", serde_json::to_string_pretty(&tree)?);
            } else {
                println!("Exploration Tree (current: `{}`):", tree.current_id);
                print!("{}", tree.render_ascii());
            }
            Ok(())
        }
        "fork" => {
            let label = args.join(" ");
            let label = if label.is_empty() {
                "Branch".to_string()
            } else {
                label
            };
            let node = tree.fork(&label, &cp)?;
            if json_mode {
                println!("{}", serde_json::to_string_pretty(&node)?);
            } else {
                println!(
                    "Forked new branch `{}` (node `{}`) from `{}`.",
                    node.label,
                    node.id,
                    node.parent_id.as_deref().unwrap_or("none")
                );
            }
            Ok(())
        }
        "switch" => {
            let node_id = args.first().cloned().unwrap_or_default();
            if node_id.is_empty() {
                eprintln!("Usage: keel tree switch <node_id> [--json]");
                std::process::exit(2);
            }
            let node = tree.switch_to(&node_id, &cp)?;
            if json_mode {
                println!("{}", serde_json::to_string_pretty(&node)?);
            } else {
                println!(
                    "Switched to node `{}` (\"{}\"). Workspace restored.",
                    node.id, node.label
                );
            }
            Ok(())
        }
        "backtrack" => {
            let parent = tree.backtrack(&cp)?;
            if json_mode {
                println!("{}", serde_json::to_string_pretty(&parent)?);
            } else {
                println!(
                    "Backtracked to parent node `{}` (\"{}\"). Workspace restored.",
                    parent.id, parent.label
                );
            }
            Ok(())
        }
        "prune" => {
            let node_id = args.first().cloned().unwrap_or_default();
            if node_id.is_empty() {
                eprintln!("Usage: keel tree prune <node_id> [--json]");
                std::process::exit(2);
            }
            let count = tree.prune(&node_id)?;
            if json_mode {
                println!("{}", serde_json::json!({ "pruned_count": count }));
            } else {
                println!("Pruned {count} nodes under `{node_id}`.");
            }
            Ok(())
        }
        "accept" => {
            let node_id = args.first().cloned().unwrap_or_default();
            if node_id.is_empty() {
                eprintln!("Usage: keel tree accept <node_id> [--json]");
                std::process::exit(2);
            }
            let node = tree.accept(&node_id)?;
            if json_mode {
                println!("{}", serde_json::to_string_pretty(&node)?);
            } else {
                println!(
                    "Accepted node `{}` (\"{}\") as active state.",
                    node.id, node.label
                );
            }
            Ok(())
        }
        other => {
            eprintln!(
                "unknown tree subcommand `{other}`. Usage: keel tree <status|fork|switch|backtrack|prune|accept> ..."
            );
            std::process::exit(2);
        }
    }
}

fn cli_vfs(mut args: Vec<String>, root_opt: Option<String>) -> anyhow::Result<()> {
    let json_mode = take_flag(&mut args, "--json");
    let root = match root_opt {
        Some(r) => std::fs::canonicalize(r)?,
        None => std::env::current_dir()?,
    };
    let vfs = TransactionalVfs::new(root, VfsMode::MaterializedCow, true);

    let subcmd = args.first().map(String::as_str).unwrap_or("status");
    match subcmd {
        "status" => {
            let items = vfs.status();
            if json_mode {
                println!("{}", serde_json::to_string_pretty(&items)?);
            } else if items.is_empty() {
                println!("No uncommitted VFS transaction changes.");
            } else {
                for i in items {
                    println!(
                        "  {} [{}] ({}B -> {}B)",
                        i.path, i.status, i.original_bytes, i.staged_bytes
                    );
                }
            }
        }
        "diff" => {
            let d = vfs.diff();
            if json_mode {
                println!("{}", serde_json::json!({ "diff": d }));
            } else if d.is_empty() {
                println!("No diff in active VFS transaction.");
            } else {
                print!("{d}");
            }
        }
        "commit" => {
            let rep = vfs.commit()?;
            if json_mode {
                println!("{}", serde_json::to_string_pretty(&rep)?);
            } else {
                println!(
                    "Committed {} file(s) in VFS transaction.",
                    rep.committed_files
                );
            }
        }
        "rollback" => {
            let rep = vfs.rollback()?;
            if json_mode {
                println!("{}", serde_json::to_string_pretty(&rep)?);
            } else {
                println!(
                    "Rolled back VFS transaction (restored {} file(s), removed {} newly created file(s)).",
                    rep.restored_files, rep.removed_files
                );
            }
        }
        other => {
            eprintln!(
                "unknown vfs subcommand `{other}`. Available: status, diff, commit, rollback"
            );
            std::process::exit(2);
        }
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();

    if take_flag(&mut args, "--help") || take_flag(&mut args, "-h") {
        print_usage();
        return Ok(());
    }

    let root_opt = take_opt(&mut args, "--root");

    if args.first().map(String::as_str) == Some("patch") {
        args.remove(0);
        return cli_patch(args);
    }
    if args.first().map(String::as_str) == Some("edit") {
        args.remove(0);
        return cli_edit(args);
    }
    if args.first().map(String::as_str) == Some("checkpoint") {
        args.remove(0);
        return cli_checkpoint(args, root_opt);
    }
    if args.first().map(String::as_str) == Some("undo") {
        args.remove(0);
        return cli_undo(args, root_opt);
    }
    if args.first().map(String::as_str) == Some("mcp") {
        args.remove(0);
        return cli_mcp(args, root_opt);
    }
    if args.first().map(String::as_str) == Some("lsp") {
        args.remove(0);
        return cli_lsp(args, root_opt);
    }
    if args.first().map(String::as_str) == Some("vfs") {
        args.remove(0);
        return cli_vfs(args, root_opt);
    }
    if args.first().map(String::as_str) == Some("memory") {
        args.remove(0);
        return cli_memory(args, root_opt);
    }
    if args.first().map(String::as_str) == Some("handoff") {
        args.remove(0);
        return cli_handoff(args, root_opt);
    }
    if args.first().map(String::as_str) == Some("oracle") {
        args.remove(0);
        return cli_oracle(args, root_opt);
    }
    if args.first().map(String::as_str) == Some("symbols") {
        args.remove(0);
        return cli_symbols(args, root_opt);
    }
    if args.first().map(String::as_str) == Some("tree") {
        args.remove(0);
        return cli_tree(args, root_opt);
    }

    let mut interactive = take_flag(&mut args, "-i") || take_flag(&mut args, "--interactive");
    if args.first().map(String::as_str) == Some("repl") {
        args.remove(0);
        interactive = true;
    }
    if args.first().map(String::as_str) == Some("run") {
        args.remove(0);
    }

    let yolo = take_flag(&mut args, "--yolo");
    let read_only = take_flag(&mut args, "--read-only");
    let shadow_vfs = take_flag(&mut args, "--shadow");
    let safety_mode_opt = take_opt(&mut args, "--safety-mode");
    let no_sandbox = take_flag(&mut args, "--no-sandbox");
    let dump_config = take_flag(&mut args, "--dump-config");
    let json_mode = take_flag(&mut args, "--json");
    let auto_rollback = take_flag(&mut args, "--auto-rollback");
    let mcp_server_opts = take_all_opts(&mut args, "--mcp-server");
    let mut log_opt = take_opt(&mut args, "--log");
    let resume_opt = take_opt(&mut args, "--resume");
    let verify_cmd = take_opt(&mut args, "--verify");
    let runs: usize = take_opt(&mut args, "--runs")
        .and_then(|r| r.parse().ok())
        .unwrap_or(1);
    let candidates_opt: usize = take_opt(&mut args, "--candidates")
        .and_then(|c| c.parse().ok())
        .unwrap_or(2);
    let cascade_model_opt =
        take_opt(&mut args, "--cascade").or_else(|| std::env::var("KEEL_CASCADE_MODEL").ok());
    let apply_winner = take_flag(&mut args, "--apply");
    let handoff_opt = take_opt(&mut args, "--handoff");
    let sentinel_opt = take_opt(&mut args, "--sentinel");
    let listen_opt = take_opt(&mut args, "--listen");
    let socket_opt = take_opt(&mut args, "--socket");
    let auth_token_opt =
        take_opt(&mut args, "--auth-token").or_else(|| std::env::var("KEEL_DAEMON_TOKEN").ok());
    let daemon_mode = args.first().map(String::as_str) == Some("daemon");
    if daemon_mode {
        args.remove(0);
    }
    let speculate_mode = args.first().map(String::as_str) == Some("speculate");
    if speculate_mode {
        args.remove(0);
    }
    let bench_mode = args.first().map(String::as_str) == Some("bench");
    if bench_mode {
        args.remove(0);
    }
    let mut prompt = args.join(" ");

    if log_opt.is_none() {
        if let Some(r) = &resume_opt {
            log_opt = Some(r.clone());
        } else if let Ok(home) = std::env::var("HOME") {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0);
            log_opt = Some(format!("{home}/.keel/sessions/session-{ts}.jsonl"));
        }
    }

    let primary_llm: Arc<dyn Llm> = Arc::new(OpenAiCompat {
        base_url: std::env::var("KEEL_BASE_URL")
            .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        api_key: std::env::var("KEEL_API_KEY")
            .or_else(|_| std::env::var("DEEPSEEK_API_KEY"))
            .ok(),
        model: std::env::var("KEEL_MODEL").unwrap_or_else(|_| "deepseek-chat".into()),
        enable_thinking: std::env::var("KEEL_THINKING").ok().map(|v| v != "0"),
    });
    let llm: Arc<dyn Llm> = if let Some(frontier_model) = cascade_model_opt {
        let frontier: Arc<dyn Llm> = Arc::new(OpenAiCompat {
            base_url: std::env::var("KEEL_CASCADE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
            api_key: std::env::var("DEEPSEEK_API_KEY")
                .or_else(|_| std::env::var("KEEL_API_KEY"))
                .ok(),
            model: frontier_model,
            enable_thinking: Some(true),
        });
        Arc::new(CascadeProvider::new(primary_llm, frontier, 2))
    } else {
        primary_llm
    };
    if bench_mode {
        return bench::run(llm, args.first().map(String::as_str), runs);
    }

    let root = match root_opt {
        Some(r) => std::fs::canonicalize(r)?,
        None => std::env::current_dir()?,
    };
    if let Some(path) = handoff_opt {
        use std::io::Read;
        let raw_json = if path == "-" {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            s
        } else {
            std::fs::read_to_string(&path)
                .with_context(|| format!("reading handoff file {path}"))?
        };
        let envelope = keel_agent::swarm::HandoffEnvelope::from_json_str(&raw_json)?;
        if prompt.trim().is_empty() {
            prompt = envelope.synth_prompt();
        }
        let epistemic = keel_agent::epistemic::EpistemicStore::new(Some(
            keel_agent::epistemic::EpistemicMemoryPlugin::persist_file(&root),
        ));
        let _ = epistemic.set_hypothesis(Some(envelope.goal.clone()));
        for f in &envelope.relevant_files {
            let _ = epistemic.add_fact(&format!("Relevant file: {f}"));
        }
        if !envelope.context_summary.is_empty() {
            let _ = epistemic.add_fact(&format!("Context summary: {}", envelope.context_summary));
        }
        let _ = epistemic.save_to_file();
    }
    if daemon_mode {
        let policy = if yolo {
            ApprovalPolicy::Yolo
        } else if read_only {
            ApprovalPolicy::ReadOnly
        } else {
            safety_mode_opt
                .or_else(|| std::env::var("KEEL_SAFETY_MODE").ok())
                .map(|s| ApprovalPolicy::from_str_lossy(&s))
                .unwrap_or(ApprovalPolicy::Standard)
        };
        let cfg = DaemonConfig {
            root,
            policy,
            landlock_enabled: !no_sandbox,
            vfs_mode: if shadow_vfs {
                VfsMode::ShadowMemory
            } else {
                VfsMode::MaterializedCow
            },
            verify_command: verify_cmd,
        };
        if let Some(addr) = listen_opt {
            return keel_agent::daemon::serve_acp_tcp(&addr, cfg, llm, auth_token_opt);
        }
        #[cfg(unix)]
        if let Some(sock_path) = socket_opt {
            return keel_agent::daemon::serve_acp_unix(
                std::path::Path::new(&sock_path),
                cfg,
                llm,
                auth_token_opt,
            );
        }
        #[cfg(not(unix))]
        if let Some(_) = socket_opt {
            anyhow::bail!("Unix domain sockets are not supported on this platform");
        }
        let stdin = std::io::stdin();
        let reader = std::io::BufReader::new(stdin.lock());
        let writer = Box::new(std::io::stdout());
        return keel_agent::daemon::serve_acp_with_auth(reader, writer, cfg, llm, auth_token_opt);
    }
    if speculate_mode {
        if prompt.trim().is_empty() {
            eprintln!(
                "usage: keel speculate \"<prompt>\" [--candidates N] [--verify \"<cmd>\"] [--apply] [--json]"
            );
            std::process::exit(2);
        }
        let engine = SpeculationEngine {
            config: SpeculationConfig {
                root,
                candidates: candidates_opt,
                max_steps: 20,
                verify_command: verify_cmd,
                apply_winner: apply_winner || yolo,
                landlock_enabled: !no_sandbox,
            },
            llm,
        };
        let rep = engine.run(&prompt)?;
        if json_mode {
            println!("{}", serde_json::to_string_pretty(&rep)?);
        } else {
            println!(
                "Speculation complete ({} candidates, winner: {:?}, applied: {}):",
                rep.candidates_run, rep.winner_id, rep.applied_to_workspace
            );
            for c in &rep.candidates {
                println!(
                    "  [Candidate #{}] verified={} steps={} time={}ms files={:?}",
                    c.candidate_id, c.verification_passed, c.steps, c.elapsed_ms, c.modified_files
                );
            }
        }
        return Ok(());
    }
    let session = log_opt
        .map(SessionLog::file)
        .unwrap_or_else(SessionLog::none);
    let approver: Arc<dyn Approver> = if yolo {
        Arc::new(AllowAll)
    } else if read_only {
        Arc::new(ReadOnly)
    } else {
        let policy = safety_mode_opt
            .or_else(|| std::env::var("KEEL_SAFETY_MODE").ok())
            .map(|s| ApprovalPolicy::from_str_lossy(&s))
            .unwrap_or(ApprovalPolicy::Standard);
        Arc::new(InteractiveApprover::new(policy, json_mode))
    };

    // Everything is mounted as a plugin onto the kernel Context.
    let ctx = Context::new();
    if json_mode {
        ctx.mount(&JsonPrinter);
    } else {
        ctx.mount(&Printer);
    }
    let _p_llm = ctx.mount(&LlmPlugin(llm));
    let _p_approver = ctx.mount(&ApproverPlugin(approver));
    let _p_log = ctx.mount(&SessionLogPlugin(session));
    let _p_landlock = ctx.mount(&LandlockPlugin {
        workspace: root.clone(),
        enabled: !no_sandbox,
    });
    let _p_tools = ctx.mount(&BuiltinToolsPlugin {
        root: root.clone(),
        landlock_enabled: !no_sandbox,
    });
    let _p_guard = ctx.mount(&RepeatToolReminderPlugin::default());
    let _p_pruner = ctx.mount(&ToolResultPrunerPlugin::default());
    let _p_ws = ctx.mount(&WorkspaceContextPlugin { root: root.clone() });
    let _p_plan = ctx.mount(&PlanPlugin {
        root: root.clone(),
        verify_command: verify_cmd.clone(),
    });
    let _p_subagent = ctx.mount(&SubagentPlugin {
        root: root.clone(),
        landlock_enabled: !no_sandbox,
    });
    let _p_speculate = ctx.mount(&SpeculatePlugin {
        root: root.clone(),
        landlock_enabled: !no_sandbox,
    });
    let _p_ckpt = ctx.mount(&CheckpointPlugin {
        root: root.clone(),
        auto_checkpoint: true,
        auto_rollback,
    });
    let _p_epistemic = ctx.mount(&EpistemicMemoryPlugin { root: root.clone() });
    let _p_lsp = ctx.mount(&LspPlugin {
        root: root.clone(),
        extra_servers: std::collections::HashMap::new(),
        auto_post_edit_check: !shadow_vfs,
    });
    let _p_vfs = ctx.mount(&VfsPlugin {
        root: root.clone(),
        mode: if shadow_vfs {
            VfsMode::ShadowMemory
        } else {
            VfsMode::MaterializedCow
        },
        persist: !shadow_vfs,
    });
    let _p_oracle = ctx.mount(&keel_agent::OraclePlugin {
        root: root.clone(),
        test_command: verify_cmd.clone(),
        mode: sentinel_opt
            .as_deref()
            .map(keel_agent::SentinelMode::from_str_lossy)
            .unwrap_or(keel_agent::SentinelMode::Warn),
        auto_baseline: sentinel_opt.is_some(),
    });
    let _p_symbols = ctx.mount(&keel_agent::SymbolPlugin { root: root.clone() });
    let _p_tree = ctx.mount(&keel_agent::TreePlugin { root: root.clone() });

    let mut extra_servers = std::collections::HashMap::new();
    for spec in mcp_server_opts {
        if let Some((name, cmd_line)) = spec.split_once('=') {
            let parts: Vec<String> = cmd_line.split_whitespace().map(String::from).collect();
            if !parts.is_empty() {
                extra_servers.insert(
                    name.to_string(),
                    McpServerConfig {
                        command: parts[0].clone(),
                        args: parts[1..].to_vec(),
                        env: std::collections::HashMap::new(),
                    },
                );
            }
        }
    }
    let _p_mcp = ctx.mount(&McpPlugin {
        root: root.clone(),
        extra_servers,
    });

    if dump_config {
        println!("# keel composed plugin tree (root: {})", root.display());
        for (i, name) in ctx.mounted_plugins().iter().enumerate() {
            println!("{}. {}", i + 1, name);
        }
        return Ok(());
    }

    let mut sys_text = format!(
        "You are a coding agent working in {}. Use tools to inspect and change files. Be concise.",
        root.display()
    );
    if let Some(ws) = ctx.service::<WorkspaceInstructions>() {
        if !ws.extra_system_prompt.is_empty() {
            sys_text.push_str("\n\n");
            sys_text.push_str(&ws.extra_system_prompt);
        }
    }
    let mut history = vec![Message::System { text: sys_text }];

    if let Some(res_path) = &resume_opt {
        match SessionLog::read_history(res_path) {
            Ok(prev) => {
                if !json_mode {
                    println!("Loaded {} previous messages from {}", prev.len(), res_path);
                }
                history.extend(prev);
            }
            Err(e) => eprintln!("warning: failed to resume from {res_path}: {e:#}"),
        }
    }

    if prompt.is_empty() || interactive {
        println!("keel v0.0.1 (interactive REPL, root: {})", root.display());
        println!("Commands: /undo, /checkpoint, /memory, /clear, /history, exit, quit.\n");
        let stdin = std::io::stdin();
        loop {
            print!("keel> ");
            let _ = std::io::stdout().flush();
            let mut line = String::new();
            if stdin.read_line(&mut line)? == 0 {
                println!();
                break;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if matches!(trimmed, "exit" | "quit") {
                break;
            }
            if trimmed == "/undo" {
                if let Some(mgr) = ctx.service::<CheckpointManager>() {
                    match mgr.undo() {
                        Ok(rep) => {
                            println!(
                                "Restored checkpoint {} (restored {} files, removed {} newly created files).\n",
                                rep.checkpoint_id, rep.restored_files, rep.removed_files
                            );
                        }
                        Err(e) => eprintln!("Failed to undo: {e:#}\n"),
                    }
                } else {
                    println!("Checkpoint manager is not active.\n");
                }
                continue;
            }
            if trimmed == "/tree" {
                if let Some(tree_arc) =
                    ctx.service::<Arc<std::sync::Mutex<keel_agent::ExplorationTree>>>()
                {
                    let tree = tree_arc.lock().unwrap();
                    print!("{}", tree.render_ascii());
                }
                continue;
            }
            if trimmed == "/checkpoint" || trimmed == "/checkpoints" {
                if let Some(mgr) = ctx.service::<CheckpointManager>() {
                    match mgr.list() {
                        Ok(list) => {
                            if list.is_empty() {
                                println!("No checkpoints recorded yet.\n");
                            } else {
                                println!("Checkpoints ({}):", list.len());
                                for c in list {
                                    println!(
                                        "  - {} [{}] ({})",
                                        c.id,
                                        c.commit_hash.chars().take(7).collect::<String>(),
                                        c.description
                                    );
                                }
                                println!();
                            }
                        }
                        Err(e) => eprintln!("Failed to list checkpoints: {e:#}\n"),
                    }
                }
                continue;
            }
            if trimmed == "/memory" {
                if let Some(store) = ctx.service::<EpistemicStore>() {
                    let state = store.get();
                    if state.is_empty() {
                        println!("Epistemic working memory is currently empty.\n");
                    } else {
                        println!("{}", state.render_markdown());
                    }
                } else {
                    println!("Epistemic store is not active.\n");
                }
                continue;
            }
            if trimmed == "/clear" {
                history.truncate(1);
                println!("History cleared.\n");
                continue;
            }
            if trimmed == "/history" {
                println!("Active conversation depth: {} messages.\n", history.len());
                continue;
            }
            if let Err(e) = Agent::run_from_context(&ctx, &mut history, trimmed, 40) {
                eprintln!("\nerror: {e:#}\n");
            } else {
                println!();
            }
        }
    } else {
        match Agent::run_from_context(&ctx, &mut history, &prompt, 40) {
            Ok(ans) => {
                if json_mode {
                    println!(
                        "{}",
                        serde_json::json!({
                            "type": "done",
                            "status": "ok",
                            "answer": ans,
                        })
                    );
                } else {
                    println!();
                }
            }
            Err(e) => {
                if json_mode {
                    println!(
                        "{}",
                        serde_json::json!({
                            "type": "done",
                            "status": "error",
                            "error": format!("{e:#}"),
                        })
                    );
                    std::process::exit(1);
                } else {
                    return Err(e);
                }
            }
        }
    }

    Ok(())
}

fn take_flag(a: &mut Vec<String>, f: &str) -> bool {
    let n = a.len();
    a.retain(|x| x != f);
    a.len() != n
}

fn take_opt(a: &mut Vec<String>, f: &str) -> Option<String> {
    let i = a.iter().position(|x| x == f)?;
    if i + 1 >= a.len() {
        return None;
    }
    let v = a.remove(i + 1);
    a.remove(i);
    Some(v)
}

fn take_all_opts(a: &mut Vec<String>, f: &str) -> Vec<String> {
    let mut out = Vec::new();
    while let Some(v) = take_opt(a, f) {
        out.push(v);
    }
    out
}
