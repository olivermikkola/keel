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
    checkpoint::{CheckpointManager, CheckpointPlugin},
    compaction::{ToolResultPrunerPlugin, WorkspaceContextPlugin, WorkspaceInstructions},
    epistemic::{EpistemicMemoryPlugin, EpistemicStore},
    guard::RepeatToolReminderPlugin,
    mcp::{McpClient, McpPlugin, McpServerConfig},
    plan::PlanPlugin,
    provider::OpenAiCompat,
    sandbox::LandlockPlugin,
    subagent::SubagentPlugin,
    tools::BuiltinToolsPlugin,
    Agent, AllowAll, ApprovalPolicy, Approver, ApproverPlugin, InteractiveApprover, LlmPlugin,
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
  keel memory <cmd>                epistemic belief state (show, add-fact, set-hypothesis, clear)
  keel bench [filter] [--runs N]   run benchmark suite

options:
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
        other => {
            eprintln!(
                "unknown memory subcommand `{other}`. Available: show, add-fact, set-hypothesis, record-failure, clear"
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
    if args.first().map(String::as_str) == Some("memory") {
        args.remove(0);
        return cli_memory(args, root_opt);
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
    let bench_mode = args.first().map(String::as_str) == Some("bench");
    if bench_mode {
        args.remove(0);
    }
    let prompt = args.join(" ");

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

    let llm = Arc::new(OpenAiCompat {
        base_url: std::env::var("KEEL_BASE_URL")
            .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        api_key: std::env::var("KEEL_API_KEY")
            .or_else(|_| std::env::var("DEEPSEEK_API_KEY"))
            .ok(),
        model: std::env::var("KEEL_MODEL").unwrap_or_else(|_| "deepseek-chat".into()),
        enable_thinking: std::env::var("KEEL_THINKING").ok().map(|v| v != "0"),
    });
    if bench_mode {
        return bench::run(llm, args.first().map(String::as_str), runs);
    }

    let root = match root_opt {
        Some(r) => std::fs::canonicalize(r)?,
        None => std::env::current_dir()?,
    };
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
        verify_command: verify_cmd,
    });
    let _p_subagent = ctx.mount(&SubagentPlugin {
        root: root.clone(),
        landlock_enabled: !no_sandbox,
    });
    let _p_ckpt = ctx.mount(&CheckpointPlugin {
        root: root.clone(),
        auto_checkpoint: true,
        auto_rollback,
    });
    let _p_epistemic = ctx.mount(&EpistemicMemoryPlugin { root: root.clone() });

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
