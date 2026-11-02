//! `keel "prompt"` - one-shot headless run. `keel bench [filter]` - run the benchmark tasks.
//!
//! Env: KEEL_THINKING=0 disables model thinking (Qwen templates). KEEL_BASE_URL (default https://api.deepseek.com), KEEL_API_KEY (or DEEPSEEK_API_KEY),
//! KEEL_MODEL (default deepseek-chat). Flags: --yolo (allow mutating tools), --no-sandbox (disable Landlock), --log <file>.
//! Default policy is read-only: mutating tools are denied unless --yolo is given.

use keel_agent::Chunk;
use keel_agent::{
    provider::OpenAiCompat, sandbox::LandlockPlugin, tools::BuiltinToolsPlugin, Agent, AllowAll,
    Approver, ApproverPlugin, LlmPlugin, Message, ReadOnly, SessionLog, SessionLogPlugin,
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

fn main() -> anyhow::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let yolo = take_flag(&mut args, "--yolo");
    let no_sandbox = take_flag(&mut args, "--no-sandbox");
    let log = take_opt(&mut args, "--log");
    let root_opt = take_opt(&mut args, "--root");
    let runs: usize = take_opt(&mut args, "--runs")
        .and_then(|r| r.parse().ok())
        .unwrap_or(1);
    let bench_mode = args.first().map(String::as_str) == Some("bench");
    if bench_mode {
        args.remove(0);
    }
    let prompt = args.join(" ");
    if prompt.is_empty() && !bench_mode {
        eprintln!("usage: keel [--yolo] [--no-sandbox] [--root DIR] [--log FILE] \"prompt\"");
        std::process::exit(2);
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
        return bench::run(llm.as_ref(), args.first().map(String::as_str), runs);
    }

    let root = match root_opt {
        Some(r) => std::fs::canonicalize(r)?,
        None => std::env::current_dir()?,
    };
    let session = log.map(SessionLog::file).unwrap_or_else(SessionLog::none);
    let approver: Arc<dyn Approver> = if yolo {
        Arc::new(AllowAll)
    } else {
        Arc::new(ReadOnly)
    };

    // Everything is mounted as a plugin onto the kernel Context.
    let ctx = Context::new();
    let _p_printer = ctx.mount(&Printer);
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

    let mut history = vec![Message::System {
        text: format!(
            "You are a coding agent working in {}. Use tools to inspect and change files. Be concise.",
            root.display()
        ),
    }];
    Agent::run_from_context(&ctx, &mut history, &prompt, 40)?;
    println!();
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
