//! `keel "prompt"` - one-shot headless run.
//!
//! Env: KEEL_BASE_URL (default https://api.deepseek.com), KEEL_API_KEY (or DEEPSEEK_API_KEY),
//! KEEL_MODEL (default deepseek-chat). Flags: --yolo (allow mutating tools), --log <file>.
//! Default policy is read-only: mutating tools are denied unless --yolo is given.

use keel_agent::{
    provider::OpenAiCompat, tools, Agent, AllowAll, Approver, Message, ReadOnly, SessionLog,
};
use keel_kernel::Context;

fn main() -> anyhow::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let yolo = take_flag(&mut args, "--yolo");
    let log = take_opt(&mut args, "--log");
    let root_opt = take_opt(&mut args, "--root");
    let prompt = args.join(" ");
    if prompt.is_empty() {
        eprintln!("usage: keel [--yolo] [--root DIR] [--log FILE] \"prompt\"");
        std::process::exit(2);
    }

    let llm = OpenAiCompat {
        base_url: std::env::var("KEEL_BASE_URL")
            .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        api_key: std::env::var("KEEL_API_KEY")
            .or_else(|_| std::env::var("DEEPSEEK_API_KEY"))
            .ok(),
        model: std::env::var("KEEL_MODEL").unwrap_or_else(|_| "deepseek-chat".into()),
    };
    let ctx = Context::new();
    let session = log.map(SessionLog::file).unwrap_or_else(SessionLog::none);
    let approver: &dyn Approver = if yolo { &AllowAll } else { &ReadOnly };
    let root = match root_opt {
        Some(r) => std::fs::canonicalize(r)?,
        None => std::env::current_dir()?,
    };
    let agent = Agent {
        ctx: &ctx,
        llm: &llm,
        tools: tools::builtin(&root),
        approver,
        log: &session,
        max_steps: 40,
    };

    let mut history = vec![Message::System {
        text: format!("You are a coding agent working in {}. Use tools to inspect and change files. Be concise.", root.display()),
    }];
    println!("{}", agent.run_turn(&mut history, &prompt)?);
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
