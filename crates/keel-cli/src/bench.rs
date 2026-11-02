//! `keel bench`: fixed coding tasks, scored per model/harness. Each task runs in a fresh temp dir.
//! Reports pass/fail, model steps and wall time. Mutating tools are allowed (temp dir only,
//! but `shell` is not sandboxed yet: run benchmarks only against models you trust).

use keel_agent::{tools, Agent, AllowAll, Llm, Message, SessionLog};
use keel_kernel::Context;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

struct Task {
    name: &'static str,
    prompt: &'static str,
    setup: &'static [(&'static str, &'static str)],
    verify: fn(dir: &Path, answer: &str) -> bool,
}

fn tasks() -> Vec<Task> {
    vec![
        Task {
            name: "create_file",
            prompt: "Create a file named hello.txt whose entire content is exactly: hi",
            setup: &[],
            verify: |d, _| std::fs::read_to_string(d.join("hello.txt")).map(|s| s.trim() == "hi").unwrap_or(false),
        },
        Task {
            name: "find_value",
            prompt: "One file in this workspace contains a line 'SECRET=<number>'. Find it and reply with the number.",
            setup: &[("a/x.txt", "nothing here"), ("b/y.txt", "noise\nSECRET=48213\nmore"), ("c.txt", "empty")],
            verify: |_, ans| ans.contains("48213"),
        },
        Task {
            name: "fix_bug",
            prompt: "test_add.py fails. Fix the bug in add.py (do not edit the test) and make sure `python3 test_add.py` passes.",
            setup: &[
                ("add.py", "def add(a, b):\n    return a - b\n"),
                ("test_add.py", "from add import add\nassert add(2, 3) == 5\nassert add(-1, 1) == 0\nprint('ok')\n"),
            ],
            verify: |d, _| Command::new("python3").arg("test_add.py").current_dir(d).output().map(|o| o.status.success()).unwrap_or(false),
        },
    ]
}

pub fn run(llm: &dyn Llm, filter: Option<&str>) -> anyhow::Result<()> {
    let mut passed = 0;
    let mut total = 0;
    println!(
        "{:<14} {:<6} {:>5} {:>8}",
        "task", "result", "steps", "secs"
    );
    for t in tasks()
        .into_iter()
        .filter(|t| filter.is_none_or(|f| t.name.contains(f)))
    {
        let dir =
            std::env::temp_dir().join(format!("keel-bench-{}-{}", t.name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir)?;
        for (path, body) in t.setup {
            let p = dir.join(path);
            std::fs::create_dir_all(p.parent().unwrap())?;
            std::fs::write(p, body)?;
        }
        let ctx = Context::new();
        let agent = Agent {
            ctx: &ctx,
            llm,
            tools: tools::builtin(&dir),
            approver: &AllowAll,
            log: &SessionLog::none(),
            max_steps: 15,
        };
        let mut history = vec![Message::System {
            text: format!("You are a coding agent working in {}. Use tools to inspect and change files. Be concise.", dir.display()),
        }];
        let start = Instant::now();
        let result = agent.run_turn(&mut history, t.prompt);
        let secs = start.elapsed().as_secs_f32();
        let steps = history
            .iter()
            .filter(|m| matches!(m, Message::Assistant { .. }))
            .count();
        let ok = result
            .as_ref()
            .map(|a| (t.verify)(&dir, a))
            .unwrap_or(false);
        total += 1;
        passed += ok as u32;
        let note = result.err().map(|e| format!("  ({e})")).unwrap_or_default();
        println!(
            "{:<14} {:<6} {:>5} {:>8.1}{note}",
            t.name,
            if ok { "PASS" } else { "FAIL" },
            steps,
            secs
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
    println!("{passed}/{total} passed");
    Ok(())
}
