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
        Task {
            name: "rename_symbol",
            prompt: "Rename the function `compute_total` to `calc_total` everywhere in this workspace (definition and all call sites). `python3 main.py` must still print 6.",
            setup: &[
                ("pricing.py", "def compute_total(items):\n    return sum(items)\n"),
                ("report.py", "from pricing import compute_total\n\ndef summary(items):\n    return compute_total(items)\n"),
                ("main.py", "from report import summary\nfrom pricing import compute_total\nprint(summary([1, 2, 3]))\nassert compute_total([1]) == 1\n"),
            ],
            verify: |d, _| {
                let any_old = ["pricing.py", "report.py", "main.py"]
                    .iter()
                    .any(|f| std::fs::read_to_string(d.join(f)).map(|s| s.contains("compute_total")).unwrap_or(true));
                let out = Command::new("python3").arg("main.py").current_dir(d).output();
                !any_old && out.map(|o| o.status.success() && String::from_utf8_lossy(&o.stdout).trim() == "6").unwrap_or(false)
            },
        },
        Task {
            name: "locate_def",
            prompt: "Which file defines the function `retry_with_backoff`? Reply with the file path only.",
            setup: &[
                ("src/a.py", "def helper():\n    pass\n"),
                ("src/net/http.py", "from util import retry_with_backoff  # imported, not defined\n"),
                ("src/net/util/retry.py", "def retry_with_backoff(fn):\n    return fn()\n"),
                ("docs/notes.md", "we should call retry_with_backoff more often\n"),
            ],
            verify: |_, ans| ans.contains("src/net/util/retry.py"),
        },
        Task {
            name: "misleading_error",
            prompt: "Running `python3 main.py` fails. Find the real root cause and fix it so it exits successfully. Do not weaken the assertion in main.py.",
            setup: &[
                ("main.py", "from stats import mean_of_pairs\nresult = mean_of_pairs([(1, 3), (2, 6)])\nassert result == 3.0, 'main.py: wrong result'\nprint('ok')\n"),
                ("stats.py", "def mean_of_pairs(pairs):\n    totals = [a + b for a, b in pairs]\n    return sum(totals) / (len(totals) + 1)\n"),
            ],
            verify: |d, _| Command::new("python3").arg("main.py").current_dir(d).output().map(|o| o.status.success()).unwrap_or(false)
                && std::fs::read_to_string(d.join("main.py")).map(|s| s.contains("== 3.0")).unwrap_or(false),
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
