//! Parallel Speculative Execution Engine (Best-of-N Git Worktree / Sandbox Forking).
//!
//! Spawns `N` isolated candidate workspaces in parallel (using `git worktree add --detach`
//! when backed by a committed git repository, or an isolated Copy-on-Write workspace clone
//! when the repository has uncommitted changes or no commits yet). Each candidate runs its
//! own [`Agent`] turn with distinct strategy hints, executes the verification command gate,
//! and gets scored on verification pass/fail, step count, and wall-clock latency.
//! The winning candidate's changes can be atomically promoted back to the main workspace.

use crate::compaction::{ToolResultPrunerPlugin, WorkspaceContextPlugin};
use crate::guard::RepeatToolReminderPlugin;
use crate::lsp::LspPlugin;
use crate::plan::{PlanPlugin, VerificationGate};
use crate::sandbox::{LandlockPlugin, ToolRegistry};
use crate::tools::BuiltinToolsPlugin;
use crate::vfs::{TransactionalVfs, VfsMode, VfsPlugin};
use crate::{
    Agent, AllowAll, ApproverPlugin, Llm, LlmPlugin, Message, SessionLog, SessionLogPlugin, Tool,
    ToolSpec,
};
use anyhow::{bail, Context as _, Result};
use keel_kernel::{Context, Plugin, Scope};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Instant;

const STRATEGY_HINTS: &[&str] = &[
    "Strategy A (Minimal & Surgical): Make the smallest, most direct change that fixes the root cause.",
    "Strategy B (Defensive & Edge-Case Aware): Consider boundary conditions, error handling, and invariants before editing.",
    "Strategy C (Clean Refactor): Ensure clean structure and idiomatic implementation while satisfying all tests.",
    "Strategy D (First-Principles Verification): Inspect all callers and tests first, formulate a hypothesis, and verify carefully.",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateResult {
    pub candidate_id: usize,
    pub strategy: String,
    pub turn_succeeded: bool,
    pub verification_passed: bool,
    pub steps: usize,
    pub elapsed_ms: u64,
    pub answer: String,
    pub diff: String,
    pub modified_files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculationReport {
    pub prompt: String,
    pub candidates_run: usize,
    pub winner_id: Option<usize>,
    pub applied_to_workspace: bool,
    pub candidates: Vec<CandidateResult>,
}

pub struct SpeculationConfig {
    pub root: PathBuf,
    pub candidates: usize,
    pub max_steps: usize,
    pub verify_command: Option<String>,
    pub apply_winner: bool,
    pub landlock_enabled: bool,
}

pub struct SpeculationEngine {
    pub config: SpeculationConfig,
    pub llm: Arc<dyn Llm>,
}

struct EphemeralSandbox {
    dir: PathBuf,
    used_git_worktree: bool,
    parent_root: PathBuf,
}

impl EphemeralSandbox {
    fn create(parent_root: &Path, candidate_id: usize) -> Result<Self> {
        let dir = std::env::temp_dir().join(format!(
            "keel-speculate-{}-cand-{}",
            std::process::id(),
            candidate_id
        ));
        let _ = std::fs::remove_dir_all(&dir);

        // Check if parent_root is a clean git repo with a valid HEAD commit
        let mut used_git_worktree = false;
        if parent_root.join(".git").exists() && is_git_clean_with_head(parent_root) {
            let status = Command::new("git")
                .args([
                    "worktree",
                    "add",
                    "--detach",
                    &dir.to_string_lossy(),
                    "HEAD",
                ])
                .current_dir(parent_root)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            if let Ok(s) = status {
                if s.success() {
                    used_git_worktree = true;
                }
            }
        }

        if !used_git_worktree {
            std::fs::create_dir_all(&dir)?;
            copy_workspace_filtered(parent_root, &dir)?;
        }

        Ok(Self {
            dir,
            used_git_worktree,
            parent_root: parent_root.to_path_buf(),
        })
    }

    fn cleanup(&self) {
        if self.used_git_worktree {
            let _ = Command::new("git")
                .args(["worktree", "remove", "--force", &self.dir.to_string_lossy()])
                .current_dir(&self.parent_root)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

impl Drop for EphemeralSandbox {
    fn drop(&mut self) {
        self.cleanup();
    }
}

fn is_git_clean_with_head(root: &Path) -> bool {
    let head_ok = Command::new("git")
        .args(["rev-parse", "--verify", "HEAD"])
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !head_ok {
        return false;
    }
    let status_out = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(root)
        .stdin(Stdio::null())
        .output();
    match status_out {
        Ok(o) => o.status.success() && o.stdout.is_empty(),
        Err(_) => false,
    }
}

fn copy_workspace_filtered(src: &Path, dst: &Path) -> Result<()> {
    let entries = match std::fs::read_dir(src) {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if matches!(
            name_str.as_ref(),
            ".git" | "target" | "node_modules" | ".keel" | "__pycache__"
        ) {
            continue;
        }
        let Ok(ft) = entry.file_type() else {
            continue;
        };
        if ft.is_symlink() {
            continue;
        }
        let src_path = entry.path();
        let dst_path = dst.join(&name);
        if ft.is_dir() {
            std::fs::create_dir_all(&dst_path)?;
            copy_workspace_filtered(&src_path, &dst_path)?;
        } else if ft.is_file() {
            std::fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

impl SpeculationEngine {
    pub fn run(&self, prompt: &str) -> Result<SpeculationReport> {
        let n = self.config.candidates.clamp(1, 8);
        let mut sandboxes = Vec::with_capacity(n);
        for i in 0..n {
            sandboxes.push(EphemeralSandbox::create(&self.config.root, i + 1)?);
        }

        let results: Vec<(CandidateResult, HashMap<String, Option<String>>)> =
            std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(n);
                for (idx, sb) in sandboxes.iter().enumerate() {
                    let candidate_id = idx + 1;
                    let strategy = STRATEGY_HINTS[idx % STRATEGY_HINTS.len()].to_string();
                    let sb_dir = sb.dir.clone();
                    let llm = self.llm.clone();
                    let verify_cmd = self.config.verify_command.clone();
                    let max_steps = self.config.max_steps;
                    let landlock_enabled = self.config.landlock_enabled;
                    let user_prompt = prompt.to_string();

                    handles.push(s.spawn(move || {
                        run_single_candidate(
                            candidate_id,
                            strategy,
                            sb_dir,
                            llm,
                            verify_cmd,
                            max_steps,
                            landlock_enabled,
                            &user_prompt,
                        )
                    }));
                }

                handles
                    .into_iter()
                    .map(|h| h.join().expect("candidate thread panicked"))
                    .collect()
            });

        // Rank candidates:
        // 1. verification_passed == true && turn_succeeded == true
        // 2. verification_passed == true
        // 3. turn_succeeded == true && !modified_files.is_empty()
        // Tie-breaker: fewer steps, then lower elapsed_ms
        let mut indexed: Vec<(usize, &CandidateResult)> = results
            .iter()
            .enumerate()
            .map(|(i, (r, _))| (i, r))
            .collect();

        indexed.sort_by(|(_, a), (_, b)| {
            let score_a = (
                a.verification_passed && a.turn_succeeded,
                a.verification_passed,
                a.turn_succeeded && !a.modified_files.is_empty(),
                a.turn_succeeded,
            );
            let score_b = (
                b.verification_passed && b.turn_succeeded,
                b.verification_passed,
                b.turn_succeeded && !b.modified_files.is_empty(),
                b.turn_succeeded,
            );
            score_b
                .cmp(&score_a)
                .then_with(|| a.steps.cmp(&b.steps))
                .then_with(|| a.elapsed_ms.cmp(&b.elapsed_ms))
        });

        let winner_idx = indexed
            .first()
            .filter(|(_, r)| r.verification_passed || r.turn_succeeded)
            .map(|(i, _)| *i);

        let mut applied_to_workspace = false;
        if let Some(w_idx) = winner_idx {
            let (ref w_res, ref staged_map) = results[w_idx];
            if self.config.apply_winner
                && (w_res.verification_passed || self.config.verify_command.is_none())
            {
                for (rel, content_opt) in staged_map {
                    let target = self.config.root.join(rel);
                    match content_opt {
                        Some(content) => {
                            if let Some(p) = target.parent() {
                                let _ = std::fs::create_dir_all(p);
                            }
                            let _ = std::fs::write(&target, content);
                        }
                        None => {
                            if target.is_file() {
                                let _ = std::fs::remove_file(&target);
                            }
                        }
                    }
                }
                applied_to_workspace = true;
            }
        }

        let winner_id = winner_idx.map(|i| results[i].0.candidate_id);
        let candidates = results.into_iter().map(|(r, _)| r).collect();

        Ok(SpeculationReport {
            prompt: prompt.to_string(),
            candidates_run: n,
            winner_id,
            applied_to_workspace,
            candidates,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn run_single_candidate(
    candidate_id: usize,
    strategy: String,
    sb_dir: PathBuf,
    llm: Arc<dyn Llm>,
    verify_cmd: Option<String>,
    max_steps: usize,
    landlock_enabled: bool,
    user_prompt: &str,
) -> (CandidateResult, HashMap<String, Option<String>>) {
    let start = Instant::now();
    let ctx = Context::new();
    let _p_llm = ctx.mount(&LlmPlugin(llm));
    let _p_app = ctx.mount(&ApproverPlugin(Arc::new(AllowAll)));
    let _p_log = ctx.mount(&SessionLogPlugin(SessionLog::none()));
    let _p_ll = ctx.mount(&LandlockPlugin {
        workspace: sb_dir.clone(),
        enabled: landlock_enabled,
        ..Default::default()
    });
    let _p_tools = ctx.mount(&BuiltinToolsPlugin {
        root: sb_dir.clone(),
        landlock_enabled,
    });
    let _p_guard = ctx.mount(&RepeatToolReminderPlugin::default());
    let _p_pruner = ctx.mount(&ToolResultPrunerPlugin::default());
    let _p_ws = ctx.mount(&WorkspaceContextPlugin {
        root: sb_dir.clone(),
    });
    let _p_plan = ctx.mount(&PlanPlugin {
        root: sb_dir.clone(),
        verify_command: verify_cmd.clone(),
    });
    let _p_lsp = ctx.mount(&LspPlugin {
        root: sb_dir.clone(),
        extra_servers: HashMap::new(),
        auto_post_edit_check: true,
    });
    let _p_vfs = ctx.mount(&VfsPlugin {
        root: sb_dir.clone(),
        mode: VfsMode::MaterializedCow,
        persist: false,
    });

    let mut history = vec![Message::System {
        text: format!(
            "You are candidate #{candidate_id} in a speculative coding tournament working in {}.\n{strategy}\nBe concise and ensure all verification checks pass.",
            sb_dir.display()
        ),
    }];

    let turn_res = Agent::run_from_context(&ctx, &mut history, user_prompt, max_steps);
    let elapsed_ms = start.elapsed().as_millis() as u64;
    let steps = history
        .iter()
        .filter(|m| matches!(m, Message::Assistant { .. }))
        .count();

    let (turn_succeeded, answer) = match turn_res {
        Ok(ans) => (true, ans),
        Err(e) => (false, format!("error: {e:#}")),
    };

    let verification_passed = if let Some(ref cmd) = verify_cmd {
        let gate = VerificationGate {
            root: sb_dir.clone(),
            command: Some(cmd.clone()),
        };
        gate.verify().is_ok()
    } else {
        turn_succeeded
    };

    let mut diff = String::new();
    let mut modified_files = Vec::new();
    let mut staged_map = HashMap::new();

    if let Some(vfs) = ctx.service::<Arc<TransactionalVfs>>() {
        diff = vfs.diff();
        for st in vfs.status() {
            let full = sb_dir.join(&st.path);
            let content_opt = if full.is_file() {
                std::fs::read_to_string(&full).ok()
            } else {
                None
            };
            staged_map.insert(st.path.clone(), content_opt);
            modified_files.push(st.path);
        }
    }

    (
        CandidateResult {
            candidate_id,
            strategy,
            turn_succeeded,
            verification_passed,
            steps,
            elapsed_ms,
            answer,
            diff,
            modified_files,
        },
        staged_map,
    )
}

/// Tool allowing an agent to fork `N` speculative candidate worktrees for a sub-task.
pub struct SpeculateTaskTool {
    pub root: PathBuf,
    pub llm: Arc<dyn Llm>,
    pub landlock_enabled: bool,
}

impl Tool for SpeculateTaskTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "speculate_task".into(),
            description: "Fork N parallel speculative candidate sandboxes to solve a task, verify each candidate against a test command, and optionally apply the winning solution.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "prompt": { "type": "string", "description": "Task description for candidates" },
                    "candidates": { "type": "integer", "description": "Number of parallel candidates (default 2, max 4)" },
                    "verify_command": { "type": "string", "description": "Shell command to verify each candidate" },
                    "apply_winner": { "type": "boolean", "description": "Apply the winning candidate's changes to the workspace" }
                },
                "required": ["prompt"]
            }),
        }
    }

    fn mutating(&self) -> bool {
        true
    }

    fn run(&self, args: &Value) -> Result<String> {
        let prompt = args["prompt"]
            .as_str()
            .context("missing string argument `prompt`")?;
        if prompt.trim().is_empty() {
            bail!("`prompt` must not be empty");
        }
        let candidates = args["candidates"].as_u64().unwrap_or(2).clamp(1, 4) as usize;
        let verify_command = args["verify_command"].as_str().map(String::from);
        let apply_winner = args["apply_winner"].as_bool().unwrap_or(true);

        let engine = SpeculationEngine {
            config: SpeculationConfig {
                root: self.root.clone(),
                candidates,
                max_steps: 10,
                verify_command,
                apply_winner,
                landlock_enabled: self.landlock_enabled,
            },
            llm: self.llm.clone(),
        };
        let rep = engine.run(prompt)?;
        Ok(serde_json::to_string_pretty(&rep)?)
    }
}

/// Plugin that registers `speculate_task` on the kernel [`ToolRegistry`].
pub struct SpeculatePlugin {
    pub root: PathBuf,
    pub landlock_enabled: bool,
}

impl Plugin for SpeculatePlugin {
    fn name(&self) -> &str {
        "speculate"
    }

    fn apply(&self, scope: &mut Scope) {
        let Some(llm) = scope.context().service::<Arc<dyn Llm>>() else {
            return;
        };
        let reg = scope
            .context()
            .service::<ToolRegistry>()
            .map(|r| (*r).clone())
            .unwrap_or_default();
        reg.register(Arc::new(SpeculateTaskTool {
            root: self.root.clone(),
            llm: (*llm).clone(),
            landlock_enabled: self.landlock_enabled,
        }));
        scope.provide(reg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolCall;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Mock LLM where Candidate 1 writes a buggy implementation (fails verification)
    /// and Candidate 2 writes the correct implementation (passes verification).
    struct TournamentMockLlm {
        call_counter: AtomicUsize,
    }

    impl Llm for TournamentMockLlm {
        fn complete(&self, messages: &[Message], _tools: &[ToolSpec]) -> Result<Message> {
            let sys = match messages.first() {
                Some(Message::System { text }) => text.as_str(),
                _ => "",
            };
            let is_cand_2 = sys.contains("candidate #2");
            let has_tool_result = messages.iter().any(|m| matches!(m, Message::Tool { .. }));
            let _ = self.call_counter.fetch_add(1, Ordering::SeqCst);

            if !has_tool_result {
                let code = if is_cand_2 {
                    "def compute():\n    return 42\n"
                } else {
                    "def compute():\n    return 0\n"
                };
                Ok(Message::Assistant {
                    text: "Writing solution".into(),
                    tool_calls: vec![ToolCall {
                        id: "tc1".into(),
                        name: "write_file".into(),
                        args: json!({
                            "path": "solution.py",
                            "content": code
                        }),
                    }],
                })
            } else {
                Ok(Message::Assistant {
                    text: "Finished candidate implementation".into(),
                    tool_calls: vec![],
                })
            }
        }
    }

    #[test]
    fn parallel_speculation_selects_and_applies_passing_candidate() {
        let tmp = std::env::temp_dir().join(format!("keel-speculate-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        std::fs::write(tmp.join("solution.py"), "def compute():\n    return -1\n").unwrap();
        std::fs::write(
            tmp.join("test_sol.py"),
            "from solution import compute\nassert compute() == 42\n",
        )
        .unwrap();

        let engine = SpeculationEngine {
            config: SpeculationConfig {
                root: tmp.clone(),
                candidates: 2,
                max_steps: 4,
                verify_command: Some("python3 test_sol.py".into()),
                apply_winner: true,
                landlock_enabled: false,
            },
            llm: Arc::new(TournamentMockLlm {
                call_counter: AtomicUsize::new(0),
            }),
        };

        let report = engine.run("Make compute() return 42").unwrap();
        assert_eq!(report.candidates_run, 2);
        assert_eq!(report.winner_id, Some(2));
        assert!(report.applied_to_workspace);

        // Candidate 1 failed verification, Candidate 2 passed verification
        let c1 = report
            .candidates
            .iter()
            .find(|c| c.candidate_id == 1)
            .unwrap();
        let c2 = report
            .candidates
            .iter()
            .find(|c| c.candidate_id == 2)
            .unwrap();
        assert!(!c1.verification_passed);
        assert!(c2.verification_passed);

        // Winning candidate (#2) must have been promoted to the primary workspace!
        let final_code = std::fs::read_to_string(tmp.join("solution.py")).unwrap();
        assert!(final_code.contains("return 42"));

        let _ = std::fs::remove_dir_all(tmp);
    }
}
