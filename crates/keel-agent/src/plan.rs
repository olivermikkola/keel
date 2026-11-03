//! Plan & task checklist management + verification gate (`keel-agent::plan`).
//!
//! 1. [`TodoWrite`]: Structured task-list tool (`todo_write`), whole-list replacement
//!    modeled on `@deepseek-ai/dsh-tool-todo`.
//! 2. [`VerificationGate`]: Post-step verification command (e.g. `--verify "cargo test"`).
//!    When the model attempts to declare completion (no tool calls), the verification command
//!    runs. If it fails, the failure output is injected back into the conversation so the model
//!    must fix the issue before finishing.

use crate::{Tool, ToolSpec};
use anyhow::{bail, Context as _, Result};
use keel_kernel::{Plugin, Scope};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TodoItem {
    pub description: String,
    pub status: TodoStatus,
}

#[derive(Default, Clone)]
pub struct PlanStore {
    todos: Arc<Mutex<Vec<TodoItem>>>,
}

impl PlanStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self) -> Vec<TodoItem> {
        self.todos.lock().unwrap().clone()
    }

    pub fn replace(&self, items: Vec<TodoItem>) -> Result<String> {
        let mut seen = HashSet::new();
        let mut pending = 0;
        let mut in_progress = 0;
        let mut completed = 0;

        for item in &items {
            let desc = item.description.trim();
            if desc.is_empty() {
                bail!("todo item description must not be empty");
            }
            if !seen.insert(desc.to_string()) {
                bail!("duplicate todo item description: `{desc}`");
            }
            match item.status {
                TodoStatus::Pending => pending += 1,
                TodoStatus::InProgress => in_progress += 1,
                TodoStatus::Completed => completed += 1,
            }
        }

        *self.todos.lock().unwrap() = items;
        Ok(format!(
            "Updated todo list: {pending} pending, {in_progress} in progress, {completed} completed."
        ))
    }
}

pub struct TodoWrite {
    store: PlanStore,
}

impl TodoWrite {
    pub fn new(store: PlanStore) -> Self {
        Self { store }
    }
}

impl Tool for TodoWrite {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "todo_write".into(),
            description: "Update the agent's task checklist. Whole-list replacement: send the full list of tasks with their status ('pending', 'in_progress', 'completed'). Use this to structure plans and track progress across multi-step tasks.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": { "type": "string" },
                                "status": { "type": "string", "enum": ["pending", "in_progress", "completed"] }
                            },
                            "required": ["description", "status"]
                        }
                    }
                },
                "required": ["todos"]
            }),
        }
    }

    fn mutating(&self) -> bool {
        false // Plan state is internal, safe even in read-only policy
    }

    fn run(&self, a: &Value) -> Result<String> {
        let arr = a["todos"]
            .as_array()
            .context("missing `todos` array in todo_write arguments")?;
        let mut items = Vec::with_capacity(arr.len());
        for v in arr {
            let desc = v["description"]
                .as_str()
                .context("missing string `description`")?
                .to_string();
            let status = match v["status"].as_str().unwrap_or("pending") {
                "in_progress" => TodoStatus::InProgress,
                "completed" => TodoStatus::Completed,
                _ => TodoStatus::Pending,
            };
            items.push(TodoItem {
                description: desc,
                status,
            });
        }
        self.store.replace(items)
    }
}

/// Verification command runner.
#[derive(Clone, Debug)]
pub struct VerificationGate {
    pub command: Option<String>,
    pub root: PathBuf,
    pub timeout: Duration,
}

impl Default for VerificationGate {
    fn default() -> Self {
        Self {
            command: None,
            root: PathBuf::new(),
            timeout: Duration::from_secs(120),
        }
    }
}

impl VerificationGate {
    pub fn new(root: PathBuf, command: Option<String>) -> Self {
        Self {
            command,
            root,
            timeout: Duration::from_secs(120),
        }
    }

    pub fn with_timeout(root: PathBuf, command: Option<String>, timeout: Duration) -> Self {
        Self {
            command,
            root,
            timeout,
        }
    }

    /// Run the verification command. Returns `Ok(())` if command passed or if no verification is set.
    /// Returns `Err(details)` if the verification command failed.
    pub fn verify(&self) -> Result<(), String> {
        let Some(cmd) = &self.command else {
            return Ok(());
        };
        let mut command = Command::new("sh");
        command
            .arg("-c")
            .arg(cmd)
            .current_dir(&self.root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }

        let mut child = command
            .spawn()
            .map_err(|e| format!("failed to spawn verification command `{cmd}`: {e:#}"))?;

        let mut stdout_pipe = child.stdout.take();
        let mut stderr_pipe = child.stderr.take();

        let stdout_thread = std::thread::spawn(move || {
            if let Some(pipe) = stdout_pipe.take() {
                crate::tools::read_pipe_capped(pipe)
            } else {
                Vec::new()
            }
        });

        let stderr_thread = std::thread::spawn(move || {
            if let Some(pipe) = stderr_pipe.take() {
                crate::tools::read_pipe_capped(pipe)
            } else {
                Vec::new()
            }
        });

        let start = std::time::Instant::now();
        let timeout = self.timeout;
        let status = loop {
            match child.try_wait() {
                Ok(Some(st)) => break st,
                Ok(None) => {
                    if start.elapsed() > timeout {
                        crate::tools::kill_process_group(&mut child);
                        let _ = stdout_thread.join();
                        let _ = stderr_thread.join();
                        return Err(format!(
                            "verification command `{cmd}` timed out after {}s",
                            timeout.as_secs()
                        ));
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(e) => {
                    crate::tools::kill_process_group(&mut child);
                    let _ = stdout_thread.join();
                    let _ = stderr_thread.join();
                    return Err(format!("error waiting on verification command: {e:#}"));
                }
            }
        };

        let stdout_bytes = stdout_thread.join().unwrap_or_default();
        let stderr_bytes = stderr_thread.join().unwrap_or_default();

        if status.success() {
            Ok(())
        } else {
            let code = status.code().unwrap_or(-1);
            let stdout = String::from_utf8_lossy(&stdout_bytes);
            let stderr = String::from_utf8_lossy(&stderr_bytes);
            Err(format!(
                "exit code: {code}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            ))
        }
    }
}

/// Plugin mounting [`PlanStore`] and [`VerificationGate`].
pub struct PlanPlugin {
    pub root: PathBuf,
    pub verify_command: Option<String>,
}

impl Plugin for PlanPlugin {
    fn name(&self) -> &str {
        "plan-and-verify"
    }

    fn apply(&self, scope: &mut Scope) {
        let store = PlanStore::new();
        scope.provide(store.clone());
        scope.provide(VerificationGate::new(
            self.root.clone(),
            self.verify_command.clone(),
        ));

        if let Some(reg) = scope.context().service::<crate::sandbox::ToolRegistry>() {
            reg.register(Arc::new(TodoWrite::new(store)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn todo_store_replace_and_validation() {
        let store = PlanStore::new();
        let tool = TodoWrite::new(store.clone());

        let res = tool
            .run(&json!({
                "todos": [
                    { "description": "task 1", "status": "completed" },
                    { "description": "task 2", "status": "in_progress" },
                    { "description": "task 3", "status": "pending" }
                ]
            }))
            .unwrap();

        assert!(res.contains("1 pending, 1 in progress, 1 completed"));
        let todos = store.get();
        assert_eq!(todos.len(), 3);
        assert_eq!(todos[0].status, TodoStatus::Completed);

        // Reject empty description
        assert!(tool
            .run(&json!({ "todos": [{ "description": "", "status": "pending" }] }))
            .is_err());
        // Reject duplicate description
        assert!(tool
            .run(&json!({ "todos": [
            { "description": "dup", "status": "pending" },
            { "description": "dup", "status": "in_progress" }
        ] }))
            .is_err());
    }

    #[test]
    fn verification_gate_reports_success_and_failure() {
        let root = std::env::temp_dir();
        let ok_gate = VerificationGate::new(root.clone(), Some("true".into()));
        assert!(ok_gate.verify().is_ok());

        let fail_gate =
            VerificationGate::new(root.clone(), Some("echo 'err text' >&2; exit 42".into()));
        let err = fail_gate.verify().unwrap_err();
        assert!(err.contains("exit code: 42"));
        assert!(err.contains("err text"));

        let timeout_gate = VerificationGate::with_timeout(
            root,
            Some("sleep 10".into()),
            Duration::from_millis(200),
        );
        let timeout_err = timeout_gate.verify().unwrap_err();
        assert!(timeout_err.contains("timed out"));
    }
}
