//! Sovereign Epistemic Handoff & Swarm Mesh Protocol (`keel handoff`).
//!
//! Provides cross-harness context distillation, ingestion, and bidirectional state synchronization
//! between Keel, Helm Swarm, and foreign coding agents (Claude Code, OpenCode, Codex, Aider, Pi).
//!
//! Converts external context envelopes into sovereign working memory:
//! - Goal -> Active epistemic hypothesis
//! - Tasks -> Structured `PlanStore` (`TodoItem` with pending/completed states)
//! - Relevant files & summaries -> Epistemic facts & compiler-verified file references
//! - Bidirectional ACP JSON-RPC (`handoff/ingest`, `handoff/export`) and CLI (`keel handoff`)

use crate::epistemic::EpistemicStore;
use crate::plan::{PlanStore, TodoItem, TodoStatus};
use crate::vfs::TransactionalVfs;
use crate::Message;
use anyhow::{bail, Context as _, Result};
use keel_kernel::Context;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;

pub const MAX_HANDOFF_JSON_BYTES: usize = 4 * 1024 * 1024; // 4MB maximum handoff envelope JSON
pub const MAX_HANDOFF_STRING_CHARS: usize = 32_768; // 32KB max per string field
pub const MAX_HANDOFF_ITEMS: usize = 200; // max 200 items per array

/// Cross-harness distilled handoff envelope, 100% interoperable with Helm Swarm.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HandoffEnvelope {
    pub source_conversation_id: String,
    pub source_harness: String,
    pub target_harness: String,
    pub goal: String,
    pub relevant_files: Vec<String>,
    pub open_tasks: Vec<String>,
    pub completed_tasks: Vec<String>,
    pub context_summary: String,
}

impl HandoffEnvelope {
    pub fn from_json_str(s: &str) -> Result<Self> {
        let trimmed = s.trim();
        if trimmed.len() > MAX_HANDOFF_JSON_BYTES {
            bail!(
                "HandoffEnvelope JSON ({} bytes) exceeds maximum allowable size of 4MB",
                trimmed.len()
            );
        }
        let env: Self = serde_json::from_str(trimmed)
            .with_context(|| "Failed to parse HandoffEnvelope JSON")?;
        env.validate()?;
        Ok(env)
    }

    pub fn validate(&self) -> Result<()> {
        if self.goal.chars().count() > MAX_HANDOFF_STRING_CHARS {
            bail!("HandoffEnvelope `goal` exceeds maximum allowable length (32,768 characters)");
        }
        if self.context_summary.chars().count() > MAX_HANDOFF_STRING_CHARS {
            bail!("HandoffEnvelope `context_summary` exceeds maximum allowable length (32,768 characters)");
        }
        if self.open_tasks.len() > MAX_HANDOFF_ITEMS {
            bail!(
                "HandoffEnvelope `open_tasks` contains {} items, exceeding maximum limit of {}",
                self.open_tasks.len(),
                MAX_HANDOFF_ITEMS
            );
        }
        if self.completed_tasks.len() > MAX_HANDOFF_ITEMS {
            bail!(
                "HandoffEnvelope `completed_tasks` contains {} items, exceeding maximum limit of {}",
                self.completed_tasks.len(),
                MAX_HANDOFF_ITEMS
            );
        }
        if self.relevant_files.len() > MAX_HANDOFF_ITEMS {
            bail!(
                "HandoffEnvelope `relevant_files` contains {} items, exceeding maximum limit of {}",
                self.relevant_files.len(),
                MAX_HANDOFF_ITEMS
            );
        }
        for f in &self.relevant_files {
            if f.contains('\0') {
                bail!("HandoffEnvelope `relevant_files` contains null byte");
            }
        }
        Ok(())
    }

    pub fn to_json_pretty(&self) -> Result<String> {
        serde_json::to_string_pretty(self).with_context(|| "Failed to serialize HandoffEnvelope")
    }

    /// Synthesizes a high-signal sovereign starting prompt from the distilled envelope.
    pub fn synth_prompt(&self) -> String {
        let mut out = format!(
            "Resuming goal from {}: {}\n\n",
            self.source_harness, self.goal
        );
        if !self.open_tasks.is_empty() {
            out.push_str("Tasks to execute:\n");
            for t in self.open_tasks.iter().take(50) {
                out.push_str(&format!("- [ ] {t}\n"));
            }
            if self.open_tasks.len() > 50 {
                out.push_str(&format!(
                    "- ... ({} more open tasks)\n",
                    self.open_tasks.len() - 50
                ));
            }
            out.push('\n');
        }
        if !self.completed_tasks.is_empty() {
            out.push_str("Already completed:\n");
            for t in self.completed_tasks.iter().take(30) {
                out.push_str(&format!("- [x] {t}\n"));
            }
            if self.completed_tasks.len() > 30 {
                out.push_str(&format!(
                    "- ... ({} more completed tasks)\n",
                    self.completed_tasks.len() - 30
                ));
            }
            out.push('\n');
        }
        if !self.relevant_files.is_empty() {
            out.push_str("Key files:\n");
            for f in self.relevant_files.iter().take(50) {
                out.push_str(&format!("- `{f}`\n"));
            }
            if self.relevant_files.len() > 50 {
                out.push_str(&format!(
                    "- ... ({} more key files)\n",
                    self.relevant_files.len() - 50
                ));
            }
            out.push('\n');
        }
        if !self.context_summary.is_empty() {
            out.push_str("Context summary:\n");
            out.push_str(&self.context_summary);
            out.push('\n');
        }
        out.push_str("\nReview key files and proceed autonomously with the remaining open tasks.");
        out
    }
}

/// Report emitted when a handoff envelope is ingested into a Keel session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffIngestReport {
    pub source_conversation_id: String,
    pub source_harness: String,
    pub goal: String,
    pub facts_injected: usize,
    pub tasks_injected: usize,
    pub relevant_files: Vec<String>,
    pub initial_prompt: String,
}

/// Ingest an external handoff envelope into Keel's epistemic working memory, todo plan, and conversation history.
pub fn ingest_handoff(
    envelope: &HandoffEnvelope,
    ctx: &Context,
    history: &mut Vec<Message>,
) -> Result<HandoffIngestReport> {
    let mut facts_injected = 0;
    let mut tasks_injected = 0;

    // 1. Epistemic memory ingestion
    if let Some(store) = ctx.service::<EpistemicStore>() {
        let _ = store.set_hypothesis(Some(envelope.goal.clone()));
        let _ = store.add_fact(&format!(
            "Handoff source harness: {} (session: {})",
            envelope.source_harness, envelope.source_conversation_id
        ));
        facts_injected += 1;

        for f in &envelope.relevant_files {
            let _ = store.add_fact(&format!("Relevant file: {f}"));
            facts_injected += 1;
        }

        if !envelope.context_summary.is_empty() {
            let _ = store.add_fact(&format!(
                "Previous context summary: {}",
                envelope.context_summary
            ));
            facts_injected += 1;
        }
        let _ = store.save_to_file();
    }

    // 2. Plan / Todo list ingestion
    if let Some(plan) = ctx.service::<PlanStore>() {
        let mut items = Vec::new();
        for t in &envelope.completed_tasks {
            let desc = t.trim();
            if !desc.is_empty() {
                items.push(TodoItem {
                    description: desc.to_string(),
                    status: TodoStatus::Completed,
                });
                tasks_injected += 1;
            }
        }
        for t in &envelope.open_tasks {
            let desc = t.trim();
            if !desc.is_empty() {
                items.push(TodoItem {
                    description: desc.to_string(),
                    status: TodoStatus::Pending,
                });
                tasks_injected += 1;
            }
        }
        if !items.is_empty() {
            let _ = plan.replace(items);
        }
    }

    // 3. Inject synthesized prompt into conversation history
    let prompt = envelope.synth_prompt();
    history.push(Message::User {
        text: prompt.clone(),
    });

    Ok(HandoffIngestReport {
        source_conversation_id: envelope.source_conversation_id.clone(),
        source_harness: envelope.source_harness.clone(),
        goal: envelope.goal.clone(),
        facts_injected,
        tasks_injected,
        relevant_files: envelope.relevant_files.clone(),
        initial_prompt: prompt,
    })
}

/// Distill and export Keel's current epistemic state, plan store, VFS diffs, and conversation
/// history into a standard HandoffEnvelope.
pub fn export_handoff(
    conv_id: &str,
    target_harness: &str,
    ctx: &Context,
    history: &[Message],
) -> Result<HandoffEnvelope> {
    // 1. Goal extraction from active hypothesis or first user message
    let mut goal = String::new();
    if let Some(store) = ctx.service::<EpistemicStore>() {
        let state = store.get();
        if let Some(ref hyp) = state.hypothesis {
            goal = hyp.clone();
        }
    }
    if goal.is_empty() {
        if let Some(first_user) = history.iter().find_map(|m| match m {
            Message::User { text } => Some(text.clone()),
            _ => None,
        }) {
            goal = first_user.lines().next().unwrap_or("").trim().to_string();
        }
    }
    if goal.is_empty() {
        goal = "Keel agent session".to_string();
    }

    // 2. Tasks extraction from PlanStore
    let mut open_tasks = Vec::new();
    let mut completed_tasks = Vec::new();
    if let Some(plan) = ctx.service::<PlanStore>() {
        for item in plan.get() {
            match item.status {
                TodoStatus::Completed => completed_tasks.push(item.description),
                TodoStatus::Pending | TodoStatus::InProgress => open_tasks.push(item.description),
            }
        }
    }

    // 3. Relevant files extraction from epistemic facts, VFS status, and tool calls
    let mut files_set = HashSet::new();
    if let Some(store) = ctx.service::<EpistemicStore>() {
        for f in &store.get().facts {
            if let Some(path) = f.strip_prefix("Relevant file: ") {
                files_set.insert(path.trim().to_string());
            }
        }
    }
    if let Some(vfs) = ctx.service::<Arc<TransactionalVfs>>() {
        for status in vfs.status() {
            files_set.insert(status.path);
        }
    }
    for m in history {
        match m {
            Message::Assistant { tool_calls, .. } => {
                for c in tool_calls {
                    if let Some(p) = c.args.get("path").and_then(|v| v.as_str()) {
                        files_set.insert(p.to_string());
                    }
                }
            }
            Message::User { text } => {
                extract_file_mentions(text, &mut files_set);
            }
            _ => {}
        }
    }
    let mut relevant_files: Vec<String> = files_set.into_iter().collect();
    relevant_files.sort();

    // 4. Context summary from epistemic memory and recent assistant messages
    let mut summary_lines = Vec::new();
    if let Some(store) = ctx.service::<EpistemicStore>() {
        let md = store.get().render_markdown();
        if !md.trim().is_empty() {
            summary_lines.push(md.trim().to_string());
        }
    }
    for m in history.iter().rev().take(3).rev() {
        if let Message::Assistant { text, .. } = m {
            if !text.is_empty() {
                let char_count = text.chars().count();
                let snippet = if char_count > 300 {
                    let truncated: String = text.chars().take(300).collect();
                    format!("{truncated}…")
                } else {
                    text.clone()
                };
                summary_lines.push(snippet.trim().to_string());
            }
        }
    }
    let context_summary = summary_lines.join("\n---\n");

    Ok(HandoffEnvelope {
        source_conversation_id: conv_id.to_string(),
        source_harness: "keel".to_string(),
        target_harness: target_harness.to_string(),
        goal,
        relevant_files,
        open_tasks,
        completed_tasks,
        context_summary,
    })
}

fn extract_file_mentions(text: &str, files: &mut HashSet<String>) {
    for token in text.split_whitespace() {
        let cleaned = token
            .trim_matches(['`', '"', '\'', '(', ')', '[', ']', '<', '>'])
            .trim_end_matches(['.', ',', ':', ';']);
        if (cleaned.contains('/')
            || cleaned.ends_with(".rs")
            || cleaned.ends_with(".toml")
            || cleaned.ends_with(".json")
            || cleaned.ends_with(".py")
            || cleaned.ends_with(".js")
            || cleaned.ends_with(".ts")
            || cleaned.ends_with(".md"))
            && !cleaned.contains("://")
            && cleaned.len() > 2
        {
            files.insert(cleaned.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epistemic::EpistemicMemoryPlugin;
    use crate::plan::PlanPlugin;

    #[test]
    fn handoff_envelope_roundtrip_and_synth() {
        let json_src = r#"{
            "source_conversation_id": "conv-1234",
            "source_harness": "helm",
            "target_harness": "keel",
            "goal": "Refactor auth middleware to use JWT",
            "relevant_files": ["src/auth.rs", "Cargo.toml"],
            "open_tasks": ["Implement token verification", "Add integration tests"],
            "completed_tasks": ["Add jsonwebtoken crate"],
            "context_summary": "Initial auth struct drafted in auth.rs"
        }"#;

        let env = HandoffEnvelope::from_json_str(json_src).unwrap();
        assert_eq!(env.source_harness, "helm");
        assert_eq!(env.goal, "Refactor auth middleware to use JWT");
        assert_eq!(env.open_tasks.len(), 2);
        assert_eq!(env.completed_tasks.len(), 1);

        let prompt = env.synth_prompt();
        assert!(prompt.contains("Resuming goal from helm: Refactor auth middleware to use JWT"));
        assert!(prompt.contains("- [ ] Implement token verification"));
        assert!(prompt.contains("- [x] Add jsonwebtoken crate"));
        assert!(prompt.contains("- `src/auth.rs`"));
    }

    #[test]
    fn handoff_ingest_and_export_roundtrip() {
        let tmp = std::env::temp_dir().join(format!("keel-swarm-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let ctx = Context::new();
        let _p_epistemic = ctx.mount(&EpistemicMemoryPlugin { root: tmp.clone() });
        let _p_plan = ctx.mount(&PlanPlugin {
            root: tmp.clone(),
            verify_command: None,
        });

        let envelope = HandoffEnvelope {
            source_conversation_id: "helm-swarm-99".into(),
            source_harness: "helm".into(),
            target_harness: "keel".into(),
            goal: "Fix database pool exhaustion".into(),
            relevant_files: vec!["src/db.rs".into(), "tests/db_test.rs".into()],
            open_tasks: vec!["Increase connection pool max capacity".into()],
            completed_tasks: vec!["Reproduce deadlock in test".into()],
            context_summary: "Pool max size 5 was exceeded under load".into(),
        };

        let mut history = vec![Message::System {
            text: "System prompt".into(),
        }];
        let report = ingest_handoff(&envelope, &ctx, &mut history).unwrap();

        assert_eq!(report.goal, "Fix database pool exhaustion");
        assert_eq!(report.tasks_injected, 2);
        assert!(report.facts_injected >= 3);
        assert_eq!(history.len(), 2); // System + Ingested user prompt

        // Verify EpistemicStore was populated
        let store = ctx.service::<EpistemicStore>().unwrap();
        let state = store.get();
        assert_eq!(
            state.hypothesis,
            Some("Fix database pool exhaustion".to_string())
        );
        assert!(state.facts.iter().any(|f| f.contains("src/db.rs")));

        // Verify PlanStore was populated
        let plan = ctx.service::<PlanStore>().unwrap();
        let todos = plan.get();
        assert_eq!(todos.len(), 2);
        assert_eq!(todos[0].status, TodoStatus::Completed);
        assert_eq!(todos[1].status, TodoStatus::Pending);

        // Now test export
        let exported = export_handoff("keel-turn-1", "claude", &ctx, &history).unwrap();
        assert_eq!(exported.source_harness, "keel");
        assert_eq!(exported.target_harness, "claude");
        assert_eq!(exported.goal, "Fix database pool exhaustion");
        assert!(exported.relevant_files.contains(&"src/db.rs".to_string()));
        assert_eq!(
            exported.open_tasks,
            vec!["Increase connection pool max capacity"]
        );
        assert_eq!(exported.completed_tasks, vec!["Reproduce deadlock in test"]);

        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn handoff_envelope_validation_bounds() {
        // Test null byte rejection
        let env_null = HandoffEnvelope {
            source_conversation_id: "s1".into(),
            source_harness: "helm".into(),
            target_harness: "keel".into(),
            goal: "Task".into(),
            relevant_files: vec!["foo\0bar.rs".into()],
            open_tasks: vec![],
            completed_tasks: vec![],
            context_summary: "".into(),
        };
        assert!(env_null.validate().is_err());

        // Test excessive task count rejection
        let env_tasks = HandoffEnvelope {
            source_conversation_id: "s1".into(),
            source_harness: "helm".into(),
            target_harness: "keel".into(),
            goal: "Task".into(),
            relevant_files: vec![],
            open_tasks: (0..201).map(|i| format!("task {i}")).collect(),
            completed_tasks: vec![],
            context_summary: "".into(),
        };
        assert!(env_tasks.validate().is_err());

        // Test oversized string rejection
        let env_goal = HandoffEnvelope {
            source_conversation_id: "s1".into(),
            source_harness: "helm".into(),
            target_harness: "keel".into(),
            goal: "x".repeat(32_769),
            relevant_files: vec![],
            open_tasks: vec![],
            completed_tasks: vec![],
            context_summary: "".into(),
        };
        assert!(env_goal.validate().is_err());
    }

    #[test]
    fn handoff_export_utf8_char_boundary_safety() {
        let tmp = std::env::temp_dir().join(format!("keel-swarm-utf8-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let ctx = Context::new();
        // Construct assistant message where byte 300 lands inside a multi-byte UTF-8 char
        let mut msg_text = "a".repeat(299);
        msg_text.push_str("🦀 Finnish text: äöå!");
        let history = vec![Message::Assistant {
            text: msg_text,
            tool_calls: vec![],
        }];

        let exported = export_handoff("conv-utf8", "helm", &ctx, &history).unwrap();
        assert!(exported.context_summary.contains('🦀'));

        let _ = std::fs::remove_dir_all(tmp);
    }
}
