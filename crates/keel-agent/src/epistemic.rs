//! Ephemeral & persistent epistemic working memory (belief state tracking).
//!
//! Tracks:
//! - **Verified Facts & Invariants** (what the agent has learned to be true about the workspace)
//! - **Active Hypothesis** (what the agent currently believes is the root cause or solution path)
//! - **Negative Knowledge / Failed Attempts** (disproven hypotheses and failed edits to NOT repeat)
//! - **Open Questions** (unresolved uncertainties to investigate)
//!
//! Especially vital for edge and quantized models (Bonsai 2B-27B, Qwen 27B) to prevent
//! context amnesia and repetitive trial-and-error loops across multi-turn sessions.

use crate::{Tool, ToolSpec};
use anyhow::{Context as _, Result};
use keel_kernel::{Plugin, Scope};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FailedAttempt {
    pub approach: String,
    pub failure_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct EpistemicState {
    #[serde(default)]
    pub hypothesis: Option<String>,
    #[serde(default)]
    pub facts: Vec<String>,
    #[serde(default)]
    pub failed_attempts: Vec<FailedAttempt>,
    #[serde(default)]
    pub open_questions: Vec<String>,
}

impl EpistemicState {
    pub fn is_empty(&self) -> bool {
        self.hypothesis.is_none()
            && self.facts.is_empty()
            && self.failed_attempts.is_empty()
            && self.open_questions.is_empty()
    }

    pub fn render_markdown(&self) -> String {
        if self.is_empty() {
            return String::new();
        }
        let mut s = String::from("## Epistemic Working Memory (Active Belief State)\n");
        if let Some(hyp) = &self.hypothesis {
            s.push_str(&format!("- **Current Hypothesis**: {hyp}\n"));
        }
        if !self.facts.is_empty() {
            s.push_str("- **Verified Facts & Invariants**:\n");
            for f in &self.facts {
                s.push_str(&format!("  * {f}\n"));
            }
        }
        if !self.failed_attempts.is_empty() {
            s.push_str("- **Disproven / Failed Attempts (DO NOT REPEAT)**:\n");
            for a in &self.failed_attempts {
                s.push_str(&format!(
                    "  * Attempt: {} -> Failed: {}\n",
                    a.approach, a.failure_reason
                ));
            }
        }
        if !self.open_questions.is_empty() {
            s.push_str("- **Open Questions**:\n");
            for q in &self.open_questions {
                s.push_str(&format!("  * {q}\n"));
            }
        }
        s
    }
}

#[derive(Clone, Default)]
pub struct EpistemicStore {
    state: Arc<Mutex<EpistemicState>>,
    persist_path: Option<PathBuf>,
}

impl EpistemicStore {
    pub fn new(persist_path: Option<PathBuf>) -> Self {
        let store = Self {
            state: Arc::new(Mutex::new(EpistemicState::default())),
            persist_path,
        };
        let _ = store.load_from_file();
        store
    }

    pub fn get(&self) -> EpistemicState {
        self.state.lock().unwrap().clone()
    }

    pub fn load_from_file(&self) -> Result<()> {
        let Some(p) = &self.persist_path else {
            return Ok(());
        };
        if !p.exists() {
            return Ok(());
        }
        let data = std::fs::read_to_string(p)
            .with_context(|| format!("reading epistemic memory from {}", p.display()))?;
        let state: EpistemicState = serde_json::from_str(&data)?;
        *self.state.lock().unwrap() = state;
        Ok(())
    }

    pub fn save_to_file(&self) -> Result<()> {
        let Some(p) = &self.persist_path else {
            return Ok(());
        };
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let state = self.state.lock().unwrap();
        let json = serde_json::to_string_pretty(&*state)?;
        std::fs::write(p, json)
            .with_context(|| format!("writing epistemic memory to {}", p.display()))?;
        Ok(())
    }

    pub fn update(&self, f: impl FnOnce(&mut EpistemicState)) -> Result<EpistemicState> {
        let mut state = self.state.lock().unwrap();
        f(&mut state);
        let cloned = state.clone();
        drop(state);
        let _ = self.save_to_file();
        Ok(cloned)
    }

    pub fn add_fact(&self, fact: &str) -> Result<()> {
        let trimmed = fact.trim();
        if trimmed.is_empty() {
            return Ok(());
        }
        self.update(|s| {
            if !s.facts.iter().any(|f| f == trimmed) {
                s.facts.push(trimmed.to_string());
            }
        })?;
        Ok(())
    }

    pub fn set_hypothesis(&self, hyp: Option<String>) -> Result<()> {
        self.update(|s| {
            s.hypothesis = hyp.map(|h| h.trim().to_string()).filter(|h| !h.is_empty());
        })?;
        Ok(())
    }

    pub fn record_failure(&self, approach: &str, reason: &str) -> Result<()> {
        let app = approach.trim();
        let rsn = reason.trim();
        if app.is_empty() {
            return Ok(());
        }
        self.update(|s| {
            if !s.failed_attempts.iter().any(|a| a.approach == app) {
                s.failed_attempts.push(FailedAttempt {
                    approach: app.to_string(),
                    failure_reason: rsn.to_string(),
                });
            }
        })?;
        Ok(())
    }

    pub fn add_question(&self, question: &str) -> Result<()> {
        let q = question.trim();
        if q.is_empty() {
            return Ok(());
        }
        self.update(|s| {
            if !s.open_questions.iter().any(|existing| existing == q) {
                s.open_questions.push(q.to_string());
            }
        })?;
        Ok(())
    }

    pub fn resolve_question(&self, question: &str) -> Result<()> {
        let q = question.trim();
        self.update(|s| {
            s.open_questions.retain(|item| item != q);
        })?;
        Ok(())
    }

    pub fn clear(&self) -> Result<()> {
        self.update(|s| {
            *s = EpistemicState::default();
        })?;
        Ok(())
    }
}

/// Tool allowing the agent to explicitly update its epistemic working memory.
pub struct EpistemicRememberTool {
    pub store: EpistemicStore,
}

impl Tool for EpistemicRememberTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "remember".into(),
            description: "Record discovered facts, update active hypothesis, log disproven/failed attempts (to never repeat them), or add open questions.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "A verified invariant or discovered fact about the codebase"
                    },
                    "hypothesis": {
                        "type": "string",
                        "description": "Current working hypothesis being investigated"
                    },
                    "failed_attempt": {
                        "type": "object",
                        "properties": {
                            "approach": { "type": "string" },
                            "reason": { "type": "string" }
                        },
                        "required": ["approach", "reason"],
                        "description": "Record a failed approach and why it failed so it is not repeated"
                    },
                    "open_question": {
                        "type": "string",
                        "description": "An open question or unknown to investigate"
                    },
                    "resolve_question": {
                        "type": "string",
                        "description": "Question that has now been answered and can be removed"
                    }
                }
            }),
        }
    }

    fn mutating(&self) -> bool {
        false
    }

    fn run(&self, a: &Value) -> Result<String> {
        let mut updates = Vec::new();

        if let Some(f) = a.get("fact").and_then(|v| v.as_str()) {
            self.store.add_fact(f)?;
            updates.push(format!("recorded fact: `{f}`"));
        }
        if let Some(h) = a.get("hypothesis").and_then(|v| v.as_str()) {
            self.store.set_hypothesis(Some(h.to_string()))?;
            updates.push(format!("updated hypothesis: `{h}`"));
        }
        if let Some(fa) = a.get("failed_attempt") {
            if let (Some(app), Some(rsn)) = (
                fa.get("approach").and_then(|v| v.as_str()),
                fa.get("reason").and_then(|v| v.as_str()),
            ) {
                self.store.record_failure(app, rsn)?;
                updates.push(format!(
                    "logged disproven attempt: `{app}` (reason: `{rsn}`)"
                ));
            }
        }
        if let Some(q) = a.get("open_question").and_then(|v| v.as_str()) {
            self.store.add_question(q)?;
            updates.push(format!("added open question: `{q}`"));
        }
        if let Some(rq) = a.get("resolve_question").and_then(|v| v.as_str()) {
            self.store.resolve_question(rq)?;
            updates.push(format!("resolved question: `{rq}`"));
        }

        if updates.is_empty() {
            let state = self.store.get();
            let rendered = state.render_markdown();
            if rendered.is_empty() {
                Ok("Epistemic working memory is currently empty.".into())
            } else {
                Ok(rendered)
            }
        } else {
            Ok(format!("Epistemic memory updated: {}.", updates.join("; ")))
        }
    }
}

/// Plugin mounting [`EpistemicStore`] and the `remember` tool.
pub struct EpistemicMemoryPlugin {
    pub root: PathBuf,
}

impl EpistemicMemoryPlugin {
    pub fn persist_file(root: &Path) -> PathBuf {
        let dot_keel = root.join(".keel");
        dot_keel.join("epistemic.json")
    }
}

impl Plugin for EpistemicMemoryPlugin {
    fn name(&self) -> &str {
        "epistemic-memory"
    }

    fn apply(&self, scope: &mut Scope) {
        let persist = Self::persist_file(&self.root);
        let store = EpistemicStore::new(Some(persist));

        scope.provide(store.clone());

        // Register remember tool in ToolRegistry if present
        let reg = scope
            .context()
            .service::<crate::sandbox::ToolRegistry>()
            .map(|r| (*r).clone())
            .unwrap_or_default();
        reg.register(Arc::new(EpistemicRememberTool { store }));
        scope.provide(reg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epistemic_store_crud_and_markdown_rendering() {
        let tmp =
            std::env::temp_dir().join(format!("keel-epistemic-test-{}.json", std::process::id()));
        let _ = std::fs::remove_file(&tmp);

        let store = EpistemicStore::new(Some(tmp.clone()));
        assert!(store.get().is_empty());

        store.add_fact("sqlite DB is in /tmp/app.db").unwrap();
        store
            .set_hypothesis(Some("syntax error in lexer".into()))
            .unwrap();
        store
            .record_failure("regex match on float", "fails on scientific notation")
            .unwrap();
        store
            .add_question("what version of Rust is required?")
            .unwrap();

        let state = store.get();
        assert_eq!(state.facts.len(), 1);
        assert_eq!(state.hypothesis.as_deref(), Some("syntax error in lexer"));
        assert_eq!(state.failed_attempts.len(), 1);
        assert_eq!(state.open_questions.len(), 1);

        let md = state.render_markdown();
        assert!(md.contains("## Epistemic Working Memory"));
        assert!(md.contains("syntax error in lexer"));
        assert!(md.contains("sqlite DB is in /tmp/app.db"));
        assert!(md.contains("regex match on float"));
        assert!(md.contains("what version of Rust is required?"));

        // Test question resolution
        store
            .resolve_question("what version of Rust is required?")
            .unwrap();
        assert_eq!(store.get().open_questions.len(), 0);

        // Test persistence reload
        let store2 = EpistemicStore::new(Some(tmp.clone()));
        let state2 = store2.get();
        assert_eq!(state2.facts.len(), 1);
        assert_eq!(state2.failed_attempts.len(), 1);

        let _ = std::fs::remove_file(tmp);
    }

    #[test]
    fn epistemic_remember_tool_works() {
        let store = EpistemicStore::new(None);
        let tool = EpistemicRememberTool {
            store: store.clone(),
        };

        let res = tool
            .run(&json!({
                "fact": "tests must run with cargo test --offline",
                "hypothesis": "missing import in lib.rs",
                "failed_attempt": {
                    "approach": "adding extern crate foo",
                    "reason": "cargo 2021 edition does not use extern crate"
                }
            }))
            .unwrap();

        assert!(res.contains("Epistemic memory updated"));
        let state = store.get();
        assert_eq!(state.facts.len(), 1);
        assert_eq!(state.failed_attempts.len(), 1);
    }
}
