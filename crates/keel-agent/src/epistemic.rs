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
use std::time::SystemTime;

/// Classification of epistemic facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactCategory {
    Invariant,
    Architecture,
    BugRootCause,
    ToolObservation,
    Environment,
}

impl FactCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Invariant => "invariant",
            Self::Architecture => "architecture",
            Self::BugRootCause => "bug_root_cause",
            Self::ToolObservation => "tool_observation",
            Self::Environment => "environment",
        }
    }

    pub fn from_str_lossy(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "invariant" => Self::Invariant,
            "architecture" | "arch" => Self::Architecture,
            "bug" | "root_cause" | "bug_root_cause" => Self::BugRootCause,
            "observation" | "tool" | "tool_observation" => Self::ToolObservation,
            _ => Self::Environment,
        }
    }
}

/// Type of semantic relationship between facts and hypotheses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationType {
    Supports,
    Contradicts,
    DependsOn,
    Falsifies,
    CausedBy,
}

impl RelationType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Supports => "supports",
            Self::Contradicts => "contradicts",
            Self::DependsOn => "depends_on",
            Self::Falsifies => "falsifies",
            Self::CausedBy => "caused_by",
        }
    }

    pub fn from_str_lossy(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "contradicts" | "conflicts" | "denies" => Self::Contradicts,
            "depends_on" | "depends" | "requires" => Self::DependsOn,
            "falsifies" | "disproves" => Self::Falsifies,
            "caused_by" | "causes" => Self::CausedBy,
            _ => Self::Supports,
        }
    }
}

/// A node in the epistemic knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FactNode {
    pub id: String,
    pub statement: String,
    pub category: FactCategory,
    pub confidence: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance: Option<String>,
    pub timestamp: u64,
}

/// A directed edge in the epistemic knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GraphEdge {
    pub from: String,
    pub to: String,
    pub relation: RelationType,
}

/// A detected contradiction between facts, hypotheses, and invariants.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContradictionAlert {
    pub source_id: String,
    pub target_id: String,
    pub reason: String,
}

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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<FactNode>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub edges: Vec<GraphEdge>,
}

impl EpistemicState {
    pub fn is_empty(&self) -> bool {
        self.hypothesis.is_none()
            && self.facts.is_empty()
            && self.failed_attempts.is_empty()
            && self.open_questions.is_empty()
            && self.nodes.is_empty()
            && self.edges.is_empty()
    }

    pub fn add_node(
        &mut self,
        statement: &str,
        category: FactCategory,
        confidence: u8,
        provenance: Option<&str>,
    ) -> String {
        let trimmed = statement.trim();
        if let Some(existing) = self.nodes.iter().find(|n| n.statement == trimmed) {
            return existing.id.clone();
        }

        let now_sec = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let id = format!("fact_{}", self.nodes.len() + 1);
        self.nodes.push(FactNode {
            id: id.clone(),
            statement: trimmed.to_string(),
            category,
            confidence: confidence.min(100),
            provenance: provenance.map(str::to_string),
            timestamp: now_sec,
        });

        // Keep flat facts list in sync
        if !self.facts.iter().any(|f| f == trimmed) {
            self.facts.push(trimmed.to_string());
        }

        id
    }

    pub fn add_edge(&mut self, from: &str, to: &str, relation: RelationType) -> bool {
        if from == to {
            return false;
        }
        let exists = self
            .edges
            .iter()
            .any(|e| e.from == from && e.to == to && e.relation == relation);
        if !exists {
            self.edges.push(GraphEdge {
                from: from.to_string(),
                to: to.to_string(),
                relation,
            });
            true
        } else {
            false
        }
    }

    pub fn find_node(&self, id_or_statement: &str) -> Option<&FactNode> {
        self.nodes
            .iter()
            .find(|n| n.id == id_or_statement || n.statement == id_or_statement)
    }

    pub fn detect_contradictions(&self) -> Vec<ContradictionAlert> {
        let mut alerts = Vec::new();
        for edge in &self.edges {
            if edge.relation == RelationType::Contradicts
                || edge.relation == RelationType::Falsifies
            {
                let from_stmt = self
                    .find_node(&edge.from)
                    .map(|n| n.statement.as_str())
                    .unwrap_or(&edge.from);
                let to_stmt = self
                    .find_node(&edge.to)
                    .map(|n| n.statement.as_str())
                    .unwrap_or(&edge.to);
                alerts.push(ContradictionAlert {
                    source_id: edge.from.clone(),
                    target_id: edge.to.clone(),
                    reason: format!("`{from_stmt}` {} `{to_stmt}`", edge.relation.as_str()),
                });
            }
        }
        alerts
    }

    pub fn render_mermaid_graph(&self) -> String {
        if self.nodes.is_empty() && self.edges.is_empty() {
            return "graph TD\n  Empty[\"Empty Epistemic Graph\"]\n".to_string();
        }

        let mut out = String::from("graph TD\n");
        for n in &self.nodes {
            let clean = n.statement.replace('"', "'");
            out.push_str(&format!(
                "  {}[\"[{} ({}%)] {}\"]\n",
                n.id,
                n.category.as_str(),
                n.confidence,
                clean
            ));
        }

        for e in &self.edges {
            out.push_str(&format!(
                "  {} -->|{}| {}\n",
                e.from,
                e.relation.as_str(),
                e.to
            ));
        }

        out
    }

    pub fn merge_from(&mut self, other: &EpistemicState) {
        if let Some(h) = &other.hypothesis {
            if self.hypothesis.is_none() {
                self.hypothesis = Some(h.clone());
            }
        }

        for f in &other.facts {
            if !self.facts.iter().any(|existing| existing == f) {
                self.facts.push(f.clone());
            }
        }

        for fa in &other.failed_attempts {
            if !self
                .failed_attempts
                .iter()
                .any(|existing| existing.approach == fa.approach)
            {
                self.failed_attempts.push(fa.clone());
            }
        }

        for q in &other.open_questions {
            if !self.open_questions.iter().any(|existing| existing == q) {
                self.open_questions.push(q.clone());
            }
        }

        for n in &other.nodes {
            if !self
                .nodes
                .iter()
                .any(|existing| existing.statement == n.statement)
            {
                let mut cloned = n.clone();
                cloned.id = format!("fact_{}", self.nodes.len() + 1);
                self.nodes.push(cloned);
            }
        }

        for e in &other.edges {
            if !self.edges.iter().any(|existing| {
                existing.from == e.from && existing.to == e.to && existing.relation == e.relation
            }) {
                self.edges.push(e.clone());
            }
        }
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

    pub fn add_fact_node(
        &self,
        statement: &str,
        category: FactCategory,
        confidence: u8,
        provenance: Option<&str>,
    ) -> Result<String> {
        let mut id = String::new();
        self.update(|s| {
            id = s.add_node(statement, category, confidence, provenance);
        })?;
        Ok(id)
    }

    pub fn add_edge(&self, from: &str, to: &str, relation: RelationType) -> Result<bool> {
        let mut added = false;
        self.update(|s| {
            added = s.add_edge(from, to, relation);
        })?;
        Ok(added)
    }

    pub fn detect_contradictions(&self) -> Vec<ContradictionAlert> {
        self.state.lock().unwrap().detect_contradictions()
    }

    pub fn render_mermaid(&self) -> String {
        self.state.lock().unwrap().render_mermaid_graph()
    }

    pub fn merge_state(&self, other: &EpistemicState) -> Result<()> {
        self.update(|s| {
            s.merge_from(other);
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

/// Tool allowing the agent to establish causal, dependency, or contradiction links between facts.
pub struct EpistemicGraphLinkTool {
    pub store: EpistemicStore,
}

impl Tool for EpistemicGraphLinkTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "link_facts".into(),
            description: "Link two facts or findings in the epistemic graph with a relationship (supports, contradicts, depends_on, falsifies, caused_by).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "from": { "type": "string", "description": "Source fact statement or ID" },
                    "to": { "type": "string", "description": "Target fact statement or ID" },
                    "relation": {
                        "type": "string",
                        "enum": ["supports", "contradicts", "depends_on", "falsifies", "caused_by"],
                        "description": "Nature of the relationship"
                    }
                },
                "required": ["from", "to", "relation"]
            }),
        }
    }

    fn mutating(&self) -> bool {
        false
    }

    fn run(&self, a: &Value) -> Result<String> {
        let from = a["from"].as_str().context("missing `from`")?;
        let to = a["to"].as_str().context("missing `to`")?;
        let rel_str = a["relation"].as_str().context("missing `relation`")?;
        let rel = RelationType::from_str_lossy(rel_str);

        let from_id = self
            .store
            .add_fact_node(from, FactCategory::ToolObservation, 80, None)?;
        let to_id = self
            .store
            .add_fact_node(to, FactCategory::ToolObservation, 80, None)?;

        self.store.add_edge(&from_id, &to_id, rel)?;

        let contradictions = self.store.detect_contradictions();
        if !contradictions.is_empty() {
            let mut warning = format!("Linked `{from_id}` -> `{to_id}` ({rel_str}). WARNING: {} contradiction(s) detected:\n", contradictions.len());
            for c in contradictions {
                warning.push_str(&format!("  ⚠️ {}\n", c.reason));
            }
            Ok(warning)
        } else {
            Ok(format!(
                "Linked `{from_id}` -> `{to_id}` with relationship `{rel_str}`."
            ))
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

        // Register remember and link tools in ToolRegistry if present
        let reg = scope
            .context()
            .service::<crate::sandbox::ToolRegistry>()
            .map(|r| (*r).clone())
            .unwrap_or_default();
        reg.register(Arc::new(EpistemicRememberTool {
            store: store.clone(),
        }));
        reg.register(Arc::new(EpistemicGraphLinkTool { store }));
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

    #[test]
    fn epistemic_graph_linking_and_contradiction_detection() {
        let store = EpistemicStore::new(None);

        let id1 = store
            .add_fact_node(
                "Port 8080 is available",
                FactCategory::Environment,
                90,
                None,
            )
            .unwrap();
        let id2 = store
            .add_fact_node(
                "Port 8080 is occupied by nginx",
                FactCategory::Environment,
                95,
                None,
            )
            .unwrap();

        store
            .add_edge(&id1, &id2, RelationType::Contradicts)
            .unwrap();

        let alerts = store.detect_contradictions();
        assert_eq!(alerts.len(), 1);
        assert!(alerts[0].reason.contains("contradicts"));

        let mermaid = store.render_mermaid();
        assert!(mermaid.contains("Port 8080"));
        assert!(mermaid.contains("contradicts"));
    }

    #[test]
    fn epistemic_state_merge_and_tool_link() {
        let store = EpistemicStore::new(None);
        let link_tool = EpistemicGraphLinkTool {
            store: store.clone(),
        };

        let res = link_tool
            .run(&json!({
                "from": "VFS transaction in memory",
                "to": "Disk is untouched until commit",
                "relation": "supports"
            }))
            .unwrap();

        assert!(res.contains("Linked"));
        let state = store.get();
        assert_eq!(state.nodes.len(), 2);
        assert_eq!(state.edges.len(), 1);

        let mut other = EpistemicState::default();
        other.add_node("Shadow VFS rollback", FactCategory::Invariant, 100, None);
        store.merge_state(&other).unwrap();

        let merged = store.get();
        assert_eq!(merged.nodes.len(), 3);
    }
}
