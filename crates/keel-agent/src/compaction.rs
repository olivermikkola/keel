//! Context compaction & workspace instructions (`keel-agent::compaction`).
//!
//! 1. [`ToolResultPrunerPlugin`]:
//!    When single tool output exceeds `threshold_chars`, trims older oversized tool outputs to
//!    `head_chars` + marker + `tail_chars` without making an LLM call.
//! 2. [`WorkspaceContextPlugin`]: Auto-discovers `AGENTS.md` / `CLAUDE.md` and `SKILL.md`
//!    summaries in the workspace and injects them into the system prompt.
//! 3. [`AdaptiveCompactor`] & [`AdaptiveCompactorPlugin`]:
//!    Monitors conversation history token pressure against a [`ContextBudget`] and performs
//!    middle-out history virtualization, preserving root intent (head) and recent interaction turns
//!    (tail) while synthesizing intermediate tool calls, files, and outputs into a dense summary.

use crate::Message;
use anyhow::Result;
use keel_kernel::{Plugin, Scope};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// 1. Tool Result Pruner
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ToolResultPruner {
    pub threshold_chars: usize,
    pub head_chars: usize,
    pub tail_chars: usize,
}

impl Default for ToolResultPruner {
    fn default() -> Self {
        Self {
            threshold_chars: 8192,
            head_chars: 4096,
            tail_chars: 1024,
        }
    }
}

impl ToolResultPruner {
    /// Trim a single string if its Unicode character count exceeds `threshold_chars`.
    pub fn prune_str(&self, s: &str) -> String {
        let total = s.chars().count();
        if total <= self.threshold_chars {
            return s.to_string();
        }
        let head: String = s.chars().take(self.head_chars).collect();
        let tail: String = s
            .chars()
            .skip(total.saturating_sub(self.tail_chars))
            .collect();
        let omitted = total.saturating_sub(self.head_chars + self.tail_chars);
        format!("{head}\n…[{omitted} chars middle-pruned by keel compaction]…\n{tail}")
    }

    /// Prune oversized `Message::Tool` entries in-place before the next model step.
    pub fn prune_history(&self, history: &mut [Message]) {
        for m in history.iter_mut() {
            if let Message::Tool { content, .. } = m {
                if content.chars().count() > self.threshold_chars {
                    *content = self.prune_str(content);
                }
            }
        }
    }
}

#[derive(Default)]
pub struct ToolResultPrunerPlugin {
    pub pruner: ToolResultPruner,
}

impl Plugin for ToolResultPrunerPlugin {
    fn name(&self) -> &str {
        "compaction-tool-result-pruner"
    }

    fn apply(&self, scope: &mut Scope) {
        scope.provide(self.pruner.clone());
    }
}

// ---------------------------------------------------------------------------
// 2. Workspace Instructions
// ---------------------------------------------------------------------------

/// Discovers `AGENTS.md`, `CLAUDE.md`, and `.agents/skills/*/SKILL.md` in the workspace.
#[derive(Clone, Debug, Default)]
pub struct WorkspaceInstructions {
    pub extra_system_prompt: String,
}

impl WorkspaceInstructions {
    pub fn discover(root: &Path) -> Self {
        let mut sections = Vec::new();
        for doc in ["AGENTS.md", "CLAUDE.md"] {
            let p = root.join(doc);
            if let Ok(text) = std::fs::read_to_string(&p) {
                let trimmed: String = text.chars().take(4000).collect();
                sections.push(format!("### Workspace `{doc}`\n{trimmed}"));
                break;
            }
        }
        let skills_dir = root.join(".agents/skills");
        if let Ok(rd) = std::fs::read_dir(&skills_dir) {
            let mut skill_names = Vec::new();
            for entry in rd.flatten() {
                let sp = entry.path().join("SKILL.md");
                if sp.is_file() {
                    skill_names.push(entry.file_name().to_string_lossy().into_owned());
                }
            }
            skill_names.sort();
            if !skill_names.is_empty() {
                sections.push(format!(
                    "### Available Workspace Skills (`.agents/skills/<name>/SKILL.md`)\n{}",
                    skill_names.join(", ")
                ));
            }
        }
        Self {
            extra_system_prompt: sections.join("\n\n"),
        }
    }
}

pub struct WorkspaceContextPlugin {
    pub root: PathBuf,
}

impl Plugin for WorkspaceContextPlugin {
    fn name(&self) -> &str {
        "workspace-context"
    }

    fn apply(&self, scope: &mut Scope) {
        scope.provide(WorkspaceInstructions::discover(&self.root));
    }
}

// ---------------------------------------------------------------------------
// 3. Adaptive Context Compaction & Budgeting
// ---------------------------------------------------------------------------

/// Configurable token budget that drives adaptive context compaction decisions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextBudget {
    /// Maximum context window size in tokens.
    pub max_context_tokens: usize,
    /// Trigger compaction when estimated history tokens cross this threshold.
    pub compaction_threshold_tokens: usize,
    /// Target size for the compacted history.
    pub target_tokens: usize,
    /// Number of recent turns to always preserve in full (verbatim).
    pub keep_recent_turns: usize,
    /// Approximate characters per token for estimation (default ~3.8).
    pub chars_per_token: f32,
}

impl Default for ContextBudget {
    fn default() -> Self {
        Self {
            max_context_tokens: 16_384,
            compaction_threshold_tokens: 12_288,
            target_tokens: 8_192,
            keep_recent_turns: 3,
            chars_per_token: 3.8,
        }
    }
}

impl ContextBudget {
    /// Creates a budget with the given maximum context window size.
    ///
    /// Threshold defaults to 75% of `max_context_tokens`, target to 50%.
    pub fn new(max_context_tokens: usize) -> Self {
        let threshold = ((max_context_tokens as f32) * 0.75) as usize;
        let target = ((max_context_tokens as f32) * 0.50) as usize;
        Self {
            max_context_tokens,
            compaction_threshold_tokens: threshold,
            target_tokens: target,
            ..Default::default()
        }
    }

    pub fn builder() -> ContextBudgetBuilder {
        ContextBudgetBuilder::default()
    }
}

#[derive(Clone, Debug, Default)]
pub struct ContextBudgetBuilder {
    max_context_tokens: Option<usize>,
    compaction_threshold_tokens: Option<usize>,
    target_tokens: Option<usize>,
    keep_recent_turns: Option<usize>,
    chars_per_token: Option<f32>,
}

impl ContextBudgetBuilder {
    pub fn max_context_tokens(mut self, v: usize) -> Self {
        self.max_context_tokens = Some(v);
        self
    }

    pub fn compaction_threshold_tokens(mut self, v: usize) -> Self {
        self.compaction_threshold_tokens = Some(v);
        self
    }

    pub fn target_tokens(mut self, v: usize) -> Self {
        self.target_tokens = Some(v);
        self
    }

    pub fn keep_recent_turns(mut self, v: usize) -> Self {
        self.keep_recent_turns = Some(v);
        self
    }

    pub fn chars_per_token(mut self, v: f32) -> Self {
        self.chars_per_token = Some(v);
        self
    }

    pub fn build(self) -> ContextBudget {
        let default_ = ContextBudget::default();
        ContextBudget {
            max_context_tokens: self
                .max_context_tokens
                .unwrap_or(default_.max_context_tokens),
            compaction_threshold_tokens: self
                .compaction_threshold_tokens
                .unwrap_or(default_.compaction_threshold_tokens),
            target_tokens: self.target_tokens.unwrap_or(default_.target_tokens),
            keep_recent_turns: self.keep_recent_turns.unwrap_or(default_.keep_recent_turns),
            chars_per_token: self.chars_per_token.unwrap_or(default_.chars_per_token),
        }
    }
}

/// Report emitted after a history compaction pass.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CompactionReport {
    /// Estimated token count of the history before compaction.
    pub before_tokens: usize,
    /// Estimated token count of the history after compaction.
    pub after_tokens: usize,
    /// Number of intermediate messages replaced by the summary.
    pub compacted_messages_count: usize,
    /// Structured summary injected into history.
    pub summary: String,
}

/// Event emitted on kernel [`keel_kernel::Context`] when compaction occurs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompactionEvent(pub CompactionReport);

/// Strategy trait for summarizing a slice of discarded conversation turns.
pub trait HistorySummarizer: Send + Sync {
    fn summarize(&self, messages: &[Message]) -> Result<String>;
}

/// Deterministic summarizer that synthesizes intermediate agent turns
/// without making an external LLM call.
#[derive(Clone, Debug, Default)]
pub struct DeterministicSummarizer;

impl DeterministicSummarizer {
    pub fn new() -> Self {
        Self
    }

    fn extract_user_guidance(messages: &[Message]) -> Vec<String> {
        messages
            .iter()
            .filter_map(|m| match m {
                Message::User { text, .. } if !text.trim().is_empty() => {
                    Some(text.trim().to_string())
                }
                _ => None,
            })
            .collect()
    }

    fn extract_tool_activity(messages: &[Message]) -> Vec<(String, usize)> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for msg in messages {
            if let Message::Assistant { tool_calls, .. } = msg {
                for tc in tool_calls {
                    *counts.entry(tc.name.clone()).or_default() += 1;
                }
            }
        }
        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        sorted
    }

    fn extract_key_commands(messages: &[Message]) -> Vec<String> {
        let mut commands: Vec<String> = Vec::new();
        for msg in messages {
            if let Message::Assistant { tool_calls, .. } = msg {
                for tc in tool_calls {
                    if matches!(tc.name.as_str(), "bash" | "shell") {
                        if let Some(cmd) = Self::extract_command_arg(&tc.args) {
                            if !commands.contains(&cmd) {
                                commands.push(cmd);
                            }
                        }
                    }
                }
            }
        }
        commands
    }

    fn extract_command_arg(args: &serde_json::Value) -> Option<String> {
        args.get("command")
            .or_else(|| args.get("cmd"))
            .or_else(|| args.get("run"))
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
    }

    fn extract_file_paths(messages: &[Message]) -> Vec<String> {
        let mut paths: Vec<String> = Vec::new();
        for msg in messages {
            if let Message::Assistant { tool_calls, .. } = msg {
                for tc in tool_calls {
                    if matches!(
                        tc.name.as_str(),
                        "read_file" | "write_file" | "edit_file" | "create_file" | "patch"
                    ) {
                        if let Some(path) = Self::extract_path_arg(&tc.args) {
                            if !paths.contains(&path) {
                                paths.push(path);
                            }
                        }
                    }
                }
            }
        }
        paths
    }

    fn extract_path_arg(args: &serde_json::Value) -> Option<String> {
        args.get("path")
            .or_else(|| args.get("file_path"))
            .or_else(|| args.get("filepath"))
            .or_else(|| args.get("file"))
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
    }

    fn extract_errors(messages: &[Message]) -> Vec<String> {
        let mut errors: Vec<String> = Vec::new();
        for msg in messages {
            if let Message::Tool { content, .. } = msg {
                for line in content.lines() {
                    let l = line.trim();
                    if l.contains("error:")
                        || l.contains("Error:")
                        || l.contains("FAILED")
                        || l.contains("PermissionDenied")
                        || l.contains("CommandNotFound")
                    {
                        let snippet: String = l.chars().take(120).collect();
                        if !errors.contains(&snippet) {
                            errors.push(snippet);
                        }
                    }
                }
            }
        }
        errors
    }
}

impl HistorySummarizer for DeterministicSummarizer {
    fn summarize(&self, messages: &[Message]) -> Result<String> {
        if messages.is_empty() {
            return Ok("No intermediate history to summarize.".to_string());
        }

        let user_guidance = Self::extract_user_guidance(messages);
        let tool_activity = Self::extract_tool_activity(messages);
        let key_commands = Self::extract_key_commands(messages);
        let file_paths = Self::extract_file_paths(messages);
        let errors = Self::extract_errors(messages);

        let mut md = String::new();
        md.push_str("### Executive Session Summary\n\n");
        md.push_str(&format!(
            "- **Compacted Turns Span**: {} intermediate message(s)\n",
            messages.len()
        ));

        if !user_guidance.is_empty() {
            md.push_str("- **Intermediate Guidance / Clarifications**:\n");
            for g in user_guidance.iter().take(3) {
                let preview: String = g.chars().take(100).collect();
                md.push_str(&format!("  * \"{preview}\"\n"));
            }
        }

        if !tool_activity.is_empty() {
            let activity_str: Vec<String> = tool_activity
                .iter()
                .map(|(tool, cnt)| format!("`{tool}` ({cnt})"))
                .collect();
            md.push_str(&format!(
                "- **Tools Executed**: {}\n",
                activity_str.join(", ")
            ));
        }

        if !key_commands.is_empty() {
            md.push_str("- **Key Shell Commands**:\n");
            for cmd in key_commands.iter().take(5) {
                let preview: String = cmd.chars().take(90).collect();
                md.push_str(&format!("  * `{preview}`\n"));
            }
        }

        if !file_paths.is_empty() {
            md.push_str("- **Modified / Inspected Files**:\n");
            for path in file_paths.iter().take(8) {
                md.push_str(&format!("  * `{path}`\n"));
            }
        }

        if !errors.is_empty() {
            md.push_str("- **Errors Encountered & Handled**:\n");
            for err in errors.iter().take(4) {
                md.push_str(&format!("  * > {err}\n"));
            }
        }

        Ok(md.trim_end().to_string())
    }
}

/// Adaptive context compactor that enforces token budget limits on history.
#[derive(Clone)]
pub struct AdaptiveCompactor {
    pub budget: ContextBudget,
    pub summarizer: Arc<dyn HistorySummarizer>,
}

impl Default for AdaptiveCompactor {
    fn default() -> Self {
        Self {
            budget: ContextBudget::default(),
            summarizer: Arc::new(DeterministicSummarizer::new()),
        }
    }
}

impl AdaptiveCompactor {
    pub fn new(budget: ContextBudget, summarizer: Arc<dyn HistorySummarizer>) -> Self {
        Self { budget, summarizer }
    }

    pub fn with_default_summarizer(budget: ContextBudget) -> Self {
        Self {
            budget,
            summarizer: Arc::new(DeterministicSummarizer::new()),
        }
    }

    /// Estimate token count of a single message using character heuristics and tool overhead.
    pub fn estimate_message_tokens(&self, msg: &Message) -> usize {
        let chars_per_tok = self.budget.chars_per_token.max(1.0);
        match msg {
            Message::System { text } | Message::User { text } => {
                ((text.chars().count() as f32) / chars_per_tok).ceil() as usize
            }
            Message::Assistant { text, tool_calls } => {
                let text_tok = ((text.chars().count() as f32) / chars_per_tok).ceil() as usize;
                let mut tool_tok = 0;
                for tc in tool_calls {
                    // tool name + args overhead
                    let args_chars = tc.args.to_string().chars().count();
                    tool_tok += 16 + ((args_chars as f32) / chars_per_tok).ceil() as usize;
                }
                text_tok + tool_tok
            }
            Message::Tool { content, .. } => {
                12 + ((content.chars().count() as f32) / chars_per_tok).ceil() as usize
            }
        }
    }

    /// Estimate total tokens across the entire conversation history.
    pub fn estimate_history_tokens(&self, history: &[Message]) -> usize {
        history
            .iter()
            .map(|m| self.estimate_message_tokens(m))
            .sum()
    }

    /// Determines whether the history has crossed the compaction threshold.
    pub fn should_compact(&self, history: &[Message]) -> bool {
        self.estimate_history_tokens(history) >= self.budget.compaction_threshold_tokens
    }

    /// Performs middle-out compaction on `history`.
    ///
    /// Preserves:
    /// - Head: Initial System messages and the initial root User prompt (task intent).
    /// - Tail: The most recent `keep_recent_turns` turns (User/Assistant/Tool interaction).
    ///
    /// The intermediate messages are condensed into a single high-density summary message.
    pub fn compact_history(&self, history: &mut Vec<Message>) -> Option<CompactionReport> {
        if !self.should_compact(history) {
            return None;
        }

        let before_tokens = self.estimate_history_tokens(history);

        // 1. Identify Head boundary (System messages + first User prompt)
        let mut head_end = 0;
        for (i, msg) in history.iter().enumerate() {
            match msg {
                Message::System { .. } => {
                    head_end = i + 1;
                }
                Message::User { .. } => {
                    head_end = i + 1;
                    break;
                }
                _ => break,
            }
        }
        if head_end == 0 && !history.is_empty() {
            head_end = 1;
        }

        // 2. Identify Tail boundary (last `keep_recent_turns` user interaction turns)
        let mut tail_start = history.len();
        let mut user_turns_seen = 0;
        for i in (head_end..history.len()).rev() {
            if matches!(history[i], Message::User { .. }) {
                user_turns_seen += 1;
                if user_turns_seen >= self.budget.keep_recent_turns {
                    tail_start = i;
                    break;
                }
            }
        }

        // Fallback: if not enough User messages, keep last (keep_recent_turns * 3) messages
        if user_turns_seen < self.budget.keep_recent_turns {
            let keep_msg_count = self.budget.keep_recent_turns.saturating_mul(3);
            tail_start = history.len().saturating_sub(keep_msg_count).max(head_end);
        }

        // If there are no intermediate messages to compact, return None
        if tail_start <= head_end {
            return None;
        }

        let intermediate: Vec<Message> = history.drain(head_end..tail_start).collect();
        let compacted_count = intermediate.len();
        if compacted_count == 0 {
            return None;
        }

        let summary = self
            .summarizer
            .summarize(&intermediate)
            .unwrap_or_else(|_| "Compacted intermediate history.".to_string());

        let summary_message = Message::User {
            text: format!(
                "[Context Virtualization: Compacted History Summary ({} messages)]\n\n{}",
                compacted_count, summary
            ),
        };

        history.insert(head_end, summary_message);

        let after_tokens = self.estimate_history_tokens(history);

        Some(CompactionReport {
            before_tokens,
            after_tokens,
            compacted_messages_count: compacted_count,
            summary,
        })
    }
}

/// Kernel plugin that mounts [`AdaptiveCompactor`] into the agent scope.
#[derive(Clone, Default)]
pub struct AdaptiveCompactorPlugin {
    pub compactor: AdaptiveCompactor,
}

impl AdaptiveCompactorPlugin {
    pub fn new(compactor: AdaptiveCompactor) -> Self {
        Self { compactor }
    }
}

impl Plugin for AdaptiveCompactorPlugin {
    fn name(&self) -> &str {
        "adaptive-context-compactor"
    }

    fn apply(&self, scope: &mut Scope) {
        scope.provide(self.compactor.clone());
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolCall;

    #[test]
    fn prunes_oversized_tool_result_preserving_head_and_tail() {
        let pruner = ToolResultPruner {
            threshold_chars: 100,
            head_chars: 20,
            tail_chars: 20,
        };
        let big = format!("HEAD_MARKER_{}_TAIL_MARKER", "x".repeat(200));
        let mut h = vec![Message::Tool {
            id: "1".into(),
            content: big,
        }];
        pruner.prune_history(&mut h);
        let Message::Tool { content, .. } = &h[0] else {
            panic!()
        };
        assert!(content.starts_with("HEAD_MARKER_"));
        assert!(content.ends_with("_TAIL_MARKER"));
        assert!(content.contains("middle-pruned by keel compaction"));
        assert!(content.len() < 120);
    }

    #[test]
    fn context_budget_builder_defaults() {
        let budget = ContextBudget::builder()
            .max_context_tokens(32_000)
            .keep_recent_turns(4)
            .build();

        assert_eq!(budget.max_context_tokens, 32_000);
        assert_eq!(budget.compaction_threshold_tokens, 12_288);
        assert_eq!(budget.keep_recent_turns, 4);
    }

    #[test]
    fn deterministic_summarizer_extracts_activity() {
        let messages = vec![
            Message::User {
                text: "Please inspect src/lib.rs and run tests".into(),
            },
            Message::Assistant {
                text: "Reading file now".into(),
                tool_calls: vec![ToolCall {
                    id: "1".into(),
                    name: "read_file".into(),
                    args: serde_json::json!({ "path": "src/lib.rs" }),
                }],
            },
            Message::Tool {
                id: "1".into(),
                content: "pub fn add() -> i32 { 42 }".into(),
            },
            Message::Assistant {
                text: "Now running cargo test".into(),
                tool_calls: vec![ToolCall {
                    id: "2".into(),
                    name: "bash".into(),
                    args: serde_json::json!({ "command": "cargo test" }),
                }],
            },
            Message::Tool {
                id: "2".into(),
                content: "error: compilation failed\ntest result: FAILED".into(),
            },
        ];

        let summarizer = DeterministicSummarizer::new();
        let summary = summarizer.summarize(&messages).unwrap();

        assert!(summary.contains("Executive Session Summary"));
        assert!(summary.contains("`read_file` (1)"));
        assert!(summary.contains("`bash` (1)"));
        assert!(summary.contains("`cargo test`"));
        assert!(summary.contains("`src/lib.rs`"));
        assert!(summary.contains("error: compilation failed"));
    }

    #[test]
    fn middle_out_compaction_preserves_head_and_tail() {
        let budget = ContextBudget {
            max_context_tokens: 1000,
            compaction_threshold_tokens: 100, // low threshold to force compaction
            target_tokens: 50,
            keep_recent_turns: 1,
            chars_per_token: 4.0,
        };

        let compactor = AdaptiveCompactor::with_default_summarizer(budget);

        let mut history = vec![
            Message::System {
                text: "System prompt instructions".into(),
            },
            Message::User {
                text: "ROOT USER INTENT: Build the feature".into(),
            },
            // Intermediate turns to be compacted
            Message::Assistant {
                text: "Step 1".into(),
                tool_calls: vec![ToolCall {
                    id: "1".into(),
                    name: "bash".into(),
                    args: serde_json::json!({ "command": "ls -la" }),
                }],
            },
            Message::Tool {
                id: "1".into(),
                content: "total 128 file1 file2 file3".repeat(10),
            },
            Message::Assistant {
                text: "Step 2".into(),
                tool_calls: vec![ToolCall {
                    id: "2".into(),
                    name: "read_file".into(),
                    args: serde_json::json!({ "path": "file1.rs" }),
                }],
            },
            Message::Tool {
                id: "2".into(),
                content: "content of file1".repeat(10),
            },
            // Recent tail turn to be preserved
            Message::User {
                text: "RECENT TURN: please verify now".into(),
            },
            Message::Assistant {
                text: "Running verification".into(),
                tool_calls: vec![],
            },
        ];

        let before_count = history.len();
        assert!(compactor.should_compact(&history));

        let report = compactor
            .compact_history(&mut history)
            .expect("should compact");
        assert_eq!(report.compacted_messages_count, 4); // 4 intermediate messages drained
        assert!(report.after_tokens < report.before_tokens);

        // Head check
        assert_eq!(
            history[0],
            Message::System {
                text: "System prompt instructions".into()
            }
        );
        assert_eq!(
            history[1],
            Message::User {
                text: "ROOT USER INTENT: Build the feature".into()
            }
        );

        // Compacted summary in middle
        match &history[2] {
            Message::User { text } => {
                assert!(text.contains("Context Virtualization: Compacted History Summary"));
                assert!(text.contains("`bash` (1)"));
                assert!(text.contains("`read_file` (1)"));
            }
            _ => panic!("Expected compacted summary message at index 2"),
        }

        // Tail check (recent turn intact)
        assert_eq!(
            history[3],
            Message::User {
                text: "RECENT TURN: please verify now".into()
            }
        );
        assert_eq!(
            history[4],
            Message::Assistant {
                text: "Running verification".into(),
                tool_calls: vec![]
            }
        );

        assert_eq!(history.len(), before_count - 4 + 1);
    }

    #[test]
    fn no_compaction_when_under_threshold() {
        let budget = ContextBudget {
            max_context_tokens: 10000,
            compaction_threshold_tokens: 5000,
            target_tokens: 2000,
            keep_recent_turns: 2,
            chars_per_token: 4.0,
        };
        let compactor = AdaptiveCompactor::with_default_summarizer(budget);

        let mut history = vec![
            Message::System {
                text: "System prompt".into(),
            },
            Message::User {
                text: "Short prompt".into(),
            },
        ];

        assert!(!compactor.should_compact(&history));
        assert!(compactor.compact_history(&mut history).is_none());
        assert_eq!(history.len(), 2);
    }
}
