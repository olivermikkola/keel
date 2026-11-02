//! Context compaction & workspace instructions (`keel-agent::compaction`).
//!
//! 1. [`ToolResultPrunerPlugin`] (modeled on `@deepseek-ai/dsh-compaction-tool-result-pruner`):
//!    When total history size crosses `pressure_chars` (or any single tool output exceeds
//!    `threshold_chars`), trims older oversized tool outputs to `head_chars` + marker + `tail_chars`
//!    without making an LLM call, while preserving the full output in the session log.
//! 2. [`WorkspaceContextPlugin`]: Auto-discovers `AGENTS.md` / `CLAUDE.md` and `SKILL.md`
//!    summaries in the workspace and injects them into the system prompt.

use crate::Message;
use keel_kernel::{Plugin, Scope};
use std::path::{Path, PathBuf};

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
