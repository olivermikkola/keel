//! Subagent delegation plugin (`keel-agent::subagent`).
//!
//! Modeled on `@deepseek-ai/dsh-subagent`, allows the primary agent (or ACP daemon / CLI)
//! to spawn isolated, read-only child agents with clean context windows to explore,
//! search symbols, navigate architecture, or answer specific subtasks without cluttering parent history.

use crate::repomap::RepoMapPlugin;
use crate::sandbox::{LandlockPlugin, ToolRegistry};
use crate::symbols::SymbolPlugin;
use crate::tools::ReadOnlyToolsPlugin;
use crate::{
    Agent, ApproverPlugin, CancelToken, CancelTokenPlugin, Llm, LlmPlugin, Message, ReadOnly,
    SessionLog, SessionLogPlugin, Tool, ToolSpec,
};
use anyhow::{Context as _, Result};
use keel_kernel::{Context, Plugin, Scope};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub struct SubagentStart(pub String);
pub struct SubagentEnd(pub String);

/// Specialized role for subagent delegation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentRole {
    Researcher,
    Navigator,
    Auditor,
}

impl SubagentRole {
    pub fn from_str_lossy(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "navigator" | "nav" | "map" => Self::Navigator,
            "auditor" | "audit" | "sec" | "security" => Self::Auditor,
            _ => Self::Researcher,
        }
    }

    pub fn system_prompt(&self, root: &Path) -> String {
        match self {
            Self::Researcher => format!(
                "You are an expert research subagent working in {}. Use read-only tools (read_file, grep, list_dir, search_symbols, outline_file, get_symbol, repomap) to investigate and provide a direct, concise, factual answer to the prompt. Once you have gathered sufficient evidence, output your final answer directly and do not invoke additional tools. Do not ask for user input.",
                root.display()
            ),
            Self::Navigator => format!(
                "You are an expert codebase navigation subagent working in {}. Use repomap, outline_file, search_symbols, and get_symbol to explore project architecture, references, and symbol definitions, then provide an architectural summary answering the prompt. Once you have gathered sufficient evidence, output your final answer directly and do not invoke additional tools. Do not ask for user input.",
                root.display()
            ),
            Self::Auditor => format!(
                "You are a thorough security and reliability code auditor working in {}. Use read-only tools (read_file, grep, list_dir, search_symbols, outline_file, get_symbol, repomap) to identify vulnerabilities, boundary condition oversights, and logic bugs related to the prompt. Report exact file paths and lines. Once your findings are compiled, output your final assessment directly and do not invoke additional tools.",
                root.display()
            ),
        }
    }
}

impl std::str::FromStr for SubagentRole {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::from_str_lossy(s))
    }
}

/// Structured outcome of a subagent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentReport {
    pub prompt: String,
    pub role: SubagentRole,
    pub answer: String,
    pub steps_taken: usize,
    pub tools_called: Vec<String>,
    pub success: bool,
}

pub const MAX_SUBAGENT_PROMPT_CHARS: usize = 65_536;

/// Execute an isolated, read-only subagent turn in a fresh kernel context.
pub fn run_subagent(
    root: &Path,
    llm: Arc<dyn Llm>,
    prompt: &str,
    role: SubagentRole,
    max_steps: usize,
    landlock_enabled: bool,
    cancel_token: Option<CancelToken>,
) -> Result<SubagentReport> {
    if prompt.chars().count() > MAX_SUBAGENT_PROMPT_CHARS {
        anyhow::bail!("subagent prompt exceeds {MAX_SUBAGENT_PROMPT_CHARS} characters");
    }
    let child_ctx = Context::new();
    let _p_cancel = cancel_token.map(|c| child_ctx.mount(&CancelTokenPlugin(c)));
    let _p_llm = child_ctx.mount(&LlmPlugin(llm));
    let _p_app = child_ctx.mount(&ApproverPlugin(Arc::new(ReadOnly)));
    let _p_log = child_ctx.mount(&SessionLogPlugin(SessionLog::none()));
    let _p_landlock = child_ctx.mount(&LandlockPlugin {
        workspace: root.to_path_buf(),
        enabled: landlock_enabled,
        ..Default::default()
    });
    let _p_tools = child_ctx.mount(&ReadOnlyToolsPlugin {
        root: root.to_path_buf(),
    });
    let _p_symbols = child_ctx.mount(&SymbolPlugin {
        root: root.to_path_buf(),
    });
    let _p_repomap = child_ctx.mount(&RepoMapPlugin::new(root.to_path_buf()));
    let _p_guard = child_ctx.mount(&crate::guard::RepeatToolReminderPlugin::default());
    let _p_pruner = child_ctx.mount(&crate::compaction::ToolResultPrunerPlugin::default());
    let _p_compactor = child_ctx.mount(&crate::compaction::AdaptiveCompactorPlugin::default());

    let mut child_history = vec![Message::System {
        text: role.system_prompt(root),
    }];

    child_ctx.emit(&SubagentStart(prompt.to_string()));
    let result = Agent::run_from_context(&child_ctx, &mut child_history, prompt, max_steps);

    let tools_called: Vec<String> = child_history
        .iter()
        .filter_map(|m| match m {
            Message::Assistant { tool_calls, .. } => {
                Some(tool_calls.iter().map(|tc| tc.name.clone()))
            }
            _ => None,
        })
        .flatten()
        .collect();

    let steps_taken = child_history
        .iter()
        .filter(|m| matches!(m, Message::Assistant { .. }))
        .count();

    match result {
        Ok(ans) => {
            child_ctx.emit(&SubagentEnd(ans.clone()));
            Ok(SubagentReport {
                prompt: prompt.to_string(),
                role,
                answer: ans,
                steps_taken,
                tools_called,
                success: true,
            })
        }
        Err(e) => {
            let partial = child_history.iter().rev().find_map(|m| match m {
                Message::Assistant { text, .. } if !text.trim().is_empty() => Some(text.clone()),
                _ => None,
            });
            let answer = match partial {
                Some(p) => format!("{p}\n\n[Note: subagent terminated: {e:#}]"),
                None => format!("failed: {e:#}"),
            };
            Ok(SubagentReport {
                prompt: prompt.to_string(),
                role,
                answer,
                steps_taken,
                tools_called,
                success: false,
            })
        }
    }
}

/// The model-facing delegation tool.
pub struct SubagentTool {
    pub root: PathBuf,
    pub llm: Arc<dyn Llm>,
    pub landlock_enabled: bool,
    pub cancel_token: Option<CancelToken>,
}

impl Tool for SubagentTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "subagent".into(),
            description: "Delegate a research, search, or isolated investigation subtask to a subagent with a fresh, clean context window. The subagent runs read-only tools (read_file, grep, list_dir, search_symbols, outline_file, get_symbol, repomap) against the workspace and returns its concise findings back to you.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Specific question or investigation task for the subagent to solve."
                    },
                    "role": {
                        "type": "string",
                        "enum": ["researcher", "navigator", "auditor"],
                        "description": "Subagent specialty: 'researcher' (default), 'navigator' (architectural layout and references), or 'auditor' (security/bug review)."
                    },
                    "max_steps": {
                        "type": "integer",
                        "description": "Maximum tool steps for the subagent (default: 8, max: 15)."
                    }
                },
                "required": ["prompt"]
            }),
        }
    }

    fn mutating(&self) -> bool {
        false // Subagent is enforced read-only
    }

    fn run(&self, a: &Value) -> Result<String> {
        let prompt = a["prompt"]
            .as_str()
            .context("missing string argument `prompt` for subagent")?;
        let role = a
            .get("role")
            .and_then(|r| r.as_str())
            .map(SubagentRole::from_str_lossy)
            .unwrap_or(SubagentRole::Researcher);
        let max_steps = a["max_steps"].as_u64().unwrap_or(8).clamp(1, 15) as usize;

        let rep = run_subagent(
            &self.root,
            self.llm.clone(),
            prompt,
            role,
            max_steps,
            self.landlock_enabled,
            self.cancel_token.clone(),
        )?;

        if rep.success {
            Ok(format!(
                "[subagent {:?} response - {} steps, tools: {:?}]\n{}",
                rep.role, rep.steps_taken, rep.tools_called, rep.answer
            ))
        } else {
            Ok(format!("[subagent failed: {}]", rep.answer))
        }
    }
}

/// Plugin mounting [`SubagentTool`] into the kernel's [`ToolRegistry`].
pub struct SubagentPlugin {
    pub root: PathBuf,
    pub landlock_enabled: bool,
}

impl Plugin for SubagentPlugin {
    fn name(&self) -> &str {
        "subagent"
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
        let cancel = scope
            .context()
            .service::<CancelToken>()
            .map(|c| (*c).clone());

        reg.register(Arc::new(SubagentTool {
            root: self.root.clone(),
            llm: (*llm).clone(),
            landlock_enabled: self.landlock_enabled,
            cancel_token: cancel,
        }));
        scope.provide(reg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct ScriptLlm(Mutex<Vec<Message>>);
    impl Llm for ScriptLlm {
        fn complete(&self, _: &[Message], _: &[crate::ToolSpec]) -> Result<Message> {
            Ok(self.0.lock().unwrap().remove(0))
        }
    }

    #[test]
    fn subagent_runs_isolated_and_returns_summary() {
        let root = std::env::temp_dir();
        let llm: Arc<dyn Llm> = Arc::new(ScriptLlm(Mutex::new(vec![Message::Assistant {
            text: "found the secret number 42".into(),
            tool_calls: vec![],
        }])));

        let tool = SubagentTool {
            root,
            llm,
            landlock_enabled: false,
            cancel_token: None,
        };

        let out = tool.run(&json!({"prompt": "find the secret"})).unwrap();
        assert!(out.contains("[subagent"));
        assert!(out.contains("found the secret number 42"));
    }

    #[test]
    fn subagent_role_system_prompts_and_reports() {
        let root = Path::new("/tmp/keel_test_workspace");
        assert!(SubagentRole::Researcher
            .system_prompt(root)
            .contains("expert research subagent"));
        assert!(SubagentRole::Navigator
            .system_prompt(root)
            .contains("expert codebase navigation subagent"));
        assert!(SubagentRole::Auditor
            .system_prompt(root)
            .contains("security and reliability"));

        assert_eq!(
            SubagentRole::from_str_lossy("navigator"),
            SubagentRole::Navigator
        );
        assert_eq!(SubagentRole::from_str_lossy("audit"), SubagentRole::Auditor);
        assert_eq!(
            SubagentRole::from_str_lossy("unknown"),
            SubagentRole::Researcher
        );
        assert_eq!(
            "navigator".parse::<SubagentRole>().unwrap(),
            SubagentRole::Navigator
        );
    }

    #[test]
    fn run_subagent_captures_structured_report() {
        let tmp = std::env::temp_dir().join(format!("keel-subagent-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let llm: Arc<dyn Llm> = Arc::new(ScriptLlm(Mutex::new(vec![
            Message::Assistant {
                text: "Checking files".into(),
                tool_calls: vec![crate::ToolCall {
                    id: "c1".into(),
                    name: "list_dir".into(),
                    args: json!({"path": "."}),
                }],
            },
            Message::Assistant {
                text: "Directory is clean and verified.".into(),
                tool_calls: vec![],
            },
        ])));

        let rep = run_subagent(
            &tmp,
            llm,
            "Inspect root dir",
            SubagentRole::Navigator,
            5,
            false,
            None,
        )
        .unwrap();

        assert_eq!(rep.prompt, "Inspect root dir");
        assert_eq!(rep.role, SubagentRole::Navigator);
        assert!(rep.success);
        assert_eq!(rep.steps_taken, 2);
        assert_eq!(rep.tools_called, vec!["list_dir"]);
        assert!(rep.answer.contains("Directory is clean and verified."));

        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn subagent_prompt_length_bound_and_cancellation() {
        let tmp = std::env::temp_dir().join(format!("keel-subagent-cancel-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let llm: Arc<dyn Llm> = Arc::new(ScriptLlm(Mutex::new(vec![])));

        // 1. Oversized prompt is rejected
        let huge_prompt = "x".repeat(MAX_SUBAGENT_PROMPT_CHARS + 10);
        let err = run_subagent(
            &tmp,
            llm.clone(),
            &huge_prompt,
            SubagentRole::Researcher,
            5,
            false,
            None,
        );
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("exceeds"));

        // 2. Cancellation stops subagent immediately
        let cancel = CancelToken::new();
        cancel.cancel();
        let res = run_subagent(
            &tmp,
            llm,
            "Valid prompt",
            SubagentRole::Researcher,
            5,
            false,
            Some(cancel),
        );
        let rep = res.expect("run_subagent returns structured report on agent error");
        assert!(!rep.success);
        assert!(rep.answer.contains("cancelled"));

        let _ = std::fs::remove_dir_all(tmp);
    }
}
