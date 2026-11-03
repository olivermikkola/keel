//! Subagent delegation plugin (`keel-agent::subagent`).
//!
//! Modeled on `@deepseek-ai/dsh-subagent`, allows the primary agent to spawn
//! isolated, read-only child agents with clean context windows to explore,
//! search, or answer specific subtasks without cluttering parent history.

use crate::sandbox::{LandlockPlugin, ToolRegistry};
use crate::tools::BuiltinToolsPlugin;
use crate::{
    Agent, ApproverPlugin, Llm, LlmPlugin, Message, ReadOnly, SessionLog, SessionLogPlugin, Tool,
    ToolSpec,
};
use anyhow::{Context as _, Result};
use keel_kernel::{Context, Plugin, Scope};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;

pub struct SubagentStart(pub String);
pub struct SubagentEnd(pub String);

/// The model-facing delegation tool.
pub struct SubagentTool {
    pub root: PathBuf,
    pub llm: Arc<dyn Llm>,
    pub landlock_enabled: bool,
}

impl Tool for SubagentTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "subagent".into(),
            description: "Delegate a research, search, or isolated investigation subtask to a subagent with a fresh, clean context window. The subagent runs read-only tools against the workspace and returns its concise findings back to you.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Specific question or investigation task for the subagent to solve."
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
        let max_steps = a["max_steps"].as_u64().unwrap_or(8).clamp(1, 15) as usize;

        // Create an isolated kernel context for the subagent
        let child_ctx = Context::new();
        let _p_llm = child_ctx.mount(&LlmPlugin(self.llm.clone()));
        let _p_app = child_ctx.mount(&ApproverPlugin(Arc::new(ReadOnly)));
        let _p_log = child_ctx.mount(&SessionLogPlugin(SessionLog::none()));
        let _p_landlock = child_ctx.mount(&LandlockPlugin {
            workspace: self.root.clone(),
            enabled: self.landlock_enabled,
            ..Default::default()
        });
        let _p_tools = child_ctx.mount(&BuiltinToolsPlugin {
            root: self.root.clone(),
            landlock_enabled: self.landlock_enabled,
        });
        let _p_guard = child_ctx.mount(&crate::guard::RepeatToolReminderPlugin::default());
        let _p_pruner = child_ctx.mount(&crate::compaction::ToolResultPrunerPlugin::default());
        let _p_compactor = child_ctx.mount(&crate::compaction::AdaptiveCompactorPlugin::default());

        let mut child_history = vec![Message::System {
            text: format!(
                "You are an expert research subagent working in {}. Use read-only tools to investigate and provide a direct, concise, factual answer to the prompt. Do not ask for user input.",
                self.root.display()
            ),
        }];

        child_ctx.emit(&SubagentStart(prompt.to_string()));
        let result = Agent::run_from_context(&child_ctx, &mut child_history, prompt, max_steps);
        match result {
            Ok(ans) => {
                child_ctx.emit(&SubagentEnd(ans.clone()));
                Ok(format!("[subagent response]\n{ans}"))
            }
            Err(e) => Ok(format!("[subagent failed: {e:#}]")),
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
        let Some(reg) = scope.context().service::<ToolRegistry>() else {
            return;
        };

        reg.register(Arc::new(SubagentTool {
            root: self.root.clone(),
            llm: (*llm).clone(),
            landlock_enabled: self.landlock_enabled,
        }));
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
        };

        let out = tool.run(&json!({"prompt": "find the secret"})).unwrap();
        assert!(out.contains("[subagent response]"));
        assert!(out.contains("found the secret number 42"));
    }
}
