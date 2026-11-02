//! keel-agent: the turn/step agent loop and its pluggable seams.
//!
//! Seams (each is a service/trait a plugin can replace):
//! - [`Llm`]      model provider
//! - [`Tool`]     capabilities the model may call
//! - [`Approver`] permission policy for tool calls
//! - [`SessionLog`] append-only JSONL record of everything
//!
//! A *step* is one model request plus the tools it calls. A *turn* is steps
//! until the model stops requesting tools. Lifecycle events are emitted on the
//! kernel [`Context`] so other plugins can observe them.

pub mod provider;
pub mod tools;

use anyhow::Result;
use keel_kernel::Context;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::Write;
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    System { text: String },
    User { text: String },
    Assistant { text: String, tool_calls: Vec<ToolCall> },
    Tool { id: String, content: String },
}

#[derive(Clone, Debug, Serialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema for the arguments.
    pub parameters: Value,
}

/// Model provider. Implementations are plugins (OpenAI-compatible, Anthropic, ...).
pub trait Llm {
    fn complete(&self, messages: &[Message], tools: &[ToolSpec]) -> Result<Message>;
}

pub trait Tool {
    fn spec(&self) -> ToolSpec;
    /// Whether this call changes state outside the session (needs approval).
    fn mutating(&self) -> bool;
    fn run(&self, args: &Value) -> Result<String>;
}

#[derive(Clone, Debug, PartialEq)]
pub enum Decision {
    Allow,
    Deny(String),
}

/// Permission policy. A phone UI, a TUI prompt or a static rule set can implement this.
pub trait Approver {
    fn decide(&self, call: &ToolCall, mutating: bool) -> Decision;
}

/// Allow everything (only for tests / explicit `--yolo`).
pub struct AllowAll;
impl Approver for AllowAll {
    fn decide(&self, _: &ToolCall, _: bool) -> Decision {
        Decision::Allow
    }
}

/// Allow read-only tools, deny anything that mutates. Safe default for headless runs.
pub struct ReadOnly;
impl Approver for ReadOnly {
    fn decide(&self, call: &ToolCall, mutating: bool) -> Decision {
        if mutating {
            Decision::Deny(format!("`{}` denied: read-only policy", call.name))
        } else {
            Decision::Allow
        }
    }
}

/// Append-only JSONL log; the source of truth for a session.
pub struct SessionLog {
    path: Option<PathBuf>,
}

impl SessionLog {
    pub fn file(path: impl Into<PathBuf>) -> Self {
        Self { path: Some(path.into()) }
    }
    pub fn none() -> Self {
        Self { path: None }
    }
    pub fn append(&self, kind: &str, data: Value) -> Result<()> {
        let Some(p) = &self.path else { return Ok(()) };
        if let Some(dir) = p.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(p)?;
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis() as u64;
        writeln!(f, "{}", serde_json::json!({"ts": ts, "kind": kind, "data": data}))?;
        Ok(())
    }
}

// ---- lifecycle events (observable via Context::emit) ----
pub struct TurnStart;
pub struct StepStart(pub usize);
pub struct ToolDenied(pub ToolCall);
pub struct ToolResult(pub String, pub String);
pub struct TurnEnd(pub String);

pub struct Agent<'a> {
    pub ctx: &'a Context,
    pub llm: &'a dyn Llm,
    pub tools: Vec<Box<dyn Tool>>,
    pub approver: &'a dyn Approver,
    pub log: &'a SessionLog,
    pub max_steps: usize,
}

impl Agent<'_> {
    /// Run one turn against `history` (mutated in place). Returns the final assistant text.
    pub fn run_turn(&self, history: &mut Vec<Message>, user: &str) -> Result<String> {
        self.ctx.emit(&TurnStart);
        history.push(Message::User { text: user.into() });
        self.log.append("user", serde_json::json!({ "text": user }))?;
        let specs: Vec<ToolSpec> = self.tools.iter().map(|t| t.spec()).collect();

        for step in 0..self.max_steps {
            self.ctx.emit(&StepStart(step));
            let reply = self.llm.complete(history, &specs)?;
            self.log.append("assistant", serde_json::to_value(&reply)?)?;
            history.push(reply.clone());
            let Message::Assistant { text, tool_calls } = reply else {
                anyhow::bail!("provider returned a non-assistant message");
            };
            if tool_calls.is_empty() {
                self.ctx.emit(&TurnEnd(text.clone()));
                return Ok(text);
            }
            for call in tool_calls {
                let out = self.execute(&call);
                self.log.append("tool", serde_json::json!({"id": call.id, "content": out}))?;
                history.push(Message::Tool { id: call.id, content: out });
            }
        }
        anyhow::bail!("step limit ({}) reached without a final answer", self.max_steps)
    }

    fn execute(&self, call: &ToolCall) -> String {
        let Some(tool) = self.tools.iter().find(|t| t.spec().name == call.name) else {
            return format!("error: unknown tool `{}`", call.name);
        };
        match self.approver.decide(call, tool.mutating()) {
            Decision::Deny(why) => {
                self.ctx.emit(&ToolDenied(call.clone()));
                format!("denied: {why}")
            }
            Decision::Allow => {
                // Tool errors go back to the model as text; they never abort the turn.
                let out = tool.run(&call.args).unwrap_or_else(|e| format!("error: {e:#}"));
                self.ctx.emit(&ToolResult(call.id.clone(), out.clone()));
                out
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// Scripted model: returns queued replies in order.
    struct Script(RefCell<Vec<Message>>);
    impl Llm for Script {
        fn complete(&self, _: &[Message], _: &[ToolSpec]) -> Result<Message> {
            Ok(self.0.borrow_mut().remove(0))
        }
    }

    struct Echo;
    impl Tool for Echo {
        fn spec(&self) -> ToolSpec {
            ToolSpec { name: "echo".into(), description: "".into(), parameters: serde_json::json!({}) }
        }
        fn mutating(&self) -> bool {
            false
        }
        fn run(&self, a: &Value) -> Result<String> {
            Ok(a["s"].as_str().unwrap_or("").to_string())
        }
    }
    struct Writer;
    impl Tool for Writer {
        fn spec(&self) -> ToolSpec {
            ToolSpec { name: "write".into(), description: "".into(), parameters: serde_json::json!({}) }
        }
        fn mutating(&self) -> bool {
            true
        }
        fn run(&self, _: &Value) -> Result<String> {
            panic!("must not run when denied")
        }
    }

    fn call(id: &str, name: &str) -> ToolCall {
        ToolCall { id: id.into(), name: name.into(), args: serde_json::json!({"s": "hi"}) }
    }
    fn asst(text: &str, calls: Vec<ToolCall>) -> Message {
        Message::Assistant { text: text.into(), tool_calls: calls }
    }

    #[test]
    fn tool_loop_then_final_answer() {
        let ctx = Context::new();
        let llm = Script(RefCell::new(vec![asst("", vec![call("1", "echo")]), asst("done", vec![])]));
        let agent = Agent { ctx: &ctx, llm: &llm, tools: vec![Box::new(Echo)], approver: &AllowAll, log: &SessionLog::none(), max_steps: 5 };
        let mut h = vec![];
        assert_eq!(agent.run_turn(&mut h, "go").unwrap(), "done");
        assert!(matches!(&h[2], Message::Tool { content, .. } if content == "hi"));
    }

    #[test]
    fn read_only_policy_denies_mutating_tool() {
        let ctx = Context::new();
        let llm = Script(RefCell::new(vec![asst("", vec![call("1", "write")]), asst("ok", vec![])]));
        let agent = Agent { ctx: &ctx, llm: &llm, tools: vec![Box::new(Writer)], approver: &ReadOnly, log: &SessionLog::none(), max_steps: 5 };
        let mut h = vec![];
        agent.run_turn(&mut h, "go").unwrap();
        assert!(matches!(&h[2], Message::Tool { content, .. } if content.starts_with("denied")));
    }

    #[test]
    fn step_limit_stops_runaway_loops() {
        let ctx = Context::new();
        let calls = (0..3).map(|i| asst("", vec![call(&i.to_string(), "echo")])).collect();
        let llm = Script(RefCell::new(calls));
        let agent = Agent { ctx: &ctx, llm: &llm, tools: vec![Box::new(Echo)], approver: &AllowAll, log: &SessionLog::none(), max_steps: 3 };
        assert!(agent.run_turn(&mut vec![], "go").is_err());
    }
}
