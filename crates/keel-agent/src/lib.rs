//! keel-agent: the turn/step agent loop and its pluggable seams.
//!
//! Seams (each is a service/trait a plugin can replace):
//! - [`Llm`]           model provider (`Arc<dyn Llm>` service on [`Context`])
//! - [`ToolRegistry`]  capabilities the model may call
//! - [`Approver`]      permission policy for tool calls (`Arc<dyn Approver>` service)
//! - [`SessionLog`]    append-only JSONL record of everything
//! - [`SandboxPolicy`] Linux Landlock filesystem confinement for subprocesses
//!
//! A *step* is one model request plus the tools it calls. A *turn* is steps
//! until the model stops requesting tools. Lifecycle events are emitted on the
//! kernel [`Context`] so other plugins can observe them.

pub mod compaction;
pub mod guard;
pub mod provider;
pub mod sandbox;
pub mod tools;

use anyhow::Result;
use keel_kernel::{Context, Plugin, Scope};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    System {
        text: String,
    },
    User {
        text: String,
    },
    Assistant {
        text: String,
        tool_calls: Vec<ToolCall>,
    },
    Tool {
        id: String,
        content: String,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema for the arguments.
    pub parameters: Value,
}

/// Incremental output from a model while it is still generating.
#[derive(Clone, Debug, PartialEq)]
pub enum Chunk {
    Text(String),
    /// Reasoning/thinking tokens. Shown to the user, never fed back into history.
    Reasoning(String),
}

/// Model provider. Implementations are plugins (OpenAI-compatible, Anthropic, ...).
pub trait Llm: Send + Sync {
    fn complete(&self, messages: &[Message], tools: &[ToolSpec]) -> Result<Message>;

    /// Streaming variant. Default: no streaming, emit the whole text at once.
    fn complete_streaming(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        sink: &mut dyn FnMut(Chunk),
    ) -> Result<Message> {
        let reply = self.complete(messages, tools)?;
        if let Message::Assistant { text, .. } = &reply {
            if !text.is_empty() {
                sink(Chunk::Text(text.clone()));
            }
        }
        Ok(reply)
    }
}

pub trait Tool: Send + Sync {
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
pub trait Approver: Send + Sync {
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
#[derive(Clone)]
pub struct SessionLog {
    path: Option<PathBuf>,
}

impl SessionLog {
    pub fn file(path: impl Into<PathBuf>) -> Self {
        Self {
            path: Some(path.into()),
        }
    }
    pub fn none() -> Self {
        Self { path: None }
    }
    pub fn append(&self, kind: &str, data: Value) -> Result<()> {
        let Some(p) = &self.path else { return Ok(()) };
        if let Some(dir) = p.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)?;
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis() as u64;
        writeln!(
            f,
            "{}",
            serde_json::json!({"ts": ts, "kind": kind, "data": data})
        )?;
        Ok(())
    }
}

// ---- Kernel Plugins for mounting standard services ----

pub struct LlmPlugin(pub Arc<dyn Llm>);
impl Plugin for LlmPlugin {
    fn name(&self) -> &str {
        "llm"
    }
    fn apply(&self, scope: &mut Scope) {
        scope.provide::<Arc<dyn Llm>>(self.0.clone());
    }
}

pub struct ApproverPlugin(pub Arc<dyn Approver>);
impl Plugin for ApproverPlugin {
    fn name(&self) -> &str {
        "approver"
    }
    fn apply(&self, scope: &mut Scope) {
        scope.provide::<Arc<dyn Approver>>(self.0.clone());
    }
}

pub struct SessionLogPlugin(pub SessionLog);
impl Plugin for SessionLogPlugin {
    fn name(&self) -> &str {
        "session-log"
    }
    fn apply(&self, scope: &mut Scope) {
        scope.provide::<SessionLog>(self.0.clone());
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
    /// Run a turn by resolving all services (`Arc<dyn Llm>`, `ToolRegistry`, `Arc<dyn Approver>`, `SessionLog`)
    /// directly from the plugin [`Context`].
    pub fn run_from_context(
        ctx: &Context,
        history: &mut Vec<Message>,
        user: &str,
        max_steps: usize,
    ) -> Result<String> {
        let llm = ctx
            .service::<Arc<dyn Llm>>()
            .ok_or_else(|| anyhow::anyhow!("missing `Llm` plugin in kernel context"))?;
        let approver = ctx
            .service::<Arc<dyn Approver>>()
            .ok_or_else(|| anyhow::anyhow!("missing `Approver` plugin in kernel context"))?;
        let log = ctx
            .service::<SessionLog>()
            .unwrap_or_else(|| Arc::new(SessionLog::none()));
        let reg_tools = ctx
            .service::<sandbox::ToolRegistry>()
            .map(|r| r.list())
            .unwrap_or_default();

        struct ArcTool(Arc<dyn Tool>);
        impl Tool for ArcTool {
            fn spec(&self) -> ToolSpec {
                self.0.spec()
            }
            fn mutating(&self) -> bool {
                self.0.mutating()
            }
            fn run(&self, args: &Value) -> Result<String> {
                self.0.run(args)
            }
        }

        let tools: Vec<Box<dyn Tool>> = reg_tools
            .into_iter()
            .map(|t| Box::new(ArcTool(t)) as Box<dyn Tool>)
            .collect();

        let agent = Agent {
            ctx,
            llm: llm.as_ref().as_ref(),
            tools,
            approver: approver.as_ref().as_ref(),
            log: log.as_ref(),
            max_steps,
        };
        agent.run_turn(history, user)
    }

    /// Extract inline `<tool_call>{"name": ..., "arguments": ...}</tool_call>` or JSON blocks
    /// when a smaller/weak model emits tool invocations in plain text instead of native `tool_calls`.
    fn fallback_parse_text_tool_calls(text: &str) -> Vec<ToolCall> {
        let mut out = Vec::new();
        // 1. `<tool_call>...</tool_call>` tags (Qwen / Hermes / Bonsai style)
        let mut rest = text;
        while let Some(start) = rest.find("<tool_call>") {
            let after = &rest[start + "<tool_call>".len()..];
            let Some(end) = after.find("</tool_call>") else {
                break;
            };
            let inner = after[..end].trim();
            if let Ok(v) = serde_json::from_str::<Value>(inner) {
                if let Some(name) = v["name"].as_str() {
                    let args = v
                        .get("arguments")
                        .or_else(|| v.get("args"))
                        .or_else(|| v.get("parameters"))
                        .cloned()
                        .unwrap_or_else(|| json_without_name(&v));
                    out.push(ToolCall {
                        id: format!("fallback-{}", out.len() + 1),
                        name: name.to_string(),
                        args,
                    });
                }
            }
            rest = &after[end + "</tool_call>".len()..];
        }
        if !out.is_empty() {
            return out;
        }
        // 2. Fenced ```json or ```tool_call blocks containing {"name": ..., "arguments": ...}
        let mut scan = text;
        while let Some(fence) = scan.find("```") {
            let after_fence = &scan[fence + 3..];
            let Some(nl) = after_fence.find('\n') else {
                break;
            };
            let body_start = &after_fence[nl + 1..];
            let Some(close) = body_start.find("```") else {
                break;
            };
            let candidate = body_start[..close].trim();
            if let Ok(v) = serde_json::from_str::<Value>(candidate) {
                if let Some(name) = v["name"].as_str() {
                    let args = v
                        .get("arguments")
                        .or_else(|| v.get("args"))
                        .or_else(|| v.get("parameters"))
                        .cloned()
                        .unwrap_or_else(|| json_without_name(&v));
                    out.push(ToolCall {
                        id: format!("fallback-{}", out.len() + 1),
                        name: name.to_string(),
                        args,
                    });
                }
            }
            scan = &body_start[close + 3..];
        }
        out
    }

    /// Run one turn against `history` (mutated in place). Returns the final assistant text.
    pub fn run_turn(&self, history: &mut Vec<Message>, user: &str) -> Result<String> {
        self.ctx.emit(&TurnStart);
        history.push(Message::User { text: user.into() });
        self.log
            .append("user", serde_json::json!({ "text": user }))?;
        let specs: Vec<ToolSpec> = self.tools.iter().map(|t| t.spec()).collect();

        for step in 0..self.max_steps {
            self.ctx.emit(&StepStart(step));
            if let Some(pruner) = self.ctx.service::<compaction::ToolResultPruner>() {
                pruner.prune_history(history);
            }
            let mut reply = self
                .llm
                .complete_streaming(history, &specs, &mut |c| self.ctx.emit(&c))?;
            if let Message::Assistant {
                ref text,
                ref mut tool_calls,
            } = reply
            {
                if tool_calls.is_empty() {
                    let recovered = Self::fallback_parse_text_tool_calls(text);
                    if !recovered.is_empty() {
                        *tool_calls = recovered;
                    }
                }
            }
            self.log
                .append("assistant", serde_json::to_value(&reply)?)?;
            history.push(reply.clone());
            let Message::Assistant { text, tool_calls } = reply else {
                anyhow::bail!("provider returned a non-assistant message");
            };
            if tool_calls.is_empty() {
                self.ctx.emit(&TurnEnd(text.clone()));
                return Ok(text);
            }
            for call in tool_calls {
                let mut out = self.execute(&call);
                if let Some(guard) = self.ctx.service::<guard::RepeatToolGuard>() {
                    if let Some(reminder) = guard.observe(&call) {
                        out.push_str(&reminder);
                    }
                }
                self.log
                    .append("tool", serde_json::json!({"id": call.id, "content": out}))?;
                history.push(Message::Tool {
                    id: call.id,
                    content: out,
                });
            }
        }
        anyhow::bail!(
            "step limit ({}) reached without a final answer",
            self.max_steps
        )
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
                let out = tool
                    .run(&call.args)
                    .unwrap_or_else(|e| format!("error: {e:#}"));
                self.ctx.emit(&ToolResult(call.id.clone(), out.clone()));
                out
            }
        }
    }
}

fn json_without_name(v: &Value) -> Value {
    if let Some(obj) = v.as_object() {
        let mut m = obj.clone();
        m.remove("name");
        Value::Object(m)
    } else {
        Value::Null
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Scripted model: returns queued replies in order.
    struct Script(Mutex<Vec<Message>>);
    impl Llm for Script {
        fn complete(&self, _: &[Message], _: &[ToolSpec]) -> Result<Message> {
            Ok(self.0.lock().unwrap().remove(0))
        }
    }

    struct Echo;
    impl Tool for Echo {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: "echo".into(),
                description: "".into(),
                parameters: serde_json::json!({}),
            }
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
            ToolSpec {
                name: "write".into(),
                description: "".into(),
                parameters: serde_json::json!({}),
            }
        }
        fn mutating(&self) -> bool {
            true
        }
        fn run(&self, _: &Value) -> Result<String> {
            panic!("must not run when denied")
        }
    }

    fn call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: name.into(),
            args: serde_json::json!({"s": "hi"}),
        }
    }
    fn asst(text: &str, calls: Vec<ToolCall>) -> Message {
        Message::Assistant {
            text: text.into(),
            tool_calls: calls,
        }
    }

    #[test]
    fn tool_loop_then_final_answer() {
        let ctx = Context::new();
        let llm = Script(Mutex::new(vec![
            asst("", vec![call("1", "echo")]),
            asst("done", vec![]),
        ]));
        let agent = Agent {
            ctx: &ctx,
            llm: &llm,
            tools: vec![Box::new(Echo)],
            approver: &AllowAll,
            log: &SessionLog::none(),
            max_steps: 5,
        };
        let mut h = vec![];
        assert_eq!(agent.run_turn(&mut h, "go").unwrap(), "done");
        assert!(matches!(&h[2], Message::Tool { content, .. } if content == "hi"));
    }

    #[test]
    fn fallback_text_tool_call_recovery() {
        let ctx = Context::new();
        let llm = Script(Mutex::new(vec![
            asst(
                "Let me run echo:\n<tool_call>{\"name\":\"echo\",\"arguments\":{\"s\":\"recovered\"}}</tool_call>",
                vec![],
            ),
            asst("all good", vec![]),
        ]));
        let agent = Agent {
            ctx: &ctx,
            llm: &llm,
            tools: vec![Box::new(Echo)],
            approver: &AllowAll,
            log: &SessionLog::none(),
            max_steps: 5,
        };
        let mut h = vec![];
        assert_eq!(agent.run_turn(&mut h, "go").unwrap(), "all good");
        assert!(matches!(&h[2], Message::Tool { content, .. } if content == "recovered"));
    }

    #[test]
    fn read_only_policy_denies_mutating_tool() {
        let ctx = Context::new();
        let llm = Script(Mutex::new(vec![
            asst("", vec![call("1", "write")]),
            asst("ok", vec![]),
        ]));
        let agent = Agent {
            ctx: &ctx,
            llm: &llm,
            tools: vec![Box::new(Writer)],
            approver: &ReadOnly,
            log: &SessionLog::none(),
            max_steps: 5,
        };
        let mut h = vec![];
        agent.run_turn(&mut h, "go").unwrap();
        assert!(matches!(&h[2], Message::Tool { content, .. } if content.starts_with("denied")));
    }

    #[test]
    fn step_limit_stops_runaway_loops() {
        let ctx = Context::new();
        let calls = (0..3)
            .map(|i| asst("", vec![call(&i.to_string(), "echo")]))
            .collect();
        let llm = Script(Mutex::new(calls));
        let agent = Agent {
            ctx: &ctx,
            llm: &llm,
            tools: vec![Box::new(Echo)],
            approver: &AllowAll,
            log: &SessionLog::none(),
            max_steps: 3,
        };
        assert!(agent.run_turn(&mut vec![], "go").is_err());
    }
}
