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

pub mod cascade;
pub mod checkpoint;
pub mod compaction;
pub mod daemon;
pub mod epistemic;
pub mod guard;
pub mod interceptor;
pub mod interrupt;
pub mod lsp;
pub mod mcp;
pub mod oracle;
pub mod patch;
pub mod plan;
pub mod provider;
pub mod recorder;
pub mod repomap;
pub mod retention;
pub mod sandbox;
pub mod speculate;
pub mod subagent;
pub mod swarm;
pub mod symbols;
pub mod tools;
pub mod tree;
pub mod vfs;

pub use compaction::{
    AdaptiveCompactor, AdaptiveCompactorPlugin, CompactionEvent, CompactionReport, ContextBudget,
    ContextBudgetBuilder, DeterministicSummarizer, HistorySummarizer, ToolResultPruner,
    ToolResultPrunerPlugin, WorkspaceContextPlugin, WorkspaceInstructions,
};
pub use epistemic::{
    ContradictionAlert, EpistemicGraphLinkTool, EpistemicMemoryPlugin, EpistemicState,
    EpistemicStore, FactCategory, FactNode, GraphEdge, RelationType,
};
pub use guard::{
    CircuitBreakerTrip, EnvironmentCircuitBreaker, EnvironmentCircuitBreakerPlugin,
    EnvironmentErrorType, RepeatToolGuard, RepeatToolReminderPlugin,
};
pub use interceptor::{
    IncrementalArgParser, PrewarmEngine, PrewarmPlugin, SentinelMode as StreamingSentinelMode,
    SentinelPlugin, SentinelRule, StreamingSentinel, VetoCategory, VetoEvent,
};
pub use oracle::{
    parse_test_output, DiagnosticOracle, OraclePlugin, RegressionDelta, SentinelMode, TestFailure,
    TestSnapshot,
};
pub use recorder::{
    FlightRecorder, FlightRecorderPlugin, ForkReport, RecordFrame, RecordedSession,
    RecordingSummary, ReplayEngine, ReplayEnginePlugin, RestoreReport, StepInspection,
};
pub use repomap::{
    compute_pagerank, LazyRepoMap, ReferenceGraph, RepoMap, RepoMapPlugin, RepoMapTool,
};
pub use sandbox::{
    LandlockPlugin, SandboxPolicy, COMMON_TOOLCHAIN_ROOTS, DEFAULT_DEVICE_NODES, TOOLCHAIN_ENV_VARS,
};
pub use subagent::{
    run_subagent, SubagentEnd, SubagentPlugin, SubagentReport, SubagentRole, SubagentStart,
};
pub use swarm::{export_handoff, ingest_handoff, HandoffEnvelope, HandoffIngestReport};
pub use symbols::{
    parse_structural_outline, render_outline_markdown, FileOutline, Symbol, SymbolIndex,
    SymbolKind, SymbolPlugin, SymbolSearchResult,
};
pub use tools::{BuiltinToolsPlugin, ReadOnlyToolsPlugin, WorkspaceSafetyPolicy};
pub use tree::{ExplorationTree, NodeStatus, TreeNode, TreePlugin};

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
    /// Incremental tool call argument delta while generating.
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        args_delta: String,
    },
    /// Early security sentinel notification emitted during streaming.
    SentinelVeto {
        tool_name: String,
        reason: String,
    },
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApprovalPolicy {
    /// Prompt on stdin for all mutating tools.
    Safe,
    /// Auto-allow low/medium non-shell workspace edits; prompt on stdin for shell or sensitive paths.
    Standard,
    /// Allow all tools without prompting.
    Yolo,
    /// Strictly deny all mutating tools without reading stdin.
    ReadOnly,
}

impl ApprovalPolicy {
    pub fn from_str_lossy(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "yolo" | "allow-all" | "allow_all" => Self::Yolo,
            "safe" | "strict" => Self::Safe,
            "read-only" | "readonly" | "read_only" => Self::ReadOnly,
            _ => Self::Standard,
        }
    }
}

/// Interactive HITL approver that reads `y` / `n` / `always` decisions from stdin
/// (compatible with both terminal users and parent harnesses like Helm).
pub struct InteractiveApprover {
    pub policy: ApprovalPolicy,
    pub json_mode: bool,
    whitelisted: std::sync::Mutex<std::collections::HashSet<String>>,
    reader: std::sync::Mutex<Option<Box<dyn std::io::BufRead + Send>>>,
    output: std::sync::Mutex<Option<Box<dyn std::io::Write + Send>>>,
    interaction: std::sync::Mutex<()>,
}

impl InteractiveApprover {
    pub fn new(policy: ApprovalPolicy, json_mode: bool) -> Self {
        Self {
            policy,
            json_mode,
            whitelisted: std::sync::Mutex::new(std::collections::HashSet::new()),
            reader: std::sync::Mutex::new(None),
            output: std::sync::Mutex::new(None),
            interaction: std::sync::Mutex::new(()),
        }
    }

    pub fn with_reader(
        policy: ApprovalPolicy,
        json_mode: bool,
        reader: Box<dyn std::io::BufRead + Send>,
    ) -> Self {
        Self {
            policy,
            json_mode,
            whitelisted: std::sync::Mutex::new(std::collections::HashSet::new()),
            reader: std::sync::Mutex::new(Some(reader)),
            output: std::sync::Mutex::new(None),
            interaction: std::sync::Mutex::new(()),
        }
    }

    /// Parent integrations can supply both streams; requests are flushed before input is read.
    pub fn with_io(
        policy: ApprovalPolicy,
        json_mode: bool,
        reader: Box<dyn std::io::BufRead + Send>,
        output: Box<dyn std::io::Write + Send>,
    ) -> Self {
        Self {
            policy,
            json_mode,
            whitelisted: std::sync::Mutex::new(std::collections::HashSet::new()),
            reader: std::sync::Mutex::new(Some(reader)),
            output: std::sync::Mutex::new(Some(output)),
            interaction: std::sync::Mutex::new(()),
        }
    }

    fn emit_request(&self, call: &ToolCall) -> std::io::Result<()> {
        static NEXT_REQUEST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let request = serde_json::json!({
            "type":"approval_request", "protocol":"stdio-line-v1",
            "id":format!("approval-{}-{}",std::process::id(),NEXT_REQUEST.fetch_add(1,std::sync::atomic::Ordering::Relaxed)),
            "tool_call_id":call.id, "tool":call.name, "arguments":call.args,
            "choices":["allow","deny","always"]
        });
        let write_request = |out: &mut dyn std::io::Write| -> std::io::Result<()> {
            writeln!(out, "{request}")?;
            out.flush()
        };
        let mut output = self
            .output
            .lock()
            .map_err(|_| std::io::Error::other("approval output lock poisoned"))?;
        if let Some(output) = output.as_mut() {
            write_request(output.as_mut())
        } else {
            write_request(&mut std::io::stdout().lock())
        }
    }

    fn is_sensitive_target(call: &ToolCall) -> bool {
        let target = call
            .args
            .get("path")
            .or_else(|| call.args.get("file"))
            .or_else(|| call.args.get("command"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        target.contains("/etc/")
            || target.contains(".ssh")
            || target.contains(".bashrc")
            || target.contains(".zshrc")
            || target.contains(".profile")
            || target.contains(".env")
            || target.contains(".git")
            || target.contains(".keel")
    }
}

impl Approver for InteractiveApprover {
    fn decide(&self, call: &ToolCall, mutating: bool) -> Decision {
        if !mutating || self.policy == ApprovalPolicy::Yolo {
            return Decision::Allow;
        }
        if self.policy == ApprovalPolicy::ReadOnly {
            return Decision::Deny(format!("`{}` denied: read-only policy", call.name));
        }
        if let Ok(wl) = self.whitelisted.lock() {
            if wl.contains(&call.name) {
                return Decision::Allow;
            }
        }
        if self.policy == ApprovalPolicy::Standard
            && call.name != "shell"
            && !Self::is_sensitive_target(call)
        {
            return Decision::Allow;
        }

        // Serialize the entire request/response exchange, not just the read.
        let Ok(_interaction) = self.interaction.lock() else {
            return Decision::Deny("approval channel unavailable".into());
        };
        if self
            .whitelisted
            .lock()
            .is_ok_and(|entries| entries.contains(&call.name))
        {
            return Decision::Allow;
        }
        if self.json_mode && self.emit_request(call).is_err() {
            return Decision::Deny("approval request could not be delivered to the parent".into());
        }
        if !self.json_mode {
            eprint!(
                "[keel] Approve tool call `{}` ({})? [y/N]: ",
                call.name, call.args
            );
            let _ = std::io::stderr().flush();
        }
        let _ = std::io::stdout().flush();

        let mut line = String::new();
        let read_res = if let Ok(mut guard) = self.reader.lock() {
            if let Some(r) = guard.as_mut() {
                r.read_line(&mut line)
            } else {
                std::io::stdin().read_line(&mut line)
            }
        } else {
            std::io::stdin().read_line(&mut line)
        };

        match read_res {
            Ok(0) | Err(_) => Decision::Deny(format!(
                "`{}` denied: awaiting approval on stdin (EOF)",
                call.name
            )),
            Ok(_) => {
                let ans = line.trim().to_ascii_lowercase();
                if matches!(ans.as_str(), "y" | "yes" | "allow" | "ok") {
                    Decision::Allow
                } else if matches!(ans.as_str(), "a" | "always" | "allow_always") {
                    if let Ok(mut wl) = self.whitelisted.lock() {
                        wl.insert(call.name.clone());
                    }
                    Decision::Allow
                } else {
                    Decision::Deny(format!("`{}` denied by user", call.name))
                }
            }
        }
    }
}

/// Append-only JSONL log; the source of truth for a session.
pub const MAX_LOG_FILE_BYTES: u64 = 64 * 1024 * 1024; // 64MB max log file
pub const MAX_LOG_LINE_BYTES: usize = 16 * 1024 * 1024; // 16MB max log line
pub const MAX_LOG_MESSAGES: usize = 50_000;

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

    /// Read an existing JSONL session log file and reconstruct previous conversation messages.
    pub fn read_history(path: impl AsRef<std::path::Path>) -> Result<Vec<Message>> {
        use std::io::BufRead;
        let p = path.as_ref();
        if !p.is_file() {
            anyhow::bail!("session log path is not a regular file: {}", p.display());
        }
        let meta = std::fs::metadata(p)?;
        if meta.len() > MAX_LOG_FILE_BYTES {
            anyhow::bail!(
                "session log file size ({} bytes) exceeds maximum allowable limit (64MB): {}",
                meta.len(),
                p.display()
            );
        }
        let file = std::fs::File::open(p)?;
        let mut reader = std::io::BufReader::new(file);
        let mut messages = Vec::new();
        let mut line_number = 0usize;
        let mut total_bytes = 0u64;
        loop {
            let mut line = String::new();
            use std::io::Read;
            let n = reader
                .by_ref()
                .take((MAX_LOG_LINE_BYTES + 1) as u64)
                .read_line(&mut line)?;
            if n == 0 {
                break;
            }
            line_number += 1;
            total_bytes += n as u64;
            if total_bytes > MAX_LOG_FILE_BYTES {
                anyhow::bail!("session log grew beyond the 64MB limit while reading");
            }
            if line.len() > MAX_LOG_LINE_BYTES {
                anyhow::bail!("session log line exceeds maximum allowable size (16MB)");
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let val: Value = serde_json::from_str(trimmed).map_err(|_| {
                anyhow::anyhow!(
                    "invalid session JSON at line {line_number}; history was not loaded"
                )
            })?;
            let kind = val["kind"].as_str().ok_or_else(|| {
                anyhow::anyhow!("missing session record kind at line {line_number}")
            })?;
            let data = &val["data"];
            let invalid = || {
                anyhow::anyhow!(
                    "invalid {kind} session record at line {line_number}; history was not loaded"
                )
            };
            match kind {
                "user" => messages.push(Message::User {
                    text: data["text"].as_str().ok_or_else(invalid)?.into(),
                }),
                "assistant" => {
                    let message =
                        serde_json::from_value::<Message>(data.clone()).map_err(|_| invalid())?;
                    if !matches!(message, Message::Assistant { .. }) {
                        return Err(invalid());
                    }
                    messages.push(message);
                }
                "tool" => messages.push(Message::Tool {
                    id: data["id"].as_str().ok_or_else(invalid)?.into(),
                    content: data["content"].as_str().ok_or_else(invalid)?.into(),
                }),
                // Telemetry and future non-conversation records do not alter history.
                _ => {}
            }
            if messages.len() > MAX_LOG_MESSAGES {
                anyhow::bail!("session log exceeds {MAX_LOG_MESSAGES} conversation messages; history was not loaded");
            }
        }
        Ok(messages)
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

/// Cooperative cancellation token checked between agent steps and tool calls.
#[derive(Clone, Default)]
pub struct CancelToken(pub Arc<std::sync::atomic::AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self(Arc::new(std::sync::atomic::AtomicBool::new(false)))
    }
    pub fn cancel(&self) {
        self.0.store(true, std::sync::atomic::Ordering::SeqCst);
    }
    pub fn reset(&self) {
        self.0.store(false, std::sync::atomic::Ordering::SeqCst);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::SeqCst)
    }
}

pub struct CancelTokenPlugin(pub CancelToken);
impl Plugin for CancelTokenPlugin {
    fn name(&self) -> &str {
        "cancel-token"
    }
    fn apply(&self, scope: &mut Scope) {
        scope.provide::<CancelToken>(self.0.clone());
    }
}

// ---- lifecycle events (observable via Context::emit) ----
pub struct TurnStart;
pub struct StepStart(pub usize);
pub struct ToolStart(pub ToolCall);
pub struct ToolDenied(pub ToolCall);
pub struct ToolResult(pub String, pub String);
pub struct TurnEnd(pub String);

/// Detailed summary of actions, metrics, and outcomes for an agent turn.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct TurnSummary {
    pub duration_ms: u64,
    pub tool_calls: usize,
    pub modified_files: Vec<String>,
    pub lines_added: usize,
    pub lines_deleted: usize,
    pub tests_passed: Option<usize>,
    pub tests_regressions: Option<usize>,
    pub estimated_tokens: Option<usize>,
}

impl TurnSummary {
    pub fn render_line(&self) -> String {
        let mut parts = Vec::new();
        parts.push(format!(
            "turn complete in {:.1}s",
            self.duration_ms as f64 / 1000.0
        ));

        if !self.modified_files.is_empty() {
            let files_word = if self.modified_files.len() == 1 {
                "file"
            } else {
                "files"
            };
            parts.push(format!(
                "{} {} modified (+{}/-{} lines)",
                self.modified_files.len(),
                files_word,
                self.lines_added,
                self.lines_deleted
            ));
        }

        if self.tool_calls > 0 {
            let calls_word = if self.tool_calls == 1 {
                "tool call"
            } else {
                "tool calls"
            };
            parts.push(format!("{} {}", self.tool_calls, calls_word));
        }

        if let Some(passed) = self.tests_passed {
            let reg = self.tests_regressions.unwrap_or(0);
            let reg_str = if reg == 0 {
                "0 regressions".to_string()
            } else {
                format!("{reg} regression(s)!")
            };
            parts.push(format!("tests: {passed} passed ({reg_str})"));
        }

        if let Some(tok) = self.estimated_tokens {
            parts.push(format!("~{tok} tokens"));
        }

        format!("[{}]", parts.join(" · "))
    }
}

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

    /// Extract inline `<tool_call>{"name": ..., "arguments": ...}</tool_call>`, `<function_call>`,
    /// or fenced JSON blocks when a smaller/local model emits tool invocations in plain text instead of native `tool_calls`.
    ///
    /// Returns `(recovered_tool_calls, cleaned_text)` with raw tool markup removed from text.
    pub fn fallback_parse_text_tool_calls(text: &str) -> (Vec<ToolCall>, String) {
        let mut out = Vec::new();
        let mut cleaned = text.to_string();

        // 1. XML-style tool calls: <tool_call>, <function_call>
        for (open_tag, close_tag) in [
            ("<tool_call>", "</tool_call>"),
            ("<function_call>", "</function_call>"),
        ] {
            while let Some(start) = cleaned.find(open_tag) {
                let after = &cleaned[start + open_tag.len()..];
                let Some(end) = after.find(close_tag) else {
                    break;
                };
                let inner = after[..end].trim();
                if let Ok(v) = serde_json::from_str::<Value>(inner) {
                    if let Some(name) = v
                        .get("name")
                        .or_else(|| v.get("tool"))
                        .and_then(|v| v.as_str())
                    {
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
                let total_len = open_tag.len() + end + close_tag.len();
                cleaned.replace_range(start..start + total_len, "");
            }
        }

        if !out.is_empty() {
            return (out, cleaned.trim().to_string());
        }

        // 2. Fenced ```json or ```tool_call blocks containing {"name": ..., "arguments": ...}
        while let Some(fence) = cleaned.find("```") {
            let after_fence = &cleaned[fence + 3..];
            let Some(nl) = after_fence.find('\n') else {
                break;
            };
            let body_start = &after_fence[nl + 1..];
            let Some(close) = body_start.find("```") else {
                break;
            };
            let candidate = body_start[..close].trim();
            if let Ok(v) = serde_json::from_str::<Value>(candidate) {
                if let Some(name) = v
                    .get("name")
                    .or_else(|| v.get("tool"))
                    .and_then(|v| v.as_str())
                {
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
                    let total_len = 3 + nl + 1 + close + 3;
                    cleaned.replace_range(fence..fence + total_len, "");
                    continue;
                }
            }
            break;
        }

        (out, cleaned.trim().to_string())
    }

    /// Run one turn against `history` (mutated in place). Returns the final assistant text.
    pub fn run_turn(&self, history: &mut Vec<Message>, user: &str) -> Result<String> {
        let policy = self
            .ctx
            .service::<checkpoint::CheckpointPolicy>()
            .map(|p| (*p).clone())
            .unwrap_or_default();
        let pre_turn_ckpt = if policy.auto_checkpoint || policy.auto_rollback {
            if let Some(mgr) = self.ctx.service::<checkpoint::CheckpointManager>() {
                // Only auto-snapshot if already inside a git repository or if auto_rollback was explicitly requested
                if mgr.root().join(".git").exists() || policy.auto_rollback {
                    mgr.save("pre-turn automatic checkpoint").ok()
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        self.ctx.emit(&TurnStart);
        let turn_idx = history
            .iter()
            .filter(|m| matches!(m, Message::User { .. }))
            .count();
        let turn_start_instant = std::time::Instant::now();
        let mut turn_tool_calls_count = 0;
        let mut turn_touched_files = std::collections::BTreeSet::new();
        if let Some(recorder) = self.ctx.service::<Arc<recorder::FlightRecorder>>() {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            let _ = recorder.record(&recorder::RecordFrame::TurnStart {
                turn_index: turn_idx,
                user_prompt: user.to_string(),
                timestamp_ms: now_ms,
                pre_checkpoint_id: pre_turn_ckpt.as_ref().map(|c| c.id.clone()),
            });
        }
        if let Some(store) = self.ctx.service::<epistemic::EpistemicStore>() {
            let memory_md = store.get().render_markdown();
            if !memory_md.is_empty() {
                if let Some(Message::System { text }) = history.first_mut() {
                    if !text.contains("## Epistemic Working Memory") {
                        text.push_str("\n\n");
                        text.push_str(&memory_md);
                    }
                }
            }
        }
        history.push(Message::User { text: user.into() });
        self.log
            .append("user", serde_json::json!({ "text": user }))?;
        let specs: Vec<ToolSpec> = self.tools.iter().map(|t| t.spec()).collect();

        let cancel_token = self.ctx.service::<CancelToken>();
        for step in 0..self.max_steps {
            let step_start_instant = std::time::Instant::now();
            if cancel_token.as_ref().is_some_and(|c| c.is_cancelled()) {
                anyhow::bail!("turn cancelled by client");
            }
            self.ctx.emit(&StepStart(step));
            if let Some(compactor) = self.ctx.service::<compaction::AdaptiveCompactor>() {
                if let Some(report) = compactor.compact_history(history) {
                    self.ctx.emit(&compaction::CompactionEvent(report.clone()));
                    let _ = self
                        .log
                        .append("compaction", serde_json::to_value(&report)?);
                }
            }
            if let Some(pruner) = self.ctx.service::<compaction::ToolResultPruner>() {
                pruner.prune_history(history);
            }
            let mut reply = match self
                .llm
                .complete_streaming(history, &specs, &mut |c| self.ctx.emit(&c))
            {
                // Our token estimate was too low. Compact what we can
                // and retry once instead of losing the whole turn.
                Err(error) if is_context_overflow(&error) => {
                    let report = self
                        .ctx
                        .service::<compaction::AdaptiveCompactor>()
                        .and_then(|c| c.force_compact(history))
                        .ok_or(error)?;
                    self.ctx.emit(&compaction::CompactionEvent(report.clone()));
                    let _ = self
                        .log
                        .append("compaction", serde_json::to_value(&report)?);
                    self.llm
                        .complete_streaming(history, &specs, &mut |c| self.ctx.emit(&c))?
                }
                other => other?,
            };
            if cancel_token.as_ref().is_some_and(|c| c.is_cancelled()) {
                anyhow::bail!("turn cancelled by client");
            }
            if let Message::Assistant {
                ref mut text,
                ref mut tool_calls,
            } = reply
            {
                if tool_calls.is_empty() {
                    let (recovered, cleaned) = Self::fallback_parse_text_tool_calls(text);
                    if !recovered.is_empty() {
                        *tool_calls = recovered;
                        *text = cleaned;
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
                if let Some(gate) = self.ctx.service::<plan::VerificationGate>() {
                    if let Err(err) = gate.verify() {
                        let cmd_str = gate.command.as_deref().unwrap_or("verification");
                        if let Some(store) = self.ctx.service::<epistemic::EpistemicStore>() {
                            let summary = err.lines().take(2).collect::<Vec<_>>().join(" ");
                            let _ = store.record_failure(
                                &format!("attempt at step {}", step + 1),
                                &format!("`{cmd_str}` failed: {summary}"),
                            );
                        }
                        let feedback = format!(
                            "[verification gate failed]\nCommand: `{cmd_str}`\n{err}\nDo not finish until verification passes. Fix the issue and verify again."
                        );
                        self.log
                            .append("user", serde_json::json!({ "text": feedback }))?;
                        history.push(Message::User { text: feedback });
                        continue;
                    }
                }
                if let Some(oracle) = self.ctx.service::<Arc<oracle::DiagnosticOracle>>() {
                    if oracle.get_baseline().is_some() {
                        if let Ok(delta) = oracle.check_delta() {
                            if !delta.is_clean {
                                let count = delta.new_regressions.len();
                                let summary = format!("{count} new test regression(s) introduced");
                                if let Some(store) = self.ctx.service::<epistemic::EpistemicStore>()
                                {
                                    let _ = store.record_failure(
                                        &format!("attempt at step {}", step + 1),
                                        &summary,
                                    );
                                }
                                if oracle.mode == oracle::SentinelMode::Strict {
                                    anyhow::bail!(
                                        "turn failed due to test regressions:\n{}",
                                        delta.render_remediation().unwrap_or_default()
                                    );
                                }
                                if oracle.mode == oracle::SentinelMode::Rollback {
                                    if let Some(vfs_svc) =
                                        self.ctx.service::<Arc<vfs::TransactionalVfs>>()
                                    {
                                        let _ = vfs_svc.rollback();
                                    }
                                }
                                let remediation = delta.render_remediation().unwrap_or(summary);
                                let feedback = format!(
                                    "[regression sentinel alert]\n{remediation}\nDo not finish until all regressions are resolved. Fix the issue or revert breaking changes."
                                );
                                self.log
                                    .append("user", serde_json::json!({ "text": feedback }))?;
                                history.push(Message::User { text: feedback });
                                continue;
                            }
                        }
                    }
                }
                if let Some(recorder) = self.ctx.service::<Arc<recorder::FlightRecorder>>() {
                    let now_ms = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0);
                    let _ = recorder.record(&recorder::RecordFrame::TurnEnd {
                        turn_index: turn_idx,
                        status: "success".into(),
                        duration_ms: turn_start_instant.elapsed().as_millis() as u64,
                        timestamp_ms: now_ms,
                    });
                }

                let elapsed_ms = turn_start_instant.elapsed().as_millis() as u64;
                let (modified_files, lines_added, lines_deleted) =
                    if let Some(vfs) = self.ctx.service::<Arc<vfs::TransactionalVfs>>() {
                        let (added, deleted) = vfs.diff_stats();
                        let files: Vec<String> = vfs.status().into_iter().map(|f| f.path).collect();
                        (files, added, deleted)
                    } else {
                        (turn_touched_files.into_iter().collect(), 0, 0)
                    };

                let (tests_passed, tests_regressions) =
                    if let Some(oracle) = self.ctx.service::<Arc<oracle::DiagnosticOracle>>() {
                        if let Some(baseline) = oracle.get_baseline() {
                            if let Ok(delta) = oracle.check_delta() {
                                (Some(baseline.passed), Some(delta.new_regressions.len()))
                            } else {
                                (Some(baseline.passed), None)
                            }
                        } else {
                            (None, None)
                        }
                    } else {
                        (None, None)
                    };

                let estimated_tokens = self
                    .ctx
                    .service::<compaction::AdaptiveCompactor>()
                    .map(|c| c.estimate_history_tokens(history));

                let summary = TurnSummary {
                    duration_ms: elapsed_ms,
                    tool_calls: turn_tool_calls_count,
                    modified_files,
                    lines_added,
                    lines_deleted,
                    tests_passed,
                    tests_regressions,
                    estimated_tokens,
                };
                self.ctx.emit(&summary);
                self.ctx.emit(&TurnEnd(text.clone()));
                return Ok(text);
            }
            let mut step_results = Vec::new();
            for call in &tool_calls {
                turn_tool_calls_count += 1;
                if matches!(
                    call.name.as_str(),
                    "write_file" | "edit_file" | "patch_file"
                ) {
                    if let Some(p) = call.args.get("path").and_then(|v| v.as_str()) {
                        turn_touched_files.insert(p.to_string());
                    }
                }
                if cancel_token.as_ref().is_some_and(|c| c.is_cancelled()) {
                    anyhow::bail!("turn cancelled by client");
                }
                self.ctx.emit(&ToolStart(call.clone()));
                let mut out = self.execute(call);
                if let Some(guard) = self.ctx.service::<guard::RepeatToolGuard>() {
                    if let Some(reminder) = guard.observe(call) {
                        out.push_str(&reminder);
                    }
                }
                if let Some(breaker) = self.ctx.service::<guard::EnvironmentCircuitBreaker>() {
                    if let Some(trip) = breaker.observe(&call.name, &out) {
                        out.push_str(&breaker.format_advisory(&trip));
                    }
                }
                self.log
                    .append("tool", serde_json::json!({"id": call.id, "content": out}))?;
                history.push(Message::Tool {
                    id: call.id.clone(),
                    content: out.clone(),
                });
                step_results.push((call.id.clone(), out));
            }

            if let Some(recorder) = self.ctx.service::<Arc<recorder::FlightRecorder>>() {
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                let post_ckpt =
                    if let Some(mgr) = self.ctx.service::<checkpoint::CheckpointManager>() {
                        mgr.save(&format!("post-step {turn_idx}-{step}"))
                            .ok()
                            .map(|c| c.id)
                    } else {
                        None
                    };
                let _ = recorder.record(&recorder::RecordFrame::StepExecution {
                    turn_index: turn_idx,
                    step_index: step,
                    thinking: None,
                    response_text: text.clone(),
                    tool_calls: tool_calls.clone(),
                    tool_results: step_results,
                    post_checkpoint_id: post_ckpt,
                    duration_ms: step_start_instant.elapsed().as_millis() as u64,
                    timestamp_ms: now_ms,
                });
            }
        }
        if let Some(recorder) = self.ctx.service::<Arc<recorder::FlightRecorder>>() {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            let _ = recorder.record(&recorder::RecordFrame::TurnEnd {
                turn_index: turn_idx,
                status: "step_limit".into(),
                duration_ms: turn_start_instant.elapsed().as_millis() as u64,
                timestamp_ms: now_ms,
            });
        }
        if policy.auto_rollback {
            if let Some(vfs_svc) = self.ctx.service::<Arc<vfs::TransactionalVfs>>() {
                let _ = vfs_svc.rollback();
            }
            if let Some(ckpt) = &pre_turn_ckpt {
                if let Some(mgr) = self.ctx.service::<checkpoint::CheckpointManager>() {
                    let _ = mgr.restore(&ckpt.id);
                }
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
        // Early security sentinel check
        if let Some(sentinel) = self.ctx.service::<Arc<interceptor::StreamingSentinel>>() {
            let mut fields = std::collections::HashMap::new();
            if let Some(obj) = call.args.as_object() {
                for (k, v) in obj {
                    if let Some(s) = v.as_str() {
                        fields.insert(k.clone(), s.to_string());
                    }
                }
            }
            if let Some(veto) = sentinel.inspect(&call.name, &fields) {
                if sentinel.mode() == interceptor::SentinelMode::Strict {
                    self.ctx.emit(&ToolDenied(call.clone()));
                    return format!("denied by streaming security sentinel: {}", veto.reason);
                }
            }
        }
        match self.approver.decide(call, tool.mutating()) {
            Decision::Deny(why) => {
                self.ctx.emit(&ToolDenied(call.clone()));
                format!("denied: {why}")
            }
            Decision::Allow => {
                let vfs_svc = self.ctx.service::<Arc<vfs::TransactionalVfs>>();
                if let Some(ref v) = vfs_svc {
                    if let Some(shadow_res) = v.execute_shadow_tool(&call.name, &call.args) {
                        let out = shadow_res.unwrap_or_else(|e| format!("error: {e:#}"));
                        self.ctx.emit(&ToolResult(call.id.clone(), out.clone()));
                        return out;
                    }
                    if matches!(
                        call.name.as_str(),
                        "write_file" | "edit_file" | "patch_file"
                    ) {
                        if let Some(rel_path) = call.args.get("path").and_then(|val| val.as_str()) {
                            let _ = v.capture_pre_image(rel_path);
                        }
                    }
                }

                // Check predictive pre-warming cache for read_file
                // Skipped in shadow mode: staged edits live only in the VFS overlay, not on disk.
                let shadow_active = vfs_svc
                    .as_ref()
                    .is_some_and(|v| v.mode() == vfs::VfsMode::ShadowMemory);
                let has_slicing = call.args.get("offset").is_some()
                    || call.args.get("limit").is_some()
                    || call.args.get("line_numbers").is_some();
                if call.name == "read_file" && !shadow_active && !has_slicing {
                    if let Some(prewarm) = self.ctx.service::<Arc<interceptor::PrewarmEngine>>() {
                        if let Some(p) = call.args.get("path").and_then(|val| val.as_str()) {
                            if let Some(content) = prewarm.get_warmed(std::path::Path::new(p)) {
                                self.ctx.emit(&ToolResult(call.id.clone(), content.clone()));
                                return content;
                            }
                        }
                    }
                }

                // Tool errors go back to the model as text; they never abort the turn.
                let mut out = tool
                    .run(&call.args)
                    .unwrap_or_else(|e| format!("error: {e:#}"));
                if !out.starts_with("error:")
                    && matches!(
                        call.name.as_str(),
                        "write_file" | "edit_file" | "patch_file"
                    )
                {
                    if let Some(prewarm) = self.ctx.service::<Arc<interceptor::PrewarmEngine>>() {
                        prewarm.invalidate_all();
                    }
                    if let Some(rel_path) = call.args.get("path").and_then(|v| v.as_str()) {
                        if let Some(ref v) = vfs_svc {
                            let _ = v.sync_post_image(rel_path);
                        }
                        if let Some(lsp_svc) = self.ctx.service::<Arc<lsp::LspService>>() {
                            if let Some(diag_feedback) = lsp_svc.post_edit_feedback(rel_path) {
                                out.push_str("\n\n");
                                out.push_str(&diag_feedback);
                            }
                        }
                    }
                }
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

/// True when a provider error says the request did not fit the context
/// window (llama.cpp, OpenAI-style and vLLM wordings).
fn is_context_overflow(error: &anyhow::Error) -> bool {
    let text = format!("{error:#}").to_ascii_lowercase();
    [
        "exceed_context_size",
        "exceeds the available context",
        "context_length_exceeded",
        "maximum context length",
    ]
    .iter()
    .any(|needle| text.contains(needle))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context as _;
    use std::sync::Mutex;

    /// Scripted model: returns queued replies in order.
    struct Script(Mutex<Vec<Message>>);
    impl Llm for Script {
        fn complete(&self, _: &[Message], _: &[ToolSpec]) -> Result<Message> {
            Ok(self.0.lock().unwrap().remove(0))
        }
    }

    /// Fails once with llama.cpp's real overflow error, then answers.
    struct OverflowOnce(Mutex<bool>);
    impl Llm for OverflowOnce {
        fn complete(&self, history: &[Message], _: &[ToolSpec]) -> Result<Message> {
            let mut failed = self.0.lock().unwrap();
            if !*failed {
                *failed = true;
                return Err(anyhow::anyhow!(
                    r#"http://127.0.0.1:8080/v1/chat/completions: status code 400: {{"error":{{"code":400,"message":"request (17000 tokens) exceeds the available context size (16384 tokens), try increasing it","type":"exceed_context_size_error"}}}}"#
                ))
                .context("provider request failed");
            }
            Ok(asst(
                &format!("done with {} messages", history.len()),
                vec![],
            ))
        }
    }

    #[test]
    fn context_overflow_forces_compaction_and_retries_once() {
        let ctx = Context::new();
        let _p = ctx.mount(&compaction::AdaptiveCompactorPlugin::new(
            compaction::AdaptiveCompactor::with_default_summarizer(
                compaction::ContextBudget::builder()
                    .keep_recent_turns(1)
                    .build(),
            ),
        ));
        let llm = OverflowOnce(Mutex::new(false));
        let agent = Agent {
            ctx: &ctx,
            llm: &llm,
            tools: vec![Box::new(Echo)],
            approver: &AllowAll,
            log: &SessionLog::none(),
            max_steps: 2,
        };
        // Small by Keel's estimate, so normal compaction would not run.
        let mut h = vec![Message::System { text: "sys".into() }];
        for i in 0..4 {
            h.push(Message::User {
                text: format!("old question {i}"),
            });
            h.push(asst(&format!("old answer {i}"), vec![]));
        }
        let before = h.len();
        agent.run_turn(&mut h, "new question").unwrap();
        assert!(
            h.len() < before,
            "history was not compacted: {} -> {}",
            before,
            h.len()
        );
        assert!(h
            .iter()
            .any(|m| matches!(m, Message::User { text } if text.contains("Compacted History"))));
        assert!(
            matches!(h.last(), Some(Message::Assistant { text, .. }) if text.starts_with("done"))
        );
    }

    #[test]
    fn other_provider_errors_are_not_treated_as_overflow() {
        let e =
            anyhow::anyhow!("status code 401: invalid api key").context("provider request failed");
        assert!(!is_context_overflow(&e));
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
        assert!(
            matches!(&h[1], Message::Assistant { text, tool_calls } if text == "Let me run echo:" && tool_calls.len() == 1)
        );
        assert!(matches!(&h[2], Message::Tool { content, .. } if content == "recovered"));
    }

    #[test]
    fn fallback_parses_function_call_and_fenced_blocks_with_text_cleaning() {
        let (calls1, clean1) = Agent::fallback_parse_text_tool_calls(
            "Executing query now:\n<function_call>{\"name\":\"grep\",\"arguments\":{\"pattern\":\"User\"}}</function_call>\nWaiting for results.",
        );
        assert_eq!(calls1.len(), 1);
        assert_eq!(calls1[0].name, "grep");
        assert_eq!(calls1[0].args["pattern"], "User");
        assert_eq!(clean1, "Executing query now:\n\nWaiting for results.");

        let (calls2, clean2) = Agent::fallback_parse_text_tool_calls(
            "Here is the change:\n```json\n{\"name\": \"patch_file\", \"arguments\": {\"path\": \"app.rs\"}}\n```\nDone.",
        );
        assert_eq!(calls2.len(), 1);
        assert_eq!(calls2[0].name, "patch_file");
        assert_eq!(calls2[0].args["path"], "app.rs");
        assert_eq!(clean2, "Here is the change:\n\nDone.");
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

    #[test]
    fn verification_gate_forces_continuation_until_passing() {
        let ctx = Context::new();
        // Model tries to finish immediately ("I'm done"), but verification fails.
        // On step 2, model fixes it and finishes ("Fixed"), and verification passes.
        let llm = Script(Mutex::new(vec![
            asst("I'm done", vec![]),
            asst("Fixed", vec![]),
        ]));

        let tmp = std::env::temp_dir().join(format!("keel-verify-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        // Use a gate command that checks an external condition: fail on first run, create marker, pass on second
        let gate_cmd = format!(
            "if [ ! -f {}/pass ]; then touch {}/pass; exit 1; else exit 0; fi",
            tmp.display(),
            tmp.display()
        );
        let _p = ctx.mount(&plan::PlanPlugin {
            root: tmp.clone(),
            verify_command: Some(gate_cmd),
        });

        let agent = Agent {
            ctx: &ctx,
            llm: &llm,
            tools: vec![Box::new(Echo)],
            approver: &AllowAll,
            log: &SessionLog::none(),
            max_steps: 5,
        };
        let mut h = vec![];
        let res = agent.run_turn(&mut h, "please solve it").unwrap();
        assert_eq!(res, "Fixed");
        // Verify that feedback was injected
        assert!(h.iter().any(
            |m| matches!(m, Message::User { text } if text.contains("[verification gate failed]"))
        ));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn session_log_append_and_read_history_roundtrip() {
        let tmp = std::env::temp_dir().join(format!("keel-log-test-{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&tmp);
        let log = SessionLog::file(&tmp);

        log.append("user", serde_json::json!({ "text": "hello world" }))
            .unwrap();
        log.append(
            "assistant",
            serde_json::to_value(Message::Assistant {
                text: "hi there".into(),
                tool_calls: vec![ToolCall {
                    id: "c1".into(),
                    name: "echo".into(),
                    args: serde_json::json!({ "s": "ok" }),
                }],
            })
            .unwrap(),
        )
        .unwrap();
        log.append("tool", serde_json::json!({ "id": "c1", "content": "ok" }))
            .unwrap();

        let history = SessionLog::read_history(&tmp).unwrap();
        assert_eq!(history.len(), 3);
        assert!(matches!(&history[0], Message::User { text } if text == "hello world"));
        assert!(
            matches!(&history[1], Message::Assistant { text, tool_calls } if text == "hi there" && tool_calls.len() == 1)
        );
        assert!(
            matches!(&history[2], Message::Tool { id, content } if id == "c1" && content == "ok")
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn auto_rollback_reverts_workspace_on_turn_failure() {
        let tmp = std::env::temp_dir().join(format!("keel-rollback-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let ctx = Context::new();
        let _p_ckpt = ctx.mount(&checkpoint::CheckpointPlugin {
            root: tmp.clone(),
            auto_checkpoint: true,
            auto_rollback: true,
        });

        // Initialize file
        std::fs::write(tmp.join("state.txt"), "original state\n").unwrap();

        struct Mutator(PathBuf);
        impl Tool for Mutator {
            fn spec(&self) -> ToolSpec {
                ToolSpec {
                    name: "mutate".into(),
                    description: "mutate state".into(),
                    parameters: serde_json::json!({}),
                }
            }
            fn mutating(&self) -> bool {
                true
            }
            fn run(&self, _: &serde_json::Value) -> Result<String> {
                std::fs::write(self.0.join("state.txt"), "corrupted in turn\n")?;
                std::fs::write(self.0.join("new_file.txt"), "unwanted file\n")?;
                Ok("mutated".into())
            }
        }

        // Step 0 calls mutating tool; Step 1 & 2 try to finish but verification fails
        let llm = Script(Mutex::new(vec![
            asst("Mutating", vec![call("c1", "mutate")]),
            asst("Done", vec![]),
            asst("Retry", vec![]),
        ]));

        let _p_plan = ctx.mount(&plan::PlanPlugin {
            root: tmp.clone(),
            verify_command: Some("exit 1".into()),
        });

        let agent = Agent {
            ctx: &ctx,
            llm: &llm,
            tools: vec![Box::new(Mutator(tmp.clone()))],
            approver: &AllowAll,
            log: &SessionLog::none(),
            max_steps: 2,
        };

        let mut h = vec![];
        let res = agent.run_turn(&mut h, "please finish");
        assert!(
            res.is_err(),
            "must fail due to verification gate & step limit"
        );

        // The auto_rollback must have restored state.txt to "original state\n" and deleted new_file.txt!
        assert_eq!(
            std::fs::read_to_string(tmp.join("state.txt")).unwrap(),
            "original state\n"
        );
        assert!(
            !tmp.join("new_file.txt").exists(),
            "new_file.txt should have been rolled back"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn epistemic_memory_injects_into_system_and_captures_gate_failure() {
        let tmp =
            std::env::temp_dir().join(format!("keel-epistemic-test-turn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let ctx = Context::new();
        let _p_epistemic = ctx.mount(&epistemic::EpistemicMemoryPlugin { root: tmp.clone() });
        let _p_plan = ctx.mount(&plan::PlanPlugin {
            root: tmp.clone(),
            verify_command: Some("exit 1".into()),
        });

        let store = ctx.service::<epistemic::EpistemicStore>().unwrap();
        store.add_fact("sqlite DB is in /var/db").unwrap();
        store
            .set_hypothesis(Some("off by one index".into()))
            .unwrap();

        let llm = Script(Mutex::new(vec![
            asst("Trying to finish", vec![]),
            asst("Second attempt", vec![]),
        ]));

        let agent = Agent {
            ctx: &ctx,
            llm: &llm,
            tools: vec![Box::new(Echo)],
            approver: &AllowAll,
            log: &SessionLog::none(),
            max_steps: 2,
        };

        let mut h = vec![Message::System {
            text: "You are a coding assistant.".into(),
        }];

        let _ = agent.run_turn(&mut h, "fix it");

        // 1. Verify system message had epistemic memory injected
        let sys_text = match &h[0] {
            Message::System { text } => text,
            _ => panic!("first message must be system"),
        };
        assert!(sys_text.contains("## Epistemic Working Memory"));
        assert!(sys_text.contains("sqlite DB is in /var/db"));
        assert!(sys_text.contains("off by one index"));

        // 2. Verify that gate failure was recorded as negative knowledge in EpistemicStore
        let state = store.get();
        assert!(!state.failed_attempts.is_empty());
        assert!(state.failed_attempts[0].failure_reason.contains("exit 1"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn json_approval_fails_closed_when_parent_output_is_unavailable() {
        struct BrokenOutput;
        impl std::io::Write for BrokenOutput {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("closed"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let approver = InteractiveApprover::with_io(
            ApprovalPolicy::Standard,
            true,
            Box::new(std::io::Cursor::new(b"allow\n")),
            Box::new(BrokenOutput),
        );
        let call = ToolCall {
            id: "test".into(),
            name: "shell".into(),
            args: serde_json::json!({"command":"true"}),
        };
        assert!(
            matches!(approver.decide(&call,true),Decision::Deny(reason) if reason.contains("could not be delivered"))
        );
    }

    #[test]
    fn interactive_approver_handles_stdin_decisions_and_standard_policy() {
        let input = b"y\nn\na\n";
        let approver = InteractiveApprover::with_reader(
            ApprovalPolicy::Standard,
            true,
            Box::new(std::io::Cursor::new(&input[..])),
        );

        // 1. Non-shell workspace write auto-passes in Standard mode without consuming stdin
        let edit_call = ToolCall {
            id: "1".into(),
            name: "edit_file".into(),
            args: serde_json::json!({"path": "src/main.rs"}),
        };
        assert_eq!(approver.decide(&edit_call, true), Decision::Allow);

        // 2. Sensitive target (.env) requires stdin approval -> consumes "y\n" -> Allow
        let env_call = ToolCall {
            id: "2".into(),
            name: "write_file".into(),
            args: serde_json::json!({"path": ".env"}),
        };
        assert_eq!(approver.decide(&env_call, true), Decision::Allow);

        // 3. Shell tool requires stdin approval -> consumes "n\n" -> Deny
        let shell_call = ToolCall {
            id: "3".into(),
            name: "shell".into(),
            args: serde_json::json!({"command": "cargo test"}),
        };
        assert!(matches!(
            approver.decide(&shell_call, true),
            Decision::Deny(_)
        ));

        // 4. Next shell call -> consumes "a\n" (always) -> Allow and whitelists "shell"
        assert_eq!(approver.decide(&shell_call, true), Decision::Allow);

        // 5. Subsequent shell call auto-passes via session whitelist even though stdin is now EOF!
        assert_eq!(approver.decide(&shell_call, true), Decision::Allow);
    }

    #[test]
    fn post_edit_diagnostics_injected_on_broken_file_write() {
        let tmp = std::env::temp_dir().join(format!("keel-lsp-turn-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let ctx = Context::new();
        let _p_tools = ctx.mount(&tools::BuiltinToolsPlugin {
            root: tmp.clone(),
            landlock_enabled: false,
            allow_protected: false,
        });
        let _p_lsp = ctx.mount(&lsp::LspPlugin {
            root: tmp.clone(),
            extra_servers: std::collections::HashMap::new(),
            auto_post_edit_check: true,
        });

        let llm = Script(Mutex::new(vec![
            asst(
                "Writing config",
                vec![ToolCall {
                    id: "w1".into(),
                    name: "write_file".into(),
                    args: serde_json::json!({
                        "path": "config.json",
                        "content": "{\"broken\": true, }"
                    }),
                }],
            ),
            asst("Fixed config", vec![]),
        ]));

        let _p_llm = ctx.mount(&LlmPlugin(Arc::new(llm)));
        let _p_app = ctx.mount(&ApproverPlugin(Arc::new(AllowAll)));
        let _p_log = ctx.mount(&SessionLogPlugin(SessionLog::none()));

        let mut h = vec![];
        let ans = Agent::run_from_context(&ctx, &mut h, "create config.json", 5).unwrap();
        assert_eq!(ans, "Fixed config");

        let tool_msg = h
            .iter()
            .find_map(|m| match m {
                Message::Tool { content, .. } => Some(content.as_str()),
                _ => None,
            })
            .expect("tool result must be recorded");
        assert!(
            tool_msg.contains("[diagnostics: 1 issue(s) detected in `config.json`]"),
            "expected post-edit diagnostics in tool result, got: {tool_msg}"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn adaptive_compactor_compacts_history_during_agent_run() {
        let tmp =
            std::env::temp_dir().join(format!("keel-compact-agent-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let ctx = Context::new();
        let _p_tools = ctx.mount(&tools::BuiltinToolsPlugin {
            root: tmp.clone(),
            landlock_enabled: false,
            allow_protected: false,
        });

        // Context budget configured with low threshold so intermediate tool turns trigger compaction
        let budget = ContextBudget {
            max_context_tokens: 500,
            compaction_threshold_tokens: 150,
            target_tokens: 80,
            keep_recent_turns: 1,
            chars_per_token: 4.0,
        };
        let _p_compactor = ctx.mount(&AdaptiveCompactorPlugin::new(
            AdaptiveCompactor::with_default_summarizer(budget),
        ));

        let llm = Script(Mutex::new(vec![
            // Step 0: generates large tool output
            asst(
                "Searching codebase",
                vec![ToolCall {
                    id: "s1".into(),
                    name: "write_file".into(),
                    args: serde_json::json!({
                        "path": "step1.txt",
                        "content": "A".repeat(400)
                    }),
                }],
            ),
            // Step 1: generates another tool output (now crossing compaction threshold)
            asst(
                "Inspecting step1",
                vec![ToolCall {
                    id: "s2".into(),
                    name: "read_file".into(),
                    args: serde_json::json!({
                        "path": "step1.txt"
                    }),
                }],
            ),
            // Step 2: final answer
            asst("All tasks completed.", vec![]),
        ]));

        let _p_llm = ctx.mount(&LlmPlugin(Arc::new(llm)));
        let _p_app = ctx.mount(&ApproverPlugin(Arc::new(AllowAll)));
        let _p_log = ctx.mount(&SessionLogPlugin(SessionLog::none()));

        let mut h = vec![];
        let ans = Agent::run_from_context(&ctx, &mut h, "Execute multistep workflow", 5).unwrap();
        assert_eq!(ans, "All tasks completed.");

        // Verify history contains the compacted summary message
        let summary_found = h.iter().any(|m| match m {
            Message::User { text } => {
                text.contains("Context Virtualization: Compacted History Summary")
            }
            _ => false,
        });
        assert!(
            summary_found,
            "Expected history to contain compacted summary"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn session_log_rejects_corruption_instead_of_returning_partial_history() {
        let path =
            std::env::temp_dir().join(format!("keel-corrupt-log-{}.jsonl", std::process::id()));
        for bad_record in [
            "not-json",
            r#"{"kind":"user","data":{}}"#,
            r#"{"kind":"assistant","data":{"role":"user","text":"wrong role"}}"#,
            r#"{"kind":"tool","data":{"content":"missing id"}}"#,
        ] {
            let content = format!(
                "{{\"kind\":\"user\",\"data\":{{\"text\":\"first turn\"}}}}\n{bad_record}\n"
            );
            std::fs::write(&path, &content).unwrap();
            let error = SessionLog::read_history(&path).unwrap_err().to_string();
            assert!(error.contains("line 2"), "{error}");
            assert_eq!(std::fs::read_to_string(&path).unwrap(), content);
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn session_log_does_not_silently_truncate_at_message_limit() {
        let path =
            std::env::temp_dir().join(format!("keel-limit-log-{}.jsonl", std::process::id()));
        let record = "{\"kind\":\"user\",\"data\":{\"text\":\"turn\"}}\n";
        std::fs::write(&path, record.repeat(MAX_LOG_MESSAGES + 1)).unwrap();
        assert!(SessionLog::read_history(&path)
            .unwrap_err()
            .to_string()
            .contains("conversation messages"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn session_log_read_history_safety() {
        let tmp = std::env::temp_dir().join(format!("keel-log-safety-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        // 1. Non-existent file errors
        assert!(SessionLog::read_history(tmp.join("missing.jsonl")).is_err());

        // 2. Directory path errors
        assert!(SessionLog::read_history(&tmp).is_err());

        // 3. Valid session log roundtrips
        let log_file = tmp.join("session.jsonl");
        let log = SessionLog::file(&log_file);
        log.append("user", serde_json::json!({"text": "Hello"}))
            .unwrap();
        log.append(
            "assistant",
            serde_json::to_value(Message::Assistant {
                text: "Hi".into(),
                tool_calls: vec![],
            })
            .unwrap(),
        )
        .unwrap();

        let msgs = SessionLog::read_history(&log_file).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(
            msgs[0],
            Message::User {
                text: "Hello".into()
            }
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn sentinel_gate_forces_continuation_on_regression() {
        let ctx = Context::new();
        let llm = Script(Mutex::new(vec![
            asst("I'm done", vec![]),
            asst("Fixed all regressions", vec![]),
        ]));

        let tmp = std::env::temp_dir().join(format!("keel-sentinel-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        // Marker script: baseline run succeeds (passes), first check introduces failure, second check passes
        let test_cmd = format!(
            "if [ ! -f {0}/step1 ]; then touch {0}/step1; echo 'running 1 tests\ntest test_a ... ok\ntest result: ok. 1 passed; 0 failed;'; elif [ ! -f {0}/step2 ]; then touch {0}/step2; echo 'running 1 tests\ntest test_a ... FAILED\ntest result: FAILED. 0 passed; 1 failed;'; else echo 'running 1 tests\ntest test_a ... ok\ntest result: ok. 1 passed; 0 failed;'; fi",
            tmp.display()
        );

        let _p = ctx.mount(&oracle::OraclePlugin {
            root: tmp.clone(),
            test_command: Some(test_cmd),
            mode: oracle::SentinelMode::Warn,
            auto_baseline: true,
        });

        let agent = Agent {
            ctx: &ctx,
            llm: &llm,
            tools: vec![Box::new(Echo)],
            approver: &AllowAll,
            log: &SessionLog::none(),
            max_steps: 5,
        };
        let mut h = vec![];
        let res = agent.run_turn(&mut h, "implement feature").unwrap();
        assert_eq!(res, "Fixed all regressions");
        assert!(h.iter().any(
            |m| matches!(m, Message::User { text } if text.contains("[regression sentinel alert]"))
        ));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn turn_summary_render_line_formatting() {
        let summary = TurnSummary {
            duration_ms: 1250,
            tool_calls: 3,
            modified_files: vec!["src/lib.rs".into(), "src/main.rs".into()],
            lines_added: 18,
            lines_deleted: 4,
            tests_passed: Some(15),
            tests_regressions: Some(0),
            estimated_tokens: Some(3200),
        };
        let line = summary.render_line();
        assert!(line.contains("turn complete in 1.2s"));
        assert!(line.contains("2 files modified (+18/-4 lines)"));
        assert!(line.contains("3 tool calls"));
        assert!(line.contains("tests: 15 passed (0 regressions)"));
        assert!(line.contains("~3200 tokens"));
    }
}
