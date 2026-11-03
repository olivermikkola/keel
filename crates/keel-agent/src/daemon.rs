//! Agent Client Protocol (ACP) Bidirectional JSON-RPC 2.0 Daemon (`keel daemon`).
//!
//! Turns `keel` into a long-lived, multiplexed JSON-RPC 2.0 server over `stdio`,
//! persistent Unix domain socket (`--socket <path>`), or TCP (`--listen <addr>`)
//! for IDE extensions, Zed/Neovim clients, and orchestrators like Helm.
//!
//! Supported JSON-RPC 2.0 methods:
//! - `initialize`: capability negotiation & optional bearer token authentication
//! - `ping`: keepalive health probe
//! - `session/new` / `session/load`: initialize or resume conversation history
//! - `session/list`: enumerate active multiplexed sessions and statuses
//! - `session/close`: terminate and unload a session
//! - `session/prompt`: run an agent turn asynchronously while streaming `session/update`
//!   notifications (`text`, `reasoning`, `tool_call`, `tool_result`, `tool_denied`)
//! - `session/cancel`: cooperatively cancel an in-flight turn via [`CancelToken`]
//! - `approval/respond`: resolve an in-flight HITL `approval/request`
//! - `vfs/status`, `vfs/diff`, `vfs/commit`, `vfs/rollback`: transactional VFS control
//! - `checkpoint/list`, `checkpoint/save`, `checkpoint/restore`: git shadow checkpoints
//! - `memory/show`, `memory/add_fact`, `memory/set_hypothesis`, `memory/record_failure`, `memory/clear`: epistemic memory
//! - `shutdown`: gracefully terminate daemon process

use crate::checkpoint::{CheckpointManager, CheckpointPlugin};
use crate::compaction::{ToolResultPrunerPlugin, WorkspaceContextPlugin, WorkspaceInstructions};
use crate::epistemic::{EpistemicMemoryPlugin, EpistemicStore};
use crate::guard::RepeatToolReminderPlugin;
use crate::lsp::LspPlugin;
use crate::oracle::{DiagnosticOracle, OraclePlugin, SentinelMode};
use crate::plan::PlanPlugin;
use crate::sandbox::LandlockPlugin;
use crate::speculate::SpeculatePlugin;
use crate::subagent::SubagentPlugin;
use crate::symbols::{SymbolIndex, SymbolPlugin};
use crate::tools::BuiltinToolsPlugin;
use crate::tree::{ExplorationTree, TreePlugin};
use crate::vfs::{TransactionalVfs, VfsMode, VfsPlugin};
use crate::{
    Agent, ApprovalPolicy, Approver, ApproverPlugin, CancelToken, CancelTokenPlugin, Chunk,
    Decision, Llm, LlmPlugin, Message, SessionLog, SessionLogPlugin, ToolCall, ToolDenied,
    ToolResult, ToolStart,
};
use anyhow::{Context as _, Result};
use keel_kernel::{Context, Plugin, Scope};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

#[derive(Clone)]
pub struct DaemonConfig {
    pub root: PathBuf,
    pub policy: ApprovalPolicy,
    pub landlock_enabled: bool,
    pub vfs_mode: VfsMode,
    pub verify_command: Option<String>,
}

/// Thread-safe JSON-RPC 2.0 writer that emits newline-delimited JSON messages.
pub struct RpcWriter {
    inner: Mutex<Box<dyn Write + Send>>,
    closed: AtomicBool,
}

impl RpcWriter {
    pub fn new(w: Box<dyn Write + Send>) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(w),
            closed: AtomicBool::new(false),
        })
    }

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Relaxed)
    }

    pub fn mark_closed(&self) {
        self.closed.store(true, Ordering::Relaxed);
    }

    pub fn send_response(&self, id: Value, result: Value) {
        let msg = json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result,
        });
        self.write_raw(&msg);
    }

    pub fn send_error(&self, id: Value, code: i64, message: &str) {
        let msg = json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {
                "code": code,
                "message": message,
            }
        });
        self.write_raw(&msg);
    }

    pub fn send_notification(&self, method: &str, params: Value) {
        let msg = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        self.write_raw(&msg);
    }

    fn write_raw(&self, val: &Value) {
        if self.is_closed() {
            return;
        }
        if let Ok(mut guard) = self.inner.lock() {
            if writeln!(guard, "{val}").is_err() || guard.flush().is_err() {
                self.closed.store(true, Ordering::Relaxed);
            }
        }
    }
}

/// Multiplexed broadcaster capable of sending JSON-RPC notifications to all connected clients.
#[derive(Clone, Default)]
pub struct RpcBroadcaster {
    clients: Arc<Mutex<Vec<Arc<RpcWriter>>>>,
}

impl RpcBroadcaster {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, writer: Arc<RpcWriter>) {
        if let Ok(mut g) = self.clients.lock() {
            g.push(writer);
        }
    }

    pub fn unregister(&self, writer: &Arc<RpcWriter>) {
        if let Ok(mut g) = self.clients.lock() {
            g.retain(|w| !Arc::ptr_eq(w, writer));
        }
    }

    pub fn send_notification(&self, method: &str, params: Value) {
        if let Ok(mut g) = self.clients.lock() {
            g.retain(|w| !w.is_closed());
            for w in g.iter() {
                w.send_notification(method, params.clone());
            }
        }
    }
}

#[derive(Default)]
struct ApprovalHubState {
    decisions: HashMap<String, String>,
    whitelisted: HashSet<String>,
    pending: HashSet<String>,
}

/// RPC-backed HITL approver that emits `approval/request` notifications and awaits
/// `approval/respond` (or `session/cancel`) from connected clients.
pub struct AcpApprover {
    policy: ApprovalPolicy,
    broadcaster: RpcBroadcaster,
    cancel_token: CancelToken,
    session_id: String,
    state: Arc<(Mutex<ApprovalHubState>, Condvar)>,
}

impl AcpApprover {
    pub fn new(
        policy: ApprovalPolicy,
        broadcaster: RpcBroadcaster,
        cancel_token: CancelToken,
        session_id: String,
    ) -> Arc<Self> {
        Arc::new(Self {
            policy,
            broadcaster,
            cancel_token,
            session_id,
            state: Arc::new((Mutex::new(ApprovalHubState::default()), Condvar::new())),
        })
    }

    pub fn resolve(&self, call_id: &str, decision: &str) {
        let (lock, cvar) = &*self.state;
        if let Ok(mut guard) = lock.lock() {
            guard
                .decisions
                .insert(call_id.to_string(), decision.to_ascii_lowercase());
            cvar.notify_all();
        }
    }

    pub fn is_pending(&self, call_id: &str) -> bool {
        let (lock, _) = &*self.state;
        if let Ok(guard) = lock.lock() {
            guard.pending.contains(call_id)
        } else {
            false
        }
    }

    pub fn notify_cancelled(&self) {
        let (_, cvar) = &*self.state;
        cvar.notify_all();
    }

    fn is_sensitive(call: &ToolCall) -> bool {
        let target = call
            .args
            .get("path")
            .or_else(|| call.args.get("file"))
            .or_else(|| call.args.get("command"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        target.contains("/etc/")
            || target.contains(".ssh/")
            || target.contains(".bashrc")
            || target.contains(".zshrc")
            || target.contains(".profile")
            || target.contains(".env")
    }
}

impl Approver for AcpApprover {
    fn decide(&self, call: &ToolCall, mutating: bool) -> Decision {
        if !mutating || self.policy == ApprovalPolicy::Yolo {
            return Decision::Allow;
        }
        if self.policy == ApprovalPolicy::ReadOnly {
            return Decision::Deny(format!("`{}` denied: read-only policy", call.name));
        }
        let (lock, cvar) = &*self.state;
        if let Ok(guard) = lock.lock() {
            if guard.whitelisted.contains(&call.name) {
                return Decision::Allow;
            }
        }
        if self.policy == ApprovalPolicy::Standard
            && call.name != "shell"
            && !Self::is_sensitive(call)
        {
            return Decision::Allow;
        }

        if let Ok(mut guard) = lock.lock() {
            guard.pending.insert(call.id.clone());
        }

        self.broadcaster.send_notification(
            "approval/request",
            json!({
                "session_id": self.session_id,
                "call_id": call.id,
                "tool": call.name,
                "arguments": call.args,
            }),
        );

        let Ok(mut guard) = lock.lock() else {
            return Decision::Deny("approval mutex poisoned".into());
        };
        let deadline = std::time::Instant::now() + Duration::from_secs(300);
        loop {
            if self.cancel_token.is_cancelled() {
                guard.pending.remove(&call.id);
                return Decision::Deny("turn cancelled while awaiting approval".into());
            }
            if let Some(dec) = guard.decisions.remove(&call.id) {
                guard.pending.remove(&call.id);
                return match dec.as_str() {
                    "allow" | "allow_once" | "y" | "yes" => Decision::Allow,
                    "allow_always" | "always" | "a" => {
                        guard.whitelisted.insert(call.name.clone());
                        Decision::Allow
                    }
                    _ => Decision::Deny(format!("`{}` denied by client", call.name)),
                };
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                guard.pending.remove(&call.id);
                return Decision::Deny(format!("`{}` approval timed out", call.name));
            }
            let Ok((g, _)) = cvar.wait_timeout(guard, Duration::from_millis(100)) else {
                return Decision::Deny("approval wait interrupted".into());
            };
            guard = g;
        }
    }
}

struct AcpEventStreamer {
    broadcaster: RpcBroadcaster,
    session_id: String,
}

impl Plugin for AcpEventStreamer {
    fn name(&self) -> &str {
        "acp-event-streamer"
    }

    fn apply(&self, scope: &mut Scope) {
        let b1 = self.broadcaster.clone();
        let sid1 = self.session_id.clone();
        scope.on::<Chunk, _>(move |c| match c {
            Chunk::Text(t) => {
                b1.send_notification(
                    "session/update",
                    json!({
                        "session_id": sid1,
                        "type": "text",
                        "content": t,
                    }),
                );
            }
            Chunk::Reasoning(t) => {
                b1.send_notification(
                    "session/update",
                    json!({
                        "session_id": sid1,
                        "type": "reasoning",
                        "content": t,
                    }),
                );
            }
            Chunk::ToolCallDelta {
                index,
                id,
                name,
                args_delta,
            } => {
                b1.send_notification(
                    "session/update",
                    json!({
                        "session_id": sid1,
                        "type": "tool_call_delta",
                        "index": index,
                        "id": id,
                        "name": name,
                        "args_delta": args_delta,
                    }),
                );
            }
            Chunk::SentinelVeto { tool_name, reason } => {
                b1.send_notification(
                    "session/update",
                    json!({
                        "session_id": sid1,
                        "type": "sentinel_veto",
                        "tool_name": tool_name,
                        "reason": reason,
                    }),
                );
            }
        });

        let b2 = self.broadcaster.clone();
        let sid2 = self.session_id.clone();
        scope.on::<ToolStart, _>(move |ts| {
            b2.send_notification(
                "session/update",
                json!({
                    "session_id": sid2,
                    "type": "tool_call",
                    "id": ts.0.id,
                    "name": ts.0.name,
                    "arguments": ts.0.args,
                }),
            );
        });

        let b3 = self.broadcaster.clone();
        let sid3 = self.session_id.clone();
        scope.on::<ToolResult, _>(move |tr| {
            b3.send_notification(
                "session/update",
                json!({
                    "session_id": sid3,
                    "type": "tool_result",
                    "id": tr.0,
                    "output": tr.1,
                }),
            );
        });

        let b4 = self.broadcaster.clone();
        let sid4 = self.session_id.clone();
        scope.on::<ToolDenied, _>(move |td| {
            b4.send_notification(
                "session/update",
                json!({
                    "session_id": sid4,
                    "type": "tool_denied",
                    "id": td.0.id,
                    "name": td.0.name,
                }),
            );
        });
    }
}

/// A managed conversation session inside the ACP daemon.
pub struct SessionEntry {
    pub id: String,
    pub ctx: Arc<Context>,
    pub history: Arc<Mutex<Vec<Message>>>,
    pub cancel_token: CancelToken,
    pub approver: Arc<AcpApprover>,
    pub busy: Arc<AtomicBool>,
    pub created_at: u64,
    pub updated_at: Arc<Mutex<u64>>,
    pub mounted: Mutex<Vec<keel_kernel::Mounted>>,
}

/// Thread-safe registry managing multiplexed sessions for the ACP daemon.
pub struct SessionRegistry {
    sessions: Mutex<HashMap<String, Arc<SessionEntry>>>,
}

impl Default for SessionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionRegistry {
    pub fn new() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
        }
    }

    pub fn get(&self, id: &str) -> Option<Arc<SessionEntry>> {
        self.sessions.lock().unwrap().get(id).cloned()
    }

    pub fn get_or_create(
        &self,
        id: &str,
        cfg: &DaemonConfig,
        llm: Arc<dyn Llm>,
        broadcaster: RpcBroadcaster,
    ) -> Arc<SessionEntry> {
        let mut g = self.sessions.lock().unwrap();
        if let Some(entry) = g.get(id) {
            return entry.clone();
        }
        let entry = Arc::new(create_session(id.to_string(), cfg, llm, broadcaster));
        g.insert(id.to_string(), entry.clone());
        entry
    }

    pub fn close(&self, id: &str) -> bool {
        let mut g = self.sessions.lock().unwrap();
        if let Some(entry) = g.remove(id) {
            entry.cancel_token.cancel();
            entry.approver.notify_cancelled();
            if let Ok(mut m) = entry.mounted.lock() {
                while let Some(h) = m.pop() {
                    h.unload();
                }
            }
            true
        } else {
            false
        }
    }

    pub fn list(&self) -> Vec<Value> {
        let g = self.sessions.lock().unwrap();
        let mut out = Vec::new();
        for (id, entry) in g.iter() {
            let msg_count = entry.history.lock().map(|h| h.len()).unwrap_or(0);
            let busy = entry.busy.load(Ordering::SeqCst);
            let updated_at = entry
                .updated_at
                .lock()
                .map(|t| *t)
                .unwrap_or(entry.created_at);
            out.push(json!({
                "session_id": id,
                "message_count": msg_count,
                "busy": busy,
                "created_at": entry.created_at,
                "updated_at": updated_at,
            }));
        }
        out.sort_by(|a, b| {
            a["session_id"]
                .as_str()
                .unwrap_or("")
                .cmp(b["session_id"].as_str().unwrap_or(""))
        });
        out
    }

    pub fn find_session_with_pending_approval(&self, call_id: &str) -> Option<Arc<SessionEntry>> {
        let g = self.sessions.lock().unwrap();
        for entry in g.values() {
            if entry.approver.is_pending(call_id) {
                return Some(entry.clone());
            }
        }
        None
    }
}

fn create_session(
    id: String,
    cfg: &DaemonConfig,
    llm: Arc<dyn Llm>,
    broadcaster: RpcBroadcaster,
) -> SessionEntry {
    let cancel_token = CancelToken::new();
    let approver = AcpApprover::new(
        cfg.policy,
        broadcaster.clone(),
        cancel_token.clone(),
        id.clone(),
    );
    let ctx = Arc::new(Context::new());
    let mounted = vec![
        ctx.mount(&AcpEventStreamer {
            broadcaster: broadcaster.clone(),
            session_id: id.clone(),
        }),
        ctx.mount(&LlmPlugin(llm)),
        ctx.mount(&ApproverPlugin(approver.clone())),
        ctx.mount(&CancelTokenPlugin(cancel_token.clone())),
        ctx.mount(&SessionLogPlugin(SessionLog::none())),
        ctx.mount(&LandlockPlugin {
            workspace: cfg.root.clone(),
            enabled: cfg.landlock_enabled,
        }),
        ctx.mount(&BuiltinToolsPlugin {
            root: cfg.root.clone(),
            landlock_enabled: cfg.landlock_enabled,
        }),
        ctx.mount(&RepeatToolReminderPlugin::default()),
        ctx.mount(&ToolResultPrunerPlugin::default()),
        ctx.mount(&WorkspaceContextPlugin {
            root: cfg.root.clone(),
        }),
        ctx.mount(&PlanPlugin {
            root: cfg.root.clone(),
            verify_command: cfg.verify_command.clone(),
        }),
        ctx.mount(&SubagentPlugin {
            root: cfg.root.clone(),
            landlock_enabled: cfg.landlock_enabled,
        }),
        ctx.mount(&SpeculatePlugin {
            root: cfg.root.clone(),
            landlock_enabled: cfg.landlock_enabled,
        }),
        ctx.mount(&CheckpointPlugin {
            root: cfg.root.clone(),
            auto_checkpoint: true,
            auto_rollback: false,
        }),
        ctx.mount(&EpistemicMemoryPlugin {
            root: cfg.root.clone(),
        }),
        ctx.mount(&LspPlugin {
            root: cfg.root.clone(),
            extra_servers: HashMap::new(),
            auto_post_edit_check: cfg.vfs_mode != VfsMode::ShadowMemory,
        }),
        ctx.mount(&VfsPlugin {
            root: cfg.root.clone(),
            mode: cfg.vfs_mode,
            persist: cfg.vfs_mode == VfsMode::MaterializedCow,
        }),
        ctx.mount(&OraclePlugin {
            root: cfg.root.clone(),
            test_command: cfg.verify_command.clone(),
            mode: SentinelMode::Warn,
            auto_baseline: false,
        }),
        ctx.mount(&SymbolPlugin {
            root: cfg.root.clone(),
        }),
        ctx.mount(&TreePlugin {
            root: cfg.root.clone(),
        }),
        ctx.mount(&crate::interceptor::SentinelPlugin(Arc::new(
            crate::interceptor::StreamingSentinel::default(),
        ))),
        ctx.mount(&crate::interceptor::PrewarmPlugin(Arc::new(
            crate::interceptor::PrewarmEngine::new(cfg.root.clone()),
        ))),
    ];

    let mut sys_text = format!(
        "You are a coding agent working in {}. Use tools to inspect and change files. Be concise.",
        cfg.root.display()
    );
    if let Some(ws) = ctx.service::<WorkspaceInstructions>() {
        if !ws.extra_system_prompt.is_empty() {
            sys_text.push_str("\n\n");
            sys_text.push_str(&ws.extra_system_prompt);
        }
    }
    let history = Arc::new(Mutex::new(vec![Message::System { text: sys_text }]));
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    SessionEntry {
        id,
        ctx,
        history,
        cancel_token,
        approver,
        busy: Arc::new(AtomicBool::new(false)),
        created_at: now,
        updated_at: Arc::new(Mutex::new(now)),
        mounted: Mutex::new(mounted),
    }
}

/// Shared daemon server state and configuration.
#[derive(Clone)]
pub struct DaemonServer {
    pub cfg: DaemonConfig,
    pub llm: Arc<dyn Llm>,
    pub auth_token: Option<String>,
    pub broadcaster: RpcBroadcaster,
    pub registry: Arc<SessionRegistry>,
    pub shutdown_signal: Arc<AtomicBool>,
}

impl DaemonServer {
    pub fn new(cfg: DaemonConfig, llm: Arc<dyn Llm>, auth_token: Option<String>) -> Self {
        let broadcaster = RpcBroadcaster::new();
        let registry = Arc::new(SessionRegistry::new());
        let shutdown_signal = Arc::new(AtomicBool::new(false));
        let default_sid = format!("acp-{}", std::process::id());
        registry.get_or_create(&default_sid, &cfg, llm.clone(), broadcaster.clone());

        Self {
            cfg,
            llm,
            auth_token,
            broadcaster,
            registry,
            shutdown_signal,
        }
    }
}

/// Run ACP JSON-RPC client connection loop over `reader` and `writer`.
pub fn handle_acp_client<R: BufRead>(
    mut reader: R,
    writer: Box<dyn Write + Send>,
    server: DaemonServer,
    default_session_id: String,
) -> Result<()> {
    let rpc_writer = RpcWriter::new(writer);
    server.broadcaster.register(rpc_writer.clone());

    let current_session_id = Arc::new(Mutex::new(default_session_id));
    let mut authenticated = server.auth_token.is_none();
    let mut active_workers: Vec<std::thread::JoinHandle<()>> = Vec::new();

    loop {
        if server.shutdown_signal.load(Ordering::Relaxed) {
            break;
        }
        let mut line = String::new();
        let n = match reader.read_line(&mut line) {
            Ok(n) => n,
            Err(_) => break,
        };
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let msg: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                rpc_writer.send_error(Value::Null, -32700, &format!("Parse error: {e}"));
                continue;
            }
        };

        let id = msg.get("id").cloned().unwrap_or(Value::Null);
        let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let params = msg.get("params").cloned().unwrap_or_else(|| json!({}));

        // Authentication enforcement
        if !authenticated {
            let provided_token = params
                .get("token")
                .or_else(|| params.get("auth_token"))
                .and_then(|v| v.as_str())
                .or_else(|| {
                    msg.get("auth")
                        .and_then(|a| a.get("token"))
                        .and_then(|v| v.as_str())
                });
            if let Some(expected) = &server.auth_token {
                if provided_token == Some(expected.as_str()) {
                    authenticated = true;
                } else if method == "initialize" || method == "auth" || method == "authenticate" {
                    rpc_writer.send_error(
                        id,
                        -32001,
                        "Unauthorized: invalid or missing daemon authentication token",
                    );
                    continue;
                } else {
                    rpc_writer.send_error(
                        id,
                        -32001,
                        "Unauthorized: please provide valid token via initialize or auth",
                    );
                    continue;
                }
            } else {
                authenticated = true;
            }
        }

        match method {
            "initialize" => {
                rpc_writer.send_response(
                    id,
                    json!({
                        "protocolVersion": "2026-11-01",
                        "serverInfo": {
                            "name": "keel-acp-daemon",
                            "version": "0.2.0",
                        },
                        "capabilities": {
                            "streaming": true,
                            "cancellation": true,
                            "approvals": true,
                            "vfs": true,
                            "checkpoints": true,
                            "speculation": true,
                            "epistemicMemory": true,
                            "epistemicGraph": true,
                            "persistentDaemon": true,
                            "sessionMultiplexing": true,
                            "swarmHandoff": true,
                            "diagnosticOracle": true,
                            "regressionSentinel": true,
                            "structuralSymbols": true,
                            "symbolOutline": true,
                            "explorationTree": true,
                            "multiStepBacktracking": true,
                            "streamingSentinel": true,
                            "predictivePrewarm": true,
                        }
                    }),
                );
            }
            "ping" => {
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0);
                rpc_writer.send_response(id, json!({ "status": "pong", "timestamp": now_ms }));
            }
            "auth" | "authenticate" => {
                rpc_writer.send_response(id, json!({ "status": "authenticated" }));
            }
            "session/list" => {
                rpc_writer.send_response(id, json!({ "sessions": server.registry.list() }));
            }
            "session/new" => {
                let new_sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| {
                        let ts = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_millis())
                            .unwrap_or(0);
                        format!("acp-{}-{}", std::process::id(), ts)
                    });
                let session = server.registry.get_or_create(
                    &new_sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                let mut sys_text = format!(
                    "You are a coding agent working in {}. Use tools to inspect and change files. Be concise.",
                    server.cfg.root.display()
                );
                if let Some(ws) = session.ctx.service::<WorkspaceInstructions>() {
                    if !ws.extra_system_prompt.is_empty() {
                        sys_text.push_str("\n\n");
                        sys_text.push_str(&ws.extra_system_prompt);
                    }
                }
                if let Ok(mut h) = session.history.lock() {
                    h.clear();
                    h.push(Message::System { text: sys_text });
                }
                if let Ok(mut s) = current_session_id.lock() {
                    *s = new_sid.clone();
                }
                rpc_writer.send_response(id, json!({ "session_id": new_sid, "status": "ok" }));
            }
            "session/load" => {
                let Some(log_path) = params.get("path").and_then(|v| v.as_str()) else {
                    rpc_writer.send_error(id, -32602, "missing `path` parameter");
                    continue;
                };
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                match SessionLog::read_history(log_path) {
                    Ok(prev) => {
                        let count = prev.len();
                        if let Ok(mut h) = session.history.lock() {
                            h.truncate(1);
                            h.extend(prev);
                        }
                        if let Ok(mut curr) = current_session_id.lock() {
                            *curr = sid.clone();
                        }
                        rpc_writer.send_response(
                            id,
                            json!({
                                "status": "ok",
                                "session_id": sid,
                                "loaded_messages": count,
                            }),
                        );
                    }
                    Err(e) => {
                        rpc_writer.send_error(id, -32000, &format!("failed to load session: {e:#}"))
                    }
                }
            }
            "session/close" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let closed = server.registry.close(&sid);
                rpc_writer.send_response(
                    id,
                    json!({ "status": "closed", "session_id": sid, "found": closed }),
                );
            }
            "session/cancel" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                if let Some(session) = server.registry.get(&sid) {
                    session.cancel_token.cancel();
                    session.approver.notify_cancelled();
                    if !id.is_null() {
                        rpc_writer
                            .send_response(id, json!({ "status": "cancelled", "session_id": sid }));
                    }
                } else if !id.is_null() {
                    rpc_writer.send_error(id, -32602, &format!("Session `{sid}` not found"));
                }
            }
            "approval/respond" => {
                let call_id = params.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                let decision = params
                    .get("decision")
                    .and_then(|v| v.as_str())
                    .unwrap_or("deny");
                let session_id_opt = params.get("session_id").and_then(|v| v.as_str());

                let target_session = if let Some(sid) = session_id_opt {
                    server.registry.get(sid)
                } else if let Some(s) = server.registry.find_session_with_pending_approval(call_id)
                {
                    Some(s)
                } else {
                    let curr = current_session_id.lock().unwrap().clone();
                    server.registry.get(&curr)
                };

                if let Some(session) = target_session {
                    session.approver.resolve(call_id, decision);
                    if !id.is_null() {
                        rpc_writer.send_response(
                            id,
                            json!({
                                "status": "ok",
                                "call_id": call_id,
                                "decision": decision,
                                "session_id": session.id,
                            }),
                        );
                    }
                } else if !id.is_null() {
                    rpc_writer.send_error(
                        id,
                        -32602,
                        &format!("No session found for approval call `{call_id}`"),
                    );
                }
            }
            "session/prompt" => {
                let Some(prompt_text) = params
                    .get("prompt")
                    .or_else(|| params.get("text"))
                    .and_then(|v| v.as_str())
                    .map(String::from)
                else {
                    rpc_writer.send_error(id, -32602, "missing `prompt` string parameter");
                    continue;
                };
                let max_steps = params
                    .get("max_steps")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(40) as usize;

                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );

                if session
                    .busy
                    .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                    .is_err()
                {
                    rpc_writer.send_error(
                        id,
                        -32002,
                        &format!("Session `{sid}` is already executing an in-flight prompt"),
                    );
                    continue;
                }

                session.cancel_token.reset();
                let session_clone = session.clone();
                let writer_clone = rpc_writer.clone();
                let sid_snapshot = sid.clone();

                let handle = std::thread::spawn(move || {
                    let result = {
                        let mut h_guard = match session_clone.history.lock() {
                            Ok(g) => g,
                            Err(_) => {
                                session_clone.busy.store(false, Ordering::SeqCst);
                                writer_clone.send_error(
                                    id,
                                    -32000,
                                    "session history mutex poisoned",
                                );
                                return;
                            }
                        };
                        Agent::run_from_context(
                            &session_clone.ctx,
                            &mut h_guard,
                            &prompt_text,
                            max_steps,
                        )
                    };
                    session_clone.busy.store(false, Ordering::SeqCst);
                    if let Ok(mut u) = session_clone.updated_at.lock() {
                        *u = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_secs())
                            .unwrap_or(0);
                    }
                    match result {
                        Ok(answer) => {
                            writer_clone.send_response(
                                id,
                                json!({
                                    "session_id": sid_snapshot,
                                    "status": "ok",
                                    "answer": answer,
                                }),
                            );
                        }
                        Err(e) => {
                            writer_clone.send_error(id, -32000, &format!("{e:#}"));
                        }
                    }
                });
                active_workers.push(handle);
            }
            "vfs/status" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(vfs) = session.ctx.service::<Arc<TransactionalVfs>>() {
                    rpc_writer
                        .send_response(id, json!({ "session_id": sid, "files": vfs.status() }));
                } else {
                    rpc_writer.send_error(id, -32000, "VFS service not mounted");
                }
            }
            "vfs/diff" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(vfs) = session.ctx.service::<Arc<TransactionalVfs>>() {
                    rpc_writer.send_response(id, json!({ "session_id": sid, "diff": vfs.diff() }));
                } else {
                    rpc_writer.send_error(id, -32000, "VFS service not mounted");
                }
            }
            "vfs/commit" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(vfs) = session.ctx.service::<Arc<TransactionalVfs>>() {
                    match vfs.commit() {
                        Ok(rep) => rpc_writer.send_response(id, serde_json::to_value(rep)?),
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "VFS service not mounted");
                }
            }
            "vfs/rollback" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(vfs) = session.ctx.service::<Arc<TransactionalVfs>>() {
                    match vfs.rollback() {
                        Ok(rep) => rpc_writer.send_response(id, serde_json::to_value(rep)?),
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "VFS service not mounted");
                }
            }
            "checkpoint/list" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(mgr) = session.ctx.service::<CheckpointManager>() {
                    match mgr.list() {
                        Ok(list) => rpc_writer
                            .send_response(id, json!({ "session_id": sid, "checkpoints": list })),
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "CheckpointManager not mounted");
                }
            }
            "checkpoint/save" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let desc = params
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("daemon checkpoint");
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(mgr) = session.ctx.service::<CheckpointManager>() {
                    match mgr.save(desc) {
                        Ok(ckpt) => rpc_writer.send_response(id, serde_json::to_value(ckpt)?),
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "CheckpointManager not mounted");
                }
            }
            "checkpoint/restore" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let target = params
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("latest");
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(mgr) = session.ctx.service::<CheckpointManager>() {
                    match mgr.restore(target) {
                        Ok(rep) => rpc_writer.send_response(id, serde_json::to_value(rep)?),
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "CheckpointManager not mounted");
                }
            }
            "memory/show" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(store) = session.ctx.service::<EpistemicStore>() {
                    let mem = store.get();
                    let md = mem.render_markdown();
                    rpc_writer.send_response(
                        id,
                        json!({
                            "session_id": sid,
                            "memory": mem,
                            "markdown": md,
                        }),
                    );
                } else {
                    rpc_writer.send_error(id, -32000, "EpistemicStore not mounted");
                }
            }
            "memory/add_fact" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let Some(fact) = params.get("fact").and_then(|v| v.as_str()) else {
                    rpc_writer.send_error(id, -32602, "missing `fact` string parameter");
                    continue;
                };
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(store) = session.ctx.service::<EpistemicStore>() {
                    match store.add_fact(fact) {
                        Ok(()) => rpc_writer.send_response(id, json!({ "status": "ok" })),
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "EpistemicStore not mounted");
                }
            }
            "memory/set_hypothesis" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let hyp = params
                    .get("hypothesis")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(store) = session.ctx.service::<EpistemicStore>() {
                    match store.set_hypothesis(hyp) {
                        Ok(()) => rpc_writer.send_response(id, json!({ "status": "ok" })),
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "EpistemicStore not mounted");
                }
            }
            "memory/record_failure" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let Some(approach) = params.get("approach").and_then(|v| v.as_str()) else {
                    rpc_writer.send_error(id, -32602, "missing `approach` parameter");
                    continue;
                };
                let Some(reason) = params.get("reason").and_then(|v| v.as_str()) else {
                    rpc_writer.send_error(id, -32602, "missing `reason` parameter");
                    continue;
                };
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(store) = session.ctx.service::<EpistemicStore>() {
                    match store.record_failure(approach, reason) {
                        Ok(()) => rpc_writer.send_response(id, json!({ "status": "ok" })),
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "EpistemicStore not mounted");
                }
            }
            "memory/clear" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(store) = session.ctx.service::<EpistemicStore>() {
                    match store.clear() {
                        Ok(()) => rpc_writer.send_response(id, json!({ "status": "ok" })),
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "EpistemicStore not mounted");
                }
            }
            "memory/graph" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(store) = session.ctx.service::<EpistemicStore>() {
                    let mermaid = store.render_mermaid();
                    let state = store.get();
                    rpc_writer.send_response(
                        id,
                        json!({
                            "session_id": sid,
                            "mermaid": mermaid,
                            "nodes": state.nodes,
                            "edges": state.edges,
                        }),
                    );
                } else {
                    rpc_writer.send_error(id, -32000, "EpistemicStore not mounted");
                }
            }
            "memory/link" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let from = params.get("from").and_then(|v| v.as_str()).unwrap_or("");
                let to = params.get("to").and_then(|v| v.as_str()).unwrap_or("");
                let rel_str = params
                    .get("relation")
                    .and_then(|v| v.as_str())
                    .unwrap_or("supports");
                let rel = crate::epistemic::RelationType::from_str_lossy(rel_str);
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(store) = session.ctx.service::<EpistemicStore>() {
                    match store.add_fact_node(
                        from,
                        crate::epistemic::FactCategory::ToolObservation,
                        80,
                        None,
                    ) {
                        Ok(from_id) => {
                            match store.add_fact_node(
                                to,
                                crate::epistemic::FactCategory::ToolObservation,
                                80,
                                None,
                            ) {
                                Ok(to_id) => {
                                    let _ = store.add_edge(&from_id, &to_id, rel);
                                    let contradictions = store.detect_contradictions();
                                    rpc_writer.send_response(
                                        id,
                                        json!({
                                            "session_id": sid,
                                            "status": "ok",
                                            "from": from_id,
                                            "to": to_id,
                                            "relation": rel.as_str(),
                                            "contradictions": contradictions,
                                        }),
                                    );
                                }
                                Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                            }
                        }
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "EpistemicStore not mounted");
                }
            }
            "memory/contradictions" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(store) = session.ctx.service::<EpistemicStore>() {
                    let contradictions = store.detect_contradictions();
                    rpc_writer.send_response(
                        id,
                        json!({
                            "session_id": sid,
                            "contradictions": contradictions,
                            "has_conflicts": !contradictions.is_empty(),
                        }),
                    );
                } else {
                    rpc_writer.send_error(id, -32000, "EpistemicStore not mounted");
                }
            }
            "memory/sync" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(store) = session.ctx.service::<EpistemicStore>() {
                    if let Some(other_state) = params.get("state") {
                        match serde_json::from_value::<crate::epistemic::EpistemicState>(
                            other_state.clone(),
                        ) {
                            Ok(state) => {
                                let _ = store.merge_state(&state);
                                rpc_writer.send_response(
                                    id,
                                    json!({
                                        "session_id": sid,
                                        "status": "synced",
                                        "memory": store.get(),
                                    }),
                                );
                            }
                            Err(e) => rpc_writer.send_error(
                                id,
                                -32602,
                                &format!("invalid state payload: {e:#}"),
                            ),
                        }
                    } else {
                        rpc_writer.send_error(id, -32602, "missing `state` parameter");
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "EpistemicStore not mounted");
                }
            }
            "handoff/ingest" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let Some(env_val) = params.get("envelope") else {
                    rpc_writer.send_error(id, -32602, "missing `envelope` object parameter");
                    continue;
                };
                let envelope: crate::swarm::HandoffEnvelope =
                    match serde_json::from_value(env_val.clone()) {
                        Ok(e) => e,
                        Err(e) => {
                            rpc_writer.send_error(
                                id,
                                -32602,
                                &format!("invalid HandoffEnvelope: {e}"),
                            );
                            continue;
                        }
                    };
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                let mut h_guard = match session.history.lock() {
                    Ok(g) => g,
                    Err(_) => {
                        rpc_writer.send_error(id, -32000, "session history mutex poisoned");
                        continue;
                    }
                };
                match crate::swarm::ingest_handoff(&envelope, &session.ctx, &mut h_guard) {
                    Ok(report) => {
                        rpc_writer.send_response(
                            id,
                            json!({
                                "session_id": sid,
                                "status": "ok",
                                "report": report,
                            }),
                        );
                    }
                    Err(e) => {
                        rpc_writer.send_error(id, -32000, &format!("{e:#}"));
                    }
                }
            }
            "handoff/export" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let target_harness = params
                    .get("target_harness")
                    .and_then(|v| v.as_str())
                    .unwrap_or("helm");
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                let h_guard = match session.history.lock() {
                    Ok(g) => g,
                    Err(_) => {
                        rpc_writer.send_error(id, -32000, "session history mutex poisoned");
                        continue;
                    }
                };
                match crate::swarm::export_handoff(&sid, target_harness, &session.ctx, &h_guard) {
                    Ok(envelope) => {
                        rpc_writer.send_response(
                            id,
                            json!({
                                "session_id": sid,
                                "status": "ok",
                                "envelope": envelope,
                            }),
                        );
                    }
                    Err(e) => {
                        rpc_writer.send_error(id, -32000, &format!("{e:#}"));
                    }
                }
            }
            "oracle/run" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(oracle) = session.ctx.service::<Arc<DiagnosticOracle>>() {
                    match oracle.run_tests() {
                        Ok(snap) => {
                            rpc_writer
                                .send_response(id, json!({ "session_id": sid, "snapshot": snap }));
                        }
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "DiagnosticOracle not mounted");
                }
            }
            "oracle/baseline" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let action = params
                    .get("action")
                    .and_then(|v| v.as_str())
                    .unwrap_or("capture");
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(oracle) = session.ctx.service::<Arc<DiagnosticOracle>>() {
                    if action == "clear" {
                        oracle.clear_baseline();
                        rpc_writer
                            .send_response(id, json!({ "session_id": sid, "status": "cleared" }));
                    } else {
                        match oracle.capture_baseline() {
                            Ok(snap) => {
                                rpc_writer.send_response(
                                    id,
                                    json!({
                                        "session_id": sid,
                                        "status": "captured",
                                        "baseline": snap
                                    }),
                                );
                            }
                            Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                        }
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "DiagnosticOracle not mounted");
                }
            }
            "oracle/diff" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(oracle) = session.ctx.service::<Arc<DiagnosticOracle>>() {
                    match oracle.check_delta() {
                        Ok(delta) => {
                            let remediation = delta.render_remediation();
                            rpc_writer.send_response(
                                id,
                                json!({
                                    "session_id": sid,
                                    "delta": delta,
                                    "remediation": remediation,
                                    "is_clean": delta.is_clean,
                                }),
                            );
                        }
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "DiagnosticOracle not mounted");
                }
            }
            "symbols/outline" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let rel = params
                    .get("path")
                    .or_else(|| params.get("file"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(index) = session.ctx.service::<Arc<SymbolIndex>>() {
                    match crate::tools::confine(&server.cfg.root, rel) {
                        Ok(abs) => match index.outline_file(&abs, rel) {
                            Ok(outline) => {
                                let md = crate::symbols::render_outline_markdown(&outline);
                                rpc_writer.send_response(
                                    id,
                                    json!({
                                        "session_id": sid,
                                        "outline": outline,
                                        "markdown": md,
                                    }),
                                );
                            }
                            Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                        },
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "SymbolIndex not mounted");
                }
            }
            "symbols/search" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let query = params.get("query").and_then(|v| v.as_str()).unwrap_or("");
                let path_filter = params.get("path").and_then(|v| v.as_str());
                let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(index) = session.ctx.service::<Arc<SymbolIndex>>() {
                    match index.search_symbols(&server.cfg.root, query, path_filter, limit) {
                        Ok(matches) => {
                            rpc_writer.send_response(
                                id,
                                json!({
                                    "session_id": sid,
                                    "query": query,
                                    "matches": matches,
                                }),
                            );
                        }
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "SymbolIndex not mounted");
                }
            }
            "symbols/get" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let rel = params
                    .get("path")
                    .or_else(|| params.get("file"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(index) = session.ctx.service::<Arc<SymbolIndex>>() {
                    match crate::tools::confine(&server.cfg.root, rel) {
                        Ok(abs) => match index.get_symbol_source(&abs, rel, name) {
                            Ok(Some((symbol, source))) => {
                                rpc_writer.send_response(
                                    id,
                                    json!({
                                        "session_id": sid,
                                        "symbol": symbol,
                                        "source": source,
                                    }),
                                );
                            }
                            Ok(None) => {
                                rpc_writer.send_error(
                                    id,
                                    -32004,
                                    &format!("Symbol `{name}` not found in `{rel}`"),
                                );
                            }
                            Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                        },
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "SymbolIndex not mounted");
                }
            }
            "tree/status" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(tree_arc) = session.ctx.service::<Arc<Mutex<ExplorationTree>>>() {
                    let tree = tree_arc.lock().unwrap();
                    let ascii = tree.render_ascii();
                    rpc_writer.send_response(
                        id,
                        json!({
                            "session_id": sid,
                            "current_id": tree.current_id,
                            "root_id": tree.root_id,
                            "nodes": tree.nodes,
                            "ascii": ascii,
                        }),
                    );
                } else {
                    rpc_writer.send_error(id, -32000, "ExplorationTree not mounted");
                }
            }
            "tree/fork" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let label = params
                    .get("label")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Branch");
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                let cp = CheckpointManager::new(server.cfg.root.clone());
                if let Some(tree_arc) = session.ctx.service::<Arc<Mutex<ExplorationTree>>>() {
                    let mut tree = tree_arc.lock().unwrap();
                    match tree.fork(label, &cp) {
                        Ok(node) => {
                            let ascii = tree.render_ascii();
                            rpc_writer.send_response(
                                id,
                                json!({
                                    "session_id": sid,
                                    "node": node,
                                    "current_id": tree.current_id,
                                    "ascii": ascii,
                                }),
                            );
                        }
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "ExplorationTree not mounted");
                }
            }
            "tree/switch" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let node_id = params.get("node_id").and_then(|v| v.as_str()).unwrap_or("");
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                let cp = CheckpointManager::new(server.cfg.root.clone());
                if let Some(tree_arc) = session.ctx.service::<Arc<Mutex<ExplorationTree>>>() {
                    let mut tree = tree_arc.lock().unwrap();
                    match tree.switch_to(node_id, &cp) {
                        Ok(node) => {
                            let ascii = tree.render_ascii();
                            rpc_writer.send_response(
                                id,
                                json!({
                                    "session_id": sid,
                                    "node": node,
                                    "current_id": tree.current_id,
                                    "ascii": ascii,
                                }),
                            );
                        }
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "ExplorationTree not mounted");
                }
            }
            "tree/backtrack" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                let cp = CheckpointManager::new(server.cfg.root.clone());
                if let Some(tree_arc) = session.ctx.service::<Arc<Mutex<ExplorationTree>>>() {
                    let mut tree = tree_arc.lock().unwrap();
                    match tree.backtrack(&cp) {
                        Ok(parent) => {
                            let ascii = tree.render_ascii();
                            rpc_writer.send_response(
                                id,
                                json!({
                                    "session_id": sid,
                                    "parent": parent,
                                    "current_id": tree.current_id,
                                    "ascii": ascii,
                                }),
                            );
                        }
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "ExplorationTree not mounted");
                }
            }
            "tree/prune" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let node_id = params.get("node_id").and_then(|v| v.as_str()).unwrap_or("");
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(tree_arc) = session.ctx.service::<Arc<Mutex<ExplorationTree>>>() {
                    let mut tree = tree_arc.lock().unwrap();
                    match tree.prune(node_id) {
                        Ok(count) => {
                            let ascii = tree.render_ascii();
                            rpc_writer.send_response(
                                id,
                                json!({
                                    "session_id": sid,
                                    "pruned_count": count,
                                    "current_id": tree.current_id,
                                    "ascii": ascii,
                                }),
                            );
                        }
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "ExplorationTree not mounted");
                }
            }
            "tree/accept" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let node_id = params.get("node_id").and_then(|v| v.as_str()).unwrap_or("");
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(tree_arc) = session.ctx.service::<Arc<Mutex<ExplorationTree>>>() {
                    let mut tree = tree_arc.lock().unwrap();
                    match tree.accept(node_id) {
                        Ok(node) => {
                            let ascii = tree.render_ascii();
                            rpc_writer.send_response(
                                id,
                                json!({
                                    "session_id": sid,
                                    "accepted_node": node,
                                    "current_id": tree.current_id,
                                    "ascii": ascii,
                                }),
                            );
                        }
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "ExplorationTree not mounted");
                }
            }
            "sentinel/status" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(sentinel) = session
                    .ctx
                    .service::<Arc<crate::interceptor::StreamingSentinel>>()
                {
                    rpc_writer.send_response(
                        id,
                        json!({
                            "session_id": sid,
                            "sentinel": sentinel.metrics(),
                        }),
                    );
                } else {
                    rpc_writer.send_error(id, -32000, "StreamingSentinel not mounted");
                }
            }
            "sentinel/mode" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let mode_str = params
                    .get("mode")
                    .and_then(|v| v.as_str())
                    .unwrap_or("strict");
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(sentinel) = session
                    .ctx
                    .service::<Arc<crate::interceptor::StreamingSentinel>>()
                {
                    let mode = match mode_str {
                        "warn" => crate::interceptor::SentinelMode::Warn,
                        "disabled" => crate::interceptor::SentinelMode::Disabled,
                        _ => crate::interceptor::SentinelMode::Strict,
                    };
                    sentinel.set_mode(mode);
                    rpc_writer.send_response(
                        id,
                        json!({
                            "session_id": sid,
                            "mode": mode,
                        }),
                    );
                } else {
                    rpc_writer.send_error(id, -32000, "StreamingSentinel not mounted");
                }
            }
            "sentinel/check" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let tool = params
                    .get("tool")
                    .and_then(|v| v.as_str())
                    .unwrap_or("bash");
                let args_val = params.get("args").cloned().unwrap_or_else(|| json!({}));
                let mut fields = HashMap::new();
                if let Some(obj) = args_val.as_object() {
                    for (k, v) in obj {
                        if let Some(s) = v.as_str() {
                            fields.insert(k.clone(), s.to_string());
                        }
                    }
                }
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(sentinel) = session
                    .ctx
                    .service::<Arc<crate::interceptor::StreamingSentinel>>()
                {
                    let veto = sentinel.inspect(tool, &fields);
                    rpc_writer.send_response(
                        id,
                        json!({
                            "session_id": sid,
                            "vetoed": veto.is_some(),
                            "veto": veto,
                        }),
                    );
                } else {
                    rpc_writer.send_error(id, -32000, "StreamingSentinel not mounted");
                }
            }
            "prewarm/status" => {
                let sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| current_session_id.lock().unwrap().clone());
                let session = server.registry.get_or_create(
                    &sid,
                    &server.cfg,
                    server.llm.clone(),
                    server.broadcaster.clone(),
                );
                if let Some(prewarm) = session
                    .ctx
                    .service::<Arc<crate::interceptor::PrewarmEngine>>()
                {
                    rpc_writer.send_response(
                        id,
                        json!({
                            "session_id": sid,
                            "prewarm": prewarm.metrics(),
                        }),
                    );
                } else {
                    rpc_writer.send_error(id, -32000, "PrewarmEngine not mounted");
                }
            }
            "shutdown" | "exit" => {
                if !id.is_null() {
                    rpc_writer.send_response(id, json!({ "status": "shutting_down" }));
                }
                server.shutdown_signal.store(true, Ordering::Relaxed);
                break;
            }
            other => {
                if !id.is_null() {
                    rpc_writer.send_error(id, -32601, &format!("Method not found: `{other}`"));
                }
            }
        }
    }

    for h in active_workers {
        let _ = h.join();
    }

    server.broadcaster.unregister(&rpc_writer);
    rpc_writer.mark_closed();

    Ok(())
}

/// Run bidirectional ACP JSON-RPC 2.0 daemon over `reader` and `writer` (stdio mode).
pub fn serve_acp<R: BufRead>(
    reader: R,
    writer: Box<dyn Write + Send>,
    cfg: DaemonConfig,
    llm: Arc<dyn Llm>,
) -> Result<()> {
    serve_acp_with_auth(reader, writer, cfg, llm, None)
}

/// Run bidirectional ACP JSON-RPC 2.0 daemon over `reader` and `writer` with optional bearer authentication.
pub fn serve_acp_with_auth<R: BufRead>(
    reader: R,
    writer: Box<dyn Write + Send>,
    cfg: DaemonConfig,
    llm: Arc<dyn Llm>,
    auth_token: Option<String>,
) -> Result<()> {
    let server = DaemonServer::new(cfg, llm, auth_token);
    let default_sid = format!("acp-{}", std::process::id());
    handle_acp_client(reader, writer, server, default_sid)
}

/// Listen on an existing TCP listener and serve multiplexed ACP JSON-RPC 2.0 sessions.
pub fn serve_acp_listener(
    listener: std::net::TcpListener,
    cfg: DaemonConfig,
    llm: Arc<dyn Llm>,
    auth_token: Option<String>,
) -> Result<()> {
    listener.set_nonblocking(true)?;

    let server = DaemonServer::new(cfg, llm, auth_token);
    let default_sid = format!("acp-{}", std::process::id());

    if let Ok(addr) = listener.local_addr() {
        eprintln!("[keel-daemon] ACP JSON-RPC 2.0 server listening on tcp://{addr}");
    }

    while !server.shutdown_signal.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _peer_addr)) => {
                let server_c = server.clone();
                let def_sid = default_sid.clone();

                std::thread::spawn(move || {
                    let reader_stream = match stream.try_clone() {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("[keel-daemon] Failed to clone stream: {e}");
                            return;
                        }
                    };
                    let reader = std::io::BufReader::new(reader_stream);
                    let writer = Box::new(stream);
                    let _ = handle_acp_client(reader, writer, server_c, def_sid);
                });
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(30));
            }
            Err(e) => {
                eprintln!("[keel-daemon] TCP accept error: {e}");
                std::thread::sleep(Duration::from_millis(30));
            }
        }
    }
    eprintln!("[keel-daemon] TCP server shutting down");
    Ok(())
}

/// Listen on TCP address and serve multiplexed ACP JSON-RPC 2.0 sessions to connected clients.
pub fn serve_acp_tcp(
    addr: &str,
    cfg: DaemonConfig,
    llm: Arc<dyn Llm>,
    auth_token: Option<String>,
) -> Result<()> {
    let listener = std::net::TcpListener::bind(addr)
        .with_context(|| format!("Failed to bind TCP listener on {addr}"))?;
    serve_acp_listener(listener, cfg, llm, auth_token)
}

/// Listen on Unix domain socket and serve multiplexed ACP JSON-RPC 2.0 sessions.
#[cfg(unix)]
pub fn serve_acp_unix(
    socket_path: &std::path::Path,
    cfg: DaemonConfig,
    llm: Arc<dyn Llm>,
    auth_token: Option<String>,
) -> Result<()> {
    if let Some(parent) = socket_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if socket_path.exists() {
        if std::os::unix::net::UnixStream::connect(socket_path).is_ok() {
            anyhow::bail!(
                "Another keel daemon is already active on socket {}",
                socket_path.display()
            );
        }
        let _ = std::fs::remove_file(socket_path);
    }

    let listener = std::os::unix::net::UnixListener::bind(socket_path)
        .with_context(|| format!("Failed to bind Unix socket at {}", socket_path.display()))?;
    listener.set_nonblocking(true)?;

    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(socket_path, std::fs::Permissions::from_mode(0o600));

    let server = DaemonServer::new(cfg, llm, auth_token);
    let default_sid = format!("acp-{}", std::process::id());

    eprintln!(
        "[keel-daemon] ACP JSON-RPC 2.0 server listening on unix:{}",
        socket_path.display()
    );

    while !server.shutdown_signal.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _)) => {
                let server_c = server.clone();
                let def_sid = default_sid.clone();

                std::thread::spawn(move || {
                    let reader_stream = match stream.try_clone() {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("[keel-daemon] Failed to clone unix stream: {e}");
                            return;
                        }
                    };
                    let reader = std::io::BufReader::new(reader_stream);
                    let writer = Box::new(stream);
                    let _ = handle_acp_client(reader, writer, server_c, def_sid);
                });
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(30));
            }
            Err(e) => {
                eprintln!("[keel-daemon] Unix accept error: {e}");
                std::thread::sleep(Duration::from_millis(30));
            }
        }
    }

    let _ = std::fs::remove_file(socket_path);
    eprintln!("[keel-daemon] Unix server shutting down");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolSpec;
    use std::io::BufReader;

    struct DaemonTestLlm;
    impl Llm for DaemonTestLlm {
        fn complete(&self, messages: &[Message], _tools: &[ToolSpec]) -> Result<Message> {
            let has_tool = messages.iter().any(|m| matches!(m, Message::Tool { .. }));
            if !has_tool {
                Ok(Message::Assistant {
                    text: "Editing file".into(),
                    tool_calls: vec![ToolCall {
                        id: "call_1".into(),
                        name: "write_file".into(),
                        args: json!({
                            "path": "hello.txt",
                            "content": "daemon wrote this\n"
                        }),
                    }],
                })
            } else {
                Ok(Message::Assistant {
                    text: "All done via ACP daemon".into(),
                    tool_calls: vec![],
                })
            }
        }
    }

    struct SharedVecWriter(Arc<Mutex<Vec<u8>>>);
    impl Write for SharedVecWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn acp_daemon_initialize_prompt_vfs_and_rollback_roundtrip() {
        let tmp = std::env::temp_dir().join(format!("keel-acp-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let out_buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));

        let requests = [
            json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}).to_string(),
            json!({"jsonrpc": "2.0", "id": 2, "method": "session/prompt", "params": {"prompt": "write hello.txt"}}).to_string(),
            json!({"jsonrpc": "2.0", "id": 3, "method": "session/list", "params": {}}).to_string(),
            json!({"jsonrpc": "2.0", "id": 4, "method": "shutdown", "params": {}}).to_string(),
        ]
        .join("\n")
            + "\n";

        let cfg = DaemonConfig {
            root: tmp.clone(),
            policy: ApprovalPolicy::Yolo,
            landlock_enabled: false,
            vfs_mode: VfsMode::MaterializedCow,
            verify_command: None,
        };

        serve_acp(
            BufReader::new(std::io::Cursor::new(requests.into_bytes())),
            Box::new(SharedVecWriter(out_buf.clone())),
            cfg,
            Arc::new(DaemonTestLlm),
        )
        .unwrap();

        let raw_output = String::from_utf8(out_buf.lock().unwrap().clone()).unwrap();
        let lines: Vec<Value> = raw_output
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .collect();

        // Verify initialize response
        assert!(lines.iter().any(|v| v["id"] == 1
            && v["result"]["capabilities"]["streaming"] == true
            && v["result"]["capabilities"]["sessionMultiplexing"] == true
            && v["result"]["capabilities"]["vfs"] == true));

        // Verify streaming tool_call & text notifications were emitted
        assert!(lines.iter().any(|v| v["method"] == "session/update"
            && v["params"]["type"] == "tool_call"
            && v["params"]["name"] == "write_file"));

        // Verify final prompt response
        assert!(lines.iter().any(|v| v["id"] == 2
            && v["result"]["status"] == "ok"
            && v["result"]["answer"] == "All done via ACP daemon"));

        // Verify session/list response
        assert!(lines
            .iter()
            .any(|v| v["id"] == 3 && v["result"]["sessions"].as_array().is_some()));

        assert_eq!(
            std::fs::read_to_string(tmp.join("hello.txt")).unwrap(),
            "daemon wrote this\n"
        );

        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn acp_daemon_tcp_multiplex_and_auth() {
        let tmp = std::env::temp_dir().join(format!("keel-tcp-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let cfg = DaemonConfig {
            root: tmp.clone(),
            policy: ApprovalPolicy::Yolo,
            landlock_enabled: false,
            vfs_mode: VfsMode::MaterializedCow,
            verify_command: None,
        };

        let token = "secret-token-12345".to_string();

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let local_addr = listener.local_addr().unwrap();
        let addr = format!("127.0.0.1:{}", local_addr.port());
        let token_clone = token.clone();

        let srv_handle = std::thread::spawn(move || {
            serve_acp_listener(listener, cfg, Arc::new(DaemonTestLlm), Some(token_clone)).unwrap();
        });

        // Wait for listener to bind
        std::thread::sleep(Duration::from_millis(50));

        // Connect client 1: try unauthorized request
        {
            let mut stream = std::net::TcpStream::connect(&addr).unwrap();
            let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());

            writeln!(
                stream,
                "{}",
                json!({"jsonrpc": "2.0", "id": 1, "method": "session/list", "params": {}})
            )
            .unwrap();

            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let err_val: Value = serde_json::from_str(&line).unwrap();
            assert_eq!(err_val["id"], 1);
            assert_eq!(err_val["error"]["code"], -32001);
        }

        // Connect client 2: authorized handshake, session/new, epistemic memory
        {
            let mut stream = std::net::TcpStream::connect(&addr).unwrap();
            let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());

            // Initialize with token
            writeln!(
                stream,
                "{}",
                json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": { "token": token }
                })
            )
            .unwrap();

            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let init_res: Value = serde_json::from_str(&line).unwrap();
            assert_eq!(init_res["id"], 1);
            assert_eq!(init_res["result"]["serverInfo"]["name"], "keel-acp-daemon");

            // Add fact to epistemic memory
            writeln!(
                stream,
                "{}",
                json!({
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "memory/add_fact",
                    "params": { "fact": "TCP daemon verified" }
                })
            )
            .unwrap();
            line.clear();
            reader.read_line(&mut line).unwrap();
            let mem_res: Value = serde_json::from_str(&line).unwrap();
            assert_eq!(mem_res["id"], 2);
            assert_eq!(mem_res["result"]["status"], "ok");

            // Inspect memory
            writeln!(
                stream,
                "{}",
                json!({
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "memory/show",
                    "params": {}
                })
            )
            .unwrap();
            line.clear();
            reader.read_line(&mut line).unwrap();
            let show_res: Value = serde_json::from_str(&line).unwrap();
            assert_eq!(show_res["id"], 3);
            let facts = show_res["result"]["memory"]["facts"].as_array().unwrap();
            assert!(facts.iter().any(|f| f == "TCP daemon verified"));

            // Graceful shutdown
            writeln!(
                stream,
                "{}",
                json!({
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "shutdown",
                    "params": {}
                })
            )
            .unwrap();
        }

        srv_handle.join().unwrap();
        let _ = std::fs::remove_dir_all(tmp);
    }

    #[cfg(unix)]
    #[test]
    fn acp_daemon_unix_socket_roundtrip() {
        let tmp = std::env::temp_dir().join(format!("keel-sock-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let sock_path = tmp.join("daemon.sock");
        let sock_path_clone = sock_path.clone();

        let cfg = DaemonConfig {
            root: tmp.clone(),
            policy: ApprovalPolicy::Yolo,
            landlock_enabled: false,
            vfs_mode: VfsMode::MaterializedCow,
            verify_command: None,
        };

        let srv_handle = std::thread::spawn(move || {
            serve_acp_unix(&sock_path_clone, cfg, Arc::new(DaemonTestLlm), None).unwrap();
        });

        std::thread::sleep(Duration::from_millis(50));

        {
            let mut stream = std::os::unix::net::UnixStream::connect(&sock_path).unwrap();
            let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());

            writeln!(
                stream,
                "{}",
                json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "ping",
                    "params": {}
                })
            )
            .unwrap();

            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let ping_res: Value = serde_json::from_str(&line).unwrap();
            assert_eq!(ping_res["id"], 1);
            assert_eq!(ping_res["result"]["status"], "pong");

            writeln!(
                stream,
                "{}",
                json!({
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "shutdown",
                    "params": {}
                })
            )
            .unwrap();
        }

        srv_handle.join().unwrap();
        assert!(!sock_path.exists());
        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn acp_daemon_handoff_ingest_and_export_rpc() {
        let tmp = std::env::temp_dir().join(format!("keel-handoff-rpc-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let out_buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));

        let handoff_payload = json!({
            "source_conversation_id": "helm-session-888",
            "source_harness": "helm",
            "target_harness": "keel",
            "goal": "Implement cross-harness bounce protocol",
            "relevant_files": ["crates/keel-agent/src/swarm.rs"],
            "open_tasks": ["Add ACP handoff JSON-RPC endpoints"],
            "completed_tasks": ["Draft HandoffEnvelope struct"],
            "context_summary": "Architecture verified on both sides"
        });

        let requests = [
            json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}).to_string(),
            json!({"jsonrpc": "2.0", "id": 2, "method": "handoff/ingest", "params": {"envelope": handoff_payload}}).to_string(),
            json!({"jsonrpc": "2.0", "id": 3, "method": "handoff/export", "params": {"target_harness": "claude"}}).to_string(),
            json!({"jsonrpc": "2.0", "id": 4, "method": "shutdown", "params": {}}).to_string(),
        ]
        .join("\n")
            + "\n";

        let cfg = DaemonConfig {
            root: tmp.clone(),
            policy: ApprovalPolicy::Yolo,
            landlock_enabled: false,
            vfs_mode: VfsMode::MaterializedCow,
            verify_command: None,
        };

        serve_acp(
            BufReader::new(std::io::Cursor::new(requests.into_bytes())),
            Box::new(SharedVecWriter(out_buf.clone())),
            cfg,
            Arc::new(DaemonTestLlm),
        )
        .unwrap();

        let raw_output = String::from_utf8(out_buf.lock().unwrap().clone()).unwrap();
        let lines: Vec<Value> = raw_output
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .collect();

        // 1. Verify capability includes swarmHandoff
        assert!(lines
            .iter()
            .any(|v| v["id"] == 1 && v["result"]["capabilities"]["swarmHandoff"] == true));

        // 2. Verify ingest response
        assert!(lines.iter().any(|v| v["id"] == 2
            && v["result"]["status"] == "ok"
            && v["result"]["report"]["tasks_injected"] == 2
            && v["result"]["report"]["goal"] == "Implement cross-harness bounce protocol"));

        // 3. Verify export response
        let export_msg = lines.iter().find(|v| v["id"] == 3).unwrap();
        assert_eq!(export_msg["result"]["status"], "ok");
        assert_eq!(
            export_msg["result"]["envelope"]["goal"],
            "Implement cross-harness bounce protocol"
        );
        assert_eq!(export_msg["result"]["envelope"]["source_harness"], "keel");
        assert_eq!(export_msg["result"]["envelope"]["target_harness"], "claude");

        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn acp_daemon_symbols_outline_and_search_rpc() {
        let tmp = std::env::temp_dir().join(format!("keel-symbols-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("src")).unwrap();

        std::fs::write(
            tmp.join("src/service.rs"),
            r#"
/// Worker service for async jobs.
pub struct WorkerService {
    pub id: u32,
}

impl WorkerService {
    pub fn start(&self) {}
}
"#,
        )
        .unwrap();

        let out_buf = Arc::new(Mutex::new(Vec::new()));

        let requests = [
            json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}).to_string(),
            json!({"jsonrpc": "2.0", "id": 2, "method": "symbols/outline", "params": {"path": "src/service.rs"}}).to_string(),
            json!({"jsonrpc": "2.0", "id": 3, "method": "symbols/search", "params": {"query": "worker"}}).to_string(),
            json!({"jsonrpc": "2.0", "id": 4, "method": "symbols/get", "params": {"path": "src/service.rs", "name": "WorkerService"}}).to_string(),
            json!({"jsonrpc": "2.0", "id": 5, "method": "shutdown", "params": {}}).to_string(),
        ]
        .join("\n")
            + "\n";

        let cfg = DaemonConfig {
            root: tmp.clone(),
            policy: ApprovalPolicy::Yolo,
            landlock_enabled: false,
            vfs_mode: VfsMode::MaterializedCow,
            verify_command: None,
        };

        serve_acp(
            BufReader::new(std::io::Cursor::new(requests.into_bytes())),
            Box::new(SharedVecWriter(out_buf.clone())),
            cfg,
            Arc::new(DaemonTestLlm),
        )
        .unwrap();

        let raw_output = String::from_utf8(out_buf.lock().unwrap().clone()).unwrap();
        let lines: Vec<Value> = raw_output
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .collect();

        // 1. Verify capability includes structuralSymbols and symbolOutline
        assert!(lines.iter().any(|v| v["id"] == 1
            && v["result"]["capabilities"]["structuralSymbols"] == true
            && v["result"]["capabilities"]["symbolOutline"] == true));

        // 2. Verify outline response
        let outline_res = lines.iter().find(|v| v["id"] == 2).unwrap();
        assert_eq!(outline_res["result"]["outline"]["language"], "rust");
        assert!(outline_res["result"]["markdown"]
            .as_str()
            .unwrap()
            .contains("WorkerService"));

        // 3. Verify search response
        let search_res = lines.iter().find(|v| v["id"] == 3).unwrap();
        let matches = search_res["result"]["matches"].as_array().unwrap();
        assert!(!matches.is_empty());
        assert!(matches
            .iter()
            .any(|m| m["symbol"]["name"] == "WorkerService"));

        // 4. Verify get symbol response
        let get_res = lines.iter().find(|v| v["id"] == 4).unwrap();
        assert_eq!(get_res["result"]["symbol"]["name"], "WorkerService");
        assert!(get_res["result"]["source"]
            .as_str()
            .unwrap()
            .contains("pub struct WorkerService"));

        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn acp_daemon_tree_fork_and_backtrack_rpc() {
        let tmp = std::env::temp_dir().join(format!("keel-tree-rpc-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(&tmp)
            .status()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.name", "Keel Test"])
            .current_dir(&tmp)
            .status()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.email", "test@keel.dev"])
            .current_dir(&tmp)
            .status()
            .unwrap();
        std::fs::write(tmp.join("init.txt"), "initial").unwrap();
        std::process::Command::new("git")
            .args(["add", "init.txt"])
            .current_dir(&tmp)
            .status()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "init", "-q"])
            .current_dir(&tmp)
            .status()
            .unwrap();

        let out_buf = Arc::new(Mutex::new(Vec::new()));

        let requests = [
            json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}).to_string(),
            json!({"jsonrpc": "2.0", "id": 2, "method": "tree/status", "params": {}}).to_string(),
            json!({"jsonrpc": "2.0", "id": 3, "method": "tree/fork", "params": {"label": "Attempt A: regex"}}).to_string(),
            json!({"jsonrpc": "2.0", "id": 4, "method": "tree/backtrack", "params": {}}).to_string(),
            json!({"jsonrpc": "2.0", "id": 5, "method": "shutdown", "params": {}}).to_string(),
        ]
        .join("\n")
            + "\n";

        let cfg = DaemonConfig {
            root: tmp.clone(),
            policy: ApprovalPolicy::Yolo,
            landlock_enabled: false,
            vfs_mode: VfsMode::MaterializedCow,
            verify_command: None,
        };

        serve_acp(
            BufReader::new(std::io::Cursor::new(requests.into_bytes())),
            Box::new(SharedVecWriter(out_buf.clone())),
            cfg,
            Arc::new(DaemonTestLlm),
        )
        .unwrap();

        let raw_output = String::from_utf8(out_buf.lock().unwrap().clone()).unwrap();
        let lines: Vec<Value> = raw_output
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .collect();

        // 1. Verify capability
        assert!(lines.iter().any(|v| v["id"] == 1
            && v["result"]["capabilities"]["explorationTree"] == true
            && v["result"]["capabilities"]["multiStepBacktracking"] == true));

        // 2. Verify tree status
        let status_res = lines.iter().find(|v| v["id"] == 2).unwrap();
        assert_eq!(status_res["result"]["current_id"], "root");

        // 3. Verify fork response
        let fork_res = lines.iter().find(|v| v["id"] == 3).unwrap();
        assert_eq!(fork_res["result"]["node"]["id"], "node_1");
        assert_eq!(fork_res["result"]["node"]["label"], "Attempt A: regex");

        // 4. Verify backtrack response
        let back_res = lines.iter().find(|v| v["id"] == 4).unwrap();
        assert_eq!(back_res["result"]["parent"]["id"], "root");
        assert_eq!(back_res["result"]["current_id"], "root");

        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn acp_daemon_epistemic_graph_and_contradictions_rpc() {
        let tmp = std::env::temp_dir().join(format!("keel-graph-rpc-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let out_buf = Arc::new(Mutex::new(Vec::new()));

        let requests = [
            json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}).to_string(),
            json!({"jsonrpc": "2.0", "id": 2, "method": "memory/link", "params": {
                "from": "Port 8000 is open",
                "to": "Port 8000 is blocked by firewall",
                "relation": "contradicts"
            }})
            .to_string(),
            json!({"jsonrpc": "2.0", "id": 3, "method": "memory/contradictions", "params": {}})
                .to_string(),
            json!({"jsonrpc": "2.0", "id": 4, "method": "memory/graph", "params": {}}).to_string(),
            json!({"jsonrpc": "2.0", "id": 5, "method": "shutdown", "params": {}}).to_string(),
        ]
        .join("\n")
            + "\n";

        let cfg = DaemonConfig {
            root: tmp.clone(),
            policy: ApprovalPolicy::Yolo,
            landlock_enabled: false,
            vfs_mode: VfsMode::MaterializedCow,
            verify_command: None,
        };

        serve_acp(
            BufReader::new(std::io::Cursor::new(requests.into_bytes())),
            Box::new(SharedVecWriter(out_buf.clone())),
            cfg,
            Arc::new(DaemonTestLlm),
        )
        .unwrap();

        let raw_output = String::from_utf8(out_buf.lock().unwrap().clone()).unwrap();
        let lines: Vec<Value> = raw_output
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .collect();

        // 1. Verify capability
        assert!(lines
            .iter()
            .any(|v| v["id"] == 1 && v["result"]["capabilities"]["epistemicGraph"] == true));

        // 2. Verify link response
        let link_res = lines.iter().find(|v| v["id"] == 2).unwrap();
        assert_eq!(link_res["result"]["status"], "ok");
        assert_eq!(link_res["result"]["relation"], "contradicts");

        // 3. Verify contradictions detected
        let contra_res = lines.iter().find(|v| v["id"] == 3).unwrap();
        assert_eq!(contra_res["result"]["has_conflicts"], true);
        let contra_list = contra_res["result"]["contradictions"].as_array().unwrap();
        assert_eq!(contra_list.len(), 1);

        // 4. Verify mermaid graph
        let graph_res = lines.iter().find(|v| v["id"] == 4).unwrap();
        let mermaid = graph_res["result"]["mermaid"].as_str().unwrap();
        assert!(mermaid.contains("Port 8000"));
        assert!(mermaid.contains("contradicts"));

        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn acp_daemon_streaming_sentinel_and_prewarm_rpc() {
        let tmp =
            std::env::temp_dir().join(format!("keel-sentinel-rpc-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let out_buf = Arc::new(Mutex::new(Vec::new()));

        let requests = [
            json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}).to_string(),
            json!({"jsonrpc": "2.0", "id": 2, "method": "sentinel/status", "params": {}}).to_string(),
            json!({"jsonrpc": "2.0", "id": 3, "method": "sentinel/check", "params": {
                "tool": "bash",
                "args": { "command": "rm -rf / --no-preserve-root" }
            }}).to_string(),
            json!({"jsonrpc": "2.0", "id": 4, "method": "sentinel/mode", "params": { "mode": "warn" }}).to_string(),
            json!({"jsonrpc": "2.0", "id": 5, "method": "prewarm/status", "params": {}}).to_string(),
            json!({"jsonrpc": "2.0", "id": 6, "method": "shutdown", "params": {}}).to_string(),
        ]
        .join("\n")
            + "\n";

        let cfg = DaemonConfig {
            root: tmp.clone(),
            policy: ApprovalPolicy::Yolo,
            landlock_enabled: false,
            vfs_mode: VfsMode::MaterializedCow,
            verify_command: None,
        };

        serve_acp(
            BufReader::new(std::io::Cursor::new(requests.into_bytes())),
            Box::new(SharedVecWriter(out_buf.clone())),
            cfg,
            Arc::new(DaemonTestLlm),
        )
        .unwrap();

        let raw_output = String::from_utf8(out_buf.lock().unwrap().clone()).unwrap();
        let lines: Vec<Value> = raw_output
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .collect();

        // 1. Verify capability
        assert!(lines.iter().any(|v| v["id"] == 1
            && v["result"]["capabilities"]["streamingSentinel"] == true
            && v["result"]["capabilities"]["predictivePrewarm"] == true));

        // 2. Verify sentinel status
        let status_res = lines.iter().find(|v| v["id"] == 2).unwrap();
        assert_eq!(status_res["result"]["sentinel"]["mode"], "strict");

        // 3. Verify check detects destructive veto
        let check_res = lines.iter().find(|v| v["id"] == 3).unwrap();
        assert_eq!(check_res["result"]["vetoed"], true);
        assert_eq!(check_res["result"]["veto"]["matched_pattern"], "rm -rf /");

        // 4. Verify mode change to warn
        let mode_res = lines.iter().find(|v| v["id"] == 4).unwrap();
        assert_eq!(mode_res["result"]["mode"], "warn");

        // 5. Verify prewarm metrics
        let prewarm_res = lines.iter().find(|v| v["id"] == 5).unwrap();
        assert!(prewarm_res["result"]["prewarm"]["prewarm_requests"].is_number());

        let _ = std::fs::remove_dir_all(tmp);
    }
}
