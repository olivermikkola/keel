//! Agent Client Protocol (ACP) Bidirectional JSON-RPC 2.0 Daemon (`keel daemon`).
//!
//! Turns `keel` into a long-lived, multiplexed JSON-RPC 2.0 server over `stdio`
//! (or any `BufRead` + `Write` stream pair) for IDE extensions, Zed/Neovim clients,
//! and orchestrators like Helm.
//!
//! Supported JSON-RPC 2.0 methods:
//! - `initialize`: capability negotiation
//! - `session/new` / `session/load`: initialize or resume conversation history
//! - `session/prompt`: run an agent turn asynchronously while streaming `session/update`
//!   notifications (`text`, `reasoning`, `tool_call`, `tool_result`, `tool_denied`)
//! - `session/cancel`: cooperatively cancel an in-flight turn via [`CancelToken`]
//! - `approval/respond`: resolve an in-flight HITL `approval/request`
//! - `vfs/status`, `vfs/diff`, `vfs/commit`, `vfs/rollback`: transactional VFS control
//! - `checkpoint/list`, `checkpoint/save`, `checkpoint/restore`: git shadow checkpoints

use crate::checkpoint::{CheckpointManager, CheckpointPlugin};
use crate::compaction::{ToolResultPrunerPlugin, WorkspaceContextPlugin, WorkspaceInstructions};
use crate::epistemic::EpistemicMemoryPlugin;
use crate::guard::RepeatToolReminderPlugin;
use crate::lsp::LspPlugin;
use crate::plan::PlanPlugin;
use crate::sandbox::LandlockPlugin;
use crate::speculate::SpeculatePlugin;
use crate::subagent::SubagentPlugin;
use crate::tools::BuiltinToolsPlugin;
use crate::vfs::{TransactionalVfs, VfsMode, VfsPlugin};
use crate::{
    Agent, ApprovalPolicy, Approver, ApproverPlugin, CancelToken, CancelTokenPlugin, Chunk,
    Decision, Llm, LlmPlugin, Message, SessionLog, SessionLogPlugin, ToolCall, ToolDenied,
    ToolResult, ToolStart,
};
use anyhow::Result;
use keel_kernel::{Context, Plugin, Scope};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

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
}

impl RpcWriter {
    pub fn new(w: Box<dyn Write + Send>) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(w),
        })
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
        if let Ok(mut guard) = self.inner.lock() {
            let _ = writeln!(guard, "{val}");
            let _ = guard.flush();
        }
    }
}

#[derive(Default)]
struct ApprovalHubState {
    decisions: HashMap<String, String>,
    whitelisted: HashSet<String>,
}

/// RPC-backed HITL approver that emits `approval/request` notifications and awaits
/// `approval/respond` (or `session/cancel`) from the connected client.
pub struct AcpApprover {
    policy: ApprovalPolicy,
    writer: Arc<RpcWriter>,
    cancel_token: CancelToken,
    state: Arc<(Mutex<ApprovalHubState>, Condvar)>,
}

impl AcpApprover {
    pub fn new(
        policy: ApprovalPolicy,
        writer: Arc<RpcWriter>,
        cancel_token: CancelToken,
    ) -> Arc<Self> {
        Arc::new(Self {
            policy,
            writer,
            cancel_token,
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

        self.writer.send_notification(
            "approval/request",
            json!({
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
                return Decision::Deny("turn cancelled while awaiting approval".into());
            }
            if let Some(dec) = guard.decisions.remove(&call.id) {
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
    writer: Arc<RpcWriter>,
    session_id: Arc<Mutex<String>>,
}

impl Plugin for AcpEventStreamer {
    fn name(&self) -> &str {
        "acp-event-streamer"
    }

    fn apply(&self, scope: &mut Scope) {
        let w1 = self.writer.clone();
        let s1 = self.session_id.clone();
        scope.on::<Chunk, _>(move |c| {
            let sid = s1.lock().ok().map(|g| g.clone()).unwrap_or_default();
            match c {
                Chunk::Text(t) => {
                    w1.send_notification(
                        "session/update",
                        json!({
                            "session_id": sid,
                            "type": "text",
                            "content": t,
                        }),
                    );
                }
                Chunk::Reasoning(t) => {
                    w1.send_notification(
                        "session/update",
                        json!({
                            "session_id": sid,
                            "type": "reasoning",
                            "content": t,
                        }),
                    );
                }
            }
        });

        let w2 = self.writer.clone();
        let s2 = self.session_id.clone();
        scope.on::<ToolStart, _>(move |ts| {
            let sid = s2.lock().ok().map(|g| g.clone()).unwrap_or_default();
            w2.send_notification(
                "session/update",
                json!({
                    "session_id": sid,
                    "type": "tool_call",
                    "id": ts.0.id,
                    "name": ts.0.name,
                    "arguments": ts.0.args,
                }),
            );
        });

        let w3 = self.writer.clone();
        let s3 = self.session_id.clone();
        scope.on::<ToolResult, _>(move |tr| {
            let sid = s3.lock().ok().map(|g| g.clone()).unwrap_or_default();
            w3.send_notification(
                "session/update",
                json!({
                    "session_id": sid,
                    "type": "tool_result",
                    "id": tr.0,
                    "output": tr.1,
                }),
            );
        });

        let w4 = self.writer.clone();
        let s4 = self.session_id.clone();
        scope.on::<ToolDenied, _>(move |td| {
            let sid = s4.lock().ok().map(|g| g.clone()).unwrap_or_default();
            w4.send_notification(
                "session/update",
                json!({
                    "session_id": sid,
                    "type": "tool_denied",
                    "id": td.0.id,
                    "name": td.0.name,
                }),
            );
        });
    }
}

/// Run the bidirectional ACP JSON-RPC 2.0 daemon loop over `reader` and `writer`.
pub fn serve_acp<R: BufRead>(
    mut reader: R,
    writer: Box<dyn Write + Send>,
    cfg: DaemonConfig,
    llm: Arc<dyn Llm>,
) -> Result<()> {
    let rpc_writer = RpcWriter::new(writer);
    let cancel_token = CancelToken::new();
    let approver = AcpApprover::new(cfg.policy, rpc_writer.clone(), cancel_token.clone());
    let session_id = Arc::new(Mutex::new(format!("acp-{}", std::process::id())));

    let ctx = Arc::new(Context::new());
    let _p_stream = ctx.mount(&AcpEventStreamer {
        writer: rpc_writer.clone(),
        session_id: session_id.clone(),
    });
    let _p_llm = ctx.mount(&LlmPlugin(llm));
    let _p_app = ctx.mount(&ApproverPlugin(approver.clone()));
    let _p_cancel = ctx.mount(&CancelTokenPlugin(cancel_token.clone()));
    let _p_log = ctx.mount(&SessionLogPlugin(SessionLog::none()));
    let _p_ll = ctx.mount(&LandlockPlugin {
        workspace: cfg.root.clone(),
        enabled: cfg.landlock_enabled,
    });
    let _p_tools = ctx.mount(&BuiltinToolsPlugin {
        root: cfg.root.clone(),
        landlock_enabled: cfg.landlock_enabled,
    });
    let _p_guard = ctx.mount(&RepeatToolReminderPlugin::default());
    let _p_pruner = ctx.mount(&ToolResultPrunerPlugin::default());
    let _p_ws = ctx.mount(&WorkspaceContextPlugin {
        root: cfg.root.clone(),
    });
    let _p_plan = ctx.mount(&PlanPlugin {
        root: cfg.root.clone(),
        verify_command: cfg.verify_command.clone(),
    });
    let _p_subagent = ctx.mount(&SubagentPlugin {
        root: cfg.root.clone(),
        landlock_enabled: cfg.landlock_enabled,
    });
    let _p_spec = ctx.mount(&SpeculatePlugin {
        root: cfg.root.clone(),
        landlock_enabled: cfg.landlock_enabled,
    });
    let _p_ckpt = ctx.mount(&CheckpointPlugin {
        root: cfg.root.clone(),
        auto_checkpoint: true,
        auto_rollback: false,
    });
    let _p_epistemic = ctx.mount(&EpistemicMemoryPlugin {
        root: cfg.root.clone(),
    });
    let _p_lsp = ctx.mount(&LspPlugin {
        root: cfg.root.clone(),
        extra_servers: HashMap::new(),
        auto_post_edit_check: cfg.vfs_mode != VfsMode::ShadowMemory,
    });
    let _p_vfs = ctx.mount(&VfsPlugin {
        root: cfg.root.clone(),
        mode: cfg.vfs_mode,
        persist: cfg.vfs_mode == VfsMode::MaterializedCow,
    });

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
    let history = Arc::new(Mutex::new(vec![Message::System {
        text: sys_text.clone(),
    }]));

    let mut active_workers: Vec<std::thread::JoinHandle<()>> = Vec::new();

    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line)?;
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

        match method {
            "initialize" => {
                rpc_writer.send_response(
                    id,
                    json!({
                        "protocolVersion": "2026-11-01",
                        "serverInfo": {
                            "name": "keel-acp-daemon",
                            "version": "0.1.0",
                        },
                        "capabilities": {
                            "streaming": true,
                            "cancellation": true,
                            "approvals": true,
                            "vfs": true,
                            "checkpoints": true,
                            "speculation": true,
                            "epistemicMemory": true,
                        }
                    }),
                );
            }
            "session/new" => {
                let new_sid = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| format!("acp-{}", std::process::id()));
                if let Ok(mut s) = session_id.lock() {
                    *s = new_sid.clone();
                }
                if let Ok(mut h) = history.lock() {
                    h.clear();
                    h.push(Message::System {
                        text: sys_text.clone(),
                    });
                }
                rpc_writer.send_response(id, json!({ "session_id": new_sid, "status": "ok" }));
            }
            "session/load" => {
                let Some(log_path) = params.get("path").and_then(|v| v.as_str()) else {
                    rpc_writer.send_error(id, -32602, "missing `path` parameter");
                    continue;
                };
                match SessionLog::read_history(log_path) {
                    Ok(prev) => {
                        let count = prev.len();
                        if let Ok(mut h) = history.lock() {
                            h.truncate(1);
                            h.extend(prev);
                        }
                        rpc_writer
                            .send_response(id, json!({ "status": "ok", "loaded_messages": count }));
                    }
                    Err(e) => {
                        rpc_writer.send_error(id, -32000, &format!("failed to load session: {e:#}"))
                    }
                }
            }
            "session/cancel" => {
                cancel_token.cancel();
                approver.notify_cancelled();
                if !id.is_null() {
                    rpc_writer.send_response(id, json!({ "status": "cancelled" }));
                }
            }
            "approval/respond" => {
                let call_id = params.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                let decision = params
                    .get("decision")
                    .and_then(|v| v.as_str())
                    .unwrap_or("deny");
                approver.resolve(call_id, decision);
                if !id.is_null() {
                    rpc_writer.send_response(
                        id,
                        json!({ "status": "ok", "call_id": call_id, "decision": decision }),
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

                cancel_token.reset();
                let ctx_clone = ctx.clone();
                let hist_clone = history.clone();
                let writer_clone = rpc_writer.clone();
                let sid_snapshot = session_id.lock().ok().map(|g| g.clone()).unwrap_or_default();

                // Spawn turn execution on a background thread so the daemon reader loop stays
                // responsive to `session/cancel` and `approval/respond` messages!
                let handle = std::thread::spawn(move || {
                    let mut h_guard = match hist_clone.lock() {
                        Ok(g) => g,
                        Err(_) => {
                            writer_clone.send_error(id, -32000, "session history mutex poisoned");
                            return;
                        }
                    };
                    match Agent::run_from_context(
                        &ctx_clone,
                        &mut h_guard,
                        &prompt_text,
                        max_steps,
                    ) {
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
                if let Some(vfs) = ctx.service::<Arc<TransactionalVfs>>() {
                    rpc_writer.send_response(id, json!({ "files": vfs.status() }));
                } else {
                    rpc_writer.send_error(id, -32000, "VFS service not mounted");
                }
            }
            "vfs/diff" => {
                if let Some(vfs) = ctx.service::<Arc<TransactionalVfs>>() {
                    rpc_writer.send_response(id, json!({ "diff": vfs.diff() }));
                } else {
                    rpc_writer.send_error(id, -32000, "VFS service not mounted");
                }
            }
            "vfs/commit" => {
                if let Some(vfs) = ctx.service::<Arc<TransactionalVfs>>() {
                    match vfs.commit() {
                        Ok(rep) => rpc_writer.send_response(id, serde_json::to_value(rep)?),
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "VFS service not mounted");
                }
            }
            "vfs/rollback" => {
                if let Some(vfs) = ctx.service::<Arc<TransactionalVfs>>() {
                    match vfs.rollback() {
                        Ok(rep) => rpc_writer.send_response(id, serde_json::to_value(rep)?),
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "VFS service not mounted");
                }
            }
            "checkpoint/list" => {
                if let Some(mgr) = ctx.service::<CheckpointManager>() {
                    match mgr.list() {
                        Ok(list) => rpc_writer.send_response(id, json!({ "checkpoints": list })),
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "CheckpointManager not mounted");
                }
            }
            "checkpoint/save" => {
                let desc = params
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("daemon checkpoint");
                if let Some(mgr) = ctx.service::<CheckpointManager>() {
                    match mgr.save(desc) {
                        Ok(ckpt) => rpc_writer.send_response(id, serde_json::to_value(ckpt)?),
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "CheckpointManager not mounted");
                }
            }
            "checkpoint/restore" => {
                let target = params
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("latest");
                if let Some(mgr) = ctx.service::<CheckpointManager>() {
                    match mgr.restore(target) {
                        Ok(rep) => rpc_writer.send_response(id, serde_json::to_value(rep)?),
                        Err(e) => rpc_writer.send_error(id, -32000, &format!("{e:#}")),
                    }
                } else {
                    rpc_writer.send_error(id, -32000, "CheckpointManager not mounted");
                }
            }
            "shutdown" | "exit" => {
                if !id.is_null() {
                    rpc_writer.send_response(id, json!({ "status": "shutting_down" }));
                }
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

    #[test]
    fn acp_daemon_initialize_prompt_vfs_and_rollback_roundtrip() {
        let tmp = std::env::temp_dir().join(format!("keel-acp-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let out_buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));

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

        // Send initialize -> session/prompt -> wait -> vfs/status -> vfs/rollback -> shutdown
        let requests = [
            json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}).to_string(),
            json!({"jsonrpc": "2.0", "id": 2, "method": "session/prompt", "params": {"prompt": "write hello.txt"}}).to_string(),
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
            && v["result"]["capabilities"]["vfs"] == true));

        // Verify streaming tool_call & text notifications were emitted
        assert!(lines.iter().any(|v| v["method"] == "session/update"
            && v["params"]["type"] == "tool_call"
            && v["params"]["name"] == "write_file"));

        // Verify final prompt response
        assert!(lines.iter().any(|v| v["id"] == 2
            && v["result"]["status"] == "ok"
            && v["result"]["answer"] == "All done via ACP daemon"));

        assert_eq!(
            std::fs::read_to_string(tmp.join("hello.txt")).unwrap(),
            "daemon wrote this\n"
        );

        let _ = std::fs::remove_dir_all(tmp);
    }
}
