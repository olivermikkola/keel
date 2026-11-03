//! Language Server Protocol (LSP 3.17) client and real-time post-edit diagnostics engine.
//!
//! Features:
//! - Standard HTTP-style `Content-Length: <N>\r\n\r\n` JSON-RPC 2.0 framing
//! - Background reader thread multiplexing request responses and asynchronous
//!   `textDocument/publishDiagnostics` notifications into a live [`DiagnosticsCache`]
//! - Fast compiler/syntax fallback (`cargo check --message-format=json`, `rustc`,
//!   `python3 -m py_compile`, `node --check`, `sh -n`, native JSON parser) when no
//!   warm language server is attached
//! - Automatic post-edit diagnostic feedback appended to `write_file`, `edit_file`,
//!   and `patch_file` tool outputs so the model repairs syntax/type errors immediately.

use crate::sandbox::ToolRegistry;
use crate::tools::confine;
use crate::{Tool, ToolSpec};
use anyhow::{bail, Context as _, Result};
use keel_kernel::{Plugin, Scope};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiagnosticItem {
    pub file: String,
    pub line: usize,
    pub column: usize,
    pub severity: String,
    pub message: String,
    pub source: String,
}

impl DiagnosticItem {
    pub fn format_compact(&self) -> String {
        format!(
            "{}:{}:{}: [{}] {} ({})",
            self.file, self.line, self.column, self.severity, self.message, self.source
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspServerConfig {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub extensions: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LspConfigFile {
    #[serde(default)]
    pub servers: HashMap<String, LspServerConfig>,
}

const MAX_LSP_FRAME_BYTES: usize = 32 * 1024 * 1024;
const MAX_LSP_HEADER_BYTES: usize = 8192;

/// Write a single LSP JSON-RPC message framed with `Content-Length: <N>\r\n\r\n`.
pub fn write_lsp_frame<W: Write>(writer: &mut W, payload: &Value) -> Result<()> {
    let body = serde_json::to_vec(payload)?;
    write!(writer, "Content-Length: {}\r\n\r\n", body.len())?;
    writer.write_all(&body)?;
    writer.flush()?;
    Ok(())
}

/// Read a single LSP JSON-RPC message framed with `Content-Length: <N>\r\n\r\n`.
pub fn read_lsp_frame<R: BufRead>(reader: &mut R) -> Result<Option<Value>> {
    let mut content_length: Option<usize> = None;
    loop {
        let mut line = String::new();
        use std::io::Read;
        let n = reader
            .by_ref()
            .take((MAX_LSP_HEADER_BYTES + 1) as u64)
            .read_line(&mut line)?;
        if n == 0 {
            return Ok(None);
        }
        if line.len() > MAX_LSP_HEADER_BYTES {
            bail!("LSP header line exceeds maximum allowable size (8KB)");
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if content_length.is_some() {
                break;
            }
            continue;
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            if k.trim().eq_ignore_ascii_case("Content-Length") {
                content_length = v.trim().parse::<usize>().ok();
            }
        }
    }
    let len = content_length.context("missing Content-Length header in LSP frame")?;
    if len > MAX_LSP_FRAME_BYTES {
        bail!("LSP frame Content-Length ({len} bytes) exceeds maximum limit (32MB)");
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    let val: Value = serde_json::from_slice(&buf).context("decoding LSP JSON-RPC body")?;
    Ok(Some(val))
}

#[derive(Default)]
struct SharedLspState {
    responses: HashMap<u64, Value>,
    diagnostics: HashMap<String, Vec<DiagnosticItem>>,
    closed: bool,
}

/// Active stdio LSP client with a background reader thread that processes
/// `Content-Length` frames and caches asynchronous `textDocument/publishDiagnostics`.
pub struct LspClient {
    name: String,
    root: PathBuf,
    stdin: Mutex<ChildStdin>,
    _child: Mutex<Child>,
    state: Arc<(Mutex<SharedLspState>, Condvar)>,
    next_id: AtomicU64,
}

impl LspClient {
    pub fn spawn(name: &str, root: &Path, cfg: &LspServerConfig) -> Result<Arc<Self>> {
        let mut cmd = Command::new(&cfg.command);
        cmd.args(&cfg.args)
            .current_dir(root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        for (k, v) in &cfg.env {
            cmd.env(k, v);
        }

        let mut child = cmd
            .spawn()
            .with_context(|| format!("spawning LSP server `{name}` ({})", cfg.command))?;
        let stdin = child.stdin.take().context("taking LSP stdin")?;
        let stdout = child.stdout.take().context("taking LSP stdout")?;

        let state = Arc::new((Mutex::new(SharedLspState::default()), Condvar::new()));
        let state_bg = state.clone();
        let root_bg = root.to_path_buf();
        let server_name = name.to_string();

        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            while let Ok(Some(msg)) = read_lsp_frame(&mut reader) {
                let (lock, cvar) = &*state_bg;
                if let Some(id) = msg.get("id").and_then(|v| v.as_u64()) {
                    if msg.get("result").is_some() || msg.get("error").is_some() {
                        if let Ok(mut guard) = lock.lock() {
                            guard.responses.insert(id, msg);
                            cvar.notify_all();
                        }
                        continue;
                    }
                }
                if msg.get("method").and_then(|m| m.as_str())
                    == Some("textDocument/publishDiagnostics")
                {
                    if let Some(params) = msg.get("params") {
                        let uri = params.get("uri").and_then(|u| u.as_str()).unwrap_or("");
                        let rel_file = uri_to_rel_path(&root_bg, uri);
                        let mut items = Vec::new();
                        if let Some(arr) = params.get("diagnostics").and_then(|d| d.as_array()) {
                            for d in arr {
                                let line =
                                    d.pointer("/range/start/line")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0) as usize
                                        + 1;
                                let col =
                                    d.pointer("/range/start/character")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0) as usize
                                        + 1;
                                let sev = match d.get("severity").and_then(|v| v.as_u64()) {
                                    Some(1) => "error",
                                    Some(2) => "warning",
                                    Some(3) => "info",
                                    Some(4) => "hint",
                                    _ => "error",
                                };
                                let message = d
                                    .get("message")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .trim()
                                    .to_string();
                                if !message.is_empty() {
                                    items.push(DiagnosticItem {
                                        file: rel_file.clone(),
                                        line,
                                        column: col,
                                        severity: sev.into(),
                                        message,
                                        source: server_name.clone(),
                                    });
                                }
                            }
                        }
                        if let Ok(mut guard) = lock.lock() {
                            guard.diagnostics.insert(rel_file, items);
                            cvar.notify_all();
                        }
                    }
                }
            }
            let (lock, cvar) = &*state_bg;
            if let Ok(mut guard) = lock.lock() {
                guard.closed = true;
                cvar.notify_all();
            }
        });

        let client = Arc::new(Self {
            name: name.to_string(),
            root: root.to_path_buf(),
            stdin: Mutex::new(stdin),
            _child: Mutex::new(child),
            state,
            next_id: AtomicU64::new(1),
        });
        client.initialize()?;
        Ok(client)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let req = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        {
            let mut stdin = self
                .stdin
                .lock()
                .map_err(|_| anyhow::anyhow!("LSP stdin mutex poisoned"))?;
            write_lsp_frame(&mut *stdin, &req)?;
        }

        let (lock, cvar) = &*self.state;
        let mut guard = lock
            .lock()
            .map_err(|_| anyhow::anyhow!("LSP state mutex poisoned"))?;
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(resp) = guard.responses.remove(&id) {
                if let Some(err) = resp.get("error") {
                    bail!("LSP `{}` error on `{method}`: {err}", self.name);
                }
                return Ok(resp.get("result").cloned().unwrap_or(Value::Null));
            }
            if guard.closed {
                bail!(
                    "LSP `{}` connection closed while awaiting `{method}`",
                    self.name
                );
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                bail!("LSP `{}` timed out waiting for `{method}`", self.name);
            }
            let (g, _) = cvar
                .wait_timeout(guard, deadline.saturating_duration_since(now))
                .map_err(|_| anyhow::anyhow!("LSP condvar poisoned"))?;
            guard = g;
        }
    }

    pub fn notify(&self, method: &str, params: Value) -> Result<()> {
        let msg = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        let mut stdin = self
            .stdin
            .lock()
            .map_err(|_| anyhow::anyhow!("LSP stdin mutex poisoned"))?;
        write_lsp_frame(&mut *stdin, &msg)
    }

    fn initialize(&self) -> Result<()> {
        let root_uri = path_to_uri(&self.root);
        let _ = self.request(
            "initialize",
            json!({
                "processId": std::process::id(),
                "rootUri": root_uri,
                "capabilities": {
                    "textDocument": {
                        "publishDiagnostics": { "relatedInformation": true },
                        "synchronization": { "didSave": true },
                        "definition": { "dynamicRegistration": false },
                        "hover": { "contentFormat": ["markdown", "plaintext"] }
                    }
                }
            }),
        )?;
        self.notify("initialized", json!({}))?;
        Ok(())
    }

    /// Notify the language server that a file was opened/updated on disk.
    pub fn sync_file(&self, rel_path: &str, content: &str) -> Result<()> {
        let full = self.root.join(rel_path);
        let uri = path_to_uri(&full);
        let lang = language_id_for_path(&full);
        self.notify(
            "textDocument/didOpen",
            json!({
                "textDocument": {
                    "uri": uri,
                    "languageId": lang,
                    "version": 1,
                    "text": content,
                }
            }),
        )?;
        self.notify(
            "textDocument/didChange",
            json!({
                "textDocument": { "uri": uri, "version": 2 },
                "contentChanges": [{ "text": content }]
            }),
        )?;
        self.notify(
            "textDocument/didSave",
            json!({
                "textDocument": { "uri": uri },
                "text": content
            }),
        )?;
        Ok(())
    }

    /// Query go-to-definition for 1-based `line` and `column`.
    pub fn definition(&self, rel_path: &str, line: usize, column: usize) -> Result<Value> {
        let full = self.root.join(rel_path);
        if let Ok(text) = std::fs::read_to_string(&full) {
            let _ = self.sync_file(rel_path, &text);
        }
        let uri = path_to_uri(&full);
        self.request(
            "textDocument/definition",
            json!({
                "textDocument": { "uri": uri },
                "position": {
                    "line": line.saturating_sub(1),
                    "character": column.saturating_sub(1),
                }
            }),
        )
    }

    /// Query hover documentation/type info for 1-based `line` and `column`.
    pub fn hover(&self, rel_path: &str, line: usize, column: usize) -> Result<Value> {
        let full = self.root.join(rel_path);
        if let Ok(text) = std::fs::read_to_string(&full) {
            let _ = self.sync_file(rel_path, &text);
        }
        let uri = path_to_uri(&full);
        self.request(
            "textDocument/hover",
            json!({
                "textDocument": { "uri": uri },
                "position": {
                    "line": line.saturating_sub(1),
                    "character": column.saturating_sub(1),
                }
            }),
        )
    }

    /// Retrieve cached diagnostics pushed by the language server for `rel_path`,
    /// waiting briefly (up to `wait_ms`) if none have arrived yet.
    pub fn diagnostics_for(&self, rel_path: &str, wait_ms: u64) -> Vec<DiagnosticItem> {
        let norm = rel_path.trim_start_matches("./");
        let (lock, cvar) = &*self.state;
        let Ok(guard) = lock.lock() else {
            return Vec::new();
        };
        if let Some(items) = guard.diagnostics.get(norm) {
            return items.clone();
        }
        if wait_ms > 0 && !guard.closed {
            let _ = cvar.wait_timeout(guard, Duration::from_millis(wait_ms));
            if let Ok(g2) = lock.lock() {
                if let Some(items) = g2.diagnostics.get(norm) {
                    return items.clone();
                }
            }
        }
        Vec::new()
    }
}

impl Drop for LspClient {
    fn drop(&mut self) {
        if let Ok(mut child) = self._child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Ok(mut guard) = self.state.0.lock() {
            guard.closed = true;
            self.state.1.notify_all();
        }
    }
}

/// Workspace-level LSP & compiler diagnostics service mounted on the kernel [`Context`].
pub struct LspService {
    root: PathBuf,
    clients: Mutex<HashMap<String, Arc<LspClient>>>,
    configs: HashMap<String, LspServerConfig>,
    pub auto_post_edit_check: bool,
}

impl LspService {
    pub fn new(
        root: PathBuf,
        configs: HashMap<String, LspServerConfig>,
        auto_post_edit_check: bool,
    ) -> Self {
        Self {
            root,
            clients: Mutex::new(HashMap::new()),
            configs,
            auto_post_edit_check,
        }
    }

    pub fn load_config(root: &Path) -> HashMap<String, LspServerConfig> {
        let p = root.join(".keel").join("lsp.json");
        if let Ok(data) = std::fs::read_to_string(&p) {
            if let Ok(parsed) = serde_json::from_str::<LspConfigFile>(&data) {
                return parsed.servers;
            }
        }
        HashMap::new()
    }

    fn client_for_extension(&self, ext: &str) -> Option<Arc<LspClient>> {
        let mut guard = self.clients.lock().ok()?;
        for (name, cfg) in &self.configs {
            let matches_ext = cfg.extensions.is_empty()
                || cfg
                    .extensions
                    .iter()
                    .any(|e| e.trim_start_matches('.').eq_ignore_ascii_case(ext));
            if !matches_ext {
                continue;
            }
            if let Some(existing) = guard.get(name) {
                return Some(existing.clone());
            }
            if let Ok(spawned) = LspClient::spawn(name, &self.root, cfg) {
                guard.insert(name.clone(), spawned.clone());
                return Some(spawned);
            }
        }
        None
    }

    /// Check a single file (relative or absolute path inside workspace) for diagnostics.
    /// Queries any configured LSP server first, then falls back to fast compiler/syntax validation.
    pub fn check_file(&self, rel_or_abs: &Path) -> Vec<DiagnosticItem> {
        let full = if rel_or_abs.is_absolute() {
            rel_or_abs.to_path_buf()
        } else {
            self.root.join(rel_or_abs)
        };
        let rel = full
            .strip_prefix(&self.root)
            .unwrap_or(&full)
            .to_string_lossy()
            .to_string();
        let ext = full
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        if let Some(client) = self.client_for_extension(&ext) {
            if let Ok(content) = std::fs::read_to_string(&full) {
                let _ = client.sync_file(&rel, &content);
                let diags = client.diagnostics_for(&rel, 120);
                if !diags.is_empty() {
                    return diags;
                }
            }
        }

        run_fast_syntax_check(&self.root, &full, &rel, &ext)
    }

    /// Format post-edit diagnostic feedback for tool output injection.
    /// Returns `None` if the edited file has zero errors/warnings.
    pub fn post_edit_feedback(&self, rel_path: &str) -> Option<String> {
        if !self.auto_post_edit_check {
            return None;
        }
        let items = self.check_file(Path::new(rel_path));
        if items.is_empty() {
            return None;
        }
        let mut lines = Vec::new();
        lines.push(format!(
            "[diagnostics: {} issue(s) detected in `{rel_path}`]",
            items.len()
        ));
        for item in items.iter().take(10) {
            lines.push(format!("  - {}", item.format_compact()));
        }
        if items.len() > 10 {
            lines.push(format!("  ... and {} more", items.len() - 10));
        }
        Some(lines.join("\n"))
    }

    pub fn definition(&self, rel_path: &str, line: usize, column: usize) -> Result<Value> {
        let ext = Path::new(rel_path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let client = self
            .client_for_extension(ext)
            .context("no LSP server configured for this file extension")?;
        client.definition(rel_path, line, column)
    }

    pub fn hover(&self, rel_path: &str, line: usize, column: usize) -> Result<Value> {
        let ext = Path::new(rel_path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let client = self
            .client_for_extension(ext)
            .context("no LSP server configured for this file extension")?;
        client.hover(rel_path, line, column)
    }
}

/// Fast zero-daemon syntax and compiler check fallback for common languages.
fn run_fast_syntax_check(
    root: &Path,
    full_path: &Path,
    rel_path: &str,
    ext: &str,
) -> Vec<DiagnosticItem> {
    if !full_path.is_file() {
        return Vec::new();
    }
    match ext {
        "json" => {
            if let Ok(text) = std::fs::read_to_string(full_path) {
                if let Err(e) = serde_json::from_str::<Value>(&text) {
                    return vec![DiagnosticItem {
                        file: rel_path.to_string(),
                        line: e.line(),
                        column: e.column(),
                        severity: "error".into(),
                        message: e.to_string(),
                        source: "json-syntax".into(),
                    }];
                }
            }
            Vec::new()
        }
        "py" => {
            let out = Command::new("python3")
                .args(["-m", "py_compile", &full_path.to_string_lossy()])
                .current_dir(root)
                .stdin(Stdio::null())
                .output();
            if let Ok(o) = out {
                if !o.status.success() {
                    let stderr = String::from_utf8_lossy(&o.stderr);
                    let line = extract_line_number(&stderr).unwrap_or(1);
                    let msg = stderr
                        .lines()
                        .rev()
                        .find(|l| !l.trim().is_empty())
                        .unwrap_or("Python syntax error")
                        .trim()
                        .to_string();
                    return vec![DiagnosticItem {
                        file: rel_path.to_string(),
                        line,
                        column: 1,
                        severity: "error".into(),
                        message: msg,
                        source: "py_compile".into(),
                    }];
                }
            }
            Vec::new()
        }
        "sh" | "bash" => {
            let out = Command::new("sh")
                .args(["-n", &full_path.to_string_lossy()])
                .current_dir(root)
                .stdin(Stdio::null())
                .output();
            if let Ok(o) = out {
                if !o.status.success() {
                    let stderr = String::from_utf8_lossy(&o.stderr).trim().to_string();
                    let line = extract_line_number(&stderr).unwrap_or(1);
                    return vec![DiagnosticItem {
                        file: rel_path.to_string(),
                        line,
                        column: 1,
                        severity: "error".into(),
                        message: stderr,
                        source: "sh -n".into(),
                    }];
                }
            }
            Vec::new()
        }
        "js" | "mjs" | "cjs" => {
            let out = Command::new("node")
                .args(["--check", &full_path.to_string_lossy()])
                .current_dir(root)
                .stdin(Stdio::null())
                .output();
            if let Ok(o) = out {
                if !o.status.success() {
                    let stderr = String::from_utf8_lossy(&o.stderr).trim().to_string();
                    let line = extract_line_number(&stderr).unwrap_or(1);
                    return vec![DiagnosticItem {
                        file: rel_path.to_string(),
                        line,
                        column: 1,
                        severity: "error".into(),
                        message: stderr.lines().take(3).collect::<Vec<_>>().join(" | "),
                        source: "node --check".into(),
                    }];
                }
            }
            Vec::new()
        }
        "rs" => {
            if root.join("Cargo.toml").is_file() {
                let out = Command::new("cargo")
                    .args(["check", "--message-format=json", "--quiet"])
                    .current_dir(root)
                    .stdin(Stdio::null())
                    .output();
                if let Ok(o) = out {
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    let mut items = Vec::new();
                    for line in stdout.lines() {
                        let Ok(val) = serde_json::from_str::<Value>(line) else {
                            continue;
                        };
                        if val.get("reason").and_then(|r| r.as_str()) != Some("compiler-message") {
                            continue;
                        }
                        let Some(msg_obj) = val.get("message") else {
                            continue;
                        };
                        let level = msg_obj
                            .get("level")
                            .and_then(|l| l.as_str())
                            .unwrap_or("error");
                        if level != "error" && level != "warning" {
                            continue;
                        }
                        let text = msg_obj
                            .get("message")
                            .and_then(|m| m.as_str())
                            .unwrap_or("")
                            .to_string();
                        if let Some(spans) = msg_obj.get("spans").and_then(|s| s.as_array()) {
                            let primary = spans
                                .iter()
                                .find(|s| {
                                    s.get("is_primary").and_then(|b| b.as_bool()) == Some(true)
                                })
                                .or_else(|| spans.first());
                            if let Some(sp) = primary {
                                let file_name = sp
                                    .get("file_name")
                                    .and_then(|f| f.as_str())
                                    .unwrap_or(rel_path)
                                    .to_string();
                                let line_num =
                                    sp.get("line_start").and_then(|l| l.as_u64()).unwrap_or(1)
                                        as usize;
                                let col_num =
                                    sp.get("column_start").and_then(|c| c.as_u64()).unwrap_or(1)
                                        as usize;
                                items.push(DiagnosticItem {
                                    file: file_name,
                                    line: line_num,
                                    column: col_num,
                                    severity: level.to_string(),
                                    message: text,
                                    source: "cargo check".into(),
                                });
                            }
                        }
                    }
                    return items;
                }
            }
            Vec::new()
        }
        _ => Vec::new(),
    }
}

fn extract_line_number(s: &str) -> Option<usize> {
    let lower = s.to_ascii_lowercase();
    if let Some(idx) = lower.find("line ") {
        let rest = &lower[idx + 5..];
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(n) = digits.parse::<usize>() {
            return Some(n);
        }
    }
    for part in s.split(':') {
        if let Ok(n) = part.trim().parse::<usize>() {
            if n > 0 {
                return Some(n);
            }
        }
    }
    None
}

fn path_to_uri(p: &Path) -> String {
    let abs = p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    format!("file://{}", abs.to_string_lossy())
}

fn uri_to_rel_path(root: &Path, uri: &str) -> String {
    let raw = uri.strip_prefix("file://").unwrap_or(uri);
    let p = Path::new(raw);
    let canon_root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    if let Ok(rel) = p.strip_prefix(&canon_root) {
        return rel.to_string_lossy().to_string();
    }
    if let Ok(rel) = p.strip_prefix(root) {
        return rel.to_string_lossy().to_string();
    }
    raw.to_string()
}

fn language_id_for_path(p: &Path) -> &'static str {
    match p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "rs" => "rust",
        "py" => "python",
        "ts" | "tsx" => "typescript",
        "js" | "jsx" | "mjs" | "cjs" => "javascript",
        "go" => "go",
        "json" => "json",
        "sh" | "bash" => "shellscript",
        "toml" => "toml",
        "md" => "markdown",
        _ => "plaintext",
    }
}

// ---- Kernel Tools exposed by LspPlugin ----

pub struct LspDiagnosticsTool {
    pub root: PathBuf,
    pub service: Arc<LspService>,
}

impl Tool for LspDiagnosticsTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "lsp_diagnostics".into(),
            description:
                "Check a workspace file for LSP or compiler/syntax diagnostics (errors and warnings)."
                    .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Relative file path in the workspace" }
                },
                "required": ["path"]
            }),
        }
    }

    fn mutating(&self) -> bool {
        false
    }

    fn run(&self, args: &Value) -> Result<String> {
        let rel = args["path"]
            .as_str()
            .context("missing string argument `path`")?;
        let full = confine(&self.root, rel)?;
        let items = self.service.check_file(&full);
        if items.is_empty() {
            return Ok(format!("no diagnostics for {rel}"));
        }
        let formatted: Vec<String> = items.iter().map(|d| d.format_compact()).collect();
        Ok(formatted.join("\n"))
    }
}

pub struct LspDefinitionTool {
    pub root: PathBuf,
    pub service: Arc<LspService>,
}

impl Tool for LspDefinitionTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "lsp_definition".into(),
            description:
                "Query the configured Language Server for symbol definition at 1-based line and column."
                    .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "line": { "type": "integer" },
                    "column": { "type": "integer" }
                },
                "required": ["path", "line", "column"]
            }),
        }
    }

    fn mutating(&self) -> bool {
        false
    }

    fn run(&self, args: &Value) -> Result<String> {
        let rel = args["path"]
            .as_str()
            .context("missing string argument `path`")?;
        let _ = confine(&self.root, rel)?;
        let line = args["line"].as_u64().unwrap_or(1) as usize;
        let column = args["column"].as_u64().unwrap_or(1) as usize;
        let res = self.service.definition(rel, line, column)?;
        Ok(serde_json::to_string_pretty(&res)?)
    }
}

pub struct LspHoverTool {
    pub root: PathBuf,
    pub service: Arc<LspService>,
}

impl Tool for LspHoverTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "lsp_hover".into(),
            description:
                "Query the configured Language Server for hover type/documentation at 1-based line and column."
                    .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "line": { "type": "integer" },
                    "column": { "type": "integer" }
                },
                "required": ["path", "line", "column"]
            }),
        }
    }

    fn mutating(&self) -> bool {
        false
    }

    fn run(&self, args: &Value) -> Result<String> {
        let rel = args["path"]
            .as_str()
            .context("missing string argument `path`")?;
        let _ = confine(&self.root, rel)?;
        let line = args["line"].as_u64().unwrap_or(1) as usize;
        let column = args["column"].as_u64().unwrap_or(1) as usize;
        let res = self.service.hover(rel, line, column)?;
        Ok(serde_json::to_string_pretty(&res)?)
    }
}

/// Plugin that mounts [`LspService`] and registers `lsp_diagnostics`, `lsp_definition`,
/// and `lsp_hover` tools into the [`ToolRegistry`].
pub struct LspPlugin {
    pub root: PathBuf,
    pub extra_servers: HashMap<String, LspServerConfig>,
    pub auto_post_edit_check: bool,
}

impl Plugin for LspPlugin {
    fn name(&self) -> &str {
        "lsp"
    }

    fn apply(&self, scope: &mut Scope) {
        let mut configs = LspService::load_config(&self.root);
        for (k, v) in &self.extra_servers {
            configs.insert(k.clone(), v.clone());
        }
        let service = Arc::new(LspService::new(
            self.root.clone(),
            configs,
            self.auto_post_edit_check,
        ));
        scope.provide::<Arc<LspService>>(service.clone());

        let reg = scope
            .context()
            .service::<ToolRegistry>()
            .map(|r| (*r).clone())
            .unwrap_or_default();
        reg.register(Arc::new(LspDiagnosticsTool {
            root: self.root.clone(),
            service: service.clone(),
        }));
        reg.register(Arc::new(LspDefinitionTool {
            root: self.root.clone(),
            service: service.clone(),
        }));
        reg.register(Arc::new(LspHoverTool {
            root: self.root.clone(),
            service,
        }));
        scope.provide(reg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn lsp_content_length_frame_roundtrip() {
        let msg = json!({
            "jsonrpc": "2.0",
            "id": 42,
            "method": "textDocument/hover",
            "params": { "line": 10 }
        });
        let mut wire = Vec::new();
        write_lsp_frame(&mut wire, &msg).unwrap();
        let header_str = String::from_utf8_lossy(&wire);
        assert!(header_str.starts_with("Content-Length: "));
        assert!(header_str.contains("\r\n\r\n"));

        let mut reader = Cursor::new(wire);
        let decoded = read_lsp_frame(&mut reader).unwrap().unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn mock_lsp_server_publish_diagnostics_and_hover() {
        let tmp = std::env::temp_dir().join(format!("keel-lsp-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let py_file = tmp.join("app.py");
        std::fs::write(&py_file, "x = 1\n").unwrap();

        // Write a tiny Python mock LSP server that speaks Content-Length JSON-RPC
        let mock_server = tmp.join("mock_lsp.py");
        std::fs::write(
            &mock_server,
            r#"import sys, json

def send(obj):
    body = json.dumps(obj).encode('utf-8')
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode('ascii') + body)
    sys.stdout.buffer.flush()

while True:
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            sys.exit(0)
        s = line.decode('ascii', errors='ignore').strip()
        if not s:
            if headers:
                break
            continue
        if ':' in s:
            k, v = s.split(':', 1)
            headers[k.strip().lower()] = int(v.strip())
    body = sys.stdin.buffer.read(headers['content-length'])
    msg = json.loads(body.decode('utf-8'))
    method = msg.get('method', '')
    if method == 'initialize':
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {"capabilities": {"hoverProvider": True, "definitionProvider": True}}})
    elif method == 'textDocument/didSave':
        uri = msg["params"]["textDocument"]["uri"]
        send({
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": uri,
                "diagnostics": [{
                    "range": {"start": {"line": 0, "character": 4}, "end": {"line": 0, "character": 5}},
                    "severity": 1,
                    "message": "type mismatch: expected str, got int"
                }]
            }
        })
    elif method == 'textDocument/hover':
        send({
            "jsonrpc": "2.0",
            "id": msg["id"],
            "result": {"contents": {"kind": "markdown", "value": "`x: int`"}}
        })
"#,
        )
        .unwrap();

        let mut configs = HashMap::new();
        configs.insert(
            "mock-py".to_string(),
            LspServerConfig {
                command: "python3".into(),
                args: vec![mock_server.to_string_lossy().to_string()],
                extensions: vec!["py".into()],
                env: HashMap::new(),
            },
        );

        let svc = LspService::new(tmp.clone(), configs, true);
        let diags = svc.check_file(Path::new("app.py"));
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].line, 1);
        assert_eq!(diags[0].column, 5);
        assert!(diags[0].message.contains("type mismatch"));

        let hover = svc.hover("app.py", 1, 1).unwrap();
        assert_eq!(hover["contents"]["value"].as_str(), Some("`x: int`"));

        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn fast_syntax_fallback_catches_invalid_json_and_python() {
        let tmp = std::env::temp_dir().join(format!("keel-lsp-fallback-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        std::fs::write(tmp.join("broken.json"), "{\"a\": 1, }\n").unwrap();
        std::fs::write(tmp.join("broken.py"), "def foo(\n    return 1\n").unwrap();

        let svc = LspService::new(tmp.clone(), HashMap::new(), true);
        let json_diags = svc.check_file(Path::new("broken.json"));
        assert_eq!(json_diags.len(), 1);
        assert_eq!(json_diags[0].source, "json-syntax");

        let py_diags = svc.check_file(Path::new("broken.py"));
        assert_eq!(py_diags.len(), 1);
        assert_eq!(py_diags[0].source, "py_compile");

        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn read_lsp_frame_enforces_size_limits() {
        // Enormous Content-Length must be rejected to prevent OOM
        let bad_header = "Content-Length: 1000000000\r\n\r\n";
        let mut cursor = std::io::Cursor::new(bad_header.as_bytes());
        let res = read_lsp_frame(&mut cursor);
        assert!(res.is_err());
        assert!(res
            .unwrap_err()
            .to_string()
            .contains("exceeds maximum limit"));
    }
}
