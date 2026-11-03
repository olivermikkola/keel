//! Streaming tool-call interception, predictive pre-warming & early security sentinel.
//!
//! Standard agents wait for an LLM to generate all tokens of a tool call before
//! parsing JSON and verifying security policies. This introduces two critical flaws:
//! 1. Security latency: Destructive or unauthorized commands (e.g., `rm -rf /`,
//!    secret scraping, path traversals) waste hundreds of tokens before detection.
//! 2. I/O latency: File reading and AST outline indexing wait until token generation
//!    completes instead of pipelining I/O during model emission.
//!
//! The [`StreamingSentinel`] incrementally inspects streaming JSON argument fragments
//! as tokens arrive, early-vetoing dangerous commands mid-stream and pre-warming
//! file read buffers before the model finishes typing.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

/// Incremental JSON argument lexer/parser that extracts partial string fields
/// from incomplete, streaming JSON fragments without requiring valid JSON closures.
#[derive(Debug, Default, Clone)]
pub struct IncrementalArgParser {
    raw_buffer: String,
    fields: HashMap<String, String>,
    pending_key: String,
    current_key: Option<String>,
    in_string: bool,
    escape_next: bool,
}

impl IncrementalArgParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed an incremental slice of streaming arguments JSON.
    pub fn feed(&mut self, chunk: &str) {
        for ch in chunk.chars() {
            self.raw_buffer.push(ch);

            if self.escape_next {
                let target = if let Some(ref key) = self.current_key {
                    self.fields.entry(key.clone()).or_default()
                } else {
                    &mut self.pending_key
                };
                match ch {
                    'n' => target.push('\n'),
                    'r' => target.push('\r'),
                    't' => target.push('\t'),
                    '"' => target.push('"'),
                    '\\' => target.push('\\'),
                    '/' => target.push('/'),
                    other => {
                        target.push('\\');
                        target.push(other);
                    }
                }
                self.escape_next = false;
                continue;
            }

            if ch == '\\' && self.in_string {
                self.escape_next = true;
                continue;
            }

            if ch == '"' {
                if !self.in_string {
                    // Starting a quoted string. Determine if this string is a key or a value.
                    self.in_string = true;
                    if self.current_key.is_none() {
                        self.pending_key.clear();
                    }
                } else {
                    // Ending a quoted string
                    self.in_string = false;
                    if self.current_key.is_none() {
                        if !self.pending_key.is_empty() {
                            self.current_key = Some(std::mem::take(&mut self.pending_key));
                        }
                    } else {
                        // Finished parsing a string value for `current_key`
                        self.current_key = None;
                    }
                }
                continue;
            }

            // If we are currently inside a string value for an established key or a key name, append the char
            if self.in_string {
                if let Some(ref key) = self.current_key {
                    self.fields.entry(key.clone()).or_default().push(ch);
                } else {
                    self.pending_key.push(ch);
                }
            } else if ch == ':' && self.current_key.is_none() {
                if !self.pending_key.is_empty() {
                    self.current_key = Some(std::mem::take(&mut self.pending_key));
                }
            } else if ch == ',' || ch == '}' {
                // Delimiter between fields
                self.current_key = None;
                self.pending_key.clear();
            }
        }
    }

    /// Read the extracted string value for a given key, if observed so far.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.fields.get(key).map(|s| s.as_str())
    }

    /// Returns all partially or fully parsed key-value string fields.
    pub fn fields(&self) -> &HashMap<String, String> {
        &self.fields
    }

    /// Raw accumulated buffer.
    pub fn raw(&self) -> &str {
        &self.raw_buffer
    }
}

/// Operational mode for the Streaming Sentinel.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SentinelMode {
    /// Strict: Veto events abort the tool execution or stream immediately.
    #[default]
    Strict,
    /// Warn: Veto events emit warnings without terminating execution.
    Warn,
    /// Disabled: Sentinel checks are bypassed.
    Disabled,
}

/// Category of security or safety violation detected by the sentinel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VetoCategory {
    DestructiveCommand,
    PathTraversal,
    ProtectedSystemRoot,
    CredentialScraping,
    ForceBranchPush,
    CustomPolicy,
}

/// Detailed record of an early security veto.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VetoEvent {
    pub id: String,
    pub category: VetoCategory,
    pub tool_name: String,
    pub matched_pattern: String,
    pub trigger_payload: String,
    pub reason: String,
    pub timestamp_epoch_ms: u64,
}

/// A configurable sentinel inspection rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentinelRule {
    pub id: String,
    pub category: VetoCategory,
    pub description: String,
    pub tool_match: Option<String>,
    pub parameter_key: String,
    pub patterns: Vec<String>,
}

/// High-velocity streaming safety sentinel.
pub struct StreamingSentinel {
    mode: RwLock<SentinelMode>,
    rules: RwLock<Vec<SentinelRule>>,
    vetoes: Mutex<Vec<VetoEvent>>,
    scanned_chunks: AtomicU64,
    veto_count: AtomicU64,
}

impl Default for StreamingSentinel {
    fn default() -> Self {
        Self::new(SentinelMode::Strict)
    }
}

/// Helper to match tool names, treating aliases like "bash" and "shell" interchangeably,
/// and supporting comma-separated lists in rule configurations (e.g. "shell,bash").
pub fn matches_tool_name(rule_tool: &str, candidate_tool: &str) -> bool {
    let candidate = candidate_tool.trim().to_lowercase();
    rule_tool.split(',').any(|t| {
        let pat = t.trim().to_lowercase();
        if pat == candidate {
            return true;
        }
        if (pat == "bash" || pat == "shell" || pat == "sh")
            && (candidate == "bash" || candidate == "shell" || candidate == "sh")
        {
            return true;
        }
        false
    })
}

impl StreamingSentinel {
    /// Create a new Streaming Sentinel with standard defense-in-depth rules.
    pub fn new(mode: SentinelMode) -> Self {
        let mut sentinel = Self {
            mode: RwLock::new(mode),
            rules: RwLock::new(Vec::new()),
            vetoes: Mutex::new(Vec::new()),
            scanned_chunks: AtomicU64::new(0),
            veto_count: AtomicU64::new(0),
        };
        sentinel.install_builtin_rules();
        sentinel
    }

    /// Install default curve-setting security rules.
    fn install_builtin_rules(&mut self) {
        let mut rules = self.rules.write().expect("lock poisoned");

        // Rule 1: Destructive file wipes and root formatting
        rules.push(SentinelRule {
            id: "destructive-shell-wipe".into(),
            category: VetoCategory::DestructiveCommand,
            description: "Detects destructive recursive root removals and disk writes".into(),
            tool_match: Some("shell,bash".into()),
            parameter_key: "command".into(),
            patterns: vec![
                "rm -rf /".into(),
                "rm -rf /*".into(),
                "rm -fr /".into(),
                "rm -rf ~".into(),
                "rm -rf $HOME".into(),
                "rm -rf /etc".into(),
                "rm -rf /usr".into(),
                "rm -rf /var".into(),
                "rm -rf /boot".into(),
                "mkfs".into(),
                "dd if=/dev/zero".into(),
                "dd if=/dev/urandom".into(),
                "> /dev/sda".into(),
                "> /dev/nvme".into(),
                ":(){ :|:& };:".into(),
                "chmod -R 777 /".into(),
                "chmod 777 /".into(),
            ],
        });

        // Rule 2: Force pushes to primary git branches
        rules.push(SentinelRule {
            id: "git-force-push-main".into(),
            category: VetoCategory::ForceBranchPush,
            description: "Detects force push targeting main or master branches".into(),
            tool_match: Some("shell,bash".into()),
            parameter_key: "command".into(),
            patterns: vec![
                "push --force origin main".into(),
                "push origin --force main".into(),
                "push origin main --force".into(),
                "push -f origin main".into(),
                "push origin -f main".into(),
                "push origin main -f".into(),
                "push --force origin master".into(),
                "push origin --force master".into(),
                "push origin master --force".into(),
                "push -f origin master".into(),
                "push origin -f master".into(),
                "push origin master -f".into(),
                "push origin +main".into(),
                "push origin +master".into(),
            ],
        });

        // Rule 3: Credential and secret exfiltration
        rules.push(SentinelRule {
            id: "credential-scraping".into(),
            category: VetoCategory::CredentialScraping,
            description: "Detects attempts to read SSH keys, system shadow, or cloud credentials"
                .into(),
            tool_match: Some("shell,bash".into()),
            parameter_key: "command".into(),
            patterns: vec![
                "cat ~/.ssh/".into(),
                "cat /etc/shadow".into(),
                "cat ~/.aws/".into(),
                "cat ~/.aws/credentials".into(),
                "cat ~/.config/gh/hosts.yml".into(),
                ".ssh/id_rsa".into(),
                ".ssh/id_ed25519".into(),
            ],
        });

        // Rule 4: Path traversal attacks
        rules.push(SentinelRule {
            id: "path-traversal-escape".into(),
            category: VetoCategory::PathTraversal,
            description: "Detects directory traversal attempting to escape project bounds".into(),
            tool_match: None, // Checks all tools (read_file, write_file, edit_file, patch_file)
            parameter_key: "path".into(),
            patterns: vec![
                "../../..".into(),
                "../..".into(),
                "../".into(),
                "..\\".into(),
                "/etc/passwd".into(),
                "/etc/shadow".into(),
                "/etc/".into(),
                "/root/".into(),
                "~/.ssh".into(),
                "/proc/".into(),
                "/sys/".into(),
                "/dev/".into(),
            ],
        });

        // Rule 5: Protected repository structures
        rules.push(SentinelRule {
            id: "protected-root-mutation".into(),
            category: VetoCategory::ProtectedSystemRoot,
            description: "Blocks direct mutations targeting .git internal objects".into(),
            tool_match: None,
            parameter_key: "path".into(),
            patterns: vec![
                ".git/objects".into(),
                ".git/refs".into(),
                ".git/HEAD".into(),
                ".git/config".into(),
            ],
        });
    }

    /// Add a custom user-defined sentinel rule.
    pub fn add_rule(&self, rule: SentinelRule) {
        let mut rules = self.rules.write().expect("lock poisoned");
        rules.push(rule);
    }

    /// Set operational mode.
    pub fn set_mode(&self, mode: SentinelMode) {
        let mut m = self.mode.write().expect("lock poisoned");
        *m = mode;
    }

    /// Get current operational mode.
    pub fn mode(&self) -> SentinelMode {
        *self.mode.read().expect("lock poisoned")
    }

    /// Inspect a streaming tool call argument map and check for early veto triggers.
    pub fn inspect(
        &self,
        tool_name: &str,
        parsed_fields: &HashMap<String, String>,
    ) -> Option<VetoEvent> {
        let mode = self.mode();
        if mode == SentinelMode::Disabled {
            return None;
        }

        self.scanned_chunks.fetch_add(1, Ordering::Relaxed);
        let rules = self.rules.read().expect("lock poisoned");

        for rule in rules.iter() {
            if let Some(ref target_tool) = rule.tool_match {
                if !matches_tool_name(target_tool, tool_name) {
                    continue;
                }
            }

            if let Some(val) = parsed_fields.get(&rule.parameter_key) {
                let norm_val = val
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
                    .to_ascii_lowercase();
                for pattern in &rule.patterns {
                    let norm_pat = pattern
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" ")
                        .to_ascii_lowercase();
                    if val.contains(pattern) || norm_val.contains(&norm_pat) {
                        let event = VetoEvent {
                            id: format!("veto-{}", self.veto_count.fetch_add(1, Ordering::SeqCst) + 1),
                            category: rule.category.clone(),
                            tool_name: tool_name.to_string(),
                            matched_pattern: pattern.clone(),
                            trigger_payload: val.clone(),
                            reason: format!(
                                "Early security veto triggered by rule '{}' ({:?}): pattern '{}' matched parameter '{}'",
                                rule.id, rule.category, pattern, rule.parameter_key
                            ),
                            timestamp_epoch_ms: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_millis() as u64)
                                .unwrap_or(0),
                        };

                        let mut v = self.vetoes.lock().expect("lock poisoned");
                        v.push(event.clone());
                        return Some(event);
                    }
                }
            }
        }

        None
    }

    /// Check an arbitrary command string against destructive shell rules.
    pub fn check_command(&self, command: &str) -> Option<VetoEvent> {
        let mut fields = HashMap::new();
        fields.insert("command".into(), command.to_string());
        self.inspect("bash", &fields)
    }

    /// Retrieve all recorded veto events.
    pub fn veto_history(&self) -> Vec<VetoEvent> {
        self.vetoes.lock().expect("lock poisoned").clone()
    }

    /// Summary telemetry metrics.
    pub fn metrics(&self) -> Value {
        serde_json::json!({
            "mode": self.mode(),
            "rule_count": self.rules.read().expect("lock poisoned").len(),
            "veto_count": self.veto_count.load(Ordering::Relaxed),
            "scanned_evaluations": self.scanned_chunks.load(Ordering::Relaxed),
            "recent_vetoes": self.veto_history().into_iter().rev().take(5).collect::<Vec<_>>(),
        })
    }
}

const MAX_PREWARM_FILE_SIZE: u64 = 1024 * 1024; // 1 MB cap on prewarmed file content
const MAX_PREWARM_ENTRIES: usize = 64; // Max entries in LRU prewarm cache
const PREWARM_TTL: std::time::Duration = std::time::Duration::from_secs(30);

/// A prewarmed file plus the on-disk fingerprint it was read under.
struct WarmEntry {
    loaded: Instant,
    mtime: Option<std::time::SystemTime>,
    len: u64,
    content: String,
}

impl WarmEntry {
    /// Still valid only if young enough and the file is unchanged on disk.
    fn is_fresh(&self, path: &Path) -> bool {
        if self.loaded.elapsed() > PREWARM_TTL {
            return false;
        }
        match std::fs::symlink_metadata(path) {
            Ok(m) => {
                m.file_type().is_file() && m.len() == self.len && m.modified().ok() == self.mtime
            }
            Err(_) => false,
        }
    }
}

/// Predictive Pre-warming Cache and Engine.
///
/// Pre-loads file contents or initiates background AST symbol outline parsing
/// as soon as the target `path` argument is streamed, eliminating disk wait times.
pub struct PrewarmEngine {
    workspace_root: PathBuf,
    file_cache: RwLock<HashMap<PathBuf, WarmEntry>>,
    prewarm_requests: AtomicU64,
    cache_hits: AtomicU64,
}

impl PrewarmEngine {
    pub fn new(workspace_root: impl Into<PathBuf>) -> Self {
        Self {
            workspace_root: workspace_root.into(),
            file_cache: RwLock::new(HashMap::new()),
            prewarm_requests: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
        }
    }

    /// Inspect a streaming tool call for pre-warmable file read/outline targets.
    pub fn check_and_prewarm(&self, tool_name: &str, fields: &HashMap<String, String>) -> bool {
        if !matches!(tool_name, "read_file" | "outline_symbols" | "get_symbol") {
            return false;
        }

        let Some(path_str) = fields.get("path") else {
            return false;
        };

        if path_str.trim().is_empty() {
            return false;
        }

        // Confine target strictly within workspace bounds to prevent out-of-bounds exfiltration
        let Ok(path) = crate::tools::confine(&self.workspace_root, path_str) else {
            return false;
        };

        let canon_root = self
            .workspace_root
            .canonicalize()
            .unwrap_or_else(|_| self.workspace_root.clone());
        let Ok(canon_path) = path.canonicalize() else {
            return false;
        };
        if !canon_path.starts_with(&canon_root) {
            return false;
        }

        // If file exists, is a regular file, and within size threshold
        let Ok(meta) = std::fs::symlink_metadata(&canon_path) else {
            return false;
        };
        if !meta.file_type().is_file() || meta.len() > MAX_PREWARM_FILE_SIZE {
            return false;
        }

        self.prewarm_requests.fetch_add(1, Ordering::Relaxed);
        let mut cache = self.file_cache.write().expect("lock poisoned");

        // Evict oldest entry if capacity reached
        if cache.len() >= MAX_PREWARM_ENTRIES && !cache.contains_key(&canon_path) {
            if let Some(oldest_key) = cache
                .iter()
                .min_by_key(|(_, e)| e.loaded)
                .map(|(k, _)| k.clone())
            {
                cache.remove(&oldest_key);
            }
        }

        // Keep a still-fresh entry; refresh anything stale.
        if cache
            .get(&canon_path)
            .is_some_and(|e| e.is_fresh(&canon_path))
        {
            return false;
        }
        use std::io::Read;
        if let Ok(mut f) = std::fs::File::open(&canon_path) {
            let mut buf = Vec::new();
            if f.by_ref()
                .take(MAX_PREWARM_FILE_SIZE)
                .read_to_end(&mut buf)
                .is_ok()
            {
                if let Ok(content) = String::from_utf8(buf) {
                    cache.insert(
                        canon_path,
                        WarmEntry {
                            loaded: Instant::now(),
                            mtime: meta.modified().ok(),
                            len: meta.len(),
                            content,
                        },
                    );
                    return true;
                }
            }
        }

        false
    }

    /// Drop every warmed entry (call after any tool that may have changed files).
    pub fn invalidate_all(&self) {
        self.file_cache.write().expect("lock poisoned").clear();
    }

    /// Retrieve warmed file contents, recording a cache hit if found.
    pub fn get_warmed(&self, path: &Path) -> Option<String> {
        let canon_root = self
            .workspace_root
            .canonicalize()
            .unwrap_or_else(|_| self.workspace_root.clone());
        let full_path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.workspace_root.join(path)
        };
        let Ok(canon_path) = full_path.canonicalize() else {
            return None;
        };
        if !canon_path.starts_with(&canon_root) {
            return None;
        }

        let hit = {
            let cache = self.file_cache.read().expect("lock poisoned");
            match cache.get(&canon_path) {
                Some(e) if e.is_fresh(&canon_path) => Some(e.content.clone()),
                Some(_) => None,
                None => return None,
            }
        };
        match hit {
            Some(content) => {
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                Some(content)
            }
            None => {
                // Stale: evict so memory is released and the next read goes to disk.
                self.file_cache
                    .write()
                    .expect("lock poisoned")
                    .remove(&canon_path);
                None
            }
        }
    }

    /// Telemetry metrics for pre-warming pipeline.
    pub fn metrics(&self) -> Value {
        serde_json::json!({
            "prewarm_requests": self.prewarm_requests.load(Ordering::Relaxed),
            "cache_hits": self.cache_hits.load(Ordering::Relaxed),
            "cached_entries": self.file_cache.read().expect("lock poisoned").len(),
        })
    }
}

/// Kernel plugin that mounts the [`StreamingSentinel`] into the context.
pub struct SentinelPlugin(pub Arc<StreamingSentinel>);

impl keel_kernel::Plugin for SentinelPlugin {
    fn name(&self) -> &str {
        "streaming-sentinel"
    }
    fn apply(&self, scope: &mut keel_kernel::Scope) {
        scope.provide::<Arc<StreamingSentinel>>(self.0.clone());
    }
}

/// Kernel plugin that mounts the [`PrewarmEngine`] into the context.
pub struct PrewarmPlugin(pub Arc<PrewarmEngine>);

impl keel_kernel::Plugin for PrewarmPlugin {
    fn name(&self) -> &str {
        "predictive-prewarm"
    }
    fn apply(&self, scope: &mut keel_kernel::Scope) {
        scope.provide::<Arc<PrewarmEngine>>(self.0.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incremental_parser_handles_split_arguments() {
        let mut parser = IncrementalArgParser::new();
        parser.feed(r#"{"command": "#);
        assert_eq!(parser.get("command"), None);

        parser.feed(r#""cargo test"#);
        assert_eq!(parser.get("command"), Some("cargo test"));

        parser.feed(r#" --workspace", "path": "src"#);
        assert_eq!(parser.get("command"), Some("cargo test --workspace"));
        assert_eq!(parser.get("path"), Some("src"));

        parser.feed(r#"/main.rs"}"#);
        assert_eq!(parser.get("path"), Some("src/main.rs"));
    }

    #[test]
    fn incremental_parser_handles_escapes_and_newlines() {
        let mut parser = IncrementalArgParser::new();
        parser.feed(r#"{"text": "line1\nline2\"quote\"\\slash"}"#);
        assert_eq!(parser.get("text"), Some("line1\nline2\"quote\"\\slash"));
    }

    #[test]
    fn sentinel_catches_destructive_rm_rf_root() {
        let sentinel = StreamingSentinel::default();
        let mut parser = IncrementalArgParser::new();

        // Feed safe initial fragment
        parser.feed(r#"{"command": "echo hello && "#);
        assert!(sentinel.inspect("bash", parser.fields()).is_none());

        // Feed destructive payload
        parser.feed(r#"rm -rf / --no-preserve-root"}"#);
        let veto = sentinel.inspect("bash", parser.fields());
        assert!(veto.is_some());
        let veto = veto.unwrap();
        assert_eq!(veto.category, VetoCategory::DestructiveCommand);
        assert_eq!(veto.matched_pattern, "rm -rf /");
        assert_eq!(sentinel.veto_history().len(), 1);
    }

    #[test]
    fn sentinel_blocks_git_force_push_to_main() {
        let sentinel = StreamingSentinel::default();
        let mut parser = IncrementalArgParser::new();
        parser.feed(r#"{"command": "git push --force origin main"}"#);
        let veto = sentinel
            .inspect("bash", parser.fields())
            .expect("must veto");
        assert_eq!(veto.category, VetoCategory::ForceBranchPush);
    }

    #[test]
    fn sentinel_intercepts_shell_alias_and_credential_scraping() {
        let sentinel = StreamingSentinel::default();

        // 1. Verify "shell" tool triggers destructive wipe rule
        let mut fields = HashMap::new();
        fields.insert("command".into(), "rm -rf / --no-preserve-root".into());
        let veto = sentinel
            .inspect("shell", &fields)
            .expect("sentinel must match 'shell' tool for bash rules");
        assert_eq!(veto.category, VetoCategory::DestructiveCommand);

        // 2. Verify "shell" tool triggers git force push rule
        let mut fields2 = HashMap::new();
        fields2.insert("command".into(), "git push --force origin main".into());
        let veto2 = sentinel
            .inspect("shell", &fields2)
            .expect("sentinel must catch git force push via 'shell'");
        assert_eq!(veto2.category, VetoCategory::ForceBranchPush);

        // 3. Verify credential scraping rule on "shell"
        let mut fields3 = HashMap::new();
        fields3.insert("command".into(), "cat ~/.ssh/id_rsa".into());
        let veto3 = sentinel
            .inspect("shell", &fields3)
            .expect("sentinel must catch credential scraping via 'shell'");
        assert_eq!(veto3.category, VetoCategory::CredentialScraping);

        // 4. Verify path traversal escape with "../"
        let mut fields4 = HashMap::new();
        fields4.insert("path".into(), "../secret.txt".into());
        let veto4 = sentinel
            .inspect("read_file", &fields4)
            .expect("sentinel must catch relative directory traversal");
        assert_eq!(veto4.category, VetoCategory::PathTraversal);
    }

    #[test]
    fn sentinel_blocks_path_traversal_on_file_tools() {
        let sentinel = StreamingSentinel::default();
        let mut parser = IncrementalArgParser::new();
        parser.feed(r#"{"path": "../../../etc/passwd", "content": "bad"}"#);

        let veto = sentinel
            .inspect("write_file", parser.fields())
            .expect("must veto");
        assert_eq!(veto.category, VetoCategory::PathTraversal);
    }

    #[test]
    fn sentinel_blocks_protected_git_mutation() {
        let sentinel = StreamingSentinel::default();
        let mut parser = IncrementalArgParser::new();
        parser.feed(r#"{"path": ".git/HEAD"}"#);

        let veto = sentinel
            .inspect("edit_file", parser.fields())
            .expect("must veto");
        assert_eq!(veto.category, VetoCategory::ProtectedSystemRoot);
    }

    #[test]
    fn prewarm_engine_caches_readable_files() {
        let tmp = std::env::temp_dir().join(format!("keel_prewarm_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let target_file = tmp.join("target.rs");
        std::fs::write(&target_file, "pub fn target() -> bool { true }").unwrap();

        let engine = PrewarmEngine::new(&tmp);
        let mut fields = HashMap::new();
        fields.insert("path".into(), "target.rs".into());

        assert!(engine.check_and_prewarm("read_file", &fields));
        let cached = engine.get_warmed(&target_file);
        assert_eq!(cached, Some("pub fn target() -> bool { true }".into()));

        let metrics = engine.metrics();
        assert_eq!(metrics["cache_hits"], 1);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn sentinel_catches_whitespace_and_case_evasion() {
        let sentinel = StreamingSentinel::default();

        // 1. Multiple spaces evasion
        let mut fields1 = HashMap::new();
        fields1.insert("command".into(), "rm    -rf    /".into());
        let veto1 = sentinel
            .inspect("shell", &fields1)
            .expect("must veto multiple spaces");
        assert_eq!(veto1.category, VetoCategory::DestructiveCommand);

        // 2. Uppercase evasion
        let mut fields2 = HashMap::new();
        fields2.insert("command".into(), "RM -RF /".into());
        let veto2 = sentinel
            .inspect("shell", &fields2)
            .expect("must veto uppercase command");
        assert_eq!(veto2.category, VetoCategory::DestructiveCommand);

        // 3. Mixed case and spaces on git push force
        let mut fields3 = HashMap::new();
        fields3.insert(
            "command".into(),
            "GIT   PUSH   ORIGIN   --FORCE   MAIN".into(),
        );
        let veto3 = sentinel
            .inspect("shell", &fields3)
            .expect("must veto uppercase git force push");
        assert_eq!(veto3.category, VetoCategory::ForceBranchPush);
    }

    #[test]
    fn prewarm_engine_rejects_path_traversal_and_escapes() {
        let tmp = std::env::temp_dir().join(format!("keel_prewarm_sec_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let engine = PrewarmEngine::new(&tmp);

        // Absolute path outside workspace must not prewarm
        let mut f1 = HashMap::new();
        f1.insert("path".into(), "/etc/passwd".into());
        assert!(!engine.check_and_prewarm("read_file", &f1));

        // Relative path traversal must not prewarm
        let mut f2 = HashMap::new();
        f2.insert("path".into(), "../secret.txt".into());
        assert!(!engine.check_and_prewarm("read_file", &f2));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn incremental_parser_handles_large_stream_efficiently() {
        let mut parser = IncrementalArgParser::new();
        parser.feed("{\"content\": \"");
        // Stream a large body of text
        let chunk = "a".repeat(10_000);
        parser.feed(&chunk);
        assert_eq!(parser.fields().get("content").unwrap().len(), 10_000);

        // Next field key extraction works cleanly even after large buffer
        parser.feed("\", \"path\": \"foo.rs\"}");
        assert_eq!(parser.fields().get("path").unwrap(), "foo.rs");
    }

    #[test]
    fn prewarm_never_serves_stale_content() {
        let tmp = std::env::temp_dir().join(format!("keel_prewarm_stale_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let f = tmp.join("a.rs");
        std::fs::write(&f, "old").unwrap();

        let engine = PrewarmEngine::new(&tmp);
        let mut fields = HashMap::new();
        fields.insert("path".into(), "a.rs".into());
        assert!(engine.check_and_prewarm("read_file", &fields));
        assert_eq!(engine.get_warmed(&f).as_deref(), Some("old"));

        // Same-length edit with a different mtime must not be served from cache.
        std::fs::write(&f, "new").unwrap();
        let later = std::time::SystemTime::now() + std::time::Duration::from_secs(5);
        std::fs::File::options()
            .write(true)
            .open(&f)
            .unwrap()
            .set_modified(later)
            .unwrap();
        assert_eq!(engine.get_warmed(&f), None);
        assert_eq!(engine.metrics()["cached_entries"], 0);

        // Re-prewarming picks up the new content; invalidate_all clears it.
        assert!(engine.check_and_prewarm("read_file", &fields));
        assert_eq!(engine.get_warmed(&f).as_deref(), Some("new"));
        engine.invalidate_all();
        assert_eq!(engine.get_warmed(&f), None);
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
