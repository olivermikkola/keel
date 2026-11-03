//! Built-in tools: read_file, list_dir, grep, edit_file, write_file, shell.
//! All paths are confined to a root, and `shell` enforces Linux Landlock when enabled.

use crate::sandbox::{SandboxPolicy, ToolRegistry};
use crate::{Tool, ToolSpec};
use anyhow::{bail, Context as _, Result};
use keel_kernel::{Plugin, Scope};
use serde_json::{json, Value};
use std::os::unix::process::CommandExt;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

const MAX_OUT: usize = 32 * 1024;

/// Resolve `rel` under `root`, rejecting absolute paths, `..` escapes, and symlinks
/// that resolve outside the canonicalized workspace root.
pub fn confine(root: &Path, rel: &str) -> Result<PathBuf> {
    let p = Path::new(rel);
    if p.is_absolute() {
        bail!("absolute paths are not allowed: {rel}");
    }
    if p.components()
        .any(|c| matches!(c, Component::ParentDir | Component::Prefix(_)))
    {
        bail!("path escapes workspace: {rel}");
    }
    let canon_root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let joined = root.join(p);

    // If the path already exists, canonicalize directly and verify prefix.
    if let Ok(canon_target) = joined.canonicalize() {
        if !canon_target.starts_with(&canon_root) {
            bail!("symlink escapes workspace: {rel}");
        }
        return Ok(joined);
    }

    // If joined does not exist (e.g. creating a new file before write_file),
    // traverse component by component to ensure no ancestor directory resolves
    // through a symlink pointing outside the workspace.
    let mut current = canon_root.clone();
    for comp in p.components() {
        match comp {
            Component::Normal(c) => {
                current = current.join(c);
                if let Ok(meta) = current.symlink_metadata() {
                    if meta.file_type().is_symlink() {
                        match current.canonicalize() {
                            Ok(resolved) => {
                                if !resolved.starts_with(&canon_root) {
                                    bail!("symlink escapes workspace: {rel}");
                                }
                                current = resolved;
                            }
                            Err(_) => {
                                bail!("broken symlink in path: {rel}");
                            }
                        }
                    }
                }
            }
            Component::CurDir => {}
            _ => bail!("invalid path component: {rel}"),
        }
    }

    Ok(joined)
}

fn truncate(mut s: String) -> String {
    if s.len() > MAX_OUT {
        let mut cut = MAX_OUT;
        while !s.is_char_boundary(cut) {
            cut -= 1;
        }
        s.truncate(cut);
        s.push_str("\n…[truncated]");
    }
    s
}

fn arg<'a>(a: &'a Value, k: &str) -> Result<&'a str> {
    a[k].as_str()
        .with_context(|| format!("missing string argument `{k}`"))
}

pub struct ReadFile(pub PathBuf);
impl Tool for ReadFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_file".into(),
            description: "Read a UTF-8 text file relative to the workspace.".into(),
            parameters: json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}),
        }
    }
    fn mutating(&self) -> bool {
        false
    }
    fn run(&self, a: &Value) -> Result<String> {
        let p = confine(&self.0, arg(a, "path")?)?;
        if p.is_dir() {
            bail!("{} is a directory; use list_dir", p.display());
        }
        Ok(truncate(
            std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?,
        ))
    }
}

/// Read-only directory listing. Exists so models never need `shell` just to `ls`
/// (a weak model wastes whole steps on denied shell calls otherwise).
pub struct ListDir(pub PathBuf);
impl Tool for ListDir {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "list_dir".into(),
            description: "List entries of a directory relative to the workspace (default '.'). Directories end with '/'.".into(),
            parameters: json!({"type":"object","properties":{"path":{"type":"string"}}}),
        }
    }
    fn mutating(&self) -> bool {
        false
    }
    fn run(&self, a: &Value) -> Result<String> {
        let p = confine(&self.0, a["path"].as_str().unwrap_or("."))?;
        let mut names: Vec<String> = std::fs::read_dir(&p)
            .with_context(|| format!("listing {}", p.display()))?
            .filter_map(|e| e.ok())
            .map(|e| {
                let n = e.file_name().to_string_lossy().into_owned();
                if e.path().is_dir() {
                    format!("{n}/")
                } else {
                    n
                }
            })
            .collect();
        names.sort();
        Ok(truncate(names.join("\n")))
    }
}

pub struct WriteFile(pub PathBuf);
impl Tool for WriteFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "write_file".into(),
            description: "Create or overwrite a file relative to the workspace.".into(),
            parameters: json!({"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}),
        }
    }
    fn mutating(&self) -> bool {
        true
    }
    fn run(&self, a: &Value) -> Result<String> {
        let p = confine(&self.0, arg(a, "path")?)?;
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d)?;
        }
        let c = arg(a, "content")?;
        std::fs::write(&p, c)?;
        Ok(format!("wrote {} bytes to {}", c.len(), p.display()))
    }
}

/// Surgical edit: replace one occurrence of `old` with `new`.
/// Supports exact match, whitespace-normalized line matching, and fuzzy matching.
pub struct EditFile(pub PathBuf);
impl Tool for EditFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "edit_file".into(),
            description: "Replace one occurrence of `old` with `new` in a file. Resilient to whitespace and indentation differences. Include surrounding lines if needed to make it unique.".into(),
            parameters: json!({"type":"object","properties":{"path":{"type":"string"},"old":{"type":"string"},"new":{"type":"string"}},"required":["path","old","new"]}),
        }
    }
    fn mutating(&self) -> bool {
        true
    }
    fn run(&self, a: &Value) -> Result<String> {
        let p = confine(&self.0, arg(a, "path")?)?;
        let (old, new) = (arg(a, "old")?, arg(a, "new")?);
        let text =
            std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
        let (updated, quality) = crate::patch::fuzzy_replace(&text, old, new)?;
        std::fs::write(&p, updated)?;
        match quality {
            crate::patch::MatchQuality::Exact => Ok(format!("edited {}", p.display())),
            other => Ok(format!("edited {} ({other})", p.display())),
        }
    }
}

/// Multi-hunk patch tool: applies search/replace blocks or unified diff hunks.
pub struct PatchFile(pub PathBuf);
impl Tool for PatchFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "patch_file".into(),
            description: "Apply unified diff hunks (@@ ... @@) or search/replace blocks (<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE) to a file. Resilient to indentation and whitespace drift.".into(),
            parameters: json!({"type":"object","properties":{"path":{"type":"string"},"patch":{"type":"string"}},"required":["path","patch"]}),
        }
    }
    fn mutating(&self) -> bool {
        true
    }
    fn run(&self, a: &Value) -> Result<String> {
        let p = confine(&self.0, arg(a, "path")?)?;
        let patch_text = arg(a, "patch")?;
        let text =
            std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
        let outcome = crate::patch::apply_patch(&text, patch_text)?;
        std::fs::write(&p, outcome.modified_text)?;
        Ok(format!(
            "patched {} ({} hunk(s) applied via {})",
            p.display(),
            outcome.hunks_applied,
            outcome.quality
        ))
    }
}

/// Literal substring search across the workspace (skips .git, target, node_modules).
pub struct Grep(pub PathBuf);
impl Tool for Grep {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "grep".into(),
            description: "Search files for a literal text pattern. Returns path:line:text. Optional `path` narrows to a subdirectory or file.".into(),
            parameters: json!({"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"}},"required":["pattern"]}),
        }
    }
    fn mutating(&self) -> bool {
        false
    }
    fn run(&self, a: &Value) -> Result<String> {
        let pat = arg(a, "pattern")?;
        let start = confine(&self.0, a["path"].as_str().unwrap_or("."))?;
        let mut hits = Vec::new();
        walk(&start, &self.0, pat, &mut hits);
        if hits.is_empty() {
            return Ok("no matches".into());
        }
        Ok(truncate(hits.join("\n")))
    }
}

const MAX_HITS: usize = 100;

fn walk(p: &Path, root: &Path, pat: &str, hits: &mut Vec<String>) {
    if hits.len() >= MAX_HITS {
        return;
    }
    if p.is_dir() {
        let Ok(rd) = std::fs::read_dir(p) else { return };
        let mut entries: Vec<_> = rd.filter_map(|e| e.ok()).collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let n = e.file_name();
            if matches!(
                n.to_string_lossy().as_ref(),
                ".git" | "target" | "node_modules"
            ) {
                continue;
            }
            // file_type() does not follow symlinks, so links can't pull in files outside the root.
            if e.file_type().map(|t| t.is_symlink()).unwrap_or(true) {
                continue;
            }
            walk(&e.path(), root, pat, hits);
        }
    } else if let Ok(text) = std::fs::read_to_string(p) {
        let rel = p.strip_prefix(root).unwrap_or(p).display().to_string();
        for (i, line) in text.lines().enumerate() {
            if line.contains(pat) && hits.len() < MAX_HITS {
                hits.push(format!("{rel}:{}:{}", i + 1, line.trim()));
            }
        }
    }
}

/// Returns true if an environment variable name is likely to contain sensitive credentials,
/// tokens, or secret keys that should not be passed to untrusted child processes.
pub fn is_sensitive_env_var(k: &str) -> bool {
    let u = k.to_uppercase();
    // Allow non-sensitive CI and build metadata
    if u == "GITHUB_ACTIONS"
        || u == "GITHUB_REPOSITORY"
        || u == "GITHUB_REF"
        || u == "GITHUB_SHA"
        || u == "GITHUB_RUN_ID"
        || u == "GITHUB_WORKFLOW"
        || u == "GITHUB_JOB"
    {
        return false;
    }
    u.ends_with("_API_KEY")
        || u.ends_with("_TOKEN")
        || u.ends_with("_SECRET")
        || u.contains("API_KEY")
        || u.contains("SECRET_ACCESS_KEY")
        || u.contains("ACCESS_TOKEN")
        || u.contains("PRIVATE_KEY")
        || u.contains("AUTH_TOKEN")
        || u == "GITHUB_TOKEN"
        || u == "GH_TOKEN"
        || u.starts_with("AWS_SECRET")
        || u.starts_with("AWS_SESSION")
        || u.starts_with("AWS_SECURITY")
        || u.starts_with("SSH_AUTH")
        || u == "DATABASE_URL"
}

pub struct Shell {
    pub root: PathBuf,
    pub sandbox: SandboxPolicy,
    pub timeout: Duration,
}

impl Shell {
    pub fn new(root: PathBuf) -> Self {
        Self {
            sandbox: SandboxPolicy::new(root.clone(), true),
            root,
            timeout: Duration::from_secs(120),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

impl Tool for Shell {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "shell".into(),
            description: "Run a shell command in the workspace; returns exit code, stdout, stderr."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" },
                    "timeout_secs": { "type": "integer", "description": "Optional timeout in seconds (default: 120)" }
                },
                "required": ["command"]
            }),
        }
    }
    fn mutating(&self) -> bool {
        true
    }
    fn run(&self, a: &Value) -> Result<String> {
        let policy = self.sandbox.clone();
        let mut cmd = Command::new("sh");
        cmd.arg("-c")
            .arg(arg(a, "command")?)
            .current_dir(&self.root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Scrub sensitive environment variables to prevent credential exfiltration
        for (k, _) in std::env::vars() {
            if is_sensitive_env_var(&k) {
                cmd.env_remove(&k);
            }
        }

        unsafe {
            cmd.pre_exec(move || {
                policy
                    .restrict_current_thread()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::PermissionDenied, e))
            });
        }

        let timeout = a
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .map(Duration::from_secs)
            .unwrap_or(self.timeout);

        let mut child = cmd.spawn()?;
        let mut stdout_pipe = child.stdout.take();
        let mut stderr_pipe = child.stderr.take();

        let stdout_thread = std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(mut pipe) = stdout_pipe.take() {
                let _ = std::io::Read::read_to_end(&mut pipe, &mut buf);
            }
            buf
        });

        let stderr_thread = std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(mut pipe) = stderr_pipe.take() {
                let _ = std::io::Read::read_to_end(&mut pipe, &mut buf);
            }
            buf
        });

        let start = std::time::Instant::now();
        let status = loop {
            match child.try_wait()? {
                Some(st) => break st,
                None => {
                    if start.elapsed() > timeout {
                        let _ = child.kill();
                        let _ = child.wait();
                        let _ = stdout_thread.join();
                        let _ = stderr_thread.join();
                        bail!("command timed out after {}s", timeout.as_secs());
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        };

        let stdout_bytes = stdout_thread.join().unwrap_or_default();
        let stderr_bytes = stderr_thread.join().unwrap_or_default();

        Ok(truncate(format!(
            "exit: {}\nstdout:\n{}\nstderr:\n{}",
            status.code().unwrap_or(-1),
            String::from_utf8_lossy(&stdout_bytes),
            String::from_utf8_lossy(&stderr_bytes)
        )))
    }
}

pub fn builtin(root: &Path) -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(ReadFile(root.into())),
        Box::new(ListDir(root.into())),
        Box::new(Grep(root.into())),
        Box::new(EditFile(root.into())),
        Box::new(PatchFile(root.into())),
        Box::new(WriteFile(root.into())),
        Box::new(Shell::new(root.into())),
    ]
}

/// Plugin that mounts all built-in tools into the kernel's [`ToolRegistry`].
pub struct BuiltinToolsPlugin {
    pub root: PathBuf,
    pub landlock_enabled: bool,
}

impl Plugin for BuiltinToolsPlugin {
    fn name(&self) -> &str {
        "builtin-tools"
    }
    fn apply(&self, scope: &mut Scope) {
        let reg = scope
            .context()
            .service::<ToolRegistry>()
            .map(|r| (*r).clone())
            .unwrap_or_default();
        let r = &self.root;
        reg.register(Arc::new(ReadFile(r.clone())));
        reg.register(Arc::new(ListDir(r.clone())));
        reg.register(Arc::new(Grep(r.clone())));
        reg.register(Arc::new(EditFile(r.clone())));
        reg.register(Arc::new(PatchFile(r.clone())));
        reg.register(Arc::new(WriteFile(r.clone())));
        let sandbox_policy = scope
            .context()
            .service::<SandboxPolicy>()
            .map(|p| (*p).clone())
            .unwrap_or_else(|| SandboxPolicy::new(r.clone(), self.landlock_enabled));

        reg.register(Arc::new(Shell {
            root: r.clone(),
            sandbox: sandbox_policy,
            timeout: Duration::from_secs(120),
        }));
        scope.provide(reg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confine_rejects_escapes() {
        let r = Path::new("/w");
        assert!(confine(r, "../etc/passwd").is_err());
        assert!(confine(r, "a/../../x").is_err());
        assert!(confine(r, "/etc/passwd").is_err());
        assert_eq!(confine(r, "a/b.txt").unwrap(), Path::new("/w/a/b.txt"));
    }

    #[test]
    fn edit_requires_unique_match_and_grep_finds() {
        let d = std::env::temp_dir().join(format!("keel-test-edit-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("f.txt"), "a\nb\na\n").unwrap();
        let e = EditFile(d.clone());
        assert!(e.run(&json!({"path":"f.txt","old":"a","new":"z"})).is_err()); // ambiguous
        assert!(e.run(&json!({"path":"f.txt","old":"q","new":"z"})).is_err()); // missing
        e.run(&json!({"path":"f.txt","old":"b","new":"B"})).unwrap();
        assert_eq!(
            std::fs::read_to_string(d.join("f.txt")).unwrap(),
            "a\nB\na\n"
        );
        let out = Grep(d.clone()).run(&json!({"pattern":"B"})).unwrap();
        assert_eq!(out, "f.txt:2:B");
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn write_then_read_roundtrip() {
        let d = std::env::temp_dir().join(format!("keel-test-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        WriteFile(d.clone())
            .run(&json!({"path":"x/y.txt","content":"hello"}))
            .unwrap();
        assert_eq!(
            ReadFile(d.clone()).run(&json!({"path":"x/y.txt"})).unwrap(),
            "hello"
        );
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn landlock_blocks_writes_outside_workspace() {
        let ws = std::env::temp_dir().join(format!("keel-ll-ws-{}", std::process::id()));
        let outside = std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/var/tmp"))
            .join(format!(".keel-ll-forbidden-{}", std::process::id()));
        std::fs::create_dir_all(&ws).unwrap();
        let sh = Shell::new(ws.clone());
        // Writing inside workspace must succeed
        let ok_out = sh
            .run(&json!({"command": "echo allowed > inside.txt"}))
            .unwrap();
        assert!(ok_out.contains("exit: 0"), "unexpected output: {ok_out}");
        assert!(ws.join("inside.txt").exists());

        // Writing outside workspace + /tmp (e.g. $HOME) must be blocked by Landlock
        let blocked_cmd = format!("echo forbidden > {}", outside.display());
        let deny_out = sh.run(&json!({"command": blocked_cmd})).unwrap();
        assert!(
            !outside.exists(),
            "Landlock failed to block write outside workspace: {deny_out}"
        );
        let _ = std::fs::remove_dir_all(ws);
        let _ = std::fs::remove_file(outside);
    }

    #[test]
    fn patch_file_tool_and_fuzzy_edit_work() {
        let d = std::env::temp_dir().join(format!("keel-test-patch-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let target = d.join("app.rs");
        std::fs::write(&target, "fn run() {\n    let count = 0;\n    count\n}\n").unwrap();

        // 1. Test resilient edit with indentation discrepancy (2 spaces vs 4)
        let edit = EditFile(d.clone());
        let res = edit
            .run(&json!({
                "path": "app.rs",
                "old": "  let count = 0;\n",
                "new": "    let count = 10;\n"
            }))
            .unwrap();
        assert!(res.contains("edited"));
        assert!(std::fs::read_to_string(&target)
            .unwrap()
            .contains("let count = 10;"));

        // 2. Test patch tool with Search/Replace block
        let patch = PatchFile(d.clone());
        let block = "<<<<<<< SEARCH\n    count\n=======\n    count + 1\n>>>>>>> REPLACE\n";
        let res2 = patch
            .run(&json!({
                "path": "app.rs",
                "patch": block
            }))
            .unwrap();
        assert!(res2.contains("patched"));
        assert!(std::fs::read_to_string(&target)
            .unwrap()
            .contains("count + 1"));

        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn confine_detects_symlink_traversal_on_nonexistent_target() {
        let tmp = std::env::temp_dir().join(format!("keel-sym-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let ws = tmp.join("ws");
        let outside = tmp.join("outside");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::create_dir_all(&outside).unwrap();

        // Create a symlink ws/sym_outside -> outside
        std::os::unix::fs::symlink(&outside, ws.join("sym_outside")).unwrap();

        // Target ws/sym_outside/new_file.txt does NOT exist yet.
        // Old buggy confine would pass because joined.canonicalize() returned Err(NotFound).
        // Hardened confine must detect that sym_outside points outside workspace!
        let err = confine(&ws, "sym_outside/new_file.txt");
        assert!(
            err.is_err(),
            "should reject symlink directory traversal to outside workspace"
        );
        let msg = format!("{:#}", err.unwrap_err());
        assert!(msg.contains("escapes workspace") || msg.contains("symlink"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn confine_rejects_broken_symlinks() {
        let tmp = std::env::temp_dir().join(format!("keel-broken-sym-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        // Create a broken symlink pointing to a non-existent path
        std::os::unix::fs::symlink(tmp.join("does_not_exist"), tmp.join("broken")).unwrap();

        let err = confine(&tmp, "broken");
        assert!(err.is_err(), "broken symlinks must be rejected");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn shell_scrubs_sensitive_environment_variables() {
        let ws = std::env::temp_dir().join(format!("keel-env-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&ws);
        std::fs::create_dir_all(&ws).unwrap();

        std::env::set_var("TEST_SECRET_API_KEY", "super_secret_token_123");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "super_secret_aws_456");

        let sh = Shell::new(ws.clone());
        let out = sh
            .run(&json!({
                "command": "echo key=$TEST_SECRET_API_KEY aws=$AWS_SECRET_ACCESS_KEY"
            }))
            .unwrap();

        assert!(!out.contains("super_secret_token_123"));
        assert!(!out.contains("super_secret_aws_456"));
        assert!(out.contains("key= aws="));

        std::env::remove_var("TEST_SECRET_API_KEY");
        std::env::remove_var("AWS_SECRET_ACCESS_KEY");
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[test]
    fn shell_enforces_execution_timeout() {
        let ws = std::env::temp_dir().join(format!("keel-timeout-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&ws);
        std::fs::create_dir_all(&ws).unwrap();

        let sh = Shell::new(ws.clone()).with_timeout(Duration::from_millis(500));
        let err = sh.run(&json!({
            "command": "sleep 10",
            "timeout_secs": 1
        }));

        assert!(err.is_err());
        let msg = format!("{:#}", err.unwrap_err());
        assert!(msg.contains("timed out"));

        let _ = std::fs::remove_dir_all(&ws);
    }
}
