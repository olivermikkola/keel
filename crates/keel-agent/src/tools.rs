//! Built-in tools: read_file, write_file, shell. All paths are confined to a root.

use crate::{Tool, ToolSpec};
use anyhow::{bail, Context as _, Result};
use serde_json::{json, Value};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};

const MAX_OUT: usize = 32 * 1024;

/// Resolve `rel` under `root`, rejecting absolute paths and `..` escapes.
pub fn confine(root: &Path, rel: &str) -> Result<PathBuf> {
    let p = Path::new(rel);
    if p.is_absolute() {
        bail!("absolute paths are not allowed: {rel}");
    }
    if p.components().any(|c| matches!(c, Component::ParentDir | Component::Prefix(_))) {
        bail!("path escapes workspace: {rel}");
    }
    Ok(root.join(p))
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
    a[k].as_str().with_context(|| format!("missing string argument `{k}`"))
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
        Ok(truncate(std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?))
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

pub struct Shell(pub PathBuf);
impl Tool for Shell {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "shell".into(),
            description: "Run a shell command in the workspace; returns exit code, stdout, stderr.".into(),
            parameters: json!({"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}),
        }
    }
    fn mutating(&self) -> bool {
        true
    }
    fn run(&self, a: &Value) -> Result<String> {
        let out = Command::new("sh")
            .arg("-c")
            .arg(arg(a, "command")?)
            .current_dir(&self.0)
            .stdin(Stdio::null())
            .output()?;
        Ok(truncate(format!(
            "exit: {}\nstdout:\n{}\nstderr:\n{}",
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )))
    }
}

pub fn builtin(root: &Path) -> Vec<Box<dyn Tool>> {
    vec![Box::new(ReadFile(root.into())), Box::new(WriteFile(root.into())), Box::new(Shell(root.into()))]
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
    fn write_then_read_roundtrip() {
        let d = std::env::temp_dir().join(format!("keel-test-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        WriteFile(d.clone()).run(&json!({"path":"x/y.txt","content":"hello"})).unwrap();
        assert_eq!(ReadFile(d.clone()).run(&json!({"path":"x/y.txt"})).unwrap(), "hello");
        std::fs::remove_dir_all(d).unwrap();
    }
}
