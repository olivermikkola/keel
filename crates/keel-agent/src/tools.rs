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
    if p.components()
        .any(|c| matches!(c, Component::ParentDir | Component::Prefix(_)))
    {
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

/// Surgical edit: replace exactly one occurrence of `old` with `new`.
/// Ambiguity is an error, so a weak model can't silently edit the wrong place.
pub struct EditFile(pub PathBuf);
impl Tool for EditFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "edit_file".into(),
            description: "Replace one exact occurrence of `old` with `new` in a file. `old` must match exactly once; include surrounding lines if needed to make it unique.".into(),
            parameters: json!({"type":"object","properties":{"path":{"type":"string"},"old":{"type":"string"},"new":{"type":"string"}},"required":["path","old","new"]}),
        }
    }
    fn mutating(&self) -> bool {
        true
    }
    fn run(&self, a: &Value) -> Result<String> {
        let p = confine(&self.0, arg(a, "path")?)?;
        let (old, new) = (arg(a, "old")?, arg(a, "new")?);
        if old.is_empty() {
            bail!("`old` must not be empty");
        }
        let text =
            std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
        match text.matches(old).count() {
            0 => bail!(
                "`old` not found in {} (it must match exactly, including whitespace)",
                p.display()
            ),
            1 => {
                std::fs::write(&p, text.replacen(old, new, 1))?;
                Ok(format!("edited {}", p.display()))
            }
            n => bail!(
                "`old` matches {n} places in {}; add more surrounding context to make it unique",
                p.display()
            ),
        }
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

pub struct Shell(pub PathBuf);
impl Tool for Shell {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "shell".into(),
            description: "Run a shell command in the workspace; returns exit code, stdout, stderr."
                .into(),
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
    vec![
        Box::new(ReadFile(root.into())),
        Box::new(ListDir(root.into())),
        Box::new(Grep(root.into())),
        Box::new(EditFile(root.into())),
        Box::new(WriteFile(root.into())),
        Box::new(Shell(root.into())),
    ]
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
}
