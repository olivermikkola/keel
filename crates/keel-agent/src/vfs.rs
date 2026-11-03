//! Transactional Virtual File System (VFS) & Copy-on-Write (CoW) Mutation Journal.
//!
//! Provides two complementary execution modes for file mutations (`write_file`,
//! `edit_file`, `patch_file`) and reads (`read_file`, `grep`, `list_dir`):
//! - [`VfsMode::MaterializedCow`]: Captures the exact pre-image of every touched file
//!   in a persisted CoW journal (`.keel/vfs_tx.json`) before writing to disk. External
//!   subprocesses (`cargo test`, `pytest`, compilers) see the changes immediately on disk,
//!   while `rollback()` atomically restores all modified/created files to their pre-turn state.
//! - [`VfsMode::ShadowMemory`]: Buffers all mutations strictly in an in-memory overlay
//!   (`--shadow` / dry-run turns). `read_file` reads through the overlay so the agent sees
//!   its own edits, while the host disk remains untouched until `commit()` is called.

use crate::sandbox::ToolRegistry;
use crate::tools::confine;
use crate::{Tool, ToolSpec};
use anyhow::{Context as _, Result};
use keel_kernel::{Plugin, Scope};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VfsMode {
    /// Materializes edits to disk after recording the original pre-image in the CoW journal.
    MaterializedCow,
    /// Buffers edits strictly in memory until `commit()` is explicitly invoked.
    ShadowMemory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VfsEntry {
    pub rel_path: String,
    /// `None` if the file did not exist prior to the transaction.
    pub original_content: Option<String>,
    /// Current content in the transaction (`None` if deleted).
    pub staged_content: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct PersistedJournal {
    entries: BTreeMap<String, VfsEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VfsFileStatus {
    pub path: String,
    pub status: String, // "modified", "created", "deleted"
    pub original_bytes: usize,
    pub staged_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VfsCommitReport {
    pub committed_files: usize,
    pub paths: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VfsRollbackReport {
    pub restored_files: usize,
    pub removed_files: usize,
    pub paths: Vec<String>,
}

/// Thread-safe transactional VFS & CoW journal service mounted on the kernel [`Context`].
pub struct TransactionalVfs {
    root: PathBuf,
    mode: VfsMode,
    journal: Mutex<PersistedJournal>,
    persist_path: Option<PathBuf>,
}

impl TransactionalVfs {
    pub fn new(root: PathBuf, mode: VfsMode, persist: bool) -> Self {
        let persist_path = if persist {
            Some(root.join(".keel").join("vfs_tx.json"))
        } else {
            None
        };
        let initial = persist_path
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str::<PersistedJournal>(&s).ok())
            .unwrap_or_default();

        Self {
            root,
            mode,
            journal: Mutex::new(initial),
            persist_path,
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn mode(&self) -> VfsMode {
        self.mode
    }

    fn normalize_rel(&self, full_or_rel: &Path) -> Result<(PathBuf, String)> {
        if full_or_rel.is_absolute() {
            let canon_root = self
                .root
                .canonicalize()
                .unwrap_or_else(|_| self.root.clone());
            let canon_full = full_or_rel
                .canonicalize()
                .unwrap_or_else(|_| full_or_rel.to_path_buf());
            if !canon_full.starts_with(&canon_root) {
                anyhow::bail!("absolute path escapes workspace: {}", full_or_rel.display());
            }
            let rel = canon_full
                .strip_prefix(&canon_root)?
                .to_string_lossy()
                .to_string();
            Ok((canon_full, rel))
        } else {
            let rel_str = full_or_rel.to_string_lossy();
            let clean = rel_str.trim_start_matches("./");
            let full = confine(&self.root, clean)?;
            Ok((full, clean.to_string()))
        }
    }

    fn flush_journal(&self, j: &PersistedJournal) -> Result<()> {
        let Some(p) = &self.persist_path else {
            return Ok(());
        };
        if j.entries.is_empty() {
            let _ = std::fs::remove_file(p);
            return Ok(());
        }
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(j)?;
        let tmp = p.with_extension(format!("tmp.{}", std::process::id()));
        std::fs::write(&tmp, data)?;
        std::fs::rename(&tmp, p)?;
        Ok(())
    }

    /// Read a UTF-8 file through the transactional VFS.
    /// In `ShadowMemory` mode, returns buffered edits if present; otherwise reads from disk.
    pub fn read_to_string(&self, path: &Path) -> Result<String> {
        let (full, rel) = self.normalize_rel(path)?;
        if self.mode == VfsMode::ShadowMemory {
            if let Ok(guard) = self.journal.lock() {
                if let Some(entry) = guard.entries.get(&rel) {
                    return match &entry.staged_content {
                        Some(text) => Ok(text.clone()),
                        None => anyhow::bail!("file `{rel}` was deleted in shadow VFS"),
                    };
                }
            }
        }
        let meta = std::fs::symlink_metadata(&full)
            .with_context(|| format!("checking metadata for {}", full.display()))?;
        if !meta.file_type().is_file() {
            anyhow::bail!("{} is not a regular file", full.display());
        }
        std::fs::read_to_string(&full).with_context(|| format!("reading {}", full.display()))
    }

    /// Write a UTF-8 file through the transactional VFS, capturing its pre-image on first touch.
    pub fn write_string(&self, path: &Path, content: &str) -> Result<()> {
        let (full, rel) = self.normalize_rel(path)?;
        if let Ok(meta) = std::fs::symlink_metadata(&full) {
            if !meta.file_type().is_file() {
                anyhow::bail!("{} is not a regular file", full.display());
            }
        }
        let mut guard = self
            .journal
            .lock()
            .map_err(|_| anyhow::anyhow!("VFS journal mutex poisoned"))?;

        if !guard.entries.contains_key(&rel) {
            let original = if full.is_file() {
                std::fs::read_to_string(&full).ok()
            } else {
                None
            };
            guard.entries.insert(
                rel.clone(),
                VfsEntry {
                    rel_path: rel.clone(),
                    original_content: original,
                    staged_content: Some(content.to_string()),
                },
            );
        } else if let Some(entry) = guard.entries.get_mut(&rel) {
            entry.staged_content = Some(content.to_string());
        }

        if self.mode == VfsMode::MaterializedCow {
            if let Some(parent) = full.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&full, content)
                .with_context(|| format!("writing {}", full.display()))?;
        }

        self.flush_journal(&guard)?;
        Ok(())
    }

    /// Record the pre-image of `rel_path` on disk before an external or built-in tool mutates it.
    pub fn capture_pre_image(&self, rel_path: &str) -> Result<()> {
        let (full, rel) = self.normalize_rel(Path::new(rel_path))?;
        let mut guard = self
            .journal
            .lock()
            .map_err(|_| anyhow::anyhow!("VFS journal mutex poisoned"))?;
        if !guard.entries.contains_key(&rel) {
            let original = if full.is_file() {
                std::fs::read_to_string(&full).ok()
            } else {
                None
            };
            guard.entries.insert(
                rel.clone(),
                VfsEntry {
                    rel_path: rel,
                    original_content: original.clone(),
                    staged_content: original,
                },
            );
            self.flush_journal(&guard)?;
        }
        Ok(())
    }

    /// Record the post-mutation content of `rel_path` from disk in `MaterializedCow` mode.
    pub fn sync_post_image(&self, rel_path: &str) -> Result<()> {
        let (full, rel) = self.normalize_rel(Path::new(rel_path))?;
        let mut guard = self
            .journal
            .lock()
            .map_err(|_| anyhow::anyhow!("VFS journal mutex poisoned"))?;
        if let Some(entry) = guard.entries.get_mut(&rel) {
            entry.staged_content = if full.is_file() {
                std::fs::read_to_string(&full).ok()
            } else {
                None
            };
            self.flush_journal(&guard)?;
        }
        Ok(())
    }

    /// Execute `read_file`, `write_file`, `edit_file`, or `patch_file` purely in `ShadowMemory` overlay.
    pub fn execute_shadow_tool(&self, tool_name: &str, args: &Value) -> Option<Result<String>> {
        if self.mode != VfsMode::ShadowMemory {
            return None;
        }
        let rel = args.get("path").and_then(|v| v.as_str())?;
        match tool_name {
            "read_file" => Some(self.read_to_string(Path::new(rel))),
            "write_file" => {
                let content = match args.get("content").and_then(|v| v.as_str()) {
                    Some(c) => c,
                    None => return Some(Err(anyhow::anyhow!("missing string argument `content`"))),
                };
                Some(
                    self.write_string(Path::new(rel), content)
                        .map(|_| format!("staged {} bytes to {rel} (shadow VFS)", content.len())),
                )
            }
            "edit_file" => {
                let old = match args.get("old").and_then(|v| v.as_str()) {
                    Some(s) => s,
                    None => return Some(Err(anyhow::anyhow!("missing string argument `old`"))),
                };
                let new = match args.get("new").and_then(|v| v.as_str()) {
                    Some(s) => s,
                    None => return Some(Err(anyhow::anyhow!("missing string argument `new`"))),
                };
                Some((|| {
                    let current = self.read_to_string(Path::new(rel))?;
                    let (updated, quality) = crate::patch::fuzzy_replace(&current, old, new)?;
                    self.write_string(Path::new(rel), &updated)?;
                    Ok(format!("staged edit to {rel} ({quality}, shadow VFS)"))
                })())
            }
            "patch_file" => {
                let patch_text = match args.get("patch").and_then(|v| v.as_str()) {
                    Some(s) => s,
                    None => return Some(Err(anyhow::anyhow!("missing string argument `patch`"))),
                };
                Some((|| {
                    let current = self.read_to_string(Path::new(rel))?;
                    let outcome = crate::patch::apply_patch(&current, patch_text)?;
                    self.write_string(Path::new(rel), &outcome.modified_text)?;
                    Ok(format!(
                        "staged patch to {rel} ({} hunk(s) via {}, shadow VFS)",
                        outcome.hunks_applied, outcome.quality
                    ))
                })())
            }
            _ => None,
        }
    }

    /// List all files currently tracked in the active VFS transaction.
    pub fn status(&self) -> Vec<VfsFileStatus> {
        let Ok(guard) = self.journal.lock() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for (rel, entry) in &guard.entries {
            let current = if self.mode == VfsMode::MaterializedCow {
                std::fs::read_to_string(self.root.join(rel)).ok()
            } else {
                entry.staged_content.clone()
            };
            if current == entry.original_content {
                continue;
            }
            let status_str = match (&entry.original_content, &current) {
                (None, Some(_)) => "created",
                (Some(_), None) => "deleted",
                (Some(_), Some(_)) => "modified",
                (None, None) => continue,
            };
            out.push(VfsFileStatus {
                path: rel.clone(),
                status: status_str.into(),
                original_bytes: entry
                    .original_content
                    .as_ref()
                    .map(|s| s.len())
                    .unwrap_or(0),
                staged_bytes: current.as_ref().map(|s| s.len()).unwrap_or(0),
            });
        }
        out
    }

    /// Render a unified diff of all uncommitted VFS changes against their pre-transaction state.
    pub fn diff(&self) -> String {
        let Ok(guard) = self.journal.lock() else {
            return String::new();
        };
        let mut out = String::new();
        for (rel, entry) in &guard.entries {
            let current = if self.mode == VfsMode::MaterializedCow {
                std::fs::read_to_string(self.root.join(rel)).ok()
            } else {
                entry.staged_content.clone()
            };
            if current == entry.original_content {
                continue;
            }
            let old_text = entry.original_content.as_deref().unwrap_or("");
            let new_text = current.as_deref().unwrap_or("");
            out.push_str(&format_simple_unified_diff(
                rel,
                entry.original_content.is_some(),
                current.is_some(),
                old_text,
                new_text,
            ));
        }
        out
    }

    /// Commit all staged changes to disk (for `ShadowMemory` mode) and clear the transaction journal.
    pub fn commit(&self) -> Result<VfsCommitReport> {
        let mut guard = self
            .journal
            .lock()
            .map_err(|_| anyhow::anyhow!("VFS journal mutex poisoned"))?;
        let mut paths = Vec::new();
        for (rel, entry) in &guard.entries {
            let full = self.root.join(rel);
            if self.mode == VfsMode::ShadowMemory {
                match &entry.staged_content {
                    Some(text) => {
                        if let Some(parent) = full.parent() {
                            std::fs::create_dir_all(parent)?;
                        }
                        std::fs::write(&full, text)?;
                    }
                    None => {
                        if full.is_file() {
                            let _ = std::fs::remove_file(&full);
                        }
                    }
                }
            }
            paths.push(rel.clone());
        }
        guard.entries.clear();
        self.flush_journal(&guard)?;
        Ok(VfsCommitReport {
            committed_files: paths.len(),
            paths,
        })
    }

    /// Atomically roll back all files touched in the current transaction to their pre-transaction state.
    pub fn rollback(&self) -> Result<VfsRollbackReport> {
        let mut guard = self
            .journal
            .lock()
            .map_err(|_| anyhow::anyhow!("VFS journal mutex poisoned"))?;
        let mut restored_files = 0;
        let mut removed_files = 0;
        let mut paths = Vec::new();

        for (rel, entry) in &guard.entries {
            let full = self.root.join(rel);
            match &entry.original_content {
                Some(orig) => {
                    if self.mode == VfsMode::MaterializedCow {
                        if let Some(parent) = full.parent() {
                            std::fs::create_dir_all(parent)?;
                        }
                        std::fs::write(&full, orig)?;
                    }
                    restored_files += 1;
                }
                None => {
                    if self.mode == VfsMode::MaterializedCow && full.is_file() {
                        let _ = std::fs::remove_file(&full);
                    }
                    removed_files += 1;
                }
            }
            paths.push(rel.clone());
        }

        guard.entries.clear();
        self.flush_journal(&guard)?;
        Ok(VfsRollbackReport {
            restored_files,
            removed_files,
            paths,
        })
    }
}

fn format_simple_unified_diff(
    rel: &str,
    existed_before: bool,
    exists_after: bool,
    old_text: &str,
    new_text: &str,
) -> String {
    let mut s = String::new();
    let a_hdr = if existed_before {
        format!("a/{rel}")
    } else {
        "/dev/null".to_string()
    };
    let b_hdr = if exists_after {
        format!("b/{rel}")
    } else {
        "/dev/null".to_string()
    };
    s.push_str(&format!("--- {a_hdr}\n+++ {b_hdr}\n"));
    let old_lines: Vec<&str> = old_text.lines().collect();
    let new_lines: Vec<&str> = new_text.lines().collect();
    s.push_str(&format!(
        "@@ -1,{} +1,{} @@\n",
        old_lines.len(),
        new_lines.len()
    ));
    for l in &old_lines {
        if !new_lines.contains(l) {
            s.push_str(&format!("-{l}\n"));
        } else {
            s.push_str(&format!(" {l}\n"));
        }
    }
    for l in &new_lines {
        if !old_lines.contains(l) {
            s.push_str(&format!("+{l}\n"));
        }
    }
    s
}

// ---- Kernel Tools for Transactional VFS control ----

pub struct VfsStatusTool(pub Arc<TransactionalVfs>);
impl Tool for VfsStatusTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "vfs_status".into(),
            description: "List all modified/created files in the active VFS transaction journal."
                .into(),
            parameters: json!({"type": "object", "properties": {}}),
        }
    }
    fn mutating(&self) -> bool {
        false
    }
    fn run(&self, _: &Value) -> Result<String> {
        let items = self.0.status();
        if items.is_empty() {
            return Ok("no uncommitted VFS changes".into());
        }
        let lines: Vec<String> = items
            .into_iter()
            .map(|i| {
                format!(
                    "{} [{}] ({}B -> {}B)",
                    i.path, i.status, i.original_bytes, i.staged_bytes
                )
            })
            .collect();
        Ok(lines.join("\n"))
    }
}

pub struct VfsDiffTool(pub Arc<TransactionalVfs>);
impl Tool for VfsDiffTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "vfs_diff".into(),
            description:
                "Show unified diff of all staged/modified files in the active VFS transaction."
                    .into(),
            parameters: json!({"type": "object", "properties": {}}),
        }
    }
    fn mutating(&self) -> bool {
        false
    }
    fn run(&self, _: &Value) -> Result<String> {
        let d = self.0.diff();
        if d.is_empty() {
            Ok("no diff in VFS transaction".into())
        } else {
            Ok(d)
        }
    }
}

pub struct VfsCommitTool(pub Arc<TransactionalVfs>);
impl Tool for VfsCommitTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "vfs_commit".into(),
            description:
                "Commit all staged VFS changes to disk and clear the transaction rollback log."
                    .into(),
            parameters: json!({"type": "object", "properties": {}}),
        }
    }
    fn mutating(&self) -> bool {
        true
    }
    fn run(&self, _: &Value) -> Result<String> {
        let rep = self.0.commit()?;
        Ok(format!(
            "committed {} file(s): {}",
            rep.committed_files,
            rep.paths.join(", ")
        ))
    }
}

pub struct VfsRollbackTool(pub Arc<TransactionalVfs>);
impl Tool for VfsRollbackTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "vfs_rollback".into(),
            description:
                "Atomically revert all files touched in the active VFS transaction back to their original state."
                    .into(),
            parameters: json!({"type": "object", "properties": {}}),
        }
    }
    fn mutating(&self) -> bool {
        true
    }
    fn run(&self, _: &Value) -> Result<String> {
        let rep = self.0.rollback()?;
        Ok(format!(
            "rolled back VFS transaction (restored {} file(s), removed {} newly created file(s))",
            rep.restored_files, rep.removed_files
        ))
    }
}

/// Plugin that mounts [`TransactionalVfs`] onto the kernel [`Context`] and registers
/// `vfs_status`, `vfs_diff`, `vfs_commit`, and `vfs_rollback` tools.
pub struct VfsPlugin {
    pub root: PathBuf,
    pub mode: VfsMode,
    pub persist: bool,
}

impl Plugin for VfsPlugin {
    fn name(&self) -> &str {
        "transactional-vfs"
    }

    fn apply(&self, scope: &mut Scope) {
        let vfs = Arc::new(TransactionalVfs::new(
            self.root.clone(),
            self.mode,
            self.persist,
        ));
        scope.provide::<Arc<TransactionalVfs>>(vfs.clone());

        let reg = scope
            .context()
            .service::<ToolRegistry>()
            .map(|r| (*r).clone())
            .unwrap_or_default();
        reg.register(Arc::new(VfsStatusTool(vfs.clone())));
        reg.register(Arc::new(VfsDiffTool(vfs.clone())));
        reg.register(Arc::new(VfsCommitTool(vfs.clone())));
        reg.register(Arc::new(VfsRollbackTool(vfs)));
        scope.provide(reg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn materialized_cow_records_preimage_and_rolls_back_atomically() {
        let tmp = std::env::temp_dir().join(format!("keel-vfs-cow-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        std::fs::write(tmp.join("existing.txt"), "line 1\nline 2\n").unwrap();
        let vfs = TransactionalVfs::new(tmp.clone(), VfsMode::MaterializedCow, true);

        // Mutate existing file twice and create a new file
        vfs.write_string(Path::new("existing.txt"), "line 1\nmodified first\n")
            .unwrap();
        vfs.write_string(Path::new("existing.txt"), "line 1\nmodified second\n")
            .unwrap();
        vfs.write_string(Path::new("created.txt"), "brand new\n")
            .unwrap();

        // Disk must reflect materialized edits for external compilers/test runners
        assert_eq!(
            std::fs::read_to_string(tmp.join("existing.txt")).unwrap(),
            "line 1\nmodified second\n"
        );
        assert!(tmp.join("created.txt").exists());

        let status = vfs.status();
        assert_eq!(status.len(), 2);

        let diff = vfs.diff();
        assert!(diff.contains("--- a/existing.txt"));
        assert!(diff.contains("+modified second"));

        // Rollback must restore the ORIGINAL pre-image (before first edit) and delete created.txt!
        let rep = vfs.rollback().unwrap();
        assert_eq!(rep.restored_files, 1);
        assert_eq!(rep.removed_files, 1);
        assert_eq!(
            std::fs::read_to_string(tmp.join("existing.txt")).unwrap(),
            "line 1\nline 2\n"
        );
        assert!(!tmp.join("created.txt").exists());

        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn shadow_memory_buffers_in_ram_until_commit() {
        let tmp = std::env::temp_dir().join(format!("keel-vfs-shadow-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        std::fs::write(tmp.join("app.py"), "x = 1\n").unwrap();
        let vfs = TransactionalVfs::new(tmp.clone(), VfsMode::ShadowMemory, false);

        vfs.write_string(Path::new("app.py"), "x = 42\n").unwrap();
        vfs.write_string(Path::new("new_mod.py"), "y = 99\n")
            .unwrap();

        // Host disk must still be untouched before commit!
        assert_eq!(
            std::fs::read_to_string(tmp.join("app.py")).unwrap(),
            "x = 1\n"
        );
        assert!(!tmp.join("new_mod.py").exists());

        // VFS read_to_string sees the shadow overlay
        assert_eq!(vfs.read_to_string(Path::new("app.py")).unwrap(), "x = 42\n");
        assert_eq!(
            vfs.read_to_string(Path::new("new_mod.py")).unwrap(),
            "y = 99\n"
        );

        // Commit flushes shadow overlay to disk
        let rep = vfs.commit().unwrap();
        assert_eq!(rep.committed_files, 2);
        assert_eq!(
            std::fs::read_to_string(tmp.join("app.py")).unwrap(),
            "x = 42\n"
        );
        assert_eq!(
            std::fs::read_to_string(tmp.join("new_mod.py")).unwrap(),
            "y = 99\n"
        );

        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn vfs_rejects_out_of_workspace_absolute_paths() {
        let tmp = std::env::temp_dir().join(format!("keel-vfs-sec-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let vfs = TransactionalVfs::new(tmp.clone(), VfsMode::MaterializedCow, false);

        // Absolute path outside workspace must be rejected
        let res_read = vfs.read_to_string(Path::new("/etc/passwd"));
        assert!(res_read.is_err());
        assert!(res_read
            .unwrap_err()
            .to_string()
            .contains("escapes workspace"));

        let res_write = vfs.write_string(Path::new("/etc/passwd"), "bad");
        assert!(res_write.is_err());
        assert!(res_write
            .unwrap_err()
            .to_string()
            .contains("escapes workspace"));

        let _ = std::fs::remove_dir_all(tmp);
    }
}
