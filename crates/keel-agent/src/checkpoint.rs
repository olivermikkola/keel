//! Ephemeral git shadow checkpointing and instant workspace rollback.
//!
//! Creates content-addressed git snapshots without altering HEAD, the user's
//! active branch, or the working git index. Enables:
//! - Instant `/undo` in REPL and interactive sessions
//! - Safe autonomous exploration with auto-revert on verification failure
//! - Scriptable CLI subcommands (`keel checkpoint save/list/restore/diff`, `keel undo`)

use anyhow::{bail, Context, Result};
use keel_kernel::{Plugin, Scope};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointInfo {
    pub id: String,
    pub description: String,
    pub created_at: String,
    pub commit_hash: String,
    pub tree_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestoreReport {
    pub checkpoint_id: String,
    pub restored_files: usize,
    pub removed_files: usize,
}

pub struct CheckpointManager {
    root: PathBuf,
}

impl CheckpointManager {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn meta_file(&self) -> PathBuf {
        let dot_git = self.root.join(".git");
        if dot_git.is_dir() {
            dot_git.join("keel_checkpoints.json")
        } else {
            self.root.join(".keel").join("checkpoints.json")
        }
    }

    fn run_git(&self, args: &[&str], extra_env: &[(&str, &str)]) -> Result<String> {
        let mut cmd = Command::new("git");
        cmd.args(args)
            .current_dir(&self.root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (k, v) in extra_env {
            cmd.env(k, v);
        }
        let out = cmd
            .output()
            .with_context(|| format!("executing git {}", args.join(" ")))?;
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            bail!("git {} failed: {err}", args.join(" "));
        }
        Ok(String::from_utf8_lossy(&out.stdout).to_string())
    }

    fn ensure_git_repo(&self) -> Result<()> {
        if is_forbidden_checkpoint_root(&self.root) {
            bail!(
                "refusing to initialize or run git checkpoint in protected directory: {}",
                self.root.display()
            );
        }
        if !self.root.join(".git").exists() {
            self.run_git(&["init"], &[])
                .context("initializing git repo for checkpoints")?;
        }
        Ok(())
    }

    pub fn list(&self) -> Result<Vec<CheckpointInfo>> {
        let p = self.meta_file();
        if !p.exists() {
            return Ok(Vec::new());
        }
        let data =
            std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
        let list: Vec<CheckpointInfo> = serde_json::from_str(&data).unwrap_or_default();
        Ok(list)
    }

    fn save_meta_list(&self, list: &[CheckpointInfo]) -> Result<()> {
        let p = self.meta_file();
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let json = serde_json::to_string_pretty(list)?;
        std::fs::write(&p, json).with_context(|| format!("writing {}", p.display()))?;
        Ok(())
    }

    /// Snapshot the entire working tree into git object storage under a shadow ref.
    pub fn save(&self, description: &str) -> Result<CheckpointInfo> {
        self.ensure_git_repo()?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let id = format!("ckpt-{}", now.as_millis());
        let created_at = format!("{}", now.as_secs());

        let dot_git = self.root.join(".git");
        let tmp_idx = dot_git.join(format!("keel_idx_{id}"));
        let tmp_idx_str = tmp_idx.to_string_lossy();

        // 1. Stage all working tree files into the temporary index
        let add_res = self.run_git(&["add", "-A"], &[("GIT_INDEX_FILE", &tmp_idx_str)]);
        if let Err(e) = add_res {
            cleanup_temp_index(&tmp_idx);
            return Err(e);
        }

        // 2. Write tree object from temporary index
        let tree_res = self.run_git(&["write-tree"], &[("GIT_INDEX_FILE", &tmp_idx_str)]);
        cleanup_temp_index(&tmp_idx);
        let tree_hash = tree_res?.trim().to_string();

        // 3. Commit the tree
        let commit_msg = format!("keel checkpoint: {description}");
        let commit_hash = self
            .run_git(&["commit-tree", &tree_hash, "-m", &commit_msg], &[])?
            .trim()
            .to_string();

        // 4. Update shadow ref
        let ref_name = format!("refs/keel/checkpoints/{id}");
        self.run_git(&["update-ref", &ref_name, &commit_hash], &[])?;

        let info = CheckpointInfo {
            id,
            description: description.to_string(),
            created_at,
            commit_hash,
            tree_hash,
        };

        let mut list = self.list().unwrap_or_default();
        list.push(info.clone());
        self.save_meta_list(&list)?;

        Ok(info)
    }

    /// Restore the working tree to the state of a checkpoint.
    pub fn restore(&self, id_or_name: &str) -> Result<RestoreReport> {
        self.ensure_git_repo()?;
        let list = self.list()?;
        if list.is_empty() {
            bail!("no checkpoints found to restore");
        }

        let target = if id_or_name.is_empty() || id_or_name == "latest" {
            list.last().unwrap()
        } else {
            list.iter()
                .rev()
                .find(|c| c.id == id_or_name || c.description == id_or_name)
                .with_context(|| format!("checkpoint `{id_or_name}` not found"))?
        };

        let dot_git = self.root.join(".git");
        let tmp_idx = dot_git.join(format!("keel_restore_idx_{}", std::process::id()));
        let tmp_idx_str = tmp_idx.to_string_lossy();

        // 1. Read checkpoint tree into temporary index
        let read_res = self.run_git(
            &["read-tree", &target.tree_hash],
            &[("GIT_INDEX_FILE", &tmp_idx_str)],
        );
        if let Err(e) = read_res {
            cleanup_temp_index(&tmp_idx);
            return Err(e);
        }

        // 2. Checkout all files from temporary index into working tree
        let checkout_res = self.run_git(
            &["checkout-index", "-a", "-f"],
            &[("GIT_INDEX_FILE", &tmp_idx_str)],
        );
        if let Err(e) = checkout_res {
            cleanup_temp_index(&tmp_idx);
            return Err(e);
        }

        // 3. Collect all files that belong to the checkpoint
        let ls_out = self.run_git(&["ls-tree", "-r", "--name-only", &target.tree_hash], &[])?;
        let tree_files: HashSet<String> = ls_out.lines().map(|s| s.trim().to_string()).collect();

        // 4. Remove any non-ignored files in the workspace that were created after the checkpoint.
        // Using `git ls-files --cached --others --exclude-standard` respects `.gitignore` rules so
        // ignored files (e.g., `.env`, `node_modules`, `.venv`, build caches) are never deleted.
        let candidates_res = self.run_git(
            &["ls-files", "--cached", "--others", "--exclude-standard"],
            &[("GIT_INDEX_FILE", &tmp_idx_str)],
        );
        cleanup_temp_index(&tmp_idx);
        let candidates = candidates_res?;

        let mut removed_count = 0;
        for line in candidates.lines() {
            let rel = line.trim();
            if rel.is_empty() || rel.starts_with(".keel/") || rel == ".keel" {
                continue;
            }
            if !tree_files.contains(rel) {
                let full = self.root.join(rel);
                if full.is_file() {
                    let _ = std::fs::remove_file(&full);
                    removed_count += 1;
                }
            }
        }

        Ok(RestoreReport {
            checkpoint_id: target.id.clone(),
            restored_files: tree_files.len(),
            removed_files: removed_count,
        })
    }

    /// Revert to the latest checkpoint.
    pub fn undo(&self) -> Result<RestoreReport> {
        self.restore("latest")
    }

    /// Compute working tree diff against a checkpoint (default: latest).
    pub fn diff(&self, id_or_name: Option<&str>) -> Result<String> {
        self.ensure_git_repo()?;
        let list = self.list()?;
        if list.is_empty() {
            bail!("no checkpoints available to diff against");
        }
        let target = match id_or_name {
            Some(name) if !name.is_empty() && name != "latest" => list
                .iter()
                .rev()
                .find(|c| c.id == name || c.description == name)
                .with_context(|| format!("checkpoint `{name}` not found"))?,
            _ => list.last().unwrap(),
        };

        let dot_git = self.root.join(".git");
        let tmp_idx = dot_git.join(format!("keel_diff_idx_{}", std::process::id()));
        let tmp_idx_str = tmp_idx.to_string_lossy();

        let add_res = self.run_git(&["add", "-A"], &[("GIT_INDEX_FILE", &tmp_idx_str)]);
        if let Err(e) = add_res {
            cleanup_temp_index(&tmp_idx);
            return Err(e);
        }
        let tree_res = self.run_git(&["write-tree"], &[("GIT_INDEX_FILE", &tmp_idx_str)]);
        cleanup_temp_index(&tmp_idx);
        let curr_tree = tree_res?.trim().to_string();

        self.run_git(&["diff", &target.tree_hash, &curr_tree], &[])
    }
}

fn cleanup_temp_index(idx_path: &Path) {
    let _ = std::fs::remove_file(idx_path);
    let lock_path = idx_path.with_extension("lock");
    let _ = std::fs::remove_file(lock_path);
    let appended_lock = PathBuf::from(format!("{}.lock", idx_path.display()));
    let _ = std::fs::remove_file(appended_lock);
}

/// Returns true if `root` is `$HOME` or a top-level system directory where
/// automatic `git init` / `git add -A` must never run.
pub fn is_forbidden_checkpoint_root(root: &Path) -> bool {
    let canon = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    if let Ok(home) = std::env::var("HOME") {
        let home_path = PathBuf::from(&home);
        let canon_home = home_path.canonicalize().unwrap_or(home_path);
        if canon == canon_home {
            return true;
        }
    }
    let s = canon.to_string_lossy();
    matches!(
        s.as_ref(),
        "/" | "/home" | "/root" | "/tmp" | "/var" | "/var/tmp" | "/etc" | "/usr"
    )
}

/// Policy controlling automatic snapshot and auto-rollback behavior.
#[derive(Clone, Debug, Default)]
pub struct CheckpointPolicy {
    pub auto_checkpoint: bool,
    pub auto_rollback: bool,
}

/// Plugin that mounts the [`CheckpointManager`] service and [`CheckpointPolicy`] onto the kernel context.
pub struct CheckpointPlugin {
    pub root: PathBuf,
    pub auto_checkpoint: bool,
    pub auto_rollback: bool,
}

impl Plugin for CheckpointPlugin {
    fn name(&self) -> &str {
        "checkpoint"
    }
    fn apply(&self, scope: &mut Scope) {
        scope.provide(CheckpointManager::new(self.root.clone()));
        scope.provide(CheckpointPolicy {
            auto_checkpoint: self.auto_checkpoint,
            auto_rollback: self.auto_rollback,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_save_restore_and_undo_roundtrip() {
        let tmp = std::env::temp_dir().join(format!("keel-ckpt-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let mgr = CheckpointManager::new(tmp.clone());
        mgr.ensure_git_repo().unwrap();

        // 1. Initial state
        std::fs::write(tmp.join("file1.txt"), "hello world\n").unwrap();
        let ckpt1 = mgr.save("initial state").unwrap();
        assert!(!ckpt1.commit_hash.is_empty());
        assert!(!ckpt1.tree_hash.is_empty());

        // 2. Mutate workspace (modify file1, create file2)
        std::fs::write(tmp.join("file1.txt"), "corrupted content\n").unwrap();
        std::fs::write(tmp.join("file2.txt"), "extraneous file\n").unwrap();
        assert_eq!(
            std::fs::read_to_string(tmp.join("file1.txt")).unwrap(),
            "corrupted content\n"
        );
        assert!(tmp.join("file2.txt").exists());

        // 3. Diff should show changes
        let diff = mgr.diff(None).unwrap();
        assert!(diff.contains("corrupted content"));

        // 4. Undo / Restore
        let report = mgr.undo().unwrap();
        assert_eq!(report.checkpoint_id, ckpt1.id);
        assert_eq!(
            std::fs::read_to_string(tmp.join("file1.txt")).unwrap(),
            "hello world\n"
        );
        assert!(
            !tmp.join("file2.txt").exists(),
            "extraneous file was not removed"
        );

        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn checkpoint_restore_preserves_gitignored_and_keel_files() {
        let tmp =
            std::env::temp_dir().join(format!("keel-ckpt-gitignore-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let mgr = CheckpointManager::new(tmp.clone());
        mgr.ensure_git_repo().unwrap();

        // Create .gitignore, tracked file, and ignored .env + node_modules file
        std::fs::write(tmp.join(".gitignore"), ".env\nnode_modules/\n").unwrap();
        std::fs::write(tmp.join("src.rs"), "fn main() {}\n").unwrap();
        std::fs::write(tmp.join(".env"), "SECRET_KEY=keep_me\n").unwrap();
        std::fs::create_dir_all(tmp.join("node_modules/pkg")).unwrap();
        std::fs::write(tmp.join("node_modules/pkg/index.js"), "module.exports = 1;\n").unwrap();
        std::fs::create_dir_all(tmp.join(".keel")).unwrap();
        std::fs::write(tmp.join(".keel/memory.json"), "{}\n").unwrap();

        let _ckpt = mgr.save("with gitignore").unwrap();

        // Mutate tracked file and add an untracked non-ignored file
        std::fs::write(tmp.join("src.rs"), "broken\n").unwrap();
        std::fs::write(tmp.join("temp_untracked.txt"), "delete me\n").unwrap();

        let rep = mgr.undo().unwrap();
        assert_eq!(rep.removed_files, 1);
        assert_eq!(
            std::fs::read_to_string(tmp.join("src.rs")).unwrap(),
            "fn main() {}\n"
        );
        assert!(!tmp.join("temp_untracked.txt").exists());
        // Ignored files and .keel metadata must still exist intact!
        assert_eq!(
            std::fs::read_to_string(tmp.join(".env")).unwrap(),
            "SECRET_KEY=keep_me\n"
        );
        assert!(tmp.join("node_modules/pkg/index.js").exists());
        assert!(tmp.join(".keel/memory.json").exists());

        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn checkpoint_refuses_protected_roots() {
        assert!(is_forbidden_checkpoint_root(Path::new("/")));
        assert!(is_forbidden_checkpoint_root(Path::new("/tmp")));
        if let Ok(home) = std::env::var("HOME") {
            let mgr = CheckpointManager::new(PathBuf::from(home));
            assert!(mgr.save("should fail").is_err());
        }
    }
}
