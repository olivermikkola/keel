//! Linux Landlock sandbox plugin (`keel-agent::sandbox`).
//!
//! Enforces kernel-level filesystem confinement on spawned `shell` child processes
//! via `CommandExt::pre_exec` (between `fork()` and `execve()`), so even a compromised
//! or hallucinating command cannot overwrite files outside the workspace + `/tmp`.

use crate::Tool;
use anyhow::Result;
use landlock::{
    Access, AccessFs, PathBeneath, PathFd, Ruleset, RulesetAttr, RulesetCreatedAttr, RulesetStatus,
    ABI,
};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct SandboxPolicy {
    pub workspace: PathBuf,
    pub enabled: bool,
}

impl SandboxPolicy {
    pub fn new(workspace: impl Into<PathBuf>, enabled: bool) -> Self {
        Self {
            workspace: workspace.into(),
            enabled,
        }
    }

    /// Apply Landlock rules inside the child process before `exec`.
    /// Read/Execute is permitted across standard system paths (`/usr`, `/bin`, `/lib`, `/etc`, `/dev`, `/proc`),
    /// while Read+Write is restricted strictly to `self.workspace` and `/tmp`.
    pub fn restrict_current_thread(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        let abi = ABI::V3;
        let read_only = AccessFs::Execute | AccessFs::ReadFile | AccessFs::ReadDir;
        let read_write = AccessFs::from_all(abi);

        let mut ruleset = Ruleset::default().handle_access(read_write)?.create()?;

        for ro_dir in [
            "/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/proc", "/dev", "/var",
        ] {
            let p = Path::new(ro_dir);
            if p.exists() {
                if let Ok(fd) = PathFd::new(p) {
                    ruleset = ruleset.add_rule(PathBeneath::new(fd, read_only))?;
                }
            }
        }

        for rw_dir in [self.workspace.as_path(), Path::new("/tmp")] {
            if rw_dir.exists() {
                if let Ok(fd) = PathFd::new(rw_dir) {
                    ruleset = ruleset.add_rule(PathBeneath::new(fd, read_write))?;
                }
            }
        }

        let status = ruleset.restrict_self()?;
        if matches!(status.ruleset, RulesetStatus::NotEnforced) {
            anyhow::bail!("Landlock is not supported by the running kernel");
        }
        Ok(())
    }
}

/// Plugin that mounts [`SandboxPolicy`] into the kernel [`keel_kernel::Context`].
pub struct LandlockPlugin {
    pub workspace: PathBuf,
    pub enabled: bool,
}

impl keel_kernel::Plugin for LandlockPlugin {
    fn name(&self) -> &str {
        "landlock-sandbox"
    }

    fn apply(&self, scope: &mut keel_kernel::Scope) {
        scope.provide(SandboxPolicy::new(self.workspace.clone(), self.enabled));
    }
}

/// Helper to register a list of tools into the [`ToolRegistry`] service on the kernel.
#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: std::sync::Arc<std::sync::Mutex<Vec<std::sync::Arc<dyn Tool>>>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, tool: std::sync::Arc<dyn Tool>) {
        self.tools.lock().unwrap().push(tool);
    }

    pub fn list(&self) -> Vec<std::sync::Arc<dyn Tool>> {
        self.tools.lock().unwrap().clone()
    }
}
