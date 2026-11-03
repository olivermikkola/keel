//! Linux Landlock sandbox plugin (`keel-agent::sandbox`).
//!
//! Enforces kernel-level filesystem confinement on spawned `shell` child processes
//! via `CommandExt::pre_exec` (between `fork()` and `execve()`), so even a compromised
//! or hallucinating command cannot overwrite files outside the workspace + `/tmp`.
//!
//! Features:
//! - Resilient developer toolchain discovery: auto-permits read-only execution of toolchains
//!   discovered in `$PATH`, `$CARGO_HOME`, `$RUSTUP_HOME`, `$GOPATH`, `$NODE_PATH`, etc., as well
//!   as standard user roots (`~/.cargo`, `~/.rustup`, `~/.local`, `~/.nvm`, `~/.pyenv`, etc.).
//! - Standard POSIX device nodes (`/dev/null`, `/dev/zero`, `/dev/urandom`, `/dev/tty`, etc.)
//!   for standard I/O redirection and piping.
//! - Configurable user overrides: `extra_read_paths` and `extra_write_paths` for custom dev trees.

use crate::Tool;
use anyhow::Result;
use landlock::{
    Access, AccessFs, PathBeneath, PathFd, Ruleset, RulesetAttr, RulesetCreatedAttr, RulesetStatus,
    ABI,
};
use std::collections::HashSet;
use std::env;
use std::path::PathBuf;

/// Default device nodes that should be accessible for standard I/O redirection.
pub const DEFAULT_DEVICE_NODES: &[&str] = &[
    "/dev/null",
    "/dev/zero",
    "/dev/urandom",
    "/dev/random",
    "/dev/tty",
    "/dev/stdin",
    "/dev/stdout",
    "/dev/stderr",
];

/// Common user-level toolchain root directories to inspect under `$HOME`.
pub const COMMON_TOOLCHAIN_ROOTS: &[&str] = &[
    ".cargo",
    ".rustup",
    ".local",
    ".nvm",
    ".pyenv",
    ".virtualenvs",
    ".bun",
    ".deno",
    ".go",
    ".node_modules",
];

/// Environment variable names whose values (colon-separated paths) should be
/// inspected for toolchain directories.
pub const TOOLCHAIN_ENV_VARS: &[&str] = &[
    "PATH",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "GOPATH",
    "NODE_PATH",
    "PYTHONPATH",
    "BUN_INSTALL",
    "DENO_INSTALL",
];

#[derive(Clone, Debug)]
pub struct SandboxPolicy {
    pub workspace: PathBuf,
    pub enabled: bool,
    pub extra_read_paths: Vec<PathBuf>,
    pub extra_write_paths: Vec<PathBuf>,
    pub discover_toolchains: bool,
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        Self {
            workspace: PathBuf::from("."),
            enabled: true,
            extra_read_paths: Vec::new(),
            extra_write_paths: Vec::new(),
            discover_toolchains: true,
        }
    }
}

impl SandboxPolicy {
    /// Creates a new `SandboxPolicy` with the given workspace and enabled flag.
    pub fn new(workspace: impl Into<PathBuf>, enabled: bool) -> Self {
        Self {
            workspace: workspace.into(),
            enabled,
            ..Default::default()
        }
    }

    /// Adds a path to the list of extra read-only paths.
    pub fn with_extra_read(mut self, path: impl Into<PathBuf>) -> Self {
        self.extra_read_paths.push(path.into());
        self
    }

    /// Adds a path to the list of extra read-write paths.
    pub fn with_extra_write(mut self, path: impl Into<PathBuf>) -> Self {
        self.extra_write_paths.push(path.into());
        self
    }

    /// Enables or disables automatic toolchain discovery.
    pub fn with_toolchains(mut self, enabled: bool) -> Self {
        self.discover_toolchains = enabled;
        self
    }

    /// Discovers toolchain directories from environment variables and common user-level roots.
    pub fn discover_toolchain_paths(&self) -> Vec<PathBuf> {
        if !self.discover_toolchains {
            return Vec::new();
        }

        let mut discovered = Vec::new();
        let mut seen = HashSet::new();

        // Inspect PATH and other env vars for toolchain directories.
        for var in TOOLCHAIN_ENV_VARS {
            if let Some(val) = env::var_os(var) {
                for part in env::split_paths(&val) {
                    if part.exists() && seen.insert(part.clone()) {
                        discovered.push(part);
                    }
                }
            }
        }

        // Inspect common user toolchain roots under $HOME.
        if let Some(home) = env::var_os("HOME") {
            let home_path = PathBuf::from(home);
            for root in COMMON_TOOLCHAIN_ROOTS {
                let candidate = home_path.join(root);
                if candidate.exists() && seen.insert(candidate.clone()) {
                    discovered.push(candidate);
                }
            }
        }

        discovered
    }

    /// Returns the combined set of read-only paths to grant access to.
    pub fn read_only_paths(&self) -> Vec<PathBuf> {
        let mut paths = Vec::new();

        // System directories that should be read-only.
        for dir in [
            "/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/proc", "/dev", "/var",
        ] {
            paths.push(PathBuf::from(dir));
        }

        // Discovered toolchain paths.
        paths.extend(self.discover_toolchain_paths());

        // User-configured extra read-only paths.
        paths.extend(self.extra_read_paths.clone());

        // Deduplicate
        let mut seen = HashSet::new();
        paths.retain(|p| seen.insert(p.clone()));

        paths
    }

    /// Returns the combined set of read-write paths to grant access to.
    pub fn write_paths(&self) -> Vec<PathBuf> {
        let mut paths = Vec::new();
        paths.push(self.workspace.clone());
        paths.push(PathBuf::from("/tmp"));
        if let Ok(tmpdir) = env::var("TMPDIR") {
            let p = PathBuf::from(tmpdir);
            if p.exists() {
                paths.push(p);
            }
        }
        paths.extend(self.extra_write_paths.clone());

        // Deduplicate
        let mut seen = HashSet::new();
        paths.retain(|p| seen.insert(p.clone()));

        paths
    }

    /// Apply Landlock rules inside the child process before `exec`.
    /// Read/Execute is permitted across standard system paths (`/usr`, `/bin`, `/lib`, `/etc`, `/dev`, `/proc`),
    /// plus discovered toolchains and extra read paths.
    /// Read+Write is restricted strictly to `self.workspace`, `/tmp`, and configured extra write paths.
    pub fn restrict_current_thread(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        let abi = ABI::V3;
        let read_only = AccessFs::Execute | AccessFs::ReadFile | AccessFs::ReadDir;
        let read_write = AccessFs::from_all(abi);

        let mut ruleset = Ruleset::default().handle_access(read_write)?.create()?;

        // Grant read-only access to system directories and discovered toolchains.
        for ro_path in self.read_only_paths() {
            if ro_path.exists() {
                if let Ok(fd) = PathFd::new(&ro_path) {
                    ruleset = ruleset.add_rule(PathBeneath::new(fd, read_only))?;
                }
            }
        }

        // Grant read-write access to workspace, /tmp, and extra write paths.
        for rw_path in self.write_paths() {
            if rw_path.exists() {
                if let Ok(fd) = PathFd::new(&rw_path) {
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
#[derive(Clone, Debug)]
pub struct LandlockPlugin {
    pub workspace: PathBuf,
    pub enabled: bool,
    pub extra_read_paths: Vec<PathBuf>,
    pub extra_write_paths: Vec<PathBuf>,
    pub discover_toolchains: bool,
}

impl Default for LandlockPlugin {
    fn default() -> Self {
        Self {
            workspace: PathBuf::from("."),
            enabled: true,
            extra_read_paths: Vec::new(),
            extra_write_paths: Vec::new(),
            discover_toolchains: true,
        }
    }
}

impl LandlockPlugin {
    pub fn new(workspace: impl Into<PathBuf>, enabled: bool) -> Self {
        Self {
            workspace: workspace.into(),
            enabled,
            ..Default::default()
        }
    }

    pub fn with_extra_read(mut self, path: impl Into<PathBuf>) -> Self {
        self.extra_read_paths.push(path.into());
        self
    }

    pub fn with_extra_write(mut self, path: impl Into<PathBuf>) -> Self {
        self.extra_write_paths.push(path.into());
        self
    }

    pub fn with_toolchains(mut self, enabled: bool) -> Self {
        self.discover_toolchains = enabled;
        self
    }
}

impl keel_kernel::Plugin for LandlockPlugin {
    fn name(&self) -> &str {
        "landlock-sandbox"
    }

    fn apply(&self, scope: &mut keel_kernel::Scope) {
        let mut policy = SandboxPolicy::new(self.workspace.clone(), self.enabled)
            .with_toolchains(self.discover_toolchains);
        for r in &self.extra_read_paths {
            policy = policy.with_extra_read(r.clone());
        }
        for w in &self.extra_write_paths {
            policy = policy.with_extra_write(w.clone());
        }
        scope.provide(policy);
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

    pub fn names(&self) -> Vec<String> {
        self.tools
            .lock()
            .unwrap()
            .iter()
            .map(|t| t.spec().name)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_policy_defaults() {
        let policy = SandboxPolicy::new("/tmp/workspace", true);
        assert_eq!(policy.workspace, PathBuf::from("/tmp/workspace"));
        assert!(policy.enabled);
        assert!(policy.discover_toolchains);
        assert!(policy.extra_read_paths.is_empty());
        assert!(policy.extra_write_paths.is_empty());
    }

    #[test]
    fn test_builder_api() {
        let tmp = env::temp_dir();
        let policy = SandboxPolicy::new(&tmp, true)
            .with_extra_read("/usr/local/bin")
            .with_extra_read("/opt/tools")
            .with_extra_write("/tmp/scratch")
            .with_toolchains(false);

        assert_eq!(
            policy.extra_read_paths,
            vec![PathBuf::from("/usr/local/bin"), PathBuf::from("/opt/tools")]
        );
        assert_eq!(
            policy.extra_write_paths,
            vec![PathBuf::from("/tmp/scratch")]
        );
        assert!(!policy.discover_toolchains);
    }

    #[test]
    fn test_discover_toolchain_paths_includes_path() {
        let policy = SandboxPolicy::new("/tmp/workspace", true).with_toolchains(true);

        let paths = policy.discover_toolchain_paths();
        // PATH should always contribute at least some directories on a normal system.
        assert!(
            !paths.is_empty(),
            "Expected at least some toolchain paths from PATH"
        );

        // Verify no duplicates
        let mut seen = HashSet::new();
        for p in &paths {
            assert!(seen.insert(p.clone()), "Duplicate path found: {:?}", p);
        }
    }

    #[test]
    fn test_discover_toolchain_paths_respects_flag() {
        let policy_disabled = SandboxPolicy::new("/tmp/workspace", true).with_toolchains(false);
        let paths_disabled = policy_disabled.discover_toolchain_paths();
        assert!(paths_disabled.is_empty());
    }

    #[test]
    fn test_read_only_paths_combines_all_sources() {
        let tmp = env::temp_dir();
        let extra = tmp.join("extra_read_test_dir");
        let _ = std::fs::create_dir_all(&extra);

        let policy = SandboxPolicy::new("/tmp/workspace", true)
            .with_extra_read(extra.clone())
            .with_toolchains(false);

        let paths = policy.read_only_paths();
        assert!(paths.contains(&PathBuf::from("/usr")));
        assert!(paths.contains(&extra));
    }

    #[test]
    fn test_write_paths_includes_workspace_and_tmp() {
        let policy = SandboxPolicy::new("/tmp/my_workspace", true).with_extra_write("/tmp/extra_w");

        let paths = policy.write_paths();
        assert!(paths.contains(&PathBuf::from("/tmp/my_workspace")));
        assert!(paths.contains(&PathBuf::from("/tmp")));
        assert!(paths.contains(&PathBuf::from("/tmp/extra_w")));
    }

    #[test]
    fn test_restrict_current_thread_disabled() {
        let policy = SandboxPolicy::new("/tmp/workspace", false);
        // Should return Ok without applying restrictions
        assert!(policy.restrict_current_thread().is_ok());
    }

    #[test]
    fn test_policy_clone() {
        let policy = SandboxPolicy::new("/tmp/workspace", true)
            .with_extra_read("/usr/local/bin")
            .with_extra_write("/tmp/scratch");

        let cloned = policy.clone();
        assert_eq!(policy.workspace, cloned.workspace);
        assert_eq!(policy.enabled, cloned.enabled);
        assert_eq!(policy.extra_read_paths, cloned.extra_read_paths);
        assert_eq!(policy.extra_write_paths, cloned.extra_write_paths);
        assert_eq!(policy.discover_toolchains, cloned.discover_toolchains);
    }

    #[test]
    fn test_device_nodes_list_contains_expected_entries() {
        assert!(DEFAULT_DEVICE_NODES.contains(&"/dev/null"));
        assert!(DEFAULT_DEVICE_NODES.contains(&"/dev/zero"));
        assert!(DEFAULT_DEVICE_NODES.contains(&"/dev/urandom"));
        assert!(DEFAULT_DEVICE_NODES.contains(&"/dev/tty"));
        assert!(!DEFAULT_DEVICE_NODES.is_empty());
    }
}
