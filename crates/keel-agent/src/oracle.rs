//! Self-Healing Test Diagnostic Oracle & Pre-Edit Regression Sentinel (`keel oracle`).
//!
//! Provides surgical regression localization and diagnostic remediation context:
//! - Pre-flight `SentinelBaseline`: captures test failure baseline before agent edits
//! - Regression Delta: segregates quarantined pre-existing test failures from new regressions
//! - Failure Parser: extracts test names, panic locations, assertion failures, and stack traces
//! - Self-Healing Oracle: injects targeted failure remediation context into agent loop
//! - Reversible Sentinel: triggers automatic VFS rollback when a regression is introduced

use anyhow::{Context as _, Result};
use keel_kernel::{Plugin, Scope};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SentinelMode {
    /// Report new regressions in agent context and continue turn.
    #[default]
    Warn,
    /// Automatically roll back mutating VFS transaction on regression.
    Rollback,
    /// Strictly fail turn if any regression is detected.
    Strict,
}

impl SentinelMode {
    pub fn from_str_lossy(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "rollback" | "revert" => Self::Rollback,
            "strict" | "deny" => Self::Strict,
            _ => Self::Warn,
        }
    }
}

/// A parsed test failure with panic details and assertion locations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TestFailure {
    pub name: String,
    pub location: Option<String>,
    pub assertion: Option<String>,
    pub snippet: String,
}

/// A point-in-time snapshot of test outcomes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct TestSnapshot {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub ignored: usize,
    pub failures: Vec<TestFailure>,
    pub elapsed_ms: u64,
    pub raw_output: String,
}

/// Delta comparison between baseline test run and post-edit test run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RegressionDelta {
    pub pre_existing_failures: Vec<String>,
    pub new_regressions: Vec<TestFailure>,
    pub resolved_tests: Vec<String>,
    pub is_clean: bool,
}

impl RegressionDelta {
    pub fn compute(baseline: &TestSnapshot, current: &TestSnapshot) -> Self {
        let baseline_set: HashSet<&str> =
            baseline.failures.iter().map(|f| f.name.as_str()).collect();
        let current_set: HashSet<&str> = current.failures.iter().map(|f| f.name.as_str()).collect();

        let mut pre_existing_failures = Vec::new();
        let mut new_regressions = Vec::new();

        for f in &current.failures {
            if baseline_set.contains(f.name.as_str()) {
                pre_existing_failures.push(f.name.clone());
            } else {
                new_regressions.push(f.clone());
            }
        }

        let mut resolved_tests = Vec::new();
        for name in &baseline_set {
            if !current_set.contains(name) {
                resolved_tests.push((*name).to_string());
            }
        }

        let is_clean = new_regressions.is_empty();

        Self {
            pre_existing_failures,
            new_regressions,
            resolved_tests,
            is_clean,
        }
    }

    /// Renders high-signal diagnostic remediation markdown for agent consumption.
    pub fn render_remediation(&self) -> Option<String> {
        if self.is_clean {
            return None;
        }

        let mut out = String::new();
        out.push_str("⚠️ **Regression Sentinel Alert**\n");
        out.push_str(&format!(
            "Your recent edits introduced {} new test regression(s):\n\n",
            self.new_regressions.len()
        ));

        for (i, f) in self.new_regressions.iter().enumerate() {
            out.push_str(&format!("{}. **`{}`**\n", i + 1, f.name));
            if let Some(loc) = &f.location {
                out.push_str(&format!("   - Location: `{loc}`\n"));
            }
            if let Some(assert) = &f.assertion {
                out.push_str(&format!("   - Assertion: `{assert}`\n"));
            }
            if !f.snippet.is_empty() {
                out.push_str("   - Failure Output:\n   ```text\n");
                for line in f.snippet.lines().take(8) {
                    out.push_str(&format!("     {line}\n"));
                }
                out.push_str("   ```\n");
            }
            out.push('\n');
        }

        if !self.pre_existing_failures.is_empty() {
            out.push_str(&format!(
                "ℹ️ *Note: {} pre-existing failure(s) ({}) were quarantined and ignored.*\n\n",
                self.pre_existing_failures.len(),
                self.pre_existing_failures.join(", ")
            ));
        }

        if !self.resolved_tests.is_empty() {
            out.push_str(&format!(
                "✅ *Fixed tests: {}*\n\n",
                self.resolved_tests.join(", ")
            ));
        }

        out.push_str("Please inspect the failure location and assertion before continuing.");
        Some(out)
    }
}

/// Parses stdout/stderr output from Cargo, Pytest, or generic test runners.
pub fn parse_test_output(raw: &str, elapsed_ms: u64) -> TestSnapshot {
    let mut failures = Vec::new();
    let mut passed = 0;
    let mut failed = 0;
    let mut ignored = 0;
    let mut total = 0;

    let lines: Vec<&str> = raw.lines().collect();

    // 1. Cargo test parsing: check summary line: `test result: FAILED. 45 passed; 3 failed; 0 ignored;`
    for line in &lines {
        let tr = line.trim();
        if tr.starts_with("test result:") {
            for part in tr.split(';') {
                let p = part.trim();
                if let Some(n) = p
                    .split_whitespace()
                    .next()
                    .and_then(|v| v.parse::<usize>().ok())
                {
                    if p.contains("passed") {
                        passed = n;
                    } else if p.contains("failed") {
                        failed = n;
                    } else if p.contains("ignored") {
                        ignored = n;
                    }
                }
            }
            total = passed + failed + ignored;
        } else if tr.starts_with("test ") && tr.ends_with(" ... ok") {
            if total == 0 {
                passed += 1;
            }
        } else if tr.starts_with("test ") && tr.ends_with(" ... FAILED") {
            if total == 0 {
                failed += 1;
            }
            let test_name = tr
                .strip_prefix("test ")
                .and_then(|s| s.strip_suffix(" ... FAILED"))
                .unwrap_or(tr)
                .trim()
                .to_string();
            if !failures.iter().any(|f: &TestFailure| f.name == test_name) {
                failures.push(TestFailure {
                    name: test_name,
                    location: None,
                    assertion: None,
                    snippet: String::new(),
                });
            }
        }
    }

    // 2. Extract failure stdout blocks: `---- <test_name> stdout ----`
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();
        if line.starts_with("---- ") && line.ends_with(" stdout ----") {
            let test_name = line
                .strip_prefix("---- ")
                .and_then(|s| s.strip_suffix(" stdout ----"))
                .unwrap_or("")
                .trim()
                .to_string();

            let mut block_lines = Vec::new();
            let mut location = None;
            let mut assertion = None;

            i += 1;
            while i < lines.len() {
                let bl = lines[i];
                if bl.starts_with("---- ")
                    || bl.starts_with("failures:")
                    || bl.starts_with("test result:")
                {
                    break;
                }
                block_lines.push(bl);

                let trimmed_bl = bl.trim();
                if trimmed_bl.contains("panicked at") {
                    if let Some(pos) = trimmed_bl.find("panicked at") {
                        let loc_str = trimmed_bl[pos + "panicked at".len()..]
                            .trim()
                            .trim_matches(['\'', ':']);
                        location = Some(loc_str.to_string());
                    }
                } else if trimmed_bl.starts_with("assertion ")
                    || trimmed_bl.starts_with("assertion `")
                {
                    assertion = Some(trimmed_bl.to_string());
                }
                i += 1;
            }

            let snippet = block_lines.join("\n").trim().to_string();

            if let Some(existing) = failures.iter_mut().find(|f| f.name == test_name) {
                existing.location = location;
                existing.assertion = assertion;
                existing.snippet = snippet;
            } else {
                failures.push(TestFailure {
                    name: test_name,
                    location,
                    assertion,
                    snippet,
                });
            }
            continue;
        }

        // 3. Pytest / generic fallback: `FAILED tests/test_foo.py::test_bar - AssertionError`
        if line.starts_with("FAILED ") {
            let rest = line.strip_prefix("FAILED ").unwrap_or(line).trim();
            let parts: Vec<&str> = rest.splitn(2, " - ").collect();
            let name = parts[0].to_string();
            let err = parts.get(1).map(|s| s.to_string());
            if !failures.iter().any(|f| f.name == name) {
                failures.push(TestFailure {
                    name,
                    location: None,
                    assertion: err,
                    snippet: line.to_string(),
                });
            }
        }

        i += 1;
    }

    if total == 0 {
        total = passed + failed + ignored;
    }

    TestSnapshot {
        total,
        passed,
        failed: failures.len().max(failed),
        ignored,
        failures,
        elapsed_ms,
        raw_output: raw.to_string(),
    }
}

/// The diagnostic oracle service managing test execution, baselining, and regression sentinel checks.
pub struct DiagnosticOracle {
    pub root: PathBuf,
    pub test_command: Option<String>,
    pub baseline: Arc<Mutex<Option<TestSnapshot>>>,
    pub mode: SentinelMode,
}

impl DiagnosticOracle {
    pub fn new(root: PathBuf, test_command: Option<String>, mode: SentinelMode) -> Self {
        Self {
            root,
            test_command,
            baseline: Arc::new(Mutex::new(None)),
            mode,
        }
    }

    /// Run the test suite and capture a baseline snapshot.
    pub fn capture_baseline(&self) -> Result<TestSnapshot> {
        let snapshot = self.run_tests()?;
        if let Ok(mut g) = self.baseline.lock() {
            *g = Some(snapshot.clone());
        }
        Ok(snapshot)
    }

    /// Clear the active baseline snapshot.
    pub fn clear_baseline(&self) {
        if let Ok(mut g) = self.baseline.lock() {
            *g = None;
        }
    }

    /// Get a copy of the current baseline snapshot if one has been recorded.
    pub fn get_baseline(&self) -> Option<TestSnapshot> {
        self.baseline.lock().ok().and_then(|g| g.clone())
    }

    /// Execute the configured test command (or `cargo test` defaulting to 2 threads).
    pub fn run_tests(&self) -> Result<TestSnapshot> {
        let start = Instant::now();
        let cmd_str = self.resolve_test_command();

        let mut cmd = Command::new("sh");
        cmd.arg("-c")
            .arg(&cmd_str)
            .current_dir(&self.root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("CARGO_BUILD_JOBS", "2")
            .env("RAYON_NUM_THREADS", "2");

        let out = cmd
            .output()
            .with_context(|| format!("executing test command: {cmd_str}"))?;

        let elapsed_ms = start.elapsed().as_millis() as u64;
        let mut raw = String::from_utf8_lossy(&out.stdout).to_string();
        if !out.stderr.is_empty() {
            raw.push('\n');
            raw.push_str(&String::from_utf8_lossy(&out.stderr));
        }

        Ok(parse_test_output(&raw, elapsed_ms))
    }

    /// Run tests and compute regression delta against the recorded baseline.
    pub fn check_delta(&self) -> Result<RegressionDelta> {
        let current = self.run_tests()?;
        let baseline = self.get_baseline().unwrap_or_default();
        Ok(RegressionDelta::compute(&baseline, &current))
    }

    fn resolve_test_command(&self) -> String {
        if let Some(ref c) = self.test_command {
            return c.clone();
        }
        if self.root.join("Cargo.toml").exists() {
            "cargo test -j 2 -- --test-threads=2".to_string()
        } else if self.root.join("package.json").exists() {
            "npm test".to_string()
        } else if self.root.join("pytest.ini").exists() || self.root.join("setup.py").exists() {
            "pytest -q".to_string()
        } else {
            "cargo test -j 2 -- --test-threads=2".to_string()
        }
    }
}

/// Plugin mounting [`DiagnosticOracle`] onto the kernel context.
pub struct OraclePlugin {
    pub root: PathBuf,
    pub test_command: Option<String>,
    pub mode: SentinelMode,
    pub auto_baseline: bool,
}

impl Plugin for OraclePlugin {
    fn name(&self) -> &str {
        "diagnostic-oracle"
    }

    fn apply(&self, scope: &mut Scope) {
        let oracle = Arc::new(DiagnosticOracle::new(
            self.root.clone(),
            self.test_command.clone(),
            self.mode,
        ));
        scope.provide(oracle.clone());

        if self.auto_baseline {
            let o_clone = oracle.clone();
            scope.on::<crate::TurnStart, _>(move |_| {
                if o_clone.get_baseline().is_none() {
                    let _ = o_clone.capture_baseline();
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cargo_test_output_with_panics_and_assertions() {
        let sample_output = r#"
running 3 tests
test tests::test_passing ... ok
test tests::test_broken ... FAILED
test tests::test_ignored ... ignored

failures:

---- tests::test_broken stdout ----
thread 'tests::test_broken' panicked at crates/keel-agent/src/vfs.rs:312:9:
assertion `left == right` failed
  left: "corrupted"
 right: "clean"
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

failures:
    tests::test_broken

test result: FAILED. 1 passed; 1 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.12s
"#;

        let snapshot = parse_test_output(sample_output, 120);
        assert_eq!(snapshot.passed, 1);
        assert_eq!(snapshot.failed, 1);
        assert_eq!(snapshot.ignored, 1);
        assert_eq!(snapshot.total, 3);
        assert_eq!(snapshot.failures.len(), 1);

        let f = &snapshot.failures[0];
        assert_eq!(f.name, "tests::test_broken");
        assert_eq!(
            f.location.as_deref(),
            Some("crates/keel-agent/src/vfs.rs:312:9")
        );
        assert!(f
            .assertion
            .as_deref()
            .unwrap()
            .contains("assertion `left == right` failed"));
        assert!(f.snippet.contains("left: \"corrupted\""));
    }

    #[test]
    fn regression_delta_quarantines_preexisting_and_detects_new_regression() {
        let baseline = TestSnapshot {
            total: 2,
            passed: 1,
            failed: 1,
            ignored: 0,
            failures: vec![TestFailure {
                name: "tests::pre_existing_flake".into(),
                location: None,
                assertion: None,
                snippet: "already broken".into(),
            }],
            elapsed_ms: 100,
            raw_output: String::new(),
        };

        // Turn introduces a new failure while pre-existing failure persists
        let post_edit = TestSnapshot {
            total: 2,
            passed: 0,
            failed: 2,
            ignored: 0,
            failures: vec![
                TestFailure {
                    name: "tests::pre_existing_flake".into(),
                    location: None,
                    assertion: None,
                    snippet: "already broken".into(),
                },
                TestFailure {
                    name: "tests::new_regression".into(),
                    location: Some("src/lib.rs:50".into()),
                    assertion: Some("assertion failed".into()),
                    snippet: "panicked".into(),
                },
            ],
            elapsed_ms: 110,
            raw_output: String::new(),
        };

        let delta = RegressionDelta::compute(&baseline, &post_edit);
        assert!(!delta.is_clean);
        assert_eq!(
            delta.pre_existing_failures,
            vec!["tests::pre_existing_flake"]
        );
        assert_eq!(delta.new_regressions.len(), 1);
        assert_eq!(delta.new_regressions[0].name, "tests::new_regression");

        let remediation = delta.render_remediation().unwrap();
        assert!(remediation.contains("Regression Sentinel Alert"));
        assert!(remediation.contains("`tests::new_regression`"));
        assert!(remediation
            .contains("pre-existing failure(s) (tests::pre_existing_flake) were quarantined"));
    }

    #[test]
    fn regression_delta_clean_when_only_preexisting_fails() {
        let baseline = TestSnapshot {
            total: 1,
            passed: 0,
            failed: 1,
            ignored: 0,
            failures: vec![TestFailure {
                name: "tests::flaky".into(),
                location: None,
                assertion: None,
                snippet: "flaky".into(),
            }],
            elapsed_ms: 50,
            raw_output: String::new(),
        };

        let current = baseline.clone();
        let delta = RegressionDelta::compute(&baseline, &current);
        assert!(delta.is_clean);
        assert_eq!(delta.new_regressions.len(), 0);
        assert_eq!(delta.render_remediation(), None);
    }
}
