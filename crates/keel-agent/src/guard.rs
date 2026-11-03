//! Guard plugins (`keel-agent::guard`), modeled on DeepSeek Harness's `@deepseek-ai/dsh-repeat-tool-reminder`.
//!
//! Detects when an agent repeats the exact same tool call with identical canonical JSON arguments
//! across consecutive invocations (at configurable thresholds, default `[2, 4, 7]`) and appends
//! an escalating advisory reminder to the tool output so smaller models break out of stuck loops.
//!
//! Also provides [`EnvironmentCircuitBreaker`], which detects consecutive environment-level
//! execution failures (Permission Denied, Command Not Found, Landlock confinement) and injects
//! high-priority advisories before models burn their turn budgets.

use crate::ToolCall;
use keel_kernel::{Plugin, Scope};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};

const GENTLE_REMINDER: &str = "\n\n[repeat-tool-reminder] You are repeating the exact same tool call with identical arguments. Carefully analyze the previous result before calling again: if the task is not complete, try a different approach or different arguments instead of repeating the call.";

fn detailed_reminder(tool_name: &str, count: usize, canonical_args: &str) -> String {
    let char_count = canonical_args.chars().count();
    let preview = if char_count > 500 {
        let truncated: String = canonical_args.chars().take(500).collect();
        format!("{}… (+{} more chars)", truncated, char_count - 500)
    } else {
        canonical_args.to_string()
    };
    format!(
        "\n\n[repeat-tool-reminder] Repeated tool call detected:\n- tool: {tool_name}\n- consecutive_calls: {count}\n- arguments: {preview}\nThe repeated calls are not making progress. Do not call this tool with these exact arguments again. Inspect the latest result and choose a different action, different arguments, or finish the task if enough evidence has been gathered."
    )
}

/// Deep key-sort of a JSON value so property order differences still canonicalize identically.
pub fn canonicalize_json(v: &Value) -> String {
    fn sort_val(v: &Value) -> Value {
        match v {
            Value::Array(arr) => Value::Array(arr.iter().map(sort_val).collect()),
            Value::Object(map) => {
                let sorted: BTreeMap<String, Value> = map
                    .iter()
                    .map(|(k, val)| (k.clone(), sort_val(val)))
                    .collect();
                let mut out = serde_json::Map::new();
                for (k, val) in sorted {
                    out.insert(k, val);
                }
                Value::Object(out)
            }
            other => other.clone(),
        }
    }
    serde_json::to_string(&sort_val(v)).unwrap_or_else(|_| "{}".into())
}

#[derive(Default)]
struct ChainState {
    last_key: String,
    count: usize,
}

#[derive(Clone)]
pub struct RepeatToolGuard {
    thresholds: Vec<usize>,
    state: Arc<Mutex<ChainState>>,
}

impl RepeatToolGuard {
    pub fn new(mut thresholds: Vec<usize>) -> Self {
        thresholds.sort_unstable();
        thresholds.dedup();
        Self {
            thresholds,
            state: Arc::new(Mutex::new(ChainState::default())),
        }
    }

    pub fn reset(&self) {
        let mut s = self.state.lock().unwrap();
        s.last_key.clear();
        s.count = 0;
    }

    /// Observe a tool call and return an optional reminder string to append to the tool result.
    pub fn observe(&self, call: &ToolCall) -> Option<String> {
        let canonical = canonicalize_json(&call.args);
        let key = format!("{}:{canonical}", call.name);
        let mut s = self.state.lock().unwrap();
        if s.last_key == key {
            s.count += 1;
        } else {
            s.last_key = key;
            s.count = 1;
        }
        let count = s.count;
        if !self.thresholds.contains(&count) {
            return None;
        }
        if Some(&count) == self.thresholds.first() {
            Some(GENTLE_REMINDER.to_string())
        } else {
            Some(detailed_reminder(&call.name, count, &canonical))
        }
    }
}

/// Plugin that mounts [`RepeatToolGuard`] and resets its chain on every [`crate::TurnStart`].
pub struct RepeatToolReminderPlugin {
    pub thresholds: Vec<usize>,
}

impl Default for RepeatToolReminderPlugin {
    fn default() -> Self {
        Self {
            thresholds: vec![2, 4, 7],
        }
    }
}

impl Plugin for RepeatToolReminderPlugin {
    fn name(&self) -> &str {
        "repeat-tool-reminder"
    }

    fn apply(&self, scope: &mut Scope) {
        let guard = RepeatToolGuard::new(self.thresholds.clone());
        let g_reset = guard.clone();
        scope.on::<crate::TurnStart, _>(move |_| {
            g_reset.reset();
        });
        scope.provide(guard);
    }
}

// ---------------------------------------------------------------------------
// Environment Circuit Breaker
// ---------------------------------------------------------------------------

/// Categorization of environment-level execution failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvironmentErrorType {
    /// Exit code 126 or output indicates a permission issue.
    PermissionDenied,
    /// Exit code 127 or output indicates a missing command/file.
    CommandNotFound,
    /// Output references sandboxing mechanisms (Landlock, seccomp, etc.).
    SandboxConfinement,
    /// Unclassified error with captured detail.
    Other(String),
}

impl fmt::Display for EnvironmentErrorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PermissionDenied => write!(f, "Permission denied"),
            Self::CommandNotFound => write!(f, "Command not found"),
            Self::SandboxConfinement => write!(f, "Sandbox confinement"),
            Self::Other(s) => write!(f, "{s}"),
        }
    }
}

/// Fired when the circuit breaker trips after repeated environmental failures.
#[derive(Debug, Clone)]
pub struct CircuitBreakerTrip {
    pub consecutive_failures: usize,
    pub error_type: EnvironmentErrorType,
    pub diagnostic_message: String,
    pub remedy_suggestion: String,
}

#[derive(Default)]
struct BreakerState {
    consecutive_failures: usize,
    last_error_type: Option<EnvironmentErrorType>,
}

/// Tracks consecutive environment-level shell command failures and trips
/// when a configurable threshold is reached.
///
/// Resets the counter on any successful (exit code 0) command execution.
#[derive(Clone)]
pub struct EnvironmentCircuitBreaker {
    trip_threshold: usize,
    state: Arc<Mutex<BreakerState>>,
}

impl Default for EnvironmentCircuitBreaker {
    fn default() -> Self {
        Self::new(3)
    }
}

impl EnvironmentCircuitBreaker {
    /// Creates a new circuit breaker that trips after `trip_threshold` consecutive environmental failures.
    pub fn new(trip_threshold: usize) -> Self {
        assert!(trip_threshold > 0, "trip_threshold must be at least 1");
        Self {
            trip_threshold,
            state: Arc::new(Mutex::new(BreakerState::default())),
        }
    }

    pub fn reset(&self) {
        let mut s = self.state.lock().unwrap();
        s.consecutive_failures = 0;
        s.last_error_type = None;
    }

    /// Extract exit code from tool output if present (e.g. "exit: 0" or "exit: 126").
    pub fn parse_exit_code(output: &str) -> Option<i32> {
        for line in output.lines() {
            if let Some(rest) = line.strip_prefix("exit:") {
                if let Ok(code) = rest.trim().parse::<i32>() {
                    return Some(code);
                }
            }
        }
        None
    }

    /// Classifies an error from a tool execution into an [`EnvironmentErrorType`].
    pub fn classify_error(output: &str, exit_code: Option<i32>) -> Option<EnvironmentErrorType> {
        let lower = output.to_lowercase();
        // Sandbox confinement is checked first — most specific to Keel
        if lower.contains("landlock") || lower.contains("sandbox") || lower.contains("seccomp") {
            return Some(EnvironmentErrorType::SandboxConfinement);
        }

        // Permission denied: exit code 126 or matching output strings
        if exit_code == Some(126)
            || output.contains("Permission denied")
            || output.contains("Operation not permitted")
            || output.contains("Read-only file system")
        {
            return Some(EnvironmentErrorType::PermissionDenied);
        }

        // Command not found: exit code 127 or matching output strings
        if exit_code == Some(127)
            || lower.contains("command not found")
            || output.contains("No such file or directory")
        {
            return Some(EnvironmentErrorType::CommandNotFound);
        }

        if let Some(code) = exit_code {
            if code != 0 {
                return Some(EnvironmentErrorType::Other(format!("exit code {code}")));
            }
        }

        None
    }

    /// Observes the result of a tool execution and returns a [`CircuitBreakerTrip`] if tripped.
    pub fn observe(&self, tool_name: &str, output: &str) -> Option<CircuitBreakerTrip> {
        let exit_code = Self::parse_exit_code(output);
        if exit_code == Some(0) {
            self.reset();
            return None;
        }

        let error_type = Self::classify_error(output, exit_code)?;

        let mut s = self.state.lock().unwrap();
        s.consecutive_failures += 1;
        s.last_error_type = Some(error_type.clone());
        let count = s.consecutive_failures;

        if count < self.trip_threshold {
            return None;
        }

        let first_line = output
            .lines()
            .find(|l| !l.starts_with("exit:"))
            .unwrap_or(output);
        let diagnostic_message = format!(
            "Tool '{tool_name}' failed {count} time(s) consecutively with {error_type}: {first_line}"
        );

        let remedy_suggestion = match &error_type {
            EnvironmentErrorType::PermissionDenied => {
                "Check file and directory permissions. Ensure developer toolchains are accessible. If Landlock sandbox is enabled, consider passing `--sandbox-allow-read <PATH>` or `--no-sandbox`."
                    .to_string()
            }
            EnvironmentErrorType::CommandNotFound => {
                "The command or executable is not installed or not in PATH. Check PATH or install the missing toolchain."
                    .to_string()
            }
            EnvironmentErrorType::SandboxConfinement => {
                "Linux Landlock filesystem confinement is blocking access. Configure `--sandbox-allow-read <PATH>` or test with `--no-sandbox`."
                    .to_string()
            }
            EnvironmentErrorType::Other(msg) => {
                format!("Environmental failure ({msg}). Verify prerequisites or workspace configuration.")
            }
        };

        Some(CircuitBreakerTrip {
            consecutive_failures: count,
            error_type,
            diagnostic_message,
            remedy_suggestion,
        })
    }

    /// Formats an advisory warning for injection into the tool output.
    pub fn format_advisory(&self, trip: &CircuitBreakerTrip) -> String {
        format!(
            "\n\n[environment-circuit-breaker] CRITICAL: Environment failure detected {} time(s) in a row.\n- Error type: {}\n- Diagnostic: {}\n- Remedy: {}\nDo not loop on identical commands. Adjust the command or notify the user if external setup is required.",
            trip.consecutive_failures,
            trip.error_type,
            trip.diagnostic_message,
            trip.remedy_suggestion,
        )
    }

    pub fn consecutive_failures(&self) -> usize {
        self.state.lock().unwrap().consecutive_failures
    }

    pub fn is_tripped(&self) -> bool {
        self.state.lock().unwrap().consecutive_failures >= self.trip_threshold
    }
}

/// Plugin that mounts [`EnvironmentCircuitBreaker`] into the kernel context.
pub struct EnvironmentCircuitBreakerPlugin {
    pub trip_threshold: usize,
}

impl Default for EnvironmentCircuitBreakerPlugin {
    fn default() -> Self {
        Self { trip_threshold: 3 }
    }
}

impl Plugin for EnvironmentCircuitBreakerPlugin {
    fn name(&self) -> &str {
        "environment-circuit-breaker"
    }

    fn apply(&self, scope: &mut Scope) {
        let breaker = EnvironmentCircuitBreaker::new(self.trip_threshold);
        let b_reset = breaker.clone();
        scope.on::<crate::TurnStart, _>(move |_| {
            b_reset.reset();
        });
        scope.provide(breaker);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn canonicalizes_object_keys_deterministically() {
        let a = json!({"b": 1, "a": {"y": 2, "x": 3}});
        let b = json!({"a": {"x": 3, "y": 2}, "b": 1});
        assert_eq!(canonicalize_json(&a), canonicalize_json(&b));
    }

    #[test]
    fn escalates_gentle_then_detailed_and_resets() {
        let g = RepeatToolGuard::new(vec![2, 4]);
        let c = ToolCall {
            id: "1".into(),
            name: "read_file".into(),
            args: json!({"path": "foo.py"}),
        };
        assert!(g.observe(&c).is_none()); // count = 1
        let r2 = g.observe(&c).unwrap(); // count = 2 -> gentle
        assert!(r2.contains("You are repeating the exact same tool call"));
        assert!(g.observe(&c).is_none()); // count = 3
        let r4 = g.observe(&c).unwrap(); // count = 4 -> detailed
        assert!(r4.contains("consecutive_calls: 4"));
        g.reset();
        assert!(g.observe(&c).is_none()); // reset back to 1
    }

    #[test]
    fn permission_denied_detection() {
        let breaker = EnvironmentCircuitBreaker::new(1);
        let trip = breaker
            .observe("shell", "exit: 126\nstderr:\nPermission denied")
            .unwrap();
        assert_eq!(trip.error_type, EnvironmentErrorType::PermissionDenied);

        let trip2 = breaker
            .observe("shell", "exit: 1\nstderr:\nOperation not permitted")
            .unwrap();
        assert_eq!(trip2.error_type, EnvironmentErrorType::PermissionDenied);
    }

    #[test]
    fn command_not_found_detection() {
        let breaker = EnvironmentCircuitBreaker::new(1);
        let trip = breaker
            .observe("shell", "exit: 127\nstderr:\nsh: cargo: command not found")
            .unwrap();
        assert_eq!(trip.error_type, EnvironmentErrorType::CommandNotFound);
    }

    #[test]
    fn sandbox_confinement_priority() {
        let breaker = EnvironmentCircuitBreaker::new(1);
        let trip = breaker
            .observe(
                "shell",
                "exit: 1\nstderr:\nlandlock: Permission denied by kernel sandbox",
            )
            .unwrap();
        assert_eq!(trip.error_type, EnvironmentErrorType::SandboxConfinement);
    }

    #[test]
    fn success_resets_counter() {
        let breaker = EnvironmentCircuitBreaker::new(3);
        breaker.observe("shell", "exit: 127\nstderr:\ncommand not found");
        breaker.observe("shell", "exit: 127\nstderr:\ncommand not found");
        assert_eq!(breaker.consecutive_failures(), 2);
        assert!(!breaker.is_tripped());

        // Success resets
        breaker.observe("shell", "exit: 0\nstdout:\nall tests passed");
        assert_eq!(breaker.consecutive_failures(), 0);
        assert!(!breaker.is_tripped());
    }

    #[test]
    fn trip_threshold_is_respected() {
        let breaker = EnvironmentCircuitBreaker::new(3);
        assert!(breaker
            .observe("shell", "exit: 126\nstderr:\nPermission denied")
            .is_none());
        assert!(breaker
            .observe("shell", "exit: 126\nstderr:\nPermission denied")
            .is_none());
        assert_eq!(breaker.consecutive_failures(), 2);
        assert!(!breaker.is_tripped());

        let trip = breaker
            .observe("shell", "exit: 126\nstderr:\nPermission denied")
            .expect("should trip at 3");
        assert_eq!(trip.consecutive_failures, 3);
        assert!(breaker.is_tripped());
    }

    #[test]
    fn format_advisory_contains_diagnostic_and_remedy() {
        let breaker = EnvironmentCircuitBreaker::new(1);
        let trip = breaker
            .observe("shell", "exit: 126\nstderr:\nPermission denied")
            .unwrap();
        let adv = breaker.format_advisory(&trip);
        assert!(adv.contains("[environment-circuit-breaker] CRITICAL"));
        assert!(adv.contains("Permission denied"));
        assert!(adv.contains("Remedy:"));
    }

    #[test]
    fn detailed_reminder_utf8_char_boundary_safety() {
        // Construct canonical args where byte 500 would land inside a 4-byte emoji or multi-byte char
        let mut args = "a".repeat(499);
        args.push_str("🦀 Finnish: äö åäö ääääääääää");
        let reminder = detailed_reminder("test_tool", 4, &args);
        assert!(reminder.contains("[repeat-tool-reminder] Repeated tool call detected"));
        assert!(reminder.contains("test_tool"));
        assert!(reminder.contains("… (+"));
    }
}
