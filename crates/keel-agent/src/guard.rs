//! Guard plugins (`keel-agent::guard`), modeled on DeepSeek Harness's `@deepseek-ai/dsh-repeat-tool-reminder`.
//!
//! Detects when an agent repeats the exact same tool call with identical canonical JSON arguments
//! across consecutive invocations (at configurable thresholds, default `[2, 4, 7]`) and appends
//! an escalating advisory reminder to the tool output so smaller models break out of stuck loops.

use crate::ToolCall;
use keel_kernel::{Plugin, Scope};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

const GENTLE_REMINDER: &str = "\n\n[repeat-tool-reminder] You are repeating the exact same tool call with identical arguments. Carefully analyze the previous result before calling again: if the task is not complete, try a different approach or different arguments instead of repeating the call.";

fn detailed_reminder(tool_name: &str, count: usize, canonical_args: &str) -> String {
    let preview = if canonical_args.len() > 500 {
        format!(
            "{}… (+{} more chars)",
            &canonical_args[..500],
            canonical_args.len() - 500
        )
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
}
