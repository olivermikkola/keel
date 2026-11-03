//! Adaptive Local-to-Cloud Model Cascade (`CascadeProvider`).
//!
//! Routes routine inspection and straightforward coding steps to a fast, zero-cost
//! local edge model (`fast`, e.g., `bonsai-9b` or `qwen-27b` on `127.0.0.1:8080`),
//! and automatically escalates to a frontier reasoning model (`frontier`, e.g.,
//! `deepseek-chat`) when:
//! 1. The local edge endpoint is unreachable or returns an error,
//! 2. The recent conversation history contains repeated tool errors, verification gate
//!    failures (`[verification gate failed]`), repeat-tool loop warnings (`[system reminder:`),
//!    or unresolved compiler diagnostics (`[diagnostics:`), or
//! 3. The local model returns a degenerate/empty response with neither text nor tool calls.

use crate::{Chunk, Llm, Message, ToolSpec};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct CascadeSnapshot {
    pub fast_calls: usize,
    pub frontier_calls: usize,
    pub last_escalation_reason: Option<String>,
}

pub struct CascadeProvider {
    pub fast: Arc<dyn Llm>,
    pub frontier: Arc<dyn Llm>,
    /// Number of recent failure signals in the tail of `messages` that triggers pre-emptive escalation.
    pub failure_threshold: usize,
    fast_calls: AtomicUsize,
    frontier_calls: AtomicUsize,
    last_reason: Mutex<Option<String>>,
}

impl CascadeProvider {
    pub fn new(fast: Arc<dyn Llm>, frontier: Arc<dyn Llm>, failure_threshold: usize) -> Self {
        Self {
            fast,
            frontier,
            failure_threshold: failure_threshold.max(1),
            fast_calls: AtomicUsize::new(0),
            frontier_calls: AtomicUsize::new(0),
            last_reason: Mutex::new(None),
        }
    }

    pub fn snapshot(&self) -> CascadeSnapshot {
        CascadeSnapshot {
            fast_calls: self.fast_calls.load(Ordering::Relaxed),
            frontier_calls: self.frontier_calls.load(Ordering::Relaxed),
            last_escalation_reason: self.last_reason.lock().ok().and_then(|g| g.clone()),
        }
    }

    fn record_escalation(&self, reason: impl Into<String>) {
        self.frontier_calls.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut g) = self.last_reason.lock() {
            *g = Some(reason.into());
        }
    }

    /// Inspect recent history for signs that the fast model is struggling or stuck.
    pub fn detect_struggle(&self, messages: &[Message]) -> Option<String> {
        let mut failure_signals = 0usize;
        for msg in messages.iter().rev().take(6) {
            match msg {
                Message::User { text } => {
                    if text.contains("[verification gate failed]") {
                        return Some("verification gate failure in recent turn".into());
                    }
                }
                Message::Tool { content, .. } => {
                    if content.contains("[system reminder: you have called") {
                        return Some("repeat-tool loop detected".into());
                    }
                    if content.starts_with("error:") || content.contains("[diagnostics:") {
                        failure_signals += 1;
                    }
                }
                _ => {}
            }
        }
        if failure_signals >= self.failure_threshold {
            Some(format!(
                "{failure_signals} recent tool/diagnostic errors (>= threshold {})",
                self.failure_threshold
            ))
        } else {
            None
        }
    }

    fn is_degenerate_reply(msg: &Message) -> bool {
        match msg {
            Message::Assistant { text, tool_calls } => {
                text.trim().is_empty() && tool_calls.is_empty()
            }
            _ => true,
        }
    }
}

impl Llm for CascadeProvider {
    fn complete(&self, messages: &[Message], tools: &[ToolSpec]) -> Result<Message> {
        if let Some(reason) = self.detect_struggle(messages) {
            self.record_escalation(reason);
            return self.frontier.complete(messages, tools);
        }

        match self.fast.complete(messages, tools) {
            Ok(reply) if !Self::is_degenerate_reply(&reply) => {
                self.fast_calls.fetch_add(1, Ordering::Relaxed);
                Ok(reply)
            }
            Ok(_) => {
                self.record_escalation("fast model returned empty/degenerate reply");
                self.frontier.complete(messages, tools)
            }
            Err(e) => {
                self.record_escalation(format!("fast model transport error: {e}"));
                self.frontier.complete(messages, tools)
            }
        }
    }

    fn complete_streaming(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        sink: &mut dyn FnMut(Chunk),
    ) -> Result<Message> {
        if let Some(reason) = self.detect_struggle(messages) {
            self.record_escalation(reason);
            return self.frontier.complete_streaming(messages, tools, sink);
        }

        let mut buffered_chunks = Vec::new();
        let fast_res = self
            .fast
            .complete_streaming(messages, tools, &mut |c| buffered_chunks.push(c));

        match fast_res {
            Ok(reply) if !Self::is_degenerate_reply(&reply) => {
                self.fast_calls.fetch_add(1, Ordering::Relaxed);
                for c in buffered_chunks {
                    sink(c);
                }
                Ok(reply)
            }
            Ok(_) => {
                self.record_escalation("fast model returned empty/degenerate streaming reply");
                self.frontier.complete_streaming(messages, tools, sink)
            }
            Err(e) => {
                self.record_escalation(format!("fast model streaming error: {e}"));
                self.frontier.complete_streaming(messages, tools, sink)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::bail;

    struct HealthyFast;
    impl Llm for HealthyFast {
        fn complete(&self, _: &[Message], _: &[ToolSpec]) -> Result<Message> {
            Ok(Message::Assistant {
                text: "handled by fast model".into(),
                tool_calls: vec![],
            })
        }
    }

    struct FailingFast;
    impl Llm for FailingFast {
        fn complete(&self, _: &[Message], _: &[ToolSpec]) -> Result<Message> {
            bail!("connection refused on 127.0.0.1:8080")
        }
    }

    struct FrontierMock;
    impl Llm for FrontierMock {
        fn complete(&self, _: &[Message], _: &[ToolSpec]) -> Result<Message> {
            Ok(Message::Assistant {
                text: "rescued by frontier model".into(),
                tool_calls: vec![],
            })
        }
    }

    #[test]
    fn cascade_uses_fast_when_healthy_and_escalates_on_struggle_or_error() {
        let cascade = CascadeProvider::new(Arc::new(HealthyFast), Arc::new(FrontierMock), 2);
        let history = vec![Message::User {
            text: "simple question".into(),
        }];
        let reply = cascade.complete(&history, &[]).unwrap();
        assert!(
            matches!(reply, Message::Assistant { ref text, .. } if text == "handled by fast model")
        );
        assert_eq!(cascade.snapshot().fast_calls, 1);
        assert_eq!(cascade.snapshot().frontier_calls, 0);

        // Inject verification failure -> must escalate to frontier
        let struggling_history = vec![Message::User {
            text: "[verification gate failed]\ncargo test failed".into(),
        }];
        let reply2 = cascade.complete(&struggling_history, &[]).unwrap();
        assert!(
            matches!(reply2, Message::Assistant { ref text, .. } if text == "rescued by frontier model")
        );
        let snap = cascade.snapshot();
        assert_eq!(snap.fast_calls, 1);
        assert_eq!(snap.frontier_calls, 1);
        assert!(snap
            .last_escalation_reason
            .unwrap()
            .contains("verification gate failure"));

        // Transport error on fast model -> must transparently fall back to frontier
        let err_cascade = CascadeProvider::new(Arc::new(FailingFast), Arc::new(FrontierMock), 2);
        let reply3 = err_cascade.complete(&history, &[]).unwrap();
        assert!(
            matches!(reply3, Message::Assistant { ref text, .. } if text == "rescued by frontier model")
        );
        assert_eq!(err_cascade.snapshot().frontier_calls, 1);
    }
}
