//! OpenAI-compatible chat-completions provider. Works with OpenAI, DeepSeek,
//! OmniRoute, llama.cpp server, OpenRouter and anything else speaking the same API.

use crate::{Chunk, Llm, Message, ToolCall, ToolSpec};
use anyhow::{Context as _, Result};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};

pub struct OpenAiCompat {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
}

impl OpenAiCompat {
    pub fn deepseek(api_key: String) -> Self {
        Self {
            base_url: "https://api.deepseek.com".into(),
            api_key: Some(api_key),
            model: "deepseek-chat".into(),
        }
    }
}

pub fn messages_to_wire(messages: &[Message]) -> Vec<Value> {
    messages
        .iter()
        .map(|m| match m {
            Message::System { text } => json!({"role":"system","content":text}),
            Message::User { text } => json!({"role":"user","content":text}),
            Message::Assistant { text, tool_calls } => {
                let mut v = json!({"role":"assistant","content":text});
                if !tool_calls.is_empty() {
                    v["tool_calls"] = tool_calls
                        .iter()
                        .map(|c| json!({"id":c.id,"type":"function","function":{"name":c.name,"arguments":c.args.to_string()}}))
                        .collect();
                }
                v
            }
            Message::Tool { id, content } => json!({"role":"tool","tool_call_id":id,"content":content}),
        })
        .collect()
}

pub fn parse_reply(body: &Value) -> Result<Message> {
    let msg = &body["choices"][0]["message"];
    anyhow::ensure!(msg.is_object(), "malformed response: {body}");
    let text = msg["content"].as_str().unwrap_or("").to_string();
    let mut tool_calls = Vec::new();
    for c in msg["tool_calls"].as_array().into_iter().flatten() {
        let raw = c["function"]["arguments"].as_str().unwrap_or("{}");
        tool_calls.push(ToolCall {
            id: c["id"].as_str().unwrap_or_default().to_string(),
            name: c["function"]["name"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
            args: serde_json::from_str(raw).unwrap_or(Value::Null),
        });
    }
    Ok(Message::Assistant { text, tool_calls })
}

/// Accumulates an OpenAI-style SSE stream into a final [`Message`].
#[derive(Default)]
pub struct StreamAcc {
    text: String,
    calls: Vec<(String, String, String)>, // (id, name, arguments-json-so-far)
}

impl StreamAcc {
    /// Feed one SSE line; returns chunks to show live.
    pub fn feed(&mut self, line: &str) -> Result<Vec<Chunk>> {
        let Some(data) = line.strip_prefix("data:") else {
            return Ok(vec![]);
        };
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            return Ok(vec![]);
        }
        let v: Value = serde_json::from_str(data).context("bad SSE json")?;
        let delta = &v["choices"][0]["delta"];
        let mut out = Vec::new();
        for key in ["reasoning_content", "reasoning"] {
            if let Some(t) = delta[key].as_str().filter(|t| !t.is_empty()) {
                out.push(Chunk::Reasoning(t.into()));
            }
        }
        if let Some(t) = delta["content"].as_str().filter(|t| !t.is_empty()) {
            self.text.push_str(t);
            out.push(Chunk::Text(t.into()));
        }
        for c in delta["tool_calls"].as_array().into_iter().flatten() {
            let i = c["index"].as_u64().unwrap_or(0) as usize;
            while self.calls.len() <= i {
                self.calls.push(Default::default());
            }
            let slot = &mut self.calls[i];
            if let Some(id) = c["id"].as_str() {
                slot.0 = id.into();
            }
            if let Some(n) = c["function"]["name"].as_str() {
                slot.1.push_str(n);
            }
            if let Some(a) = c["function"]["arguments"].as_str() {
                slot.2.push_str(a);
            }
        }
        Ok(out)
    }

    pub fn finish(self) -> Message {
        let tool_calls = self
            .calls
            .into_iter()
            .filter(|c| !c.1.is_empty())
            .map(|(id, name, args)| ToolCall {
                id,
                name,
                args: serde_json::from_str(if args.is_empty() { "{}" } else { &args })
                    .unwrap_or(Value::Null),
            })
            .collect();
        Message::Assistant {
            text: self.text,
            tool_calls,
        }
    }
}

impl OpenAiCompat {
    fn request(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        stream: bool,
    ) -> Result<ureq::Response> {
        let mut body =
            json!({"model": self.model, "messages": messages_to_wire(messages), "stream": stream});
        if !tools.is_empty() {
            body["tools"] = tools
                .iter()
                .map(|t| json!({"type":"function","function":{"name":t.name,"description":t.description,"parameters":t.parameters}}))
                .collect();
        }
        // Generous read timeout: local 27B models generate at ~5 tok/s, but a dead server must still fail.
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(std::time::Duration::from_secs(10))
            .timeout_read(std::time::Duration::from_secs(600))
            .build();
        let mut req = agent.post(&format!(
            "{}/chat/completions",
            self.base_url.trim_end_matches('/')
        ));
        if let Some(k) = &self.api_key {
            req = req.set("Authorization", &format!("Bearer {k}"));
        }
        req.send_json(body).context("provider request failed")
    }
}

impl Llm for OpenAiCompat {
    fn complete(&self, messages: &[Message], tools: &[ToolSpec]) -> Result<Message> {
        let resp: Value = self.request(messages, tools, false)?.into_json()?;
        parse_reply(&resp)
    }

    fn complete_streaming(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        sink: &mut dyn FnMut(Chunk),
    ) -> Result<Message> {
        let resp = self.request(messages, tools, true)?;
        let mut acc = StreamAcc::default();
        for line in BufReader::new(resp.into_reader()).lines() {
            for c in acc.feed(&line?)? {
                sink(c);
            }
        }
        Ok(acc.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tool_calls() {
        let body = json!({"choices":[{"message":{"content":null,"tool_calls":[
            {"id":"c1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.rs\"}"}}]}}]});
        let Message::Assistant { tool_calls, .. } = parse_reply(&body).unwrap() else {
            panic!()
        };
        assert_eq!(tool_calls[0].name, "read_file");
        assert_eq!(tool_calls[0].args["path"], "a.rs");
    }

    #[test]
    fn stream_accumulates_text_reasoning_and_split_tool_args() {
        let lines = [
            r#"data: {"choices":[{"delta":{"reasoning_content":"hmm"}}]}"#,
            r#"data: {"choices":[{"delta":{"content":"Hel"}}]}"#,
            r#"data: {"choices":[{"delta":{"content":"lo"}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"read_file","arguments":"{\"pa"}}]}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"a.rs\"}"}}]}}]}"#,
            "data: [DONE]",
        ];
        let mut acc = StreamAcc::default();
        let mut seen = vec![];
        for l in lines {
            seen.extend(acc.feed(l).unwrap());
        }
        assert_eq!(seen[0], Chunk::Reasoning("hmm".into()));
        let Message::Assistant { text, tool_calls } = acc.finish() else {
            panic!()
        };
        assert_eq!(text, "Hello");
        assert_eq!(tool_calls[0].args["path"], "a.rs");
    }

    #[test]
    fn wire_roundtrip_shapes() {
        let w = messages_to_wire(&[Message::Tool {
            id: "c1".into(),
            content: "x".into(),
        }]);
        assert_eq!(w[0]["tool_call_id"], "c1");
    }
}
