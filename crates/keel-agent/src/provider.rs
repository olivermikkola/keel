//! OpenAI-compatible chat-completions provider. Works with OpenAI, DeepSeek,
//! OmniRoute, llama.cpp server, OpenRouter and anything else speaking the same API.

use crate::{Llm, Message, ToolCall, ToolSpec};
use anyhow::{Context as _, Result};
use serde_json::{json, Value};

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

impl Llm for OpenAiCompat {
    fn complete(&self, messages: &[Message], tools: &[ToolSpec]) -> Result<Message> {
        let mut body = json!({"model": self.model, "messages": messages_to_wire(messages)});
        if !tools.is_empty() {
            body["tools"] = tools
                .iter()
                .map(|t| json!({"type":"function","function":{"name":t.name,"description":t.description,"parameters":t.parameters}}))
                .collect();
        }
        // Generous timeout: local 27B models generate at ~5 tok/s, but a dead server must still fail.
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
        let resp: Value = req
            .send_json(body)
            .context("provider request failed")?
            .into_json()?;
        parse_reply(&resp)
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
    fn wire_roundtrip_shapes() {
        let w = messages_to_wire(&[Message::Tool {
            id: "c1".into(),
            content: "x".into(),
        }]);
        assert_eq!(w[0]["tool_call_id"], "c1");
    }
}
