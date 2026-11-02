//! Model Context Protocol (MCP) client plugin and dynamic tool bridge.
//!
//! Connects Keel to external MCP servers over stdio JSON-RPC 2.0.
//! Supports:
//! - Loading `.keel/mcp.json` configuration
//! - Protocol handshake (`initialize`, `notifications/initialized`, `tools/list`)
//! - Dynamically registering remote MCP tools into Keel's [`ToolRegistry`]
//! - Invoking tools via `tools/call` with transparent JSON-RPC request-response cycles

use crate::sandbox::ToolRegistry;
use crate::{Tool, ToolSpec};
use anyhow::{bail, Context, Result};
use keel_kernel::{Plugin, Scope};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct McpConfigFile {
    #[serde(rename = "mcpServers", default)]
    pub mcp_servers: HashMap<String, McpServerConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDef {
    pub name: String,
    pub description: Option<String>,
    #[serde(rename = "inputSchema", default)]
    pub input_schema: Value,
}

struct McpProcess {
    _child: Child,
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
    next_id: u64,
}

impl McpProcess {
    fn send_request(&mut self, method: &str, params: Value) -> Result<Value> {
        self.next_id += 1;
        let id = self.next_id;
        let req = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let mut line = serde_json::to_string(&req)?;
        line.push('\n');
        self.stdin
            .write_all(line.as_bytes())
            .with_context(|| format!("writing request to MCP server for `{method}`"))?;
        self.stdin.flush()?;

        loop {
            let mut res_line = String::new();
            if self.reader.read_line(&mut res_line)? == 0 {
                bail!("MCP server terminated stream unexpectedly while awaiting `{method}`");
            }
            let trimmed = res_line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let val: Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(_) => continue, // Skip any non-JSON log lines
            };

            // Ignore notifications received while waiting for response
            if val.get("id").and_then(|v| v.as_u64()) == Some(id) {
                if let Some(err) = val.get("error") {
                    bail!("MCP server returned error for `{method}`: {err}");
                }
                return Ok(val.get("result").cloned().unwrap_or(Value::Null));
            }
        }
    }

    fn send_notification(&mut self, method: &str, params: Value) -> Result<()> {
        let notif = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        let mut line = serde_json::to_string(&notif)?;
        line.push('\n');
        self.stdin.write_all(line.as_bytes())?;
        self.stdin.flush()?;
        Ok(())
    }
}

pub struct McpClient {
    server_name: String,
    process: Mutex<McpProcess>,
    tools: Vec<McpToolDef>,
}

impl McpClient {
    /// Spawn an MCP server process, complete handshake, and list tools.
    pub fn spawn(server_name: &str, config: &McpServerConfig) -> Result<Arc<Self>> {
        let mut cmd = Command::new(&config.command);
        cmd.args(&config.args)
            .envs(&config.env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        let mut child = cmd.spawn().with_context(|| {
            format!("spawning MCP server `{}`: {}", server_name, config.command)
        })?;

        let stdin = child
            .stdin
            .take()
            .with_context(|| format!("capturing stdin for `{server_name}`"))?;
        let stdout = child
            .stdout
            .take()
            .with_context(|| format!("capturing stdout for `{server_name}`"))?;
        let reader = BufReader::new(stdout);

        let mut proc = McpProcess {
            _child: child,
            stdin,
            reader,
            next_id: 0,
        };

        // 1. Initialize
        let init_params = json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "keel",
                "version": "0.0.1"
            }
        });
        proc.send_request("initialize", init_params)
            .with_context(|| format!("MCP initialize handshake failed with `{server_name}`"))?;

        // 2. Initialized notification
        let _ = proc.send_notification("notifications/initialized", json!({}));

        // 3. List tools
        let tools_res = proc
            .send_request("tools/list", json!({}))
            .with_context(|| format!("listing tools for MCP server `{server_name}`"))?;

        let mut tools = Vec::new();
        if let Some(arr) = tools_res.get("tools").and_then(|t| t.as_array()) {
            for item in arr {
                if let Ok(def) = serde_json::from_value::<McpToolDef>(item.clone()) {
                    tools.push(def);
                }
            }
        }

        Ok(Arc::new(Self {
            server_name: server_name.to_string(),
            process: Mutex::new(proc),
            tools,
        }))
    }

    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    pub fn tools(&self) -> &[McpToolDef] {
        &self.tools
    }

    /// Invoke a tool on the remote MCP server.
    pub fn call_tool(&self, name: &str, arguments: &Value) -> Result<String> {
        let mut proc = self.process.lock().unwrap();
        let res = proc.send_request(
            "tools/call",
            json!({
                "name": name,
                "arguments": arguments,
            }),
        )?;

        let is_error = res
            .get("isError")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let mut text_out = String::new();
        if let Some(content) = res.get("content").and_then(|c| c.as_array()) {
            for item in content {
                if let Some(t) = item.get("text").and_then(|v| v.as_str()) {
                    if !text_out.is_empty() {
                        text_out.push('\n');
                    }
                    text_out.push_str(t);
                }
            }
        } else if let Some(t) = res.as_str() {
            text_out = t.to_string();
        } else {
            text_out = res.to_string();
        }

        if is_error {
            bail!("MCP tool `{name}` returned error: {text_out}");
        }

        Ok(text_out)
    }
}

/// Dynamic bridge wrapping a remote MCP tool into a local Keel [`Tool`].
pub struct McpToolBridge {
    pub client: Arc<McpClient>,
    pub mcp_tool_name: String,
    pub exposed_name: String,
    pub description: String,
    pub input_schema: Value,
}

impl Tool for McpToolBridge {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.exposed_name.clone(),
            description: self.description.clone(),
            parameters: self.input_schema.clone(),
        }
    }
    fn mutating(&self) -> bool {
        true
    }
    fn run(&self, a: &Value) -> Result<String> {
        self.client.call_tool(&self.mcp_tool_name, a)
    }
}

/// Plugin that discovers and registers MCP servers and their tools.
pub struct McpPlugin {
    pub root: PathBuf,
    pub extra_servers: HashMap<String, McpServerConfig>,
}

impl McpPlugin {
    pub fn load_config(root: &Path) -> HashMap<String, McpServerConfig> {
        let config_path = root.join(".keel").join("mcp.json");
        if !config_path.exists() {
            return HashMap::new();
        }
        let data = match std::fs::read_to_string(&config_path) {
            Ok(d) => d,
            Err(_) => return HashMap::new(),
        };
        let cfg: McpConfigFile = serde_json::from_str(&data).unwrap_or_default();
        cfg.mcp_servers
    }
}

impl Plugin for McpPlugin {
    fn name(&self) -> &str {
        "mcp-client"
    }

    fn apply(&self, scope: &mut Scope) {
        let mut servers = Self::load_config(&self.root);
        for (k, v) in &self.extra_servers {
            servers.insert(k.clone(), v.clone());
        }

        if servers.is_empty() {
            return;
        }

        let reg = scope
            .context()
            .service::<ToolRegistry>()
            .map(|r| (*r).clone())
            .unwrap_or_default();

        let mut clients: Vec<Arc<McpClient>> = Vec::new();

        for (name, cfg) in servers {
            match McpClient::spawn(&name, &cfg) {
                Ok(client) => {
                    for tool in client.tools() {
                        let exposed = format!("mcp__{}__{}", name, tool.name);
                        let desc = tool
                            .description
                            .clone()
                            .unwrap_or_else(|| format!("MCP tool `{}` from `{name}`", tool.name));
                        let schema = if tool.input_schema.is_object() {
                            tool.input_schema.clone()
                        } else {
                            json!({"type": "object", "properties": {}})
                        };

                        reg.register(Arc::new(McpToolBridge {
                            client: client.clone(),
                            mcp_tool_name: tool.name.clone(),
                            exposed_name: exposed,
                            description: desc,
                            input_schema: schema,
                        }));
                    }
                    clients.push(client);
                }
                Err(e) => {
                    eprintln!("warning: failed to launch MCP server `{name}`: {e:#}");
                }
            }
        }

        scope.provide(reg);
        scope.provide(Arc::new(clients));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcp_handshake_and_tool_call_mock() {
        let tmp = std::env::temp_dir().join(format!("keel-mcp-mock-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        // Create a mock MCP server shell script that responds to JSON-RPC over stdio
        let script = r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *initialize*)
      echo '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","serverInfo":{"name":"test-mcp","version":"0.1"}}}'
      ;;
    *tools/list*)
      echo '{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"ping","description":"test ping tool","inputSchema":{"type":"object","properties":{"msg":{"type":"string"}},"required":["msg"]}}]}}'
      ;;
    *tools/call*)
      echo '{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"pong: hello from mock mcp"}],"isError":false}}'
      ;;
  esac
done
"#;
        let script_path = tmp.join("mock_mcp.sh");
        std::fs::write(&script_path, script).unwrap();
        let _ = Command::new("chmod").arg("+x").arg(&script_path).status();

        let cfg = McpServerConfig {
            command: "sh".into(),
            args: vec![script_path.to_str().unwrap().into()],
            env: HashMap::new(),
        };

        let client = McpClient::spawn("mock", &cfg).unwrap();
        assert_eq!(client.server_name(), "mock");
        assert_eq!(client.tools().len(), 1);
        assert_eq!(client.tools()[0].name, "ping");

        // Test tool call
        let out = client.call_tool("ping", &json!({"msg": "hello"})).unwrap();
        assert_eq!(out, "pong: hello from mock mcp");

        let _ = std::fs::remove_dir_all(tmp);
    }
}
