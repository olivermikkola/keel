use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn check_decision(allow: bool) {
    let root =
        std::env::temp_dir().join(format!("keel-approval-wire-{}-{allow}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        for round in 0..2 {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(10)))
                .unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
                if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = value.trim().parse::<usize>().unwrap();
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            let delta = if round == 0 {
                json!({"tool_calls":[{"index":0,"id":"call-proof","function":{"name":"shell","arguments":"{\"command\":\"printf approved > approval-marker.txt\"}"}}]})
            } else {
                json!({"content":"Finished."})
            };
            let body = format!(
                "data: {}\n\ndata: [DONE]\n\n",
                json!({"choices":[{"delta":delta}]})
            );
            write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();
            stream.flush().unwrap();
        }
    });
    let child = Command::new(env!("CARGO_BIN_EXE_keel"))
        .args(["run", "--json", "--safety-mode", "standard", "--root"])
        .arg(&root)
        .arg("--log")
        .arg(root.join("session.jsonl"))
        .args([
            "--model",
            "approval-test",
            "--api-key",
            "test-only-placeholder",
            "--base-url",
            &format!("http://{address}/v1"),
            "Run the requested test tool",
        ])
        // The fake server scripts exactly two chat calls; skip the
        // context-window probe so it gets no extra GET requests.
        .env("KEEL_MAX_CONTEXT_TOKENS", "16384")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut child = ChildGuard(child);
    let stdout = child.0.stdout.take().unwrap();
    let stderr = child.0.stderr.take().unwrap();
    let stderr_reader = std::thread::spawn(move || {
        let mut text = String::new();
        BufReader::new(stderr).read_to_string(&mut text).unwrap();
        text
    });
    let (tx, rx) = mpsc::channel();
    let output_reader = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            if tx.send(line.unwrap()).is_err() {
                break;
            }
        }
    });
    let mut events = Vec::<Value>::new();
    loop {
        let line = rx
            .recv_timeout(Duration::from_secs(15))
            .expect("native approval request was not emitted before waiting for input");
        let event: Value = serde_json::from_str(&line).unwrap();
        let requested = event["type"] == "approval_request";
        events.push(event);
        if requested {
            break;
        }
    }
    assert_eq!(events[0]["approval_protocol"], "stdio-line-v1");
    let request = events.last().unwrap();
    assert_eq!(request["tool"], "shell");
    assert_eq!(request["tool_call_id"], "call-proof");
    assert!(events
        .iter()
        .any(|e| e["type"] == "tool_call" && e["id"] == "call-proof"));
    assert!(
        !root.join("approval-marker.txt").exists(),
        "tool executed before parent approval"
    );
    let mut stdin = child.0.stdin.take().unwrap();
    writeln!(stdin, "{}", if allow { "allow" } else { "deny" }).unwrap();
    drop(stdin);
    loop {
        let line = rx
            .recv_timeout(Duration::from_secs(15))
            .expect("turn did not finish after approval response");
        let event: Value = serde_json::from_str(&line).unwrap();
        let done = event["type"] == "done";
        events.push(event);
        if done {
            break;
        }
    }
    let status = child.0.wait().unwrap();
    output_reader.join().unwrap();
    let stderr = stderr_reader.join().unwrap();
    assert!(status.success(), "{stderr}");
    assert_eq!(
        events
            .iter()
            .filter(|e| e["type"] == "approval_request")
            .count(),
        1
    );
    assert_eq!(
        root.join("approval-marker.txt").exists(),
        allow,
        "decision was not enforced"
    );
    server.join().unwrap();
    std::fs::remove_dir_all(root).unwrap();
}
#[test]
fn native_json_approval_blocks_execution_and_honors_allow_and_deny() {
    check_decision(false);
    check_decision(true);
}
