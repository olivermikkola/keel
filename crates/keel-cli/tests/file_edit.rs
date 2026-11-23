//! Prove `--json` emits a dedicated `file_edit` NDJSON event when file tools
//! apply under `--shadow` (Desktop Diff projection must not invent Diff from
//! `tool_call` alone).

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

#[test]
fn json_shadow_write_emits_file_edit_with_unified_diff() {
    let root = std::env::temp_dir().join(format!("keel-file-edit-wire-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("src_main.rs"), "fn main() {}\n").unwrap();

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
                json!({"tool_calls":[{"index":0,"id":"call-edit","function":{"name":"edit_file","arguments":"{\"path\":\"src_main.rs\",\"old\":\"fn main() {}\",\"new\":\"fn main() { println!(\\\"hi\\\"); }\"}"}}]})
            } else {
                json!({"content":"Edited under shadow."})
            };
            let body = format!(
                "data: {}\n\ndata: [DONE]\n\n",
                json!({"choices":[{"delta":delta}]})
            );
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
            stream.flush().unwrap();
        }
    });

    let child = Command::new(env!("CARGO_BIN_EXE_keel"))
        .args(["run", "--json", "--shadow", "--yolo", "--root"])
        .arg(&root)
        .arg("--log")
        .arg(root.join("session.jsonl"))
        .args([
            "--model",
            "file-edit-test",
            "--api-key",
            "test-only-placeholder",
            "--base-url",
            &format!("http://{address}/v1"),
            "Edit src_main.rs under shadow",
        ])
        .env("KEEL_MAX_CONTEXT_TOKENS", "16384")
        .stdin(Stdio::null())
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
            .recv_timeout(Duration::from_secs(20))
            .expect("turn did not finish");
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
    assert!(status.success(), "{stderr}\nevents={events:?}");

    // Disk must stay untouched under --shadow
    assert_eq!(
        std::fs::read_to_string(root.join("src_main.rs")).unwrap(),
        "fn main() {}\n"
    );

    let file_edits: Vec<_> = events.iter().filter(|e| e["type"] == "file_edit").collect();
    assert_eq!(
        file_edits.len(),
        1,
        "expected one file_edit event, got {events:?}"
    );
    let edit = file_edits[0];
    assert_eq!(edit["path"], "src_main.rs");
    assert_eq!(edit["tool_call_id"], "call-edit");
    let diff = edit["diff"].as_str().expect("diff string");
    assert!(
        diff.contains("--- a/src_main.rs") && diff.contains("+++ b/src_main.rs"),
        "unified diff headers missing: {diff}"
    );
    assert!(
        diff.contains("-fn main() {}") && diff.contains("+fn main() { println!(\"hi\"); }"),
        "unified diff body missing: {diff}"
    );
    // No invented parallel "patch" event
    assert!(events.iter().all(|e| e["type"] != "patch"));

    server.join().unwrap();
    let _ = std::fs::remove_dir_all(root);
}
