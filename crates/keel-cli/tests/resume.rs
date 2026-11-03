use std::process::Command;

#[test]
fn invalid_resume_fails_before_creating_a_log_or_contacting_the_provider() {
    let root = std::env::temp_dir().join(format!("keel-resume-cli-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    for (name, content) in [
        ("missing", None),
        ("empty", Some("")),
        ("corrupt", Some("{not-json\n")),
        ("broken-message", Some("{\"kind\":\"user\",\"data\":{}}\n")),
    ] {
        let source = root.join(format!("{name}.jsonl"));
        let destination = root.join(format!("{name}-output.jsonl"));
        if let Some(content) = content {
            std::fs::write(&source, content).unwrap();
        }
        let output = Command::new(env!("CARGO_BIN_EXE_keel"))
            .args(["run", "--json", "--root"])
            .arg(&root)
            .arg("--resume")
            .arg(&source)
            .arg("--log")
            .arg(&destination)
            .args([
                "--base-url",
                "http://127.0.0.1:9/v1",
                "--api-key",
                "test-only-placeholder",
                "continue",
            ])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{name}");
        let stdout = String::from_utf8(output.stdout).unwrap();
        let events: Vec<serde_json::Value> = stdout
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(events.len(), 1, "{name}: {stdout}");
        assert_eq!(events[0]["type"], "done");
        assert_eq!(events[0]["status"], "error");
        assert!(events[0]["error"]
            .as_str()
            .unwrap()
            .contains("Cannot resume session"));
        assert!(
            !destination.exists(),
            "plugins wrote a new log after failed resume"
        );
        match content {
            Some(content) => assert_eq!(std::fs::read_to_string(&source).unwrap(), content),
            None => assert!(!source.exists(), "missing source was created"),
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}
