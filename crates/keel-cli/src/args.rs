//! Strict validation for the agent-path command line.
//!
//! The agent entrypoint parses flags by repeatedly *removing* known ones from
//! the argument vector and treating whatever is left as the prompt. That is
//! convenient but unforgiving: a typo such as `--yoloo` or a dangling `--model`
//! would silently become part of the prompt and be sent to a (possibly paid)
//! model. This module runs after all known flags were removed and rejects
//! anything flag-shaped that remains, with a "did you mean" hint.

/// Flags that take no value on the agent path.
pub const BOOL_FLAGS: &[&str] = &[
    "--interactive",
    "--yolo",
    "--read-only",
    "--shadow",
    "--no-sandbox",
    "--no-toolchains",
    "--dump-config",
    "--json",
    "--auto-rollback",
    "--apply",
    "--help",
    "--version",
    "--allow-protected",
    "--diff",
];

/// Flags that consume the following argument as their value.
pub const VALUE_OPTS: &[&str] = &[
    "--root",
    "--safety-mode",
    "--sandbox-allow-read",
    "--sandbox-allow-write",
    "--max-context-tokens",
    "--compaction-keep-turns",
    "--mcp-server",
    "--log",
    "--resume",
    "--verify",
    "--model",
    "--base-url",
    "--api-key",
    "--runs",
    "--candidates",
    "--cascade",
    "--handoff",
    "--sentinel",
    "--listen",
    "--socket",
    "--auth-token",
    "--role",
    "--steps",
];

/// Top-level subcommands (used to catch single-word typos of them).
pub const SUBCOMMANDS: &[&str] = &[
    "run",
    "repl",
    "patch",
    "edit",
    "checkpoint",
    "undo",
    "mcp",
    "lsp",
    "vfs",
    "speculate",
    "daemon",
    "memory",
    "handoff",
    "oracle",
    "symbols",
    "tree",
    "sentinel",
    "replay",
    "repomap",
    "subagent",
    "bench",
];

/// Removes and returns everything after the first literal `--`, which lets
/// users pass prompts that start with `-` or mention flags verbatim.
pub fn split_literal_tail(args: &mut Vec<String>) -> Vec<String> {
    match args.iter().position(|a| a == "--") {
        Some(i) => {
            let tail = args.split_off(i + 1);
            args.pop();
            tail
        }
        None => Vec::new(),
    }
}

/// Classic two-row Levenshtein distance over `char`s.
fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != *cb);
            cur.push(sub.min(prev[j + 1] + 1).min(cur[j] + 1));
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Closest candidate within `max_dist` edits, if any.
pub fn suggest<'a>(word: &str, candidates: &[&'a str], max_dist: usize) -> Option<&'a str> {
    candidates
        .iter()
        .map(|c| (edit_distance(word, c), *c))
        .filter(|(d, _)| *d <= max_dist)
        .min_by_key(|(d, _)| *d)
        .map(|(_, c)| c)
}

/// Validates the arguments that remain after every known flag was consumed.
///
/// Returns a user-facing error message for the first offending argument.
pub fn validate_remaining(args: &[String]) -> Result<(), String> {
    for a in args {
        if !a.starts_with('-') || a == "-" {
            continue;
        }
        // A bare negative number is plausible prompt text ("-1 is wrong").
        if a.parse::<f64>().is_ok() {
            continue;
        }
        if VALUE_OPTS.contains(&a.as_str()) {
            return Err(format!(
                "option {a} requires a value (or was given more than once)"
            ));
        }
        let all: Vec<&str> = BOOL_FLAGS.iter().chain(VALUE_OPTS).copied().collect();
        let mut msg = format!("unknown option {a}");
        if let Some(s) = suggest(a, &all, 2) {
            msg.push_str(&format!(" (did you mean {s}?)"));
        }
        msg.push_str("\nuse `--` before the prompt to pass text that starts with '-'");
        return Err(msg);
    }
    Ok(())
}

/// Rejects a lone word that is one typo away from a subcommand (`keel pach`),
/// which would otherwise be sent to the model as a one-word prompt.
pub fn check_subcommand_typo(args: &[String]) -> Result<(), String> {
    let [word] = args else { return Ok(()) };
    if word.contains(char::is_whitespace) || SUBCOMMANDS.contains(&word.as_str()) {
        return Ok(());
    }
    match suggest(word, SUBCOMMANDS, 1) {
        Some(s) if word.len() >= 4 => Err(format!(
            "unknown command '{word}' (did you mean `keel {s}`?)\nuse `--` before the prompt to send it to the model verbatim"
        )),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn plain_prompt_is_accepted() {
        assert!(validate_remaining(&v(&["fix", "the", "bug"])).is_ok());
        assert!(validate_remaining(&v(&["-5", "is", "wrong"])).is_ok());
        assert!(validate_remaining(&v(&["-"])).is_ok());
    }

    #[test]
    fn unknown_flag_gets_suggestion() {
        let e = validate_remaining(&v(&["--yoloo", "fix"])).unwrap_err();
        assert!(e.contains("unknown option --yoloo"), "{e}");
        assert!(e.contains("did you mean --yolo?"), "{e}");
    }

    #[test]
    fn dangling_value_option_is_reported() {
        let e = validate_remaining(&v(&["--model"])).unwrap_err();
        assert!(e.contains("--model requires a value"), "{e}");
    }

    #[test]
    fn far_off_flag_has_no_suggestion() {
        let e = validate_remaining(&v(&["--zzzzzzzzzz"])).unwrap_err();
        assert!(!e.contains("did you mean"), "{e}");
    }

    #[test]
    fn literal_tail_is_split_off() {
        let mut a = v(&["--json", "--", "--yolo", "is", "a", "flag"]);
        let tail = split_literal_tail(&mut a);
        assert_eq!(a, v(&["--json"]));
        assert_eq!(tail, v(&["--yolo", "is", "a", "flag"]));

        let mut none = v(&["hello"]);
        assert!(split_literal_tail(&mut none).is_empty());
        assert_eq!(none, v(&["hello"]));
    }

    #[test]
    fn subcommand_typo_is_caught_but_prompts_pass() {
        let e = check_subcommand_typo(&v(&["pach"])).unwrap_err();
        assert!(e.contains("did you mean `keel patch`?"), "{e}");
        assert!(check_subcommand_typo(&v(&["patch"])).is_ok());
        assert!(check_subcommand_typo(&v(&["explain this repo"])).is_ok());
        assert!(check_subcommand_typo(&v(&["hello"])).is_ok());
        assert!(check_subcommand_typo(&v(&["fix", "pach"])).is_ok());
    }

    #[test]
    fn edit_distance_basics() {
        assert_eq!(edit_distance("", ""), 0);
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("yolo", "yoloo"), 1);
    }

    #[test]
    fn every_flag_is_dash_dash_and_unique() {
        let mut all: Vec<&str> = BOOL_FLAGS.iter().chain(VALUE_OPTS).copied().collect();
        assert!(all.iter().all(|f| f.starts_with("--")));
        let n = all.len();
        all.sort_unstable();
        all.dedup();
        assert_eq!(n, all.len());
    }
}
