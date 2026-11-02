//! Resilient fuzzy diff, search-replace hunk applicant, and unified diff engine.
//!
//! Small and edge-quantized models (Bonsai 2B-27B, Qwen 27B) frequently output
//! edits with slight indentation drift, trailing whitespace, or minor comment
//! discrepancies. This module provides a 3-tier matcher:
//! 1. Exact substring match
//! 2. Whitespace- & indentation-normalized line block match
//! 3. Levenshtein sliding-window similarity match (with confidence >= 80% and uniqueness margin)
//!
//! It also parses and applies:
//! - Search/Replace blocks (`<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE`)
//! - Unified diff hunks (`@@ ... @@` with `-`, `+`, ` ` lines)

use anyhow::{bail, Context, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum MatchQuality {
    Exact,
    WhitespaceNormalized,
    Fuzzy(f64),
}

impl std::fmt::Display for MatchQuality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exact => write!(f, "exact match"),
            Self::WhitespaceNormalized => write!(f, "whitespace-normalized match"),
            Self::Fuzzy(conf) => write!(f, "fuzzy match ({:.0}% confidence)", conf * 100.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PatchOutcome {
    pub modified_text: String,
    pub hunks_applied: usize,
    pub quality: MatchQuality,
}

/// Compute Levenshtein edit distance between two strings.
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let (m, n) = (a_chars.len(), b_chars.len());
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut dp: Vec<usize> = (0..=n).collect();
    for i in 1..=m {
        let mut prev = dp[0];
        dp[0] = i;
        for j in 1..=n {
            let temp = dp[j];
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            dp[j] = (dp[j] + 1).min(dp[j - 1] + 1).min(prev + cost);
            prev = temp;
        }
    }
    dp[n]
}

/// Line similarity in range [0.0, 1.0].
fn line_similarity(a: &str, b: &str) -> f64 {
    let a_t = a.trim();
    let b_t = b.trim();
    if a_t == b_t {
        return 1.0;
    }
    let max_len = a_t.chars().count().max(b_t.chars().count());
    if max_len == 0 {
        return 1.0;
    }
    let dist = levenshtein(a_t, b_t);
    if dist >= max_len {
        0.0
    } else {
        1.0 - (dist as f64 / max_len as f64)
    }
}

/// Normalize line by stripping trailing `\r` and whitespace.
fn normalize_line(l: &str) -> &str {
    l.trim_end_matches(['\r', ' ', '\t'])
}

/// Replace one occurrence of `old` with `new` using the 3-tier resilient matcher.
pub fn fuzzy_replace(text: &str, old: &str, new: &str) -> Result<(String, MatchQuality)> {
    if old.is_empty() {
        bail!("target pattern `old` must not be empty");
    }

    // Tier 1: Exact substring match
    let exact_count = text.matches(old).count();
    if exact_count == 1 {
        return Ok((text.replacen(old, new, 1), MatchQuality::Exact));
    }
    if exact_count > 1 {
        bail!("`old` matches {exact_count} places in text; include more context lines to disambiguate");
    }

    // Prepare line representation for Tiers 2 and 3
    let text_lines: Vec<&str> = text.lines().collect();
    let old_lines: Vec<&str> = old.lines().collect();
    if old_lines.is_empty() {
        bail!("target pattern `old` has no lines");
    }

    // Compute line byte offsets in `text` so we can do precise byte slicing
    let mut line_byte_offsets = Vec::with_capacity(text_lines.len() + 1);
    let mut offset = 0;
    for _line in &text_lines {
        line_byte_offsets.push(offset);
        // Find next newline after line
        if let Some(nl_pos) = text[offset..].find('\n') {
            offset += nl_pos + 1;
        } else {
            offset = text.len();
        }
    }
    line_byte_offsets.push(text.len());

    let k = old_lines.len();

    // Tier 2: Whitespace- & indentation-normalized line match
    let old_norm: Vec<&str> = old_lines.iter().map(|l| normalize_line(l)).collect();
    let old_trimmed: Vec<&str> = old_lines.iter().map(|l| l.trim()).collect();

    if text_lines.len() >= k {
        let mut matches_norm = Vec::new();
        for i in 0..=text_lines.len() - k {
            let window = &text_lines[i..i + k];
            let all_match = window
                .iter()
                .zip(&old_norm)
                .all(|(w, o)| normalize_line(w) == *o);
            if all_match {
                matches_norm.push(i);
            }
        }

        if matches_norm.len() == 1 {
            let start_line = matches_norm[0];
            let end_line = start_line + k;
            let start_byte = line_byte_offsets[start_line];
            let end_byte = if end_line < line_byte_offsets.len() - 1 {
                line_byte_offsets[end_line]
            } else {
                text.len()
            };

            let mut res = String::with_capacity(text.len() + new.len());
            res.push_str(&text[..start_byte]);
            res.push_str(new);
            if !new.ends_with('\n')
                && end_line < text_lines.len()
                && !text[..start_byte].ends_with('\n')
            {
                res.push('\n');
            }
            res.push_str(&text[end_byte..]);
            return Ok((res, MatchQuality::WhitespaceNormalized));
        }
        if matches_norm.len() > 1 {
            bail!(
                "`old` matches {} places (whitespace-normalized); add more surrounding context",
                matches_norm.len()
            );
        }

        // Also try fully trimmed (handles indentation changes e.g. 2 spaces vs 4 spaces)
        let mut matches_trim = Vec::new();
        for i in 0..=text_lines.len() - k {
            let window = &text_lines[i..i + k];
            let all_match = window.iter().zip(&old_trimmed).all(|(w, o)| w.trim() == *o);
            if all_match {
                matches_trim.push(i);
            }
        }

        if matches_trim.len() == 1 {
            let start_line = matches_trim[0];
            let end_line = start_line + k;
            let start_byte = line_byte_offsets[start_line];
            let end_byte = if end_line < line_byte_offsets.len() - 1 {
                line_byte_offsets[end_line]
            } else {
                text.len()
            };

            let mut res = String::with_capacity(text.len() + new.len());
            res.push_str(&text[..start_byte]);
            res.push_str(new);
            if !new.ends_with('\n')
                && end_line < text_lines.len()
                && !text[..start_byte].ends_with('\n')
            {
                res.push('\n');
            }
            res.push_str(&text[end_byte..]);
            return Ok((res, MatchQuality::WhitespaceNormalized));
        }
        if matches_trim.len() > 1 {
            bail!(
                "`old` matches {} places (trimmed); add more surrounding context to disambiguate",
                matches_trim.len()
            );
        }
    }

    // Tier 3: Sliding window Levenshtein similarity on context lines
    if text_lines.len() >= k {
        let mut best_score = 0.0;
        let mut best_idx = None;
        let mut second_best_score = 0.0;

        for i in 0..=text_lines.len() - k {
            let window = &text_lines[i..i + k];
            let sum_sim: f64 = window
                .iter()
                .zip(&old_lines)
                .map(|(w, o)| line_similarity(w, o))
                .sum();
            let avg_sim = sum_sim / (k as f64);

            if avg_sim > best_score {
                second_best_score = best_score;
                best_score = avg_sim;
                best_idx = Some(i);
            } else if avg_sim > second_best_score {
                second_best_score = avg_sim;
            }
        }

        // Accept if confidence >= 80% and sufficiently distinct from second best
        if best_score >= 0.80
            && (best_score - second_best_score >= 0.08 || second_best_score < 0.60)
        {
            if let Some(start_line) = best_idx {
                let end_line = start_line + k;
                let start_byte = line_byte_offsets[start_line];
                let end_byte = if end_line < line_byte_offsets.len() - 1 {
                    line_byte_offsets[end_line]
                } else {
                    text.len()
                };

                let mut res = String::with_capacity(text.len() + new.len());
                res.push_str(&text[..start_byte]);
                res.push_str(new);
                if !new.ends_with('\n')
                    && end_line < text_lines.len()
                    && !text[..start_byte].ends_with('\n')
                {
                    res.push('\n');
                }
                res.push_str(&text[end_byte..]);
                return Ok((res, MatchQuality::Fuzzy(best_score)));
            }
        }
    }

    bail!("`old` block could not be located in file (tried exact, whitespace-normalized, and fuzzy match)")
}

/// A parsed hunk from either Search/Replace format or unified diff format.
#[derive(Debug, Clone)]
pub struct Hunk {
    pub search: String,
    pub replace: String,
}

/// Parse patch string into hunks.
/// Supports:
/// 1. Aider / Search-Replace blocks:
///    ```text
///    <<<<<<< SEARCH
///    [lines to find]
///    =======
///    [lines to replace with]
///    >>>>>>> REPLACE
///    ```
/// 2. Unified diff format:
///    ```diff
///    @@ -10,4 +10,4 @@
///     context line
///    -old line
///    +new line
///    ```
pub fn parse_hunks(patch_str: &str) -> Result<Vec<Hunk>> {
    let mut hunks = Vec::new();

    // Check for SEARCH/REPLACE blocks
    if patch_str.contains("<<<<<<< SEARCH") {
        let parts: Vec<&str> = patch_str.split("<<<<<<< SEARCH").collect();
        for part in parts.iter().skip(1) {
            let Some((search_part, rest)) = part.split_once("=======") else {
                continue;
            };
            let Some((replace_part, _)) = rest.split_once(">>>>>>> REPLACE") else {
                continue;
            };

            // Trim leading newline from search & replace if present
            let search = search_part.strip_prefix('\n').unwrap_or(search_part);
            let replace = replace_part.strip_prefix('\n').unwrap_or(replace_part);

            hunks.push(Hunk {
                search: search.to_string(),
                replace: replace.to_string(),
            });
        }
        if !hunks.is_empty() {
            return Ok(hunks);
        }
    }

    // Check for Unified diff format
    let lines: Vec<&str> = patch_str.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if line.starts_with("@@") {
            i += 1;
            let mut search_buf = String::new();
            let mut replace_buf = String::new();

            while i < lines.len() && !lines[i].starts_with("@@") && !lines[i].starts_with("diff ") {
                let hline = lines[i];
                if let Some(stripped) = hline.strip_prefix('-') {
                    search_buf.push_str(stripped);
                    search_buf.push('\n');
                } else if let Some(stripped) = hline.strip_prefix('+') {
                    replace_buf.push_str(stripped);
                    replace_buf.push('\n');
                } else if let Some(stripped) = hline.strip_prefix(' ') {
                    search_buf.push_str(stripped);
                    search_buf.push('\n');
                    replace_buf.push_str(stripped);
                    replace_buf.push('\n');
                } else if hline.is_empty() {
                    // Empty lines in diffs are often context lines without leading space
                    search_buf.push('\n');
                    replace_buf.push('\n');
                }
                i += 1;
            }

            if !search_buf.is_empty() || !replace_buf.is_empty() {
                hunks.push(Hunk {
                    search: search_buf,
                    replace: replace_buf,
                });
            }
            continue;
        }
        i += 1;
    }

    if hunks.is_empty() {
        bail!("no valid hunks found; expected Search/Replace blocks (<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE) or unified diff hunks (@@ ... @@)");
    }

    Ok(hunks)
}

/// Apply a patch string (containing one or more hunks) to `target_text`.
pub fn apply_patch(target_text: &str, patch_str: &str) -> Result<PatchOutcome> {
    let hunks = parse_hunks(patch_str)?;
    let mut current = target_text.to_string();
    let mut lowest_quality = MatchQuality::Exact;
    let total = hunks.len();

    for (idx, hunk) in hunks.into_iter().enumerate() {
        let (updated, q) = fuzzy_replace(&current, &hunk.search, &hunk.replace)
            .with_context(|| format!("applying hunk #{}", idx + 1))?;
        current = updated;
        match (&lowest_quality, &q) {
            (MatchQuality::Exact, other) => lowest_quality = other.clone(),
            (MatchQuality::WhitespaceNormalized, MatchQuality::Fuzzy(c)) => {
                lowest_quality = MatchQuality::Fuzzy(*c);
            }
            _ => {}
        }
    }

    Ok(PatchOutcome {
        modified_text: current,
        hunks_applied: total,
        quality: lowest_quality,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_replace_works() {
        let original = "fn main() {\n    let a = 1;\n    println!(\"{a}\");\n}\n";
        let (res, q) = fuzzy_replace(original, "let a = 1;", "let a = 42;").unwrap();
        assert_eq!(q, MatchQuality::Exact);
        assert!(res.contains("let a = 42;"));
    }

    #[test]
    fn whitespace_normalized_replace_handles_indentation_and_blank_lines() {
        let original = "fn compute() {\n    let x = 10;\n    let y = 20;\n    x + y\n}\n";
        // Search pattern has 2 spaces indentation instead of 4, and extra carriage return
        let search = "  let x = 10;\r\n  let y = 20;\r\n";
        let replacement = "    let x = 100;\n    let y = 200;\n";
        let (res, q) = fuzzy_replace(original, search, replacement).unwrap();
        assert_eq!(q, MatchQuality::WhitespaceNormalized);
        assert!(res.contains("let x = 100;\n    let y = 200;"));
    }

    #[test]
    fn fuzzy_replace_recovers_from_minor_comment_drift() {
        let original = "def calculate(a, b):\n    # old comment version 1\n    result = a * 2\n    return result + b\n";
        // Search has slightly drifted comment line
        let search = "def calculate(a, b):\n    # old comment version 2\n    result = a * 2\n";
        let replacement = "def calculate(a, b):\n    # updated computation\n    result = a * 4\n";
        let (res, q) = fuzzy_replace(original, search, replacement).unwrap();
        assert!(matches!(q, MatchQuality::Fuzzy(conf) if conf >= 0.85));
        assert!(res.contains("result = a * 4"));
    }

    #[test]
    fn parse_and_apply_search_replace_blocks() {
        let original = "alpha\nbeta\ngamma\ndelta\n";
        let patch = r#"
<<<<<<< SEARCH
beta
gamma
=======
BETA_MODIFIED
GAMMA_MODIFIED
>>>>>>> REPLACE
"#;
        let outcome = apply_patch(original, patch).unwrap();
        assert_eq!(outcome.hunks_applied, 1);
        assert_eq!(
            outcome.modified_text,
            "alpha\nBETA_MODIFIED\nGAMMA_MODIFIED\ndelta\n"
        );
    }

    #[test]
    fn parse_and_apply_unified_diff_hunks() {
        let original = "line1\nline2\nline3\nline4\n";
        let diff = r#"
--- a/file.txt
+++ b/file.txt
@@ -2,2 +2,2 @@
 line1
-line2
+line2_updated
 line3
"#;
        let outcome = apply_patch(original, diff).unwrap();
        assert_eq!(outcome.hunks_applied, 1);
        assert_eq!(
            outcome.modified_text,
            "line1\nline2_updated\nline3\nline4\n"
        );
    }
}
