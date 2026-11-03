//! Interactive terminal line reader with persistent history, arrow navigation, and tab completion.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;

const MAX_HISTORY: usize = 1_000;

pub struct LineReader {
    pub history: Vec<String>,
    history_file: Option<PathBuf>,
}

impl LineReader {
    pub fn new() -> Self {
        let history_file = std::env::var_os("HOME")
            .map(PathBuf::from)
            .map(|h| h.join(".keel").join("history"));

        let mut history = Vec::new();
        if let Some(ref path) = history_file {
            if let Ok(file) = File::open(path) {
                let reader = BufReader::new(file);
                for line in reader.lines().map_while(Result::ok) {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        history.push(trimmed.to_string());
                    }
                }
                if history.len() > MAX_HISTORY {
                    history.drain(0..history.len() - MAX_HISTORY);
                }
            }
        }

        Self {
            history,
            history_file,
        }
    }

    pub fn add_history(&mut self, line: &str) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return;
        }
        if self.history.last().map(|s| s.as_str()) == Some(trimmed) {
            return;
        }

        self.history.push(trimmed.to_string());
        if self.history.len() > MAX_HISTORY {
            self.history.remove(0);
        }

        if let Some(ref path) = self.history_file {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
                let _ = writeln!(f, "{trimmed}");
            }
        }
    }

    pub fn read_line(
        &mut self,
        prompt: &str,
        commands: &[&str],
    ) -> std::io::Result<Option<String>> {
        // Fallback to standard read_line if not a TTY
        let is_tty = unsafe { libc::isatty(libc::STDIN_FILENO) == 1 };
        if !is_tty {
            print!("{prompt}");
            let _ = std::io::stdout().flush();
            let mut line = String::new();
            if std::io::stdin().read_line(&mut line)? == 0 {
                return Ok(None);
            }
            return Ok(Some(line));
        }

        let _guard = match RawModeGuard::enter() {
            Ok(g) => g,
            Err(_) => {
                print!("{prompt}");
                let _ = std::io::stdout().flush();
                let mut line = String::new();
                if std::io::stdin().read_line(&mut line)? == 0 {
                    return Ok(None);
                }
                return Ok(Some(line));
            }
        };

        let mut buf: Vec<char> = Vec::new();
        let mut cursor = 0usize;
        let mut history_idx: Option<usize> = None;
        let mut in_progress = String::new();

        print!("{prompt}");
        let _ = std::io::stdout().flush();

        let mut stdin = std::io::stdin();
        let mut byte_buf = [0u8; 1];

        loop {
            if stdin.read_exact(&mut byte_buf).is_err() {
                println!();
                return Ok(None);
            }
            let b = byte_buf[0];

            match b {
                b'\r' | b'\n' => {
                    println!();
                    let line: String = buf.iter().collect();
                    return Ok(Some(line));
                }
                0x03 => {
                    // Ctrl-C
                    print!("^C\r\n{prompt}");
                    let _ = std::io::stdout().flush();
                    buf.clear();
                    cursor = 0;
                    history_idx = None;
                    in_progress.clear();
                }
                0x04 => {
                    // Ctrl-D
                    if buf.is_empty() {
                        println!();
                        return Ok(None);
                    }
                    if cursor < buf.len() {
                        buf.remove(cursor);
                        Self::redraw(prompt, &buf, cursor)?;
                    }
                }
                0x08 | 0x7f => {
                    // Backspace
                    if cursor > 0 {
                        cursor -= 1;
                        buf.remove(cursor);
                        Self::redraw(prompt, &buf, cursor)?;
                    }
                }
                0x01 => {
                    // Ctrl-A / Home
                    cursor = 0;
                    Self::redraw(prompt, &buf, cursor)?;
                }
                0x05 => {
                    // Ctrl-E / End
                    cursor = buf.len();
                    Self::redraw(prompt, &buf, cursor)?;
                }
                0x0b => {
                    // Ctrl-K (kill line to right)
                    buf.truncate(cursor);
                    Self::redraw(prompt, &buf, cursor)?;
                }
                0x15 => {
                    // Ctrl-U (kill entire line)
                    buf.clear();
                    cursor = 0;
                    Self::redraw(prompt, &buf, cursor)?;
                }
                0x0c => {
                    // Ctrl-L (clear screen)
                    print!("\x1b[2J\x1b[H");
                    Self::redraw(prompt, &buf, cursor)?;
                }
                b'\t' => {
                    // Tab completion
                    let current_str: String = buf.iter().collect();
                    if current_str.starts_with('/') && !current_str.contains(' ') {
                        let matches: Vec<&&str> = commands
                            .iter()
                            .filter(|cmd| cmd.starts_with(&current_str))
                            .collect();

                        if matches.len() == 1 {
                            let completed = format!("{} ", matches[0]);
                            buf = completed.chars().collect();
                            cursor = buf.len();
                            Self::redraw(prompt, &buf, cursor)?;
                        } else if matches.len() > 1 {
                            print!("\r\n");
                            for m in &matches {
                                print!("  {} ", m);
                            }
                            print!("\r\n");
                            Self::redraw(prompt, &buf, cursor)?;
                        }
                    }
                }
                0x1b => {
                    // Escape sequence
                    let mut seq = [0u8; 2];
                    if stdin.read_exact(&mut seq[0..1]).is_ok()
                        && seq[0] == b'['
                        && stdin.read_exact(&mut seq[1..2]).is_ok()
                    {
                        match seq[1] {
                            b'A' => {
                                // Up arrow: browse history backwards
                                if history_idx.is_none() {
                                    if !self.history.is_empty() {
                                        in_progress = buf.iter().collect();
                                        let new_idx = self.history.len() - 1;
                                        history_idx = Some(new_idx);
                                        buf = self.history[new_idx].chars().collect();
                                        cursor = buf.len();
                                        Self::redraw(prompt, &buf, cursor)?;
                                    }
                                } else if let Some(idx) = history_idx {
                                    if idx > 0 {
                                        let new_idx = idx - 1;
                                        history_idx = Some(new_idx);
                                        buf = self.history[new_idx].chars().collect();
                                        cursor = buf.len();
                                        Self::redraw(prompt, &buf, cursor)?;
                                    }
                                }
                            }
                            b'B' => {
                                // Down arrow: browse history forwards
                                if let Some(idx) = history_idx {
                                    if idx + 1 < self.history.len() {
                                        let new_idx = idx + 1;
                                        history_idx = Some(new_idx);
                                        buf = self.history[new_idx].chars().collect();
                                        cursor = buf.len();
                                        Self::redraw(prompt, &buf, cursor)?;
                                    } else {
                                        history_idx = None;
                                        buf = in_progress.chars().collect();
                                        cursor = buf.len();
                                        Self::redraw(prompt, &buf, cursor)?;
                                    }
                                }
                            }
                            b'C' => {
                                // Right arrow
                                if cursor < buf.len() {
                                    cursor += 1;
                                    Self::redraw(prompt, &buf, cursor)?;
                                }
                            }
                            b'D' => {
                                // Left arrow
                                if cursor > 0 {
                                    cursor -= 1;
                                    Self::redraw(prompt, &buf, cursor)?;
                                }
                            }
                            b'H' => {
                                // Home
                                cursor = 0;
                                Self::redraw(prompt, &buf, cursor)?;
                            }
                            b'F' => {
                                // End
                                cursor = buf.len();
                                Self::redraw(prompt, &buf, cursor)?;
                            }
                            b'3' => {
                                // Delete key (\x1b[3~)
                                let mut tilde = [0u8; 1];
                                if stdin.read_exact(&mut tilde).is_ok()
                                    && tilde[0] == b'~'
                                    && cursor < buf.len()
                                {
                                    buf.remove(cursor);
                                    Self::redraw(prompt, &buf, cursor)?;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                c if c >= 0x20 => {
                    // UTF-8 lead byte detection
                    let mut bytes = vec![c];
                    let needed = if c < 0x80 {
                        0
                    } else if (c & 0xe0) == 0xc0 {
                        1
                    } else if (c & 0xf0) == 0xe0 {
                        2
                    } else if (c & 0xf8) == 0xf0 {
                        3
                    } else {
                        0
                    };

                    for _ in 0..needed {
                        let mut next = [0u8; 1];
                        if stdin.read_exact(&mut next).is_ok() {
                            bytes.push(next[0]);
                        }
                    }

                    if let Ok(s) = std::str::from_utf8(&bytes) {
                        for ch in s.chars() {
                            buf.insert(cursor, ch);
                            cursor += 1;
                        }
                        Self::redraw(prompt, &buf, cursor)?;
                    }
                }
                _ => {}
            }
        }
    }

    fn redraw(prompt: &str, buf: &[char], cursor: usize) -> std::io::Result<()> {
        let mut stdout = std::io::stdout();
        let content: String = buf.iter().collect();
        write!(stdout, "\r\x1b[2K{}{}", prompt, content)?;
        let col = prompt.chars().count() + cursor + 1;
        write!(stdout, "\x1b[{}G", col)?;
        stdout.flush()?;
        Ok(())
    }
}

struct RawModeGuard {
    orig: libc::termios,
    active: bool,
}

impl RawModeGuard {
    fn enter() -> Result<Self, std::io::Error> {
        unsafe {
            let mut orig = std::mem::zeroed();
            if libc::tcgetattr(libc::STDIN_FILENO, &mut orig) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            let mut raw = orig;
            libc::cfmakeraw(&mut raw);
            // Re-enable output post-processing so newline is \r\n
            raw.c_oflag |= libc::OPOST | libc::ONLCR;
            if libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &raw) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(Self { orig, active: true })
        }
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        if self.active {
            unsafe {
                libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &self.orig);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_reader_history_operations() {
        let tmp = std::env::temp_dir().join(format!("keel-history-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let hist_file = tmp.join("history");

        let mut lr = LineReader {
            history: Vec::new(),
            history_file: Some(hist_file.clone()),
        };

        // Adding entries
        lr.add_history("/help");
        lr.add_history("inspect main.rs");
        // Consecutive duplicate is ignored
        lr.add_history("inspect main.rs");
        // Empty is ignored
        lr.add_history("   ");

        assert_eq!(lr.history, vec!["/help", "inspect main.rs"]);

        // Verify persisted to disk
        let disk_content = std::fs::read_to_string(&hist_file).unwrap();
        assert_eq!(disk_content, "/help\ninspect main.rs\n");

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
