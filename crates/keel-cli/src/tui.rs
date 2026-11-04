//! Keel's terminal cockpit: a session-centric "wake" over the JSON event
//! stream of `keel run --json`.
//!
//! The TUI deliberately does not reimplement the agent loop. It launches the
//! same `keel` binary, renders the streaming NDJSON events, keeps the session
//! log as the source of truth, and lets the composer queue the next prompt
//! while the current turn is still running.

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};
use serde_json::Value;
use std::io::{BufRead, BufReader, IsTerminal, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const TEAL: Color = Color::Rgb(45, 212, 191);
const SKY: Color = Color::Rgb(56, 189, 248);
const AMBER: Color = Color::Rgb(251, 191, 36);
const ROSE: Color = Color::Rgb(251, 113, 133);
const MUTED: Color = Color::Rgb(100, 116, 139);
const POLL: Duration = Duration::from_millis(60);

const KNOWN_MODELS: [(&str, &str, &str); 7] = [
    (
        "bonsai-9b",
        "Bonsai 2 9B (Local)",
        "http://127.0.0.1:8080/v1",
    ),
    (
        "bonsai-4b",
        "Bonsai 2 4B (Local)",
        "http://127.0.0.1:8080/v1",
    ),
    (
        "bonsai-2b",
        "Bonsai 2 2B (Local)",
        "http://127.0.0.1:8080/v1",
    ),
    (
        "bonsai-27b",
        "Bonsai 2 27B (Local)",
        "http://127.0.0.1:8080/v1",
    ),
    (
        "qwen-27b",
        "Qwen 3.8 27B (Local)",
        "http://127.0.0.1:8080/v1",
    ),
    (
        "kat-coder",
        "Kat Coder (Local :8085)",
        "http://127.0.0.1:8085/v1",
    ),
    (
        "deepseek-chat",
        "DeepSeek V3 (API)",
        "https://api.deepseek.com",
    ),
];

pub struct Config {
    pub root: PathBuf,
    pub model: String,
    pub base_url: String,
    pub api_key: Option<String>,
    pub yolo: bool,
    pub read_only: bool,
    pub safety_mode: Option<String>,
    pub shadow_vfs: bool,
    pub no_sandbox: bool,
    pub allow_protected: bool,
    pub thinking: Option<String>,
    pub steps: Option<usize>,
    pub max_context_tokens: Option<usize>,
    pub prompt: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Focus {
    Logbook,
    Wake,
    Composer,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PaletteAction {
    NewSession,
    SearchSessions,
    ModelPicker,
    ToggleSidebar,
    ToggleDetails,
    ShowDiff,
    Help,
    Interrupt,
    ClearWake,
    Quit,
}

impl PaletteAction {
    const ALL: [PaletteAction; 10] = [
        PaletteAction::NewSession,
        PaletteAction::SearchSessions,
        PaletteAction::ModelPicker,
        PaletteAction::ToggleSidebar,
        PaletteAction::ToggleDetails,
        PaletteAction::ShowDiff,
        PaletteAction::Help,
        PaletteAction::Interrupt,
        PaletteAction::ClearWake,
        PaletteAction::Quit,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::NewSession => "New wake",
            Self::SearchSessions => "Search log book",
            Self::ModelPicker => "Choose model",
            Self::ToggleSidebar => "Toggle ship status",
            Self::ToggleDetails => "Toggle reasoning/tool detail",
            Self::ShowDiff => "Show git diff",
            Self::Help => "Keyboard help",
            Self::Interrupt => "Interrupt current turn",
            Self::ClearWake => "Clear the wake view",
            Self::Quit => "Quit Keel",
        }
    }

    fn hint(self) -> &'static str {
        match self {
            Self::NewSession => "start a fresh session log",
            Self::SearchSessions => "filter the log book by title",
            Self::ModelPicker => "pick a local or API model",
            Self::ToggleSidebar => "show model, sandbox, usage and diffs",
            Self::ToggleDetails => "expand reasoning and tool output",
            Self::ShowDiff => "review the workspace git diff",
            Self::Help => "show every key binding",
            Self::Interrupt => "stop the running keel process",
            Self::ClearWake => "hide the current transcript without deleting the log",
            Self::Quit => "leave the TUI",
        }
    }
}

struct PaletteState {
    query: String,
    selected: usize,
}

struct ModelPickerState {
    selected: usize,
}

struct SessionMeta {
    path: PathBuf,
    title: String,
    turns: usize,
    updated: SystemTime,
}

enum Item {
    User(String),
    Assistant(String),
    Reasoning(String),
    ToolCall { name: String, args: String },
    ToolResult { id: String, output: String },
    Notice(String),
    Veto { tool: String, reason: String },
}

struct TurnStats {
    duration_ms: u64,
    tool_calls: usize,
    modified_files: Vec<String>,
    lines_added: usize,
    lines_deleted: usize,
    tests_passed: Option<usize>,
    tests_regressions: Option<usize>,
    estimated_tokens: Option<usize>,
}

impl TurnStats {
    fn from_json(value: &Value) -> Option<Self> {
        Some(Self {
            duration_ms: value.get("duration_ms")?.as_u64().unwrap_or(0),
            tool_calls: value.get("tool_calls").and_then(Value::as_u64).unwrap_or(0) as usize,
            modified_files: value
                .get("modified_files")
                .and_then(Value::as_array)
                .map(|files| {
                    files
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default(),
            lines_added: value
                .get("lines_added")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize,
            lines_deleted: value
                .get("lines_deleted")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize,
            tests_passed: value
                .get("tests_passed")
                .and_then(Value::as_u64)
                .map(|n| n as usize),
            tests_regressions: value
                .get("tests_regressions")
                .and_then(Value::as_u64)
                .map(|n| n as usize),
            estimated_tokens: value
                .get("estimated_tokens")
                .and_then(Value::as_u64)
                .map(|n| n as usize),
        })
    }
}

struct App {
    config: Config,
    sessions: Vec<SessionMeta>,
    session_idx: usize,
    session_state: ListState,
    current_log: Option<PathBuf>,
    session_title: String,
    items: Vec<Item>,
    model: String,
    base_url: String,
    running: bool,
    interrupted: bool,
    child_pid: Option<i32>,
    pending_prompts: Vec<String>,
    input: Vec<char>,
    cursor: usize,
    focus: Focus,
    message_scroll: usize,
    auto_scroll: bool,
    show_sidebar: bool,
    show_details: bool,
    show_help: bool,
    show_diff: bool,
    diff_text: String,
    diff_error: Option<String>,
    diff_scroll: usize,
    palette: Option<PaletteState>,
    model_picker: Option<ModelPickerState>,
    search_active: bool,
    search_query: String,
    last_stats: Option<TurnStats>,
    stderr_tail: Vec<String>,
    status: String,
    error: Option<String>,
    animation: u64,
    quit: bool,
    rx: Option<Receiver<AgentEvent>>,
}

enum AgentEvent {
    Line(String),
    Stderr(String),
    Exited(Option<i32>),
}

impl App {
    fn new(config: Config) -> Self {
        let model = config.model.clone();
        let base_url = config.base_url.clone();
        Self {
            config,
            sessions: Vec::new(),
            session_idx: 0,
            session_state: ListState::default(),
            current_log: None,
            session_title: "New wake".to_string(),
            items: Vec::new(),
            model,
            base_url,
            running: false,
            interrupted: false,
            child_pid: None,
            pending_prompts: Vec::new(),
            input: Vec::new(),
            cursor: 0,
            focus: Focus::Composer,
            message_scroll: 0,
            auto_scroll: true,
            show_sidebar: true,
            show_details: false,
            show_help: false,
            show_diff: false,
            diff_text: String::new(),
            diff_error: None,
            diff_scroll: 0,
            palette: None,
            model_picker: None,
            search_active: false,
            search_query: String::new(),
            last_stats: None,
            stderr_tail: Vec::new(),
            status: "Ready".to_string(),
            error: None,
            animation: 0,
            quit: false,
            rx: None,
        }
    }

    fn scan_sessions(&mut self) {
        self.sessions = scan_sessions();
        self.session_idx = self.session_idx.min(self.sessions.len().saturating_sub(1));
        self.session_state
            .select((!self.sessions.is_empty()).then_some(self.session_idx));
    }

    fn filtered_sessions(&self) -> Vec<usize> {
        let query = self.search_query.trim().to_lowercase();
        self.sessions
            .iter()
            .enumerate()
            .filter(|(_, session)| {
                query.is_empty() || session.title.to_lowercase().contains(&query)
            })
            .map(|(index, _)| index)
            .collect()
    }

    fn move_session(&mut self, delta: isize) {
        let visible = self.filtered_sessions();
        if visible.is_empty() {
            return;
        }
        let current = visible
            .iter()
            .position(|index| *index == self.session_idx)
            .unwrap_or(0);
        let next = (current as isize + delta).rem_euclid(visible.len() as isize) as usize;
        self.session_idx = visible[next];
    }

    fn load_session(&mut self, index: usize) {
        let Some(session) = self.sessions.get(index) else {
            return;
        };
        self.current_log = Some(session.path.clone());
        self.session_title = session.title.clone();
        self.items = load_session_items(&session.path);
        self.session_idx = index;
        self.message_scroll = 0;
        self.auto_scroll = true;
        self.focus = Focus::Wake;
        self.status = format!("Loaded {}", session.path.display());
        self.last_stats = None;
        self.stderr_tail.clear();
    }

    fn new_session(&mut self) {
        if self.running {
            self.interrupt();
        }
        self.current_log = None;
        self.session_title = "New wake".to_string();
        self.items.clear();
        self.last_stats = None;
        self.stderr_tail.clear();
        self.message_scroll = 0;
        self.auto_scroll = true;
        self.focus = Focus::Composer;
        self.status = "New wake".to_string();
    }

    fn log_path(&self) -> PathBuf {
        if let Some(path) = &self.current_log {
            return path.clone();
        }
        let sessions = sessions_dir();
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis())
            .unwrap_or(0);
        sessions.join(format!("tui-{stamp}.jsonl"))
    }

    fn start_turn(&mut self, prompt: String) {
        if prompt.trim().is_empty() || self.running {
            return;
        }
        let log = self.log_path();
        self.current_log = Some(log.clone());
        if self.session_title == "New wake" {
            self.session_title = truncate(&prompt, 36);
        }
        self.items.push(Item::User(prompt.clone()));
        let mut command = Command::new(std::env::current_exe().unwrap_or_else(|_| "keel".into()));
        command
            .arg("run")
            .arg("--json")
            .arg("--root")
            .arg(&self.config.root)
            .arg("--model")
            .arg(&self.model)
            .arg("--base-url")
            .arg(&self.base_url)
            .arg("--log")
            .arg(&log)
            .current_dir(&self.config.root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if log.exists()
            && std::fs::metadata(&log)
                .map(|m| m.len() > 0)
                .unwrap_or(false)
        {
            command.arg("--resume").arg(&log);
        }
        if self.config.yolo {
            command.arg("--yolo");
        }
        if self.config.read_only {
            command.arg("--read-only");
        }
        if self.config.shadow_vfs {
            command.arg("--shadow");
        }
        if self.config.no_sandbox {
            command.arg("--no-sandbox");
        }
        if self.config.allow_protected {
            command.arg("--allow-protected");
        }
        if let Some(mode) = &self.config.safety_mode {
            command.arg("--safety-mode").arg(mode);
        }
        if let Some(steps) = self.config.steps {
            command.arg("--steps").arg(steps.to_string());
        }
        if let Some(tokens) = self.config.max_context_tokens {
            command.arg("--max-context-tokens").arg(tokens.to_string());
        }
        if let Some(thinking) = &self.config.thinking {
            command.env("KEEL_THINKING", thinking);
        }
        if let Some(api_key) = &self.config.api_key {
            command.env("KEEL_API_KEY", api_key);
        }
        command.arg("--").arg(&prompt);
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }

        match command.spawn() {
            Ok(mut child) => {
                let pid = child.id() as i32;
                let stdout = child.stdout.take();
                let stderr = child.stderr.take();
                let (tx, rx) = mpsc::channel();
                if let Some(stdout) = stdout {
                    let tx = tx.clone();
                    std::thread::spawn(move || read_lines(stdout, tx, false));
                }
                if let Some(stderr) = stderr {
                    let tx = tx.clone();
                    std::thread::spawn(move || read_lines(stderr, tx, true));
                }
                std::thread::spawn(move || {
                    let code = child.wait().ok().and_then(|status| status.code());
                    // Give the stdout reader a moment to flush the last events
                    // before the TUI decides the turn is over.
                    std::thread::sleep(Duration::from_millis(120));
                    let _ = tx.send(AgentEvent::Exited(code));
                });
                self.rx = Some(rx);
                self.child_pid = Some(pid);
                self.running = true;
                self.interrupted = false;
                self.error = None;
                self.status = format!("Running · {}", self.model);
                self.auto_scroll = true;
            }
            Err(error) => {
                self.error = Some(format!("cannot start keel: {error}"));
                self.running = false;
            }
        }
    }

    fn interrupt(&mut self) {
        if !self.running {
            return;
        }
        if let Some(pid) = self.child_pid {
            #[cfg(unix)]
            unsafe {
                if libc::kill(-pid, libc::SIGTERM) != 0 {
                    libc::kill(pid, libc::SIGTERM);
                }
            }
            #[cfg(not(unix))]
            {
                let _ = pid;
            }
        }
        self.interrupted = true;
        self.status = "Interrupting…".to_string();
    }

    fn submit(&mut self) {
        let prompt: String = self.input.iter().collect();
        let prompt = prompt.trim().to_string();
        if prompt.is_empty() {
            return;
        }
        self.input.clear();
        self.cursor = 0;
        if self.running {
            self.pending_prompts.push(prompt);
            self.status = format!("Running · {} queued", self.pending_prompts.len());
            return;
        }
        self.start_turn(prompt);
    }

    fn drain_agent(&mut self) {
        let Some(rx) = self.rx.take() else {
            return;
        };
        let mut keep_receiver = true;
        loop {
            match rx.try_recv() {
                Ok(AgentEvent::Line(line)) => self.handle_agent_line(&line),
                Ok(AgentEvent::Stderr(line)) => {
                    if !line.trim().is_empty() {
                        self.stderr_tail.push(line);
                        if self.stderr_tail.len() > 20 {
                            self.stderr_tail.remove(0);
                        }
                    }
                }
                Ok(AgentEvent::Exited(code)) => {
                    self.running = false;
                    self.child_pid = None;
                    self.rx = None;
                    keep_receiver = false;
                    if let Some(prompt) = self.pending_prompts.first().cloned() {
                        self.pending_prompts.remove(0);
                        self.start_turn(prompt);
                    } else if self.interrupted {
                        self.status = "Interrupted".to_string();
                        self.interrupted = false;
                    } else if code == Some(0) {
                        self.status = "Turn complete".to_string();
                    } else {
                        self.status = format!("Exited with {:?}", code);
                    }
                    self.scan_sessions();
                    break;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.rx = None;
                    keep_receiver = false;
                    break;
                }
            }
        }
        if keep_receiver {
            self.rx = Some(rx);
        }
    }

    fn handle_agent_line(&mut self, line: &str) {
        let Ok(value) = serde_json::from_str::<Value>(line.trim()) else {
            return;
        };
        match value.get("type").and_then(Value::as_str).unwrap_or("") {
            "text" => {
                let text = value
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if text.is_empty() {
                    return;
                }
                if let Some(Item::Assistant(existing)) = self.items.last_mut() {
                    existing.push_str(text);
                } else {
                    self.items.push(Item::Assistant(text.to_string()));
                }
                self.auto_scroll = true;
            }
            "reasoning" => {
                let text = value
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if text.is_empty() {
                    return;
                }
                if let Some(Item::Reasoning(existing)) = self.items.last_mut() {
                    existing.push_str(text);
                } else {
                    self.items.push(Item::Reasoning(text.to_string()));
                }
                self.auto_scroll = true;
            }
            "tool_call" => {
                let name = value
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("tool")
                    .to_string();
                let args = value
                    .get("arguments")
                    .map(|args| compact_json(args, 180))
                    .unwrap_or_default();
                self.items.push(Item::ToolCall { name, args });
                self.auto_scroll = true;
            }
            "tool_result" => {
                let id = value
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let output = value
                    .get("output")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                self.items.push(Item::ToolResult { id, output });
                self.auto_scroll = true;
            }
            "turn_summary" => {
                if let Some(summary) = value.get("summary") {
                    self.last_stats = TurnStats::from_json(summary);
                }
            }
            "sentinel_veto" => {
                self.items.push(Item::Veto {
                    tool: value
                        .get("tool_name")
                        .and_then(Value::as_str)
                        .unwrap_or("tool")
                        .to_string(),
                    reason: value
                        .get("reason")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                });
            }
            "done" => {
                let status = value.get("status").and_then(Value::as_str).unwrap_or("ok");
                if status != "ok" {
                    let error = value
                        .get("error")
                        .and_then(Value::as_str)
                        .unwrap_or("turn failed");
                    self.items.push(Item::Notice(error.to_string()));
                }
            }
            _ => {}
        }
    }

    fn handle_key(&mut self, key: KeyEvent) {
        if key.kind == KeyEventKind::Release {
            return;
        }
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.quit = true;
            return;
        }
        if self.show_diff {
            match key.code {
                KeyCode::Esc | KeyCode::Char('q') => self.show_diff = false,
                KeyCode::Char('j') | KeyCode::Down => {
                    self.diff_scroll = self.diff_scroll.saturating_add(1);
                }
                KeyCode::Char('k') | KeyCode::Up => {
                    self.diff_scroll = self.diff_scroll.saturating_sub(1);
                }
                KeyCode::PageDown => {
                    self.diff_scroll = self.diff_scroll.saturating_add(15);
                }
                KeyCode::PageUp => {
                    self.diff_scroll = self.diff_scroll.saturating_sub(15);
                }
                KeyCode::Char('g') => self.diff_scroll = 0,
                KeyCode::Char('G') => self.diff_scroll = usize::MAX / 2,
                _ => {}
            }
            return;
        }
        if self.show_help {
            if matches!(
                key.code,
                KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('?')
            ) {
                self.show_help = false;
            }
            return;
        }
        if self.palette.is_some() {
            self.handle_palette_key(key);
            return;
        }
        if self.model_picker.is_some() {
            self.handle_model_picker_key(key);
            return;
        }
        if key.code == KeyCode::Char('p') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.palette = Some(PaletteState {
                query: String::new(),
                selected: 0,
            });
            return;
        }
        if self.search_active {
            match key.code {
                KeyCode::Esc => {
                    self.search_active = false;
                    self.search_query.clear();
                }
                KeyCode::Enter => {
                    self.search_active = false;
                    self.load_session(self.session_idx);
                }
                KeyCode::Backspace => {
                    self.search_query.pop();
                }
                KeyCode::Down => self.move_session(1),
                KeyCode::Up => self.move_session(-1),
                KeyCode::Char(character)
                    if !key.modifiers.contains(KeyModifiers::CONTROL)
                        && !key.modifiers.contains(KeyModifiers::ALT) =>
                {
                    self.search_query.push(character);
                }
                _ => {}
            }
            return;
        }

        match key.code {
            KeyCode::Tab => {
                self.focus = match self.focus {
                    Focus::Logbook => Focus::Wake,
                    Focus::Wake => Focus::Composer,
                    Focus::Composer => Focus::Logbook,
                };
            }
            KeyCode::Esc => {
                if self.focus == Focus::Composer {
                    self.focus = Focus::Wake;
                } else if self.running {
                    self.interrupt();
                } else {
                    self.focus = Focus::Wake;
                }
            }
            KeyCode::PageUp => {
                self.auto_scroll = false;
                self.message_scroll = self.message_scroll.saturating_sub(10);
            }
            KeyCode::PageDown => {
                self.message_scroll = self.message_scroll.saturating_add(10);
            }
            _ => {}
        }

        match self.focus {
            Focus::Logbook => match key.code {
                KeyCode::Char('q') => self.quit = true,
                KeyCode::Char('j') | KeyCode::Down => self.move_session(1),
                KeyCode::Char('k') | KeyCode::Up => self.move_session(-1),
                KeyCode::Enter => self.load_session(self.session_idx),
                KeyCode::Char('n') => self.new_session(),
                KeyCode::Char('/') => {
                    self.search_active = true;
                    self.search_query.clear();
                }
                KeyCode::Char('i') => self.focus = Focus::Composer,
                KeyCode::Char('?') => self.show_help = true,
                KeyCode::Char('D') => self.open_diff(),
                KeyCode::Char('m') => self.open_model_picker(),
                KeyCode::Char(':') => {
                    self.palette = Some(PaletteState {
                        query: String::new(),
                        selected: 0,
                    })
                }
                _ => {}
            },
            Focus::Wake => match key.code {
                KeyCode::Char('q') => self.quit = true,
                KeyCode::Char('j') | KeyCode::Down => {
                    self.auto_scroll = false;
                    self.message_scroll = self.message_scroll.saturating_add(1);
                }
                KeyCode::Char('k') | KeyCode::Up => {
                    self.auto_scroll = false;
                    self.message_scroll = self.message_scroll.saturating_sub(1);
                }
                KeyCode::Char('g') => {
                    self.auto_scroll = false;
                    self.message_scroll = 0;
                }
                KeyCode::Char('G') => self.auto_scroll = true,
                KeyCode::Char('i') => self.focus = Focus::Composer,
                KeyCode::Char('n') => self.new_session(),
                KeyCode::Char('s') => self.focus = Focus::Logbook,
                KeyCode::Char('v') => self.show_sidebar = !self.show_sidebar,
                KeyCode::Char('d') => self.show_details = !self.show_details,
                KeyCode::Char('D') => self.open_diff(),
                KeyCode::Char('m') => self.open_model_picker(),
                KeyCode::Char('?') => self.show_help = true,
                KeyCode::Char(':') => {
                    self.palette = Some(PaletteState {
                        query: String::new(),
                        selected: 0,
                    })
                }
                KeyCode::Char('x') => self.interrupt(),
                _ => {}
            },
            Focus::Composer => match key.code {
                KeyCode::Enter => self.submit(),
                KeyCode::Backspace => {
                    if self.cursor > 0 {
                        self.cursor -= 1;
                        self.input.remove(self.cursor);
                    }
                }
                KeyCode::Delete => {
                    if self.cursor < self.input.len() {
                        self.input.remove(self.cursor);
                    }
                }
                KeyCode::Left => self.cursor = self.cursor.saturating_sub(1),
                KeyCode::Right => {
                    if self.cursor < self.input.len() {
                        self.cursor += 1;
                    }
                }
                KeyCode::Home => self.cursor = 0,
                KeyCode::End => self.cursor = self.input.len(),
                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.input.clear();
                    self.cursor = 0;
                }
                KeyCode::Char('w') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    while self.cursor > 0 && self.input[self.cursor - 1].is_whitespace() {
                        self.cursor -= 1;
                        self.input.remove(self.cursor);
                    }
                    while self.cursor > 0 && !self.input[self.cursor - 1].is_whitespace() {
                        self.cursor -= 1;
                        self.input.remove(self.cursor);
                    }
                }
                KeyCode::Char(character)
                    if !key.modifiers.contains(KeyModifiers::CONTROL)
                        && !key.modifiers.contains(KeyModifiers::ALT) =>
                {
                    self.input.insert(self.cursor, character);
                    self.cursor += 1;
                }
                _ => {}
            },
        }
    }

    fn open_model_picker(&mut self) {
        let selected = KNOWN_MODELS
            .iter()
            .position(|(id, _, _)| *id == self.model)
            .unwrap_or(0);
        self.model_picker = Some(ModelPickerState { selected });
    }

    fn open_diff(&mut self) {
        self.show_diff = true;
        self.diff_scroll = 0;
        self.diff_error = None;
        self.diff_text.clear();
        let mut command = Command::new("git");
        command.arg("diff").arg("--no-color");
        if let Some(stats) = &self.last_stats {
            if !stats.modified_files.is_empty() {
                command.arg("--");
                for file in &stats.modified_files {
                    command.arg(file);
                }
            }
        }
        command.current_dir(&self.config.root);
        match command.output() {
            Ok(output) if output.status.success() => {
                let text = String::from_utf8_lossy(&output.stdout).to_string();
                if text.trim().is_empty() {
                    self.diff_error = Some("No changes in the workspace.".to_string());
                } else {
                    self.diff_text = text;
                }
            }
            Ok(output) => {
                self.diff_error = Some(String::from_utf8_lossy(&output.stderr).trim().to_string());
            }
            Err(error) => {
                self.diff_error = Some(format!("cannot run git diff: {error}"));
            }
        }
    }

    fn handle_model_picker_key(&mut self, key: KeyEvent) {
        if self.model_picker.is_none() {
            return;
        }
        match key.code {
            KeyCode::Esc => self.model_picker = None,
            KeyCode::Up | KeyCode::Char('k') => {
                if let Some(picker) = self.model_picker.as_mut() {
                    picker.selected = picker
                        .selected
                        .checked_sub(1)
                        .unwrap_or(KNOWN_MODELS.len() - 1);
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if let Some(picker) = self.model_picker.as_mut() {
                    picker.selected = (picker.selected + 1) % KNOWN_MODELS.len();
                }
            }
            KeyCode::Enter => {
                let selected = self
                    .model_picker
                    .as_ref()
                    .map(|picker| picker.selected)
                    .unwrap_or(0);
                let (model, label, base_url) = KNOWN_MODELS[selected];
                self.model = model.to_string();
                self.base_url = base_url.to_string();
                self.status = format!("Model: {label}");
                self.model_picker = None;
            }
            _ => {}
        }
    }

    fn handle_palette_key(&mut self, key: KeyEvent) {
        if self.palette.is_none() {
            return;
        }
        match key.code {
            KeyCode::Esc => self.palette = None,
            KeyCode::Enter => {
                let selected = self
                    .palette
                    .as_ref()
                    .map(|palette| palette.selected)
                    .unwrap_or(0);
                let action = self.filtered_palette_actions().get(selected).copied();
                self.palette = None;
                if let Some(action) = action {
                    self.execute_palette_action(action);
                }
            }
            KeyCode::Up => {
                let count = self.filtered_palette_actions().len();
                if let Some(palette) = self.palette.as_mut() {
                    if count > 0 {
                        palette.selected = palette.selected.checked_sub(1).unwrap_or(count - 1);
                    }
                }
            }
            KeyCode::Down => {
                let count = self.filtered_palette_actions().len();
                if let Some(palette) = self.palette.as_mut() {
                    if count > 0 {
                        palette.selected = (palette.selected + 1) % count;
                    }
                }
            }
            KeyCode::Backspace => {
                if let Some(palette) = self.palette.as_mut() {
                    palette.query.pop();
                    palette.selected = 0;
                }
            }
            KeyCode::Char(character)
                if !key.modifiers.contains(KeyModifiers::CONTROL)
                    && !key.modifiers.contains(KeyModifiers::ALT) =>
            {
                if let Some(palette) = self.palette.as_mut() {
                    palette.query.push(character);
                    palette.selected = 0;
                }
            }
            _ => {}
        }
    }

    fn filtered_palette_actions(&self) -> Vec<PaletteAction> {
        let query = self
            .palette
            .as_ref()
            .map(|palette| palette.query.trim().to_lowercase())
            .unwrap_or_default();
        PaletteAction::ALL
            .into_iter()
            .filter(|action| {
                query.is_empty()
                    || action.label().to_lowercase().contains(&query)
                    || action.hint().to_lowercase().contains(&query)
            })
            .collect()
    }

    fn execute_palette_action(&mut self, action: PaletteAction) {
        match action {
            PaletteAction::NewSession => self.new_session(),
            PaletteAction::SearchSessions => {
                self.focus = Focus::Logbook;
                self.search_active = true;
                self.search_query.clear();
            }
            PaletteAction::ModelPicker => self.open_model_picker(),
            PaletteAction::ToggleSidebar => self.show_sidebar = !self.show_sidebar,
            PaletteAction::ToggleDetails => self.show_details = !self.show_details,
            PaletteAction::ShowDiff => self.open_diff(),
            PaletteAction::Help => self.show_help = true,
            PaletteAction::Interrupt => self.interrupt(),
            PaletteAction::ClearWake => {
                self.items.clear();
                self.auto_scroll = true;
            }
            PaletteAction::Quit => self.quit = true,
        }
    }

    fn render(&mut self, frame: &mut Frame) {
        let area = frame.area();
        if self.show_help {
            self.render_help(frame, area);
            return;
        }
        if self.palette.is_some() {
            self.render_palette(frame, area);
            return;
        }
        if self.model_picker.is_some() {
            self.render_model_picker(frame, area);
            return;
        }
        if self.show_diff {
            self.render_diff(frame, area);
            return;
        }
        let vertical = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(5),
                Constraint::Length(1),
            ])
            .split(area);
        self.render_header(frame, vertical[0]);
        let mut constraints = vec![Constraint::Length(30), Constraint::Min(30)];
        if self.show_sidebar {
            constraints.push(Constraint::Length(32));
        }
        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(constraints)
            .split(vertical[1]);
        self.render_logbook(frame, body[0]);
        self.render_wake(frame, body[1]);
        if self.show_sidebar {
            self.render_ship(frame, body[2]);
        }
        self.render_footer(frame, vertical[2]);
    }

    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let state = if self.running {
            format!("RUNNING · {}", self.model)
        } else {
            format!("IDLE · {}", self.model)
        };
        let line = Line::from(vec![
            Span::styled(
                " ⚓ KEEL ",
                Style::default()
                    .bg(TEAL)
                    .fg(Color::Black)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" ≈ ", Style::default().fg(TEAL)),
            Span::styled(state, Style::default().fg(Color::White)),
            Span::styled("  ·  ", Style::default().fg(MUTED)),
            Span::styled(
                truncate(&self.config.root.display().to_string(), 46),
                Style::default().fg(MUTED),
            ),
            Span::styled("  ·  ", Style::default().fg(MUTED)),
            Span::styled(
                safety_label(&self.config),
                Style::default().fg(if self.config.yolo { ROSE } else { AMBER }),
            ),
        ]);
        frame.render_widget(Paragraph::new(line), area);
    }

    fn render_logbook(&mut self, frame: &mut Frame, area: Rect) {
        let visible = self.filtered_sessions();
        let items: Vec<ListItem> = visible
            .iter()
            .map(|index| {
                let session = &self.sessions[*index];
                let selected = *index == self.session_idx;
                let marker = if selected { "▸ " } else { "  " };
                let first = Line::from(vec![
                    Span::styled(marker, Style::default().fg(TEAL)),
                    Span::styled(
                        truncate(&session.title, 24),
                        Style::default()
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]);
                let second = Line::from(vec![Span::styled(
                    format!(
                        "  {} turns · {}",
                        session.turns,
                        relative_time(session.updated)
                    ),
                    Style::default().fg(MUTED),
                )]);
                ListItem::new(vec![first, second])
            })
            .collect();
        let title = if self.search_active {
            format!(" LOG BOOK · /{} ", truncate(&self.search_query, 14))
        } else {
            format!(" LOG BOOK · {} ", visible.len())
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(MUTED))
            .title(Span::styled(
                title,
                Style::default().fg(TEAL).add_modifier(Modifier::BOLD),
            ));
        if items.is_empty() {
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    "No sessions yet.\nPress n to start a wake.",
                    Style::default().fg(MUTED),
                )))
                .block(block)
                .wrap(Wrap { trim: false }),
                area,
            );
            return;
        }
        let selected = visible
            .iter()
            .position(|index| *index == self.session_idx)
            .unwrap_or(0);
        let list = List::new(items)
            .block(block)
            .highlight_style(
                Style::default()
                    .bg(Color::Rgb(20, 45, 50))
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("");
        self.session_state.select(Some(selected));
        frame.render_stateful_widget(list, area, &mut self.session_state);
    }

    fn render_wake(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(4), Constraint::Length(3)])
            .split(area);
        let mut lines: Vec<Line<'static>> = Vec::new();
        if self.items.is_empty() {
            lines.push(Line::from(Span::styled(
                "⚓ The wake is clear. Type a prompt below.",
                Style::default().fg(MUTED),
            )));
        }
        for item in &self.items {
            match item {
                Item::User(text) => {
                    lines.push(Line::from(""));
                    lines.push(Line::from(Span::styled(
                        "≈ YOU",
                        Style::default().fg(SKY).add_modifier(Modifier::BOLD),
                    )));
                    push_text(&mut lines, text, Color::White);
                }
                Item::Assistant(text) => {
                    lines.push(Line::from(Span::styled(
                        "⚓ KEEL",
                        Style::default().fg(TEAL).add_modifier(Modifier::BOLD),
                    )));
                    push_text(&mut lines, text, Color::White);
                }
                Item::Reasoning(text) => {
                    if self.show_details {
                        lines.push(Line::from(Span::styled(
                            "  THINK",
                            Style::default().fg(MUTED),
                        )));
                        push_text(&mut lines, text, MUTED);
                    }
                }
                Item::ToolCall { name, args } => {
                    lines.push(Line::from(vec![
                        Span::styled("  ⚓ ", Style::default().fg(TEAL)),
                        Span::styled(
                            name.clone(),
                            Style::default().fg(TEAL).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            format!("  {}", truncate(args, 90)),
                            Style::default().fg(MUTED),
                        ),
                    ]));
                }
                Item::ToolResult { id, output } => {
                    if self.show_details {
                        lines.push(Line::from(Span::styled(
                            format!("  ↳ {id}"),
                            Style::default().fg(MUTED),
                        )));
                        for line in output.lines().take(8) {
                            lines.push(Line::from(Span::styled(
                                format!("    {}", truncate(line, 100)),
                                Style::default().fg(MUTED),
                            )));
                        }
                    }
                }
                Item::Notice(text) => {
                    lines.push(Line::from(Span::styled(
                        format!("  ! {}", truncate(text, 120)),
                        Style::default().fg(AMBER),
                    )));
                }
                Item::Veto { tool, reason } => {
                    lines.push(Line::from(Span::styled(
                        format!("  ⛔ {tool}: {}", truncate(reason, 100)),
                        Style::default().fg(ROSE).add_modifier(Modifier::BOLD),
                    )));
                }
            }
        }
        if let Some(stats) = &self.last_stats {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!(
                    "≈ turn complete in {:.1}s · {} tool calls · {} files (+{}/-{})",
                    stats.duration_ms as f64 / 1000.0,
                    stats.tool_calls,
                    stats.modified_files.len(),
                    stats.lines_added,
                    stats.lines_deleted
                ),
                Style::default().fg(MUTED),
            )));
        }
        let title = format!(
            " WAKE · {} · {} ",
            truncate(&self.session_title, 34),
            self.model
        );
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(MUTED))
            .title(Span::styled(
                title,
                Style::default().fg(TEAL).add_modifier(Modifier::BOLD),
            ));
        let content_height = wrapped_line_count(&lines, chunks[0].width.saturating_sub(2));
        let paragraph = Paragraph::new(Text::from(lines))
            .block(block)
            .wrap(Wrap { trim: false });
        let max_scroll = content_height.saturating_sub(chunks[0].height.saturating_sub(2) as usize);
        if self.auto_scroll {
            self.message_scroll = max_scroll;
        } else {
            self.message_scroll = self.message_scroll.min(max_scroll);
        }
        let paragraph = paragraph.scroll((self.message_scroll.min(u16::MAX as usize) as u16, 0));
        frame.render_widget(paragraph, chunks[0]);
        self.render_composer(frame, chunks[1]);
    }

    fn render_composer(&self, frame: &mut Frame, area: Rect) {
        let focused = self.focus == Focus::Composer;
        let input: String = self.input.iter().collect();
        let accent = if focused { TEAL } else { MUTED };
        let text = if input.is_empty() {
            Line::from(vec![
                Span::styled("⟩ ", Style::default().fg(accent)),
                Span::styled("Speak to the wake…", Style::default().fg(MUTED)),
            ])
        } else {
            Line::from(vec![
                Span::styled("⟩ ", Style::default().fg(accent)),
                Span::raw(input),
            ])
        };
        let title = if self.running {
            format!(
                " WAKE LINE · running{} · Enter queues · x interrupts ",
                if self.pending_prompts.is_empty() {
                    String::new()
                } else {
                    format!(" · {} queued", self.pending_prompts.len())
                }
            )
        } else if focused {
            " WAKE LINE · Enter sends · Esc wake ".to_string()
        } else {
            " WAKE LINE · i to type ".to_string()
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(accent))
            .title(Span::styled(title, Style::default().fg(accent)));
        frame.render_widget(Paragraph::new(text).block(block), area);
        if focused {
            let x = area
                .x
                .saturating_add(3)
                .saturating_add(self.cursor as u16)
                .min(area.right().saturating_sub(2));
            let y = area.y.saturating_add(1);
            frame.set_cursor_position((x, y));
        }
    }

    fn render_ship(&self, frame: &mut Frame, area: Rect) {
        let mut lines: Vec<Line<'static>> = Vec::new();
        lines.push(ship_line("model", &self.model));
        lines.push(ship_line("provider", &self.base_url));
        lines.push(ship_line("root", &self.config.root.display().to_string()));
        lines.push(ship_line(
            "thinking",
            self.config.thinking.as_deref().unwrap_or("auto"),
        ));
        lines.push(ship_line("safety", &safety_label(&self.config)));
        lines.push(ship_line(
            "sandbox",
            if self.config.no_sandbox {
                "disabled"
            } else {
                "Landlock"
            },
        ));
        lines.push(ship_line(
            "vfs",
            if self.config.shadow_vfs {
                "shadow"
            } else {
                "direct"
            },
        ));
        if let Some(tokens) = self.config.max_context_tokens {
            lines.push(ship_line("context", &format!("{tokens} tokens")));
        }
        if let Some(stats) = &self.last_stats {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "LAST TURN",
                Style::default().fg(TEAL).add_modifier(Modifier::BOLD),
            )));
            lines.push(ship_line(
                "duration",
                &format!("{:.1}s", stats.duration_ms as f64 / 1000.0),
            ));
            lines.push(ship_line("tools", &stats.tool_calls.to_string()));
            lines.push(ship_line(
                "tokens",
                &stats
                    .estimated_tokens
                    .map(|tokens| tokens.to_string())
                    .unwrap_or_else(|| "?".to_string()),
            ));
            if let Some(passed) = stats.tests_passed {
                lines.push(ship_line("tests", &passed.to_string()));
            }
            if let Some(regressions) = stats.tests_regressions {
                lines.push(ship_line("regressions", &regressions.to_string()));
            }
            if !stats.modified_files.is_empty() {
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "CHANGED",
                    Style::default().fg(TEAL).add_modifier(Modifier::BOLD),
                )));
                for file in stats.modified_files.iter().take(8) {
                    lines.push(Line::from(Span::styled(
                        format!("  {}", truncate(file, 28)),
                        Style::default().fg(MUTED),
                    )));
                }
            }
        }
        if !self.stderr_tail.is_empty() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "STDERR",
                Style::default().fg(AMBER).add_modifier(Modifier::BOLD),
            )));
            for line in self.stderr_tail.iter().rev().take(5).rev() {
                lines.push(Line::from(Span::styled(
                    format!("  {}", truncate(line, 28)),
                    Style::default().fg(MUTED),
                )));
            }
        }
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(MUTED))
            .title(Span::styled(
                " SHIP ",
                Style::default().fg(TEAL).add_modifier(Modifier::BOLD),
            ));
        frame.render_widget(
            Paragraph::new(Text::from(lines))
                .block(block)
                .wrap(Wrap { trim: false }),
            area,
        );
    }

    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let hints = match self.focus {
            Focus::Logbook => {
                "j/k sessions · Enter load · / search · n new · Tab wake · Ctrl-P palette · q quit"
            }
            Focus::Wake => {
                "j/k scroll · d details · D diff · v ship · i type · n new · Ctrl-P palette · Esc interrupt · q quit"
            }
            Focus::Composer => {
                "Enter send/queue · Esc wake · Ctrl-U clear · Ctrl-C quit"
            }
        };
        let line = if let Some(error) = &self.error {
            Line::from(vec![
                Span::styled(" ! ", Style::default().fg(ROSE)),
                Span::styled(truncate(error, 110), Style::default().fg(ROSE)),
            ])
        } else {
            Line::from(vec![
                Span::styled(truncate(&self.status, 52), Style::default().fg(MUTED)),
                Span::raw("  "),
                Span::styled(hints, Style::default().fg(MUTED)),
            ])
        };
        frame.render_widget(Paragraph::new(line), area);
    }

    fn render_help(&self, frame: &mut Frame, area: Rect) {
        frame.render_widget(Clear, area);
        let vertical = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Fill(1),
                Constraint::Length(22.min(area.height)),
                Constraint::Fill(1),
            ])
            .split(area);
        let horizontal = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Fill(1),
                Constraint::Length(76.min(area.width)),
                Constraint::Fill(1),
            ])
            .split(vertical[1]);
        let text = Text::from(vec![
            Line::from(Span::styled(
                "KEEL WAKE",
                Style::default().fg(TEAL).add_modifier(Modifier::BOLD),
            )),
            Line::from("  Tab              cycle Log book → Wake → Wake line"),
            Line::from("  j / k, ↑ / ↓     move or scroll"),
            Line::from("  Enter            send, or queue while a turn runs"),
            Line::from("  Esc              leave composer, or interrupt a running turn"),
            Line::from("  n                new wake (new session log)"),
            Line::from("  s                focus the log book"),
            Line::from("  /                search the log book"),
            Line::from("  d                expand reasoning and tool output"),
            Line::from("  D                review the workspace git diff"),
            Line::from("  v                toggle ship status rail"),
            Line::from("  m                model picker"),
            Line::from("  Ctrl-P or :      command palette"),
            Line::from("  x                interrupt the running turn"),
            Line::from("  ?                this help"),
            Line::from("  q / Ctrl-C       quit"),
            Line::from(""),
            Line::from(Span::styled(
                "The log book is the source of truth. Closing the TUI does not delete a wake.",
                Style::default().fg(MUTED),
            )),
        ]);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(TEAL))
            .title(" Keys ");
        frame.render_widget(
            Paragraph::new(text).block(block).wrap(Wrap { trim: false }),
            horizontal[1],
        );
    }

    fn render_palette(&self, frame: &mut Frame, area: Rect) {
        frame.render_widget(Clear, area);
        let vertical = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Fill(1),
                Constraint::Length(19.min(area.height)),
                Constraint::Fill(1),
            ])
            .split(area);
        let horizontal = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Fill(1),
                Constraint::Length(78.min(area.width)),
                Constraint::Fill(1),
            ])
            .split(vertical[1]);
        let box_area = horizontal[1];
        let Some(palette) = &self.palette else {
            return;
        };
        let actions = self.filtered_palette_actions();
        let mut lines: Vec<Line<'static>> = Vec::new();
        lines.push(Line::from(vec![
            Span::styled(" ⟩ ", Style::default().fg(TEAL)),
            Span::styled(palette.query.clone(), Style::default().fg(Color::White)),
        ]));
        lines.push(Line::from(""));
        if actions.is_empty() {
            lines.push(Line::from(Span::styled(
                "No matching actions.",
                Style::default().fg(MUTED),
            )));
        } else {
            for (index, action) in actions.iter().enumerate() {
                let selected = index == palette.selected;
                let style = if selected {
                    Style::default().bg(Color::Rgb(20, 45, 50)).fg(Color::White)
                } else {
                    Style::default().fg(Color::Gray)
                };
                lines.push(Line::from(vec![
                    Span::styled(
                        if selected { " ▸ " } else { "   " },
                        Style::default().fg(TEAL),
                    ),
                    Span::styled(action.label(), style.add_modifier(Modifier::BOLD)),
                    Span::raw("  "),
                    Span::styled(truncate(action.hint(), 44), Style::default().fg(MUTED)),
                ]));
            }
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Enter run · ↑/↓ move · Esc close",
            Style::default().fg(MUTED),
        )));
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(TEAL))
            .title(" Command palette ");
        frame.render_widget(
            Paragraph::new(Text::from(lines))
                .block(block)
                .wrap(Wrap { trim: false }),
            box_area,
        );
    }

    fn render_diff(&self, frame: &mut Frame, area: Rect) {
        frame.render_widget(Clear, area);
        let vertical = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Fill(1),
                Constraint::Length(30.min(area.height)),
                Constraint::Fill(1),
            ])
            .split(area);
        let horizontal = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Fill(1),
                Constraint::Length(110.min(area.width)),
                Constraint::Fill(1),
            ])
            .split(vertical[1]);
        let box_area = horizontal[1];
        let mut lines: Vec<Line<'static>> = Vec::new();
        if let Some(error) = &self.diff_error {
            lines.push(Line::from(Span::styled(
                error.clone(),
                Style::default().fg(AMBER),
            )));
        } else {
            for line in self.diff_text.lines() {
                let color = if line.starts_with("+++") || line.starts_with("---") {
                    MUTED
                } else if line.starts_with('+') {
                    TEAL
                } else if line.starts_with('-') {
                    ROSE
                } else if line.starts_with("@@") || line.starts_with("diff --git") {
                    SKY
                } else {
                    Color::Gray
                };
                lines.push(Line::from(Span::styled(
                    line.to_string(),
                    Style::default().fg(color),
                )));
            }
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "j/k scroll · PgUp/PgDn page · g/G top/bottom · Esc close",
            Style::default().fg(MUTED),
        )));
        let content_height = wrapped_line_count(&lines, box_area.width.saturating_sub(2));
        let max_scroll = content_height.saturating_sub(box_area.height.saturating_sub(2) as usize);
        let scroll = self.diff_scroll.min(max_scroll);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(TEAL))
            .title(Span::styled(
                format!(
                    " DIFF · {} ",
                    truncate(&self.config.root.display().to_string(), 60)
                ),
                Style::default().fg(TEAL).add_modifier(Modifier::BOLD),
            ));
        frame.render_widget(
            Paragraph::new(Text::from(lines))
                .block(block)
                .wrap(Wrap { trim: false })
                .scroll((scroll.min(u16::MAX as usize) as u16, 0)),
            box_area,
        );
    }

    fn render_model_picker(&self, frame: &mut Frame, area: Rect) {
        frame.render_widget(Clear, area);
        let vertical = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Fill(1),
                Constraint::Length(20.min(area.height)),
                Constraint::Fill(1),
            ])
            .split(area);
        let horizontal = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Fill(1),
                Constraint::Length(78.min(area.width)),
                Constraint::Fill(1),
            ])
            .split(vertical[1]);
        let box_area = horizontal[1];
        let Some(picker) = &self.model_picker else {
            return;
        };
        let mut lines: Vec<Line<'static>> = Vec::new();
        for (index, (model, label, base_url)) in KNOWN_MODELS.iter().enumerate() {
            let selected = index == picker.selected;
            let style = if selected {
                Style::default().bg(Color::Rgb(20, 45, 50)).fg(Color::White)
            } else {
                Style::default().fg(Color::Gray)
            };
            lines.push(Line::from(vec![
                Span::styled(
                    if selected { " ▸ " } else { "   " },
                    Style::default().fg(TEAL),
                ),
                Span::styled(*label, style.add_modifier(Modifier::BOLD)),
                Span::styled(format!("  {model}"), Style::default().fg(MUTED)),
            ]));
            lines.push(Line::from(Span::styled(
                format!("     {base_url}"),
                Style::default().fg(MUTED),
            )));
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "j/k move · Enter choose · Esc close",
            Style::default().fg(MUTED),
        )));
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(TEAL))
            .title(" Model / provider ");
        frame.render_widget(
            Paragraph::new(Text::from(lines))
                .block(block)
                .wrap(Wrap { trim: false }),
            box_area,
        );
    }
}

fn ship_line(label: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!(" {label:<9}"), Style::default().fg(MUTED)),
        Span::styled(truncate(value, 20), Style::default().fg(Color::White)),
    ])
}

fn safety_label(config: &Config) -> String {
    if config.yolo {
        "YOLO".to_string()
    } else if config.read_only {
        "READ-ONLY".to_string()
    } else if let Some(mode) = &config.safety_mode {
        mode.to_uppercase()
    } else {
        "STANDARD".to_string()
    }
}

fn push_text(lines: &mut Vec<Line<'static>>, text: &str, color: Color) {
    if text.is_empty() {
        lines.push(Line::from(""));
        return;
    }
    for line in text.lines() {
        lines.push(Line::from(Span::styled(
            line.to_string(),
            Style::default().fg(color),
        )));
    }
}

fn wrapped_line_count(lines: &[Line], width: u16) -> usize {
    let width = width.max(1) as usize;
    lines
        .iter()
        .map(|line| {
            let line_width = line.width();
            if line_width == 0 {
                1
            } else {
                line_width.div_ceil(width)
            }
        })
        .sum()
}

fn read_lines<R: Read>(reader: R, tx: mpsc::Sender<AgentEvent>, stderr: bool) {
    for line in BufReader::new(reader).lines() {
        match line {
            Ok(line) => {
                let event = if stderr {
                    AgentEvent::Stderr(line)
                } else {
                    AgentEvent::Line(line)
                };
                if tx.send(event).is_err() {
                    break;
                }
            }
            Err(_) => break,
        }
    }
}

fn sessions_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".keel")
        .join("sessions")
}

fn scan_sessions() -> Vec<SessionMeta> {
    let dir = sessions_dir();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut sessions: Vec<SessionMeta> = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
                return None;
            }
            let metadata = entry.metadata().ok()?;
            let (title, turns) = session_title_and_turns(&path);
            Some(SessionMeta {
                path,
                title,
                turns,
                updated: metadata.modified().unwrap_or(UNIX_EPOCH),
            })
        })
        .collect();
    sessions.sort_by_key(|session| std::cmp::Reverse(session.updated));
    sessions.truncate(100);
    sessions
}

fn session_title_and_turns(path: &Path) -> (String, usize) {
    let Ok(file) = std::fs::File::open(path) else {
        return (file_stem(path), 0);
    };
    let mut title = None;
    let mut turns = 0;
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let Ok(value) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        if value.get("kind").and_then(Value::as_str) == Some("user") {
            turns += 1;
            if title.is_none() {
                if let Some(text) = value
                    .get("data")
                    .and_then(|data| data.get("text"))
                    .and_then(Value::as_str)
                {
                    title = Some(text.trim().replace('\n', " "));
                }
            }
        }
    }
    (
        title
            .filter(|title| !title.is_empty())
            .map(|title| truncate(&title, 36))
            .unwrap_or_else(|| file_stem(path)),
        turns,
    )
}

fn load_session_items(path: &Path) -> Vec<Item> {
    let Ok(file) = std::fs::File::open(path) else {
        return Vec::new();
    };
    let mut items = Vec::new();
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let Ok(value) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let Some(kind) = value.get("kind").and_then(Value::as_str) else {
            continue;
        };
        let data = &value["data"];
        match kind {
            "user" => {
                if let Some(text) = data.get("text").and_then(Value::as_str) {
                    items.push(Item::User(text.to_string()));
                }
            }
            "assistant" => {
                if let Some(text) = data.get("text").and_then(Value::as_str) {
                    if !text.is_empty() {
                        items.push(Item::Assistant(text.to_string()));
                    }
                }
                if let Some(tool_calls) = data.get("tool_calls").and_then(Value::as_array) {
                    for tool_call in tool_calls {
                        let name = tool_call
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("tool");
                        let args = tool_call
                            .get("args")
                            .map(|args| compact_json(args, 180))
                            .unwrap_or_default();
                        items.push(Item::ToolCall {
                            name: name.to_string(),
                            args,
                        });
                    }
                }
            }
            "tool" => {
                if let Some(content) = data.get("content").and_then(Value::as_str) {
                    items.push(Item::ToolResult {
                        id: data
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        output: content.to_string(),
                    });
                }
            }
            _ => {}
        }
    }
    items
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("session")
        .to_string()
}

fn compact_json(value: &Value, max: usize) -> String {
    let text = value.to_string();
    truncate(&text, max)
}

fn truncate(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        return value.to_string();
    }
    let mut truncated: String = value.chars().take(max.saturating_sub(1)).collect();
    truncated.push('…');
    truncated
}

fn relative_time(updated: SystemTime) -> String {
    let elapsed = SystemTime::now()
        .duration_since(updated)
        .unwrap_or_default()
        .as_secs();
    if elapsed < 60 {
        "now".to_string()
    } else if elapsed < 3_600 {
        format!("{}m", elapsed / 60)
    } else if elapsed < 86_400 {
        format!("{}h", elapsed / 3_600)
    } else {
        format!("{}d", elapsed / 86_400)
    }
}

/// Run the terminal cockpit. Restores the terminal before returning.
pub fn run(config: Config) -> Result<()> {
    if !std::io::stdout().is_terminal() {
        anyhow::bail!("keel tui needs an interactive terminal (stdout is not a TTY)");
    }
    let initial_prompt = config.prompt.clone();
    let mut app = App::new(config);
    app.scan_sessions();
    if let Some(prompt) = initial_prompt {
        app.start_turn(prompt);
    }

    let mut terminal = ratatui::init();
    let result: Result<()> = (|| {
        loop {
            app.drain_agent();
            terminal.draw(|frame| app.render(frame))?;
            if event::poll(POLL)? {
                match event::read()? {
                    Event::Key(key) => app.handle_key(key),
                    Event::Paste(text) if app.focus == Focus::Composer => {
                        for character in text.chars() {
                            app.input.insert(app.cursor, character);
                            app.cursor += 1;
                        }
                    }
                    _ => {}
                }
            }
            app.animation = app.animation.wrapping_add(1);
            if app.quit {
                break;
            }
        }
        Ok(())
    })();
    ratatui::restore();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncates_on_character_boundaries() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 6), "hello…");
        assert_eq!(truncate("ääkkösiä", 4), "ääk…");
    }

    #[test]
    fn relative_time_has_compact_buckets() {
        assert_eq!(relative_time(SystemTime::now()), "now");
    }

    #[test]
    fn parses_turn_summary() {
        let value = serde_json::json!({
            "duration_ms": 1500,
            "tool_calls": 3,
            "modified_files": ["src/main.rs"],
            "lines_added": 4,
            "lines_deleted": 1,
            "tests_passed": 12,
            "tests_regressions": 0,
            "estimated_tokens": 99
        });
        let stats = TurnStats::from_json(&value).unwrap();
        assert_eq!(stats.tool_calls, 3);
        assert_eq!(stats.modified_files, vec!["src/main.rs"]);
        assert_eq!(stats.estimated_tokens, Some(99));
    }
}
