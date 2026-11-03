//! Deterministic session flight recorder & time-travel replay engine.
//!
//! Autonomous multi-step coding agents face non-deterministic trajectories,
//! complex refactoring cascades, and sudden hallucinations. When an agent turns
//! off-course on step 11 of a 20-step run, post-mortem debugging using static text
//! logs is slow and incomplete: developers cannot inspect intermediate VFS pre-images
//! or branch alternative hypotheses from historical checkpoints without re-running
//! from scratch.
//!
//! The [`FlightRecorder`] writes an append-only, deterministic event stream
//! (`.keelrec`) capturing exact conversation history, reasoning tokens, tool invocations,
//! intermediate shadow checkpoints, and oracle regression verdicts.
//!
//! The [`ReplayEngine`] provides time-travel capabilities:
//! 1. Step-by-step state inspection across past sessions.
//! 2. Instant VFS restoration to any turn/step pre-image via shadow checkpoints.
//! 3. Time-travel branching / forking: branch a new session from step N with a modified
//!    instruction or model without touching the working tree.

use crate::checkpoint::CheckpointManager;
use crate::{Message, ToolCall};
use anyhow::{bail, Context as _, Result};
use keel_kernel::{Plugin, Scope};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// A discrete, structured event frame recorded into the flight recorder stream.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RecordFrame {
    SessionInit {
        session_id: String,
        timestamp_ms: u64,
        workspace_root: String,
        model: String,
        system_prompt: Option<String>,
    },
    TurnStart {
        turn_index: usize,
        user_prompt: String,
        timestamp_ms: u64,
        pre_checkpoint_id: Option<String>,
    },
    StepExecution {
        turn_index: usize,
        step_index: usize,
        thinking: Option<String>,
        response_text: String,
        tool_calls: Vec<ToolCall>,
        tool_results: Vec<(String, String)>, // (call_id, result_summary)
        post_checkpoint_id: Option<String>,
        duration_ms: u64,
        timestamp_ms: u64,
    },
    VfsMutation {
        turn_index: usize,
        step_index: usize,
        path: String,
        mutation_type: String, // "write", "edit", "patch"
        checkpoint_id: Option<String>,
    },
    OracleVerdict {
        turn_index: usize,
        step_index: usize,
        passed: usize,
        failed: usize,
        regressions: Vec<String>,
    },
    TurnEnd {
        turn_index: usize,
        status: String, // "success", "max_steps", "error", "vetoed"
        duration_ms: u64,
        timestamp_ms: u64,
    },
    SessionEnd {
        total_turns: usize,
        total_steps: usize,
        duration_ms: u64,
        status: String,
    },
}

/// Maximum permitted recording file size (64MB) to prevent unbounded memory consumption.
pub const MAX_RECORDING_FILE_BYTES: u64 = 64 * 1024 * 1024;

/// Maximum permitted length of an individual `.keelrec` JSONL line (16MB).
pub const MAX_RECORDING_LINE_BYTES: usize = 16 * 1024 * 1024;

/// Validate a session identifier to prevent path traversal and unsafe filesystem characters.
pub fn validate_session_id(id: &str) -> Result<()> {
    let trimmed = id.trim();
    if trimmed.is_empty() {
        bail!("session ID must not be empty");
    }
    if trimmed.contains('/') || trimmed.contains('\\') || trimmed.contains("..") {
        bail!("session ID contains forbidden path traversal characters: {id}");
    }
    if !trimmed
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        bail!(
            "session ID contains invalid characters (must be ASCII alphanumeric, '-', '_', or '.'): {id}"
        );
    }
    Ok(())
}

/// Append-only flight recorder logging every event to a deterministic `.keelrec` file.
pub struct FlightRecorder {
    path: PathBuf,
    file: Mutex<File>,
    session_id: String,
    workspace_root: PathBuf,
    model: String,
    start_time_ms: u64,
}

impl FlightRecorder {
    /// Create or open a recording file in `recordings_dir` for `session_id`.
    pub fn create(
        recordings_dir: impl AsRef<Path>,
        session_id: &str,
        workspace_root: PathBuf,
        model: &str,
        system_prompt: Option<&str>,
    ) -> Result<Self> {
        validate_session_id(session_id)?;
        let dir = recordings_dir.as_ref();
        std::fs::create_dir_all(dir)
            .with_context(|| format!("creating recordings directory: {}", dir.display()))?;

        let path = dir.join(format!("{session_id}.keelrec"));
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("opening recording file: {}", path.display()))?;

        let start_time_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let recorder = Self {
            path,
            file: Mutex::new(file),
            session_id: session_id.to_string(),
            workspace_root: workspace_root.clone(),
            model: model.to_string(),
            start_time_ms,
        };

        recorder.record(&RecordFrame::SessionInit {
            session_id: session_id.to_string(),
            timestamp_ms: start_time_ms,
            workspace_root: workspace_root.display().to_string(),
            model: model.to_string(),
            system_prompt: system_prompt.map(String::from),
        })?;

        Ok(recorder)
    }

    /// Record a single frame into the `.keelrec` stream atomically.
    pub fn record(&self, frame: &RecordFrame) -> Result<()> {
        let line = serde_json::to_string(frame).context("serializing RecordFrame")?;
        let mut guard = self.file.lock().expect("flight recorder mutex poisoned");
        writeln!(guard, "{line}").context("writing flight recorder frame")?;
        guard.flush().context("flushing flight recorder frame")?;
        Ok(())
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn recording_path(&self) -> &Path {
        &self.path
    }

    pub fn start_time_ms(&self) -> u64 {
        self.start_time_ms
    }

    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    pub fn model(&self) -> &str {
        &self.model
    }
}

/// Summary overview of a recorded session on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingSummary {
    pub session_id: String,
    pub path: PathBuf,
    pub model: String,
    pub total_turns: usize,
    pub total_steps: usize,
    pub duration_ms: u64,
    pub start_timestamp_ms: u64,
    pub last_status: String,
    pub modified_files: Vec<String>,
}

/// Full loaded flight recorder session containing all frames.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedSession {
    pub session_id: String,
    pub workspace_root: PathBuf,
    pub model: String,
    pub system_prompt: Option<String>,
    pub start_time_ms: u64,
    pub frames: Vec<RecordFrame>,
}

impl RecordedSession {
    /// Load a `.keelrec` recording file from disk.
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self> {
        let p = path.as_ref();
        let meta = std::fs::symlink_metadata(p)
            .with_context(|| format!("checking metadata for {}", p.display()))?;
        if !meta.file_type().is_file() {
            bail!("recording path is not a regular file: {}", p.display());
        }
        if meta.len() > MAX_RECORDING_FILE_BYTES {
            bail!(
                "recording file {} exceeds maximum allowed size ({MAX_RECORDING_FILE_BYTES} bytes)",
                p.display()
            );
        }

        let file = File::open(p).with_context(|| format!("opening recording {}", p.display()))?;
        let reader = BufReader::new(file);

        let mut frames = Vec::new();
        let mut session_id = String::new();
        let mut workspace_root = PathBuf::new();
        let mut model = String::new();
        let mut system_prompt = None;
        let mut start_time_ms = 0;

        for (line_idx, line_res) in reader.lines().enumerate() {
            let line = line_res.with_context(|| format!("reading line {}", line_idx + 1))?;
            if line.len() > MAX_RECORDING_LINE_BYTES {
                bail!(
                    "line {} exceeds maximum allowed length ({MAX_RECORDING_LINE_BYTES} bytes)",
                    line_idx + 1
                );
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let frame: RecordFrame = serde_json::from_str(trimmed)
                .with_context(|| format!("parsing RecordFrame at line {}", line_idx + 1))?;

            if let RecordFrame::SessionInit {
                session_id: sid,
                workspace_root: ws,
                model: m,
                system_prompt: sp,
                timestamp_ms: ts,
            } = &frame
            {
                session_id = sid.clone();
                workspace_root = PathBuf::from(ws);
                model = m.clone();
                system_prompt = sp.clone();
                start_time_ms = *ts;
            }

            frames.push(frame);
        }

        if session_id.is_empty() {
            session_id = path
                .as_ref()
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
        }

        Ok(Self {
            session_id,
            workspace_root,
            model,
            system_prompt,
            start_time_ms,
            frames,
        })
    }

    pub fn total_turns(&self) -> usize {
        self.frames
            .iter()
            .filter(|f| matches!(f, RecordFrame::TurnStart { .. }))
            .count()
    }

    pub fn total_steps(&self) -> usize {
        self.frames
            .iter()
            .filter(|f| matches!(f, RecordFrame::StepExecution { .. }))
            .count()
    }

    pub fn duration_ms(&self) -> u64 {
        if let Some(RecordFrame::SessionEnd { duration_ms, .. }) = self.frames.last() {
            *duration_ms
        } else {
            let last_ts = self
                .frames
                .iter()
                .filter_map(|f| match f {
                    RecordFrame::TurnStart { timestamp_ms, .. } => Some(*timestamp_ms),
                    RecordFrame::StepExecution { timestamp_ms, .. } => Some(*timestamp_ms),
                    RecordFrame::TurnEnd { timestamp_ms, .. } => Some(*timestamp_ms),
                    _ => None,
                })
                .max()
                .unwrap_or(self.start_time_ms);
            last_ts.saturating_sub(self.start_time_ms)
        }
    }

    pub fn user_prompts(&self) -> Vec<String> {
        self.frames
            .iter()
            .filter_map(|f| match f {
                RecordFrame::TurnStart { user_prompt, .. } => Some(user_prompt.clone()),
                _ => None,
            })
            .collect()
    }

    pub fn modified_files(&self) -> Vec<String> {
        let mut files = Vec::new();
        for f in &self.frames {
            if let RecordFrame::VfsMutation { path, .. } = f {
                if !files.contains(path) {
                    files.push(path.clone());
                }
            }
        }
        files
    }

    /// Reconstruct conversation history up to turn `target_turn` and step `target_step`.
    pub fn history_up_to(&self, target_turn: usize, target_step: usize) -> Vec<Message> {
        let mut history = Vec::new();
        if let Some(ref sys) = self.system_prompt {
            history.push(Message::System { text: sys.clone() });
        }

        for f in &self.frames {
            match f {
                RecordFrame::TurnStart {
                    turn_index,
                    user_prompt,
                    ..
                } => {
                    if *turn_index > target_turn {
                        break;
                    }
                    history.push(Message::User {
                        text: user_prompt.clone(),
                    });
                }
                RecordFrame::StepExecution {
                    turn_index,
                    step_index,
                    response_text,
                    tool_calls,
                    tool_results,
                    ..
                } => {
                    if *turn_index > target_turn
                        || (*turn_index == target_turn && *step_index > target_step)
                    {
                        break;
                    }
                    history.push(Message::Assistant {
                        text: response_text.clone(),
                        tool_calls: tool_calls.clone(),
                    });
                    for (id, content) in tool_results {
                        history.push(Message::Tool {
                            id: id.clone(),
                            content: content.clone(),
                        });
                    }
                }
                _ => {}
            }
        }

        history
    }
}

/// Inspection details of a specific turn and step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepInspection {
    pub session_id: String,
    pub turn_index: usize,
    pub step_index: usize,
    pub thinking: Option<String>,
    pub response_text: String,
    pub tool_calls: Vec<ToolCall>,
    pub tool_results: Vec<(String, String)>,
    pub checkpoint_id: Option<String>,
    pub duration_ms: u64,
    pub timestamp_ms: u64,
}

/// Result of a time-travel restore operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestoreReport {
    pub session_id: String,
    pub turn_index: usize,
    pub step_index: usize,
    pub restored_checkpoint_id: String,
    pub restored_files_count: usize,
    pub status: String,
}

/// Result of a time-travel fork operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForkReport {
    pub source_session_id: String,
    pub new_session_id: String,
    pub turn_index: usize,
    pub step_index: usize,
    pub checkpoint_id: String,
    pub conversation_length: usize,
}

/// Replay and time-travel management engine.
pub struct ReplayEngine {
    workspace_root: PathBuf,
    recordings_dir: PathBuf,
}

impl ReplayEngine {
    pub fn new(workspace_root: impl Into<PathBuf>) -> Self {
        let ws = workspace_root.into();
        let rec = ws.join(".keel").join("recordings");
        Self {
            workspace_root: ws,
            recordings_dir: rec,
        }
    }

    pub fn with_recordings_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.recordings_dir = dir.into();
        self
    }

    pub fn recordings_dir(&self) -> &Path {
        &self.recordings_dir
    }

    /// List all recorded sessions in chronological order.
    pub fn list_recordings(&self) -> Result<Vec<RecordingSummary>> {
        if !self.recordings_dir.is_dir() {
            return Ok(Vec::new());
        }

        let mut out = Vec::new();
        for entry in std::fs::read_dir(&self.recordings_dir)
            .with_context(|| format!("reading {}", self.recordings_dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("keelrec") {
                if let Ok(sess) = RecordedSession::load_from_file(&path) {
                    let total_turns = sess.total_turns();
                    let total_steps = sess.total_steps();
                    let duration_ms = sess.duration_ms();
                    let modified_files = sess.modified_files();
                    let last_status = sess
                        .frames
                        .iter()
                        .rev()
                        .find_map(|f| match f {
                            RecordFrame::TurnEnd { status, .. } => Some(status.clone()),
                            RecordFrame::SessionEnd { status, .. } => Some(status.clone()),
                            _ => None,
                        })
                        .unwrap_or_else(|| "in_progress".into());

                    out.push(RecordingSummary {
                        session_id: sess.session_id,
                        path,
                        model: sess.model,
                        total_turns,
                        total_steps,
                        duration_ms,
                        start_timestamp_ms: sess.start_time_ms,
                        last_status,
                        modified_files,
                    });
                }
            }
        }

        out.sort_by_key(|s| std::cmp::Reverse(s.start_timestamp_ms));
        Ok(out)
    }

    /// Load a specific recorded session by ID.
    pub fn load_session(&self, session_id: &str) -> Result<RecordedSession> {
        validate_session_id(session_id)?;
        let path = self.recordings_dir.join(format!("{session_id}.keelrec"));
        if !path.exists() {
            bail!(
                "recording not found for session `{session_id}` at {}",
                path.display()
            );
        }
        RecordedSession::load_from_file(path)
    }

    /// Inspect a given turn and step in a recorded session.
    pub fn inspect_step(
        &self,
        session_id: &str,
        turn_idx: usize,
        step_idx: usize,
    ) -> Result<StepInspection> {
        let session = self.load_session(session_id)?;

        for f in &session.frames {
            if let RecordFrame::StepExecution {
                turn_index,
                step_index,
                thinking,
                response_text,
                tool_calls,
                tool_results,
                post_checkpoint_id,
                duration_ms,
                timestamp_ms,
            } = f
            {
                if *turn_index == turn_idx && *step_index == step_idx {
                    return Ok(StepInspection {
                        session_id: session_id.to_string(),
                        turn_index: *turn_index,
                        step_index: *step_index,
                        thinking: thinking.clone(),
                        response_text: response_text.clone(),
                        tool_calls: tool_calls.clone(),
                        tool_results: tool_results.clone(),
                        checkpoint_id: post_checkpoint_id.clone(),
                        duration_ms: *duration_ms,
                        timestamp_ms: *timestamp_ms,
                    });
                }
            }
        }

        bail!("step {step_idx} not found in turn {turn_idx} for session `{session_id}`")
    }

    /// Restore the workspace filesystem to the snapshot captured at a target turn and step.
    pub fn restore_step(
        &self,
        session_id: &str,
        turn_idx: usize,
        step_idx: usize,
        cp: &CheckpointManager,
    ) -> Result<RestoreReport> {
        let step = self.inspect_step(session_id, turn_idx, step_idx)?;
        let Some(checkpoint_id) = step.checkpoint_id else {
            bail!(
                "step {step_idx} in turn {turn_idx} does not have an associated shadow checkpoint"
            );
        };

        let report = cp
            .restore(&checkpoint_id)
            .with_context(|| format!("restoring checkpoint `{checkpoint_id}`"))?;

        Ok(RestoreReport {
            session_id: session_id.to_string(),
            turn_index: turn_idx,
            step_index: step_idx,
            restored_checkpoint_id: checkpoint_id,
            restored_files_count: report.restored_files + report.removed_files,
            status: "restored".into(),
        })
    }

    /// Fork a new session trajectory from historical step N.
    pub fn fork_step(
        &self,
        session_id: &str,
        turn_idx: usize,
        step_idx: usize,
        new_session_id: &str,
        cp: &CheckpointManager,
    ) -> Result<ForkReport> {
        validate_session_id(session_id)?;
        validate_session_id(new_session_id)?;
        let session = self.load_session(session_id)?;
        let step = self.inspect_step(session_id, turn_idx, step_idx)?;
        let Some(checkpoint_id) = step.checkpoint_id else {
            bail!("cannot fork from step {step_idx}: no shadow checkpoint recorded");
        };

        // 1. Restore workspace state to that exact step
        cp.restore(&checkpoint_id)
            .with_context(|| format!("restoring checkpoint `{checkpoint_id}` for fork"))?;

        // 2. Extract conversation history up to this step
        let history = session.history_up_to(turn_idx, step_idx);

        // 3. Create a new recording initialized with this history
        let new_recorder = FlightRecorder::create(
            &self.recordings_dir,
            new_session_id,
            self.workspace_root.clone(),
            &session.model,
            session.system_prompt.as_deref(),
        )?;

        // Replay historical frames into new recorder
        for h in &history {
            match h {
                Message::User { text } => {
                    new_recorder.record(&RecordFrame::TurnStart {
                        turn_index: 0,
                        user_prompt: text.clone(),
                        timestamp_ms: SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .map(|d| d.as_millis() as u64)
                            .unwrap_or(0),
                        pre_checkpoint_id: Some(checkpoint_id.clone()),
                    })?;
                }
                Message::Assistant { text, tool_calls } => {
                    new_recorder.record(&RecordFrame::StepExecution {
                        turn_index: 0,
                        step_index: 0,
                        thinking: None,
                        response_text: text.clone(),
                        tool_calls: tool_calls.clone(),
                        tool_results: Vec::new(),
                        post_checkpoint_id: Some(checkpoint_id.clone()),
                        duration_ms: 0,
                        timestamp_ms: SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .map(|d| d.as_millis() as u64)
                            .unwrap_or(0),
                    })?;
                }
                _ => {}
            }
        }

        Ok(ForkReport {
            source_session_id: session_id.to_string(),
            new_session_id: new_session_id.to_string(),
            turn_index: turn_idx,
            step_index: step_idx,
            checkpoint_id,
            conversation_length: history.len(),
        })
    }
}

/// Kernel plugin that mounts the [`FlightRecorder`] service into the context.
pub struct FlightRecorderPlugin(pub Arc<FlightRecorder>);

impl Plugin for FlightRecorderPlugin {
    fn name(&self) -> &str {
        "flight-recorder"
    }
    fn apply(&self, scope: &mut Scope) {
        scope.provide::<Arc<FlightRecorder>>(self.0.clone());
    }
}

/// Kernel plugin that mounts the [`ReplayEngine`] service into the context.
pub struct ReplayEnginePlugin(pub Arc<ReplayEngine>);

impl Plugin for ReplayEnginePlugin {
    fn name(&self) -> &str {
        "replay-engine"
    }
    fn apply(&self, scope: &mut Scope) {
        scope.provide::<Arc<ReplayEngine>>(self.0.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flight_recorder_writes_and_reloads_frames() {
        let tmp = std::env::temp_dir().join(format!("keel_rec_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let rec_dir = tmp.join(".keel").join("recordings");
        let recorder = FlightRecorder::create(
            &rec_dir,
            "session-test-01",
            tmp.clone(),
            "kat-coder-v2.5",
            Some("You are an expert coder."),
        )
        .unwrap();

        recorder
            .record(&RecordFrame::TurnStart {
                turn_index: 0,
                user_prompt: "Implement fast sort".into(),
                timestamp_ms: 100,
                pre_checkpoint_id: Some("ckpt-01".into()),
            })
            .unwrap();

        recorder
            .record(&RecordFrame::StepExecution {
                turn_index: 0,
                step_index: 0,
                thinking: Some("Let's write sort.rs".into()),
                response_text: "Creating sort.rs".into(),
                tool_calls: vec![ToolCall {
                    id: "call-1".into(),
                    name: "write_file".into(),
                    args: serde_json::json!({ "path": "sort.rs" }),
                }],
                tool_results: vec![("call-1".into(), "ok".into())],
                post_checkpoint_id: Some("ckpt-02".into()),
                duration_ms: 250,
                timestamp_ms: 350,
            })
            .unwrap();

        recorder
            .record(&RecordFrame::TurnEnd {
                turn_index: 0,
                status: "success".into(),
                duration_ms: 350,
                timestamp_ms: 450,
            })
            .unwrap();

        let loaded =
            RecordedSession::load_from_file(rec_dir.join("session-test-01.keelrec")).unwrap();
        assert_eq!(loaded.session_id, "session-test-01");
        assert_eq!(loaded.model, "kat-coder-v2.5");
        assert_eq!(loaded.total_turns(), 1);
        assert_eq!(loaded.total_steps(), 1);
        assert_eq!(loaded.user_prompts(), vec!["Implement fast sort"]);

        let history = loaded.history_up_to(0, 0);
        assert_eq!(history.len(), 4); // System, User, Assistant, Tool

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn replay_engine_lists_and_inspects_steps() {
        let tmp =
            std::env::temp_dir().join(format!("keel_replay_engine_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let rec_dir = tmp.join(".keel").join("recordings");
        let recorder = FlightRecorder::create(
            &rec_dir,
            "session-alpha",
            tmp.clone(),
            "deepseek-chat",
            None,
        )
        .unwrap();

        recorder
            .record(&RecordFrame::TurnStart {
                turn_index: 0,
                user_prompt: "Refactor database".into(),
                timestamp_ms: 1000,
                pre_checkpoint_id: None,
            })
            .unwrap();

        recorder
            .record(&RecordFrame::StepExecution {
                turn_index: 0,
                step_index: 0,
                thinking: Some("Reviewing schema".into()),
                response_text: "Checking schema.rs".into(),
                tool_calls: vec![ToolCall {
                    id: "call-99".into(),
                    name: "read_file".into(),
                    args: serde_json::json!({ "path": "schema.rs" }),
                }],
                tool_results: vec![("call-99".into(), "table users".into())],
                post_checkpoint_id: Some("ckpt-alpha-1".into()),
                duration_ms: 50,
                timestamp_ms: 1050,
            })
            .unwrap();

        let engine = ReplayEngine::new(&tmp).with_recordings_dir(&rec_dir);
        let list = engine.list_recordings().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].session_id, "session-alpha");

        let step = engine.inspect_step("session-alpha", 0, 0).unwrap();
        assert_eq!(step.tool_calls[0].name, "read_file");
        assert_eq!(step.checkpoint_id, Some("ckpt-alpha-1".into()));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn session_id_validation_and_path_traversal_defense() {
        assert!(validate_session_id("session-123_abc.def").is_ok());
        assert!(validate_session_id("").is_err());
        assert!(validate_session_id("../../etc/passwd").is_err());
        assert!(validate_session_id("session/child").is_err());
        assert!(validate_session_id("session\\child").is_err());
        assert!(validate_session_id("session;rm -rf").is_err());

        let tmp = std::env::temp_dir().join(format!("keel_sec_replay_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let engine = ReplayEngine::new(&tmp);
        assert!(engine.load_session("../escaped").is_err());
        assert!(engine
            .fork_step(
                "valid-src",
                0,
                0,
                "../evil-fork",
                &CheckpointManager::new(tmp.clone())
            )
            .is_err());

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
