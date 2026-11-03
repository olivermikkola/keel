//! Branching Exploration Tree & Multi-Step Backtracking in Shadow VFS (`keel tree`).
//!
//! Provides a non-linear decision DAG enabling autonomous agents to explore
//! alternative implementation paths, compare diagnostic deltas, backtrack to
//! intermediate decision points, and prune dead ends without corrupting git HEAD.

use crate::checkpoint::CheckpointManager;
use crate::sandbox::ToolRegistry;
use crate::{Tool, ToolSpec};
use anyhow::{bail, Context as _, Result};
use keel_kernel::{Plugin, Scope};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// Status lifecycle of a decision node in the exploration tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NodeStatus {
    #[default]
    Active,
    Backtracked,
    Pruned,
    Accepted,
}

impl NodeStatus {
    pub fn icon(&self) -> &'static str {
        match self {
            Self::Active => "●",
            Self::Backtracked => "○",
            Self::Pruned => "✗",
            Self::Accepted => "★",
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Backtracked => "backtracked",
            Self::Pruned => "pruned",
            Self::Accepted => "accepted",
        }
    }
}

/// A decision node in the exploration tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode {
    pub id: String,
    pub parent_id: Option<String>,
    pub depth: usize,
    pub label: String,
    pub checkpoint_id: Option<String>,
    pub status: NodeStatus,
    pub children: Vec<String>,
    pub created_at_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

/// Exploration Tree state containing all nodes and active navigation pointer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationTree {
    pub root_id: String,
    pub current_id: String,
    pub nodes: HashMap<String, TreeNode>,
    next_node_num: usize,
    #[serde(skip)]
    root_path: PathBuf,
}

impl ExplorationTree {
    /// Initialize or load tree from `.keel/tree.json` or `.git/keel_tree.json`.
    pub fn load_or_init(root: &Path, cp: &CheckpointManager) -> Result<Self> {
        let path = tree_file_path(root);
        if path.exists() {
            if let Ok(content) = fs::read_to_string(&path) {
                if let Ok(mut tree) = serde_json::from_str::<ExplorationTree>(&content) {
                    tree.root_path = root.to_path_buf();
                    return Ok(tree);
                }
            }
        }

        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let root_snap = cp.save("Initial workspace state").ok().map(|s| s.id);

        let root_node = TreeNode {
            id: "root".to_string(),
            parent_id: None,
            depth: 0,
            label: "Initial workspace state".to_string(),
            checkpoint_id: root_snap,
            status: NodeStatus::Active,
            children: Vec::new(),
            created_at_ms: now_ms,
            notes: None,
        };

        let mut nodes = HashMap::new();
        nodes.insert("root".to_string(), root_node);

        let tree = Self {
            root_id: "root".to_string(),
            current_id: "root".to_string(),
            nodes,
            next_node_num: 1,
            root_path: root.to_path_buf(),
        };

        tree.save()?;
        Ok(tree)
    }

    /// Persist tree metadata to disk.
    pub fn save(&self) -> Result<()> {
        let p = tree_file_path(&self.root_path);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(self)?;
        fs::write(&p, data)?;
        Ok(())
    }

    /// Fork a new decision branch from the current active node.
    pub fn fork(&mut self, label: &str, cp: &CheckpointManager) -> Result<TreeNode> {
        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Snapshot current state of active node at the moment of branching
        let cur_label = self
            .nodes
            .get(&self.current_id)
            .map(|n| n.label.clone())
            .unwrap_or_else(|| "current".to_string());
        let parent_snap = cp.save(&format!("Tree node: {cur_label}"))?;
        if let Some(cur) = self.nodes.get_mut(&self.current_id) {
            cur.checkpoint_id = Some(parent_snap.id.clone());
        }

        let current_depth = self
            .nodes
            .get(&self.current_id)
            .map(|n| n.depth)
            .unwrap_or(0);

        let new_id = format!("node_{}", self.next_node_num);
        self.next_node_num += 1;

        let new_node = TreeNode {
            id: new_id.clone(),
            parent_id: Some(self.current_id.clone()),
            depth: current_depth + 1,
            label: label.to_string(),
            checkpoint_id: Some(parent_snap.id),
            status: NodeStatus::Active,
            children: Vec::new(),
            created_at_ms: now_ms,
            notes: None,
        };

        if let Some(parent) = self.nodes.get_mut(&self.current_id) {
            parent.children.push(new_id.clone());
        }

        self.nodes.insert(new_id.clone(), new_node.clone());
        self.current_id = new_id;
        self.save()?;

        Ok(new_node)
    }

    /// Backtrack to the parent node of the currently active node.
    pub fn backtrack(&mut self, cp: &CheckpointManager) -> Result<TreeNode> {
        let (parent_id, cur_label) = {
            let cur = self
                .nodes
                .get(&self.current_id)
                .context("current node not found")?;
            match &cur.parent_id {
                Some(pid) => (pid.clone(), cur.label.clone()),
                None => bail!("cannot backtrack: already at root node"),
            }
        };

        // Snapshot current node so work on it is preserved if later switched to
        if let Ok(cur_snap) = cp.save(&format!("Tree node: {cur_label}")) {
            if let Some(cur) = self.nodes.get_mut(&self.current_id) {
                cur.checkpoint_id = Some(cur_snap.id);
                cur.status = NodeStatus::Backtracked;
            }
        } else if let Some(cur) = self.nodes.get_mut(&self.current_id) {
            cur.status = NodeStatus::Backtracked;
        }

        // Restore parent's checkpoint
        let parent_node = self
            .nodes
            .get(&parent_id)
            .cloned()
            .context("parent node not found")?;

        if let Some(ref ckpt) = parent_node.checkpoint_id {
            cp.restore(ckpt)?;
        }

        self.current_id = parent_id;
        self.save()?;

        Ok(parent_node)
    }

    /// Switch workspace to a specific node by ID.
    pub fn switch_to(&mut self, target_id: &str, cp: &CheckpointManager) -> Result<TreeNode> {
        if target_id == self.current_id {
            return self
                .nodes
                .get(target_id)
                .cloned()
                .context("current node not found");
        }

        // Save current node's state before leaving it
        let cur_label = self
            .nodes
            .get(&self.current_id)
            .map(|n| n.label.clone())
            .unwrap_or_else(|| "current".to_string());
        if let Ok(cur_snap) = cp.save(&format!("Tree node: {cur_label}")) {
            if let Some(cur) = self.nodes.get_mut(&self.current_id) {
                cur.checkpoint_id = Some(cur_snap.id);
            }
        }

        let target = self
            .nodes
            .get(target_id)
            .cloned()
            .with_context(|| format!("node `{target_id}` not found in exploration tree"))?;

        if let Some(ref ckpt) = target.checkpoint_id {
            cp.restore(ckpt)?;
        }

        self.current_id = target.id.clone();
        self.save()?;

        Ok(target)
    }

    /// Recursively prune a node and all of its descendants.
    pub fn prune(&mut self, target_id: &str) -> Result<usize> {
        if !self.nodes.contains_key(target_id) {
            bail!("node `{target_id}` not found in exploration tree");
        }

        let mut pruned_count = 0;
        let mut stack = vec![target_id.to_string()];

        while let Some(id) = stack.pop() {
            if let Some(node) = self.nodes.get_mut(&id) {
                if node.status != NodeStatus::Pruned {
                    node.status = NodeStatus::Pruned;
                    pruned_count += 1;
                }
                stack.extend(node.children.clone());
            }
        }

        // If current was pruned, reset active to root
        if self.current_id == target_id {
            self.current_id = self.root_id.clone();
        }

        self.save()?;
        Ok(pruned_count)
    }

    /// Mark a node as Accepted and active.
    pub fn accept(&mut self, target_id: &str) -> Result<TreeNode> {
        let node = self
            .nodes
            .get_mut(target_id)
            .with_context(|| format!("node `{target_id}` not found"))?;

        node.status = NodeStatus::Accepted;
        self.current_id = target_id.to_string();
        let cloned = node.clone();
        self.save()?;
        Ok(cloned)
    }

    /// Render visual ASCII tree diagram.
    pub fn render_ascii(&self) -> String {
        let mut out = String::new();
        if let Some(root) = self.nodes.get(&self.root_id) {
            self.render_node_recursive(root, "", true, &mut out);
        }
        out
    }

    fn render_node_recursive(
        &self,
        node: &TreeNode,
        prefix: &str,
        is_last: bool,
        out: &mut String,
    ) {
        let is_current = node.id == self.current_id;
        let current_marker = if is_current { " <-- [CURRENT]" } else { "" };
        let connector = if node.id == self.root_id {
            ""
        } else if is_last {
            "└── "
        } else {
            "├── "
        };

        let ckpt_tag = node
            .checkpoint_id
            .as_deref()
            .map(|c| format!(" ({})", c))
            .unwrap_or_default();

        out.push_str(&format!(
            "{}{}{} [{}] \"{}\" [depth {}]{}{}\n",
            prefix,
            connector,
            node.status.icon(),
            node.id,
            node.label,
            node.depth,
            ckpt_tag,
            current_marker
        ));

        let next_prefix = if node.id == self.root_id {
            String::new()
        } else if is_last {
            format!("{prefix}    ")
        } else {
            format!("{prefix}│   ")
        };

        for (idx, child_id) in node.children.iter().enumerate() {
            if let Some(child) = self.nodes.get(child_id) {
                let child_is_last = idx + 1 == node.children.len();
                self.render_node_recursive(child, &next_prefix, child_is_last, out);
            }
        }
    }
}

fn tree_file_path(root: &Path) -> PathBuf {
    let dot_git = root.join(".git");
    if dot_git.is_dir() {
        dot_git.join("keel_tree.json")
    } else {
        root.join(".keel").join("tree.json")
    }
}

// ---------------------------------------------------------------------------
// Kernel Plugin & Agent Tools
// ---------------------------------------------------------------------------

pub struct TreePlugin {
    pub root: PathBuf,
}

impl Plugin for TreePlugin {
    fn name(&self) -> &str {
        "exploration-tree"
    }

    fn apply(&self, scope: &mut Scope) {
        let cp = Arc::new(CheckpointManager::new(self.root.clone()));
        let tree = match ExplorationTree::load_or_init(&self.root, &cp) {
            Ok(t) => Arc::new(Mutex::new(t)),
            Err(_) => return,
        };

        scope.provide(tree.clone());

        let reg = scope
            .context()
            .service::<ToolRegistry>()
            .map(|r| (*r).clone())
            .unwrap_or_default();

        reg.register(Arc::new(TreeForkTool {
            tree: tree.clone(),
            cp: cp.clone(),
        }));
        reg.register(Arc::new(TreeBacktrackTool {
            tree: tree.clone(),
            cp: cp.clone(),
        }));
        reg.register(Arc::new(TreeSwitchTool {
            tree: tree.clone(),
            cp,
        }));
        reg.register(Arc::new(TreeStatusTool { tree }));

        scope.provide(reg);
    }
}

pub struct TreeForkTool {
    pub tree: Arc<Mutex<ExplorationTree>>,
    pub cp: Arc<CheckpointManager>,
}

impl Tool for TreeForkTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "tree_fork".into(),
            description: "Fork a new decision branch in the exploration tree to explore an alternative hypothesis or speculative edit path.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "label": { "type": "string", "description": "Short description of the branch hypothesis" }
                },
                "required": ["label"]
            }),
        }
    }

    fn mutating(&self) -> bool {
        false
    }

    fn run(&self, a: &Value) -> Result<String> {
        let label = a["label"].as_str().context("missing `label`")?;
        let mut tree = self.tree.lock().unwrap();
        let node = tree.fork(label, &self.cp)?;
        Ok(format!(
            "Forked new branch `{}` (node `{}`) from `{}`.",
            node.label,
            node.id,
            node.parent_id.as_deref().unwrap_or("none")
        ))
    }
}

pub struct TreeBacktrackTool {
    pub tree: Arc<Mutex<ExplorationTree>>,
    pub cp: Arc<CheckpointManager>,
}

impl Tool for TreeBacktrackTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "tree_backtrack".into(),
            description: "Backtrack to the parent decision node, automatically reverting workspace modifications back to the parent snapshot.".into(),
            parameters: json!({ "type": "object" }),
        }
    }

    fn mutating(&self) -> bool {
        true
    }

    fn run(&self, _a: &Value) -> Result<String> {
        let mut tree = self.tree.lock().unwrap();
        let parent = tree.backtrack(&self.cp)?;
        Ok(format!(
            "Backtracked to parent node `{}` (\"{}\"). Workspace restored.",
            parent.id, parent.label
        ))
    }
}

pub struct TreeSwitchTool {
    pub tree: Arc<Mutex<ExplorationTree>>,
    pub cp: Arc<CheckpointManager>,
}

impl Tool for TreeSwitchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "tree_switch".into(),
            description: "Switch workspace state to an arbitrary node in the exploration tree."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "node_id": { "type": "string", "description": "ID of the target node (e.g. node_1, root)" }
                },
                "required": ["node_id"]
            }),
        }
    }

    fn mutating(&self) -> bool {
        true
    }

    fn run(&self, a: &Value) -> Result<String> {
        let node_id = a["node_id"].as_str().context("missing `node_id`")?;
        let mut tree = self.tree.lock().unwrap();
        let target = tree.switch_to(node_id, &self.cp)?;
        Ok(format!(
            "Switched to node `{}` (\"{}\"). Workspace restored.",
            target.id, target.label
        ))
    }
}

pub struct TreeStatusTool {
    pub tree: Arc<Mutex<ExplorationTree>>,
}

impl Tool for TreeStatusTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "tree_status".into(),
            description: "View the exploration tree visual ASCII diagram and node states.".into(),
            parameters: json!({ "type": "object" }),
        }
    }

    fn mutating(&self) -> bool {
        false
    }

    fn run(&self, _a: &Value) -> Result<String> {
        let tree = self.tree.lock().unwrap();
        let diagram = tree.render_ascii();
        Ok(format!(
            "### Exploration Tree (current: `{}`):\n\n```\n{}\n```",
            tree.current_id, diagram
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn init_test_git_repo(path: &Path) {
        Command::new("git")
            .args(["init", "-q"])
            .current_dir(path)
            .status()
            .unwrap();
        Command::new("git")
            .args(["config", "user.name", "Keel Test"])
            .current_dir(path)
            .status()
            .unwrap();
        Command::new("git")
            .args(["config", "user.email", "test@keel.dev"])
            .current_dir(path)
            .status()
            .unwrap();
        fs::write(path.join("init.txt"), "initial").unwrap();
        Command::new("git")
            .args(["add", "init.txt"])
            .current_dir(path)
            .status()
            .unwrap();
        Command::new("git")
            .args(["commit", "-m", "init", "-q"])
            .current_dir(path)
            .status()
            .unwrap();
    }

    #[test]
    fn exploration_tree_fork_backtrack_and_switch() {
        let tmp = std::env::temp_dir().join(format!("keel-tree-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        init_test_git_repo(&tmp);

        let cp = CheckpointManager::new(tmp.clone());
        let mut tree = ExplorationTree::load_or_init(&tmp, &cp).unwrap();
        assert_eq!(tree.current_id, "root");

        // 1. Fork Branch A
        let node_a = tree.fork("Attempt A: regex", &cp).unwrap();
        assert_eq!(node_a.id, "node_1");
        assert_eq!(tree.current_id, "node_1");
        fs::write(tmp.join("file_a.txt"), "branch a content").unwrap();
        assert!(tmp.join("file_a.txt").exists());

        // 2. Fork nested Branch A.1
        let node_a1 = tree.fork("Attempt A.1: nested fix", &cp).unwrap();
        assert_eq!(node_a1.id, "node_2");
        assert_eq!(tree.current_id, "node_2");
        fs::write(tmp.join("file_a1.txt"), "branch a1 content").unwrap();
        assert!(tmp.join("file_a1.txt").exists());

        // 3. Backtrack back to Branch A
        let back_a = tree.backtrack(&cp).unwrap();
        assert_eq!(back_a.id, "node_1");
        assert_eq!(tree.current_id, "node_1");
        assert!(tmp.join("file_a.txt").exists());
        assert!(!tmp.join("file_a1.txt").exists()); // Reverted!

        // 4. Backtrack to root
        let back_root = tree.backtrack(&cp).unwrap();
        assert_eq!(back_root.id, "root");
        assert_eq!(tree.current_id, "root");
        assert!(!tmp.join("file_a.txt").exists()); // Reverted!

        // 5. Fork Branch B
        let node_b = tree.fork("Attempt B: state machine", &cp).unwrap();
        assert_eq!(node_b.id, "node_3");
        assert_eq!(tree.current_id, "node_3");
        fs::write(tmp.join("file_b.txt"), "branch b content").unwrap();
        assert!(tmp.join("file_b.txt").exists());

        // 6. Switch directly back to Branch A (node_1)
        tree.switch_to("node_1", &cp).unwrap();
        assert_eq!(tree.current_id, "node_1");
        assert!(tmp.join("file_a.txt").exists());
        assert!(!tmp.join("file_b.txt").exists());

        // 7. Render ASCII diagram
        let diagram = tree.render_ascii();
        assert!(diagram.contains("root"));
        assert!(diagram.contains("node_1"));
        assert!(diagram.contains("node_3"));
        assert!(diagram.contains("<-- [CURRENT]"));

        let _ = fs::remove_dir_all(tmp);
    }
}
