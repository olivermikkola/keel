//! Repository Architecture Map & PageRank Centrality Index (`keel repomap`).
//!
//! Analyzes cross-file symbol definitions and reference topologies to build a global
//! code dependency graph. Applies power-iteration PageRank (with optional personalization / focus bias)
//! to score architectural centrality, and renders an ultra-compact, token-budgeted structural map
//! for LLM context injection.

use crate::symbols::{parse_structural_outline, FileOutline, SymbolKind};
use crate::{Tool, ToolSpec};
use anyhow::Result;
use keel_kernel::{Plugin, Scope};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// 1. Reference Graph
// ---------------------------------------------------------------------------

/// A directed graph of workspace files and cross-file symbol references.
///
/// An edge `A → B` with weight `w` indicates that file `A` references
/// identifiers defined in file `B` `w` times.
#[derive(Debug, Clone, Default)]
pub struct ReferenceGraph {
    /// File paths in the graph.
    nodes: Vec<String>,
    /// Lookup from path to node index.
    nodes_to_idx: HashMap<String, usize>,
    /// Outgoing edges: `out_edges[src_idx]` -> `Vec<(dst_idx, weight)>`.
    out_edges: Vec<Vec<(usize, usize)>>,
    /// Incoming edges: `in_edges[dst_idx]` -> `Vec<(src_idx, weight)>`.
    in_edges: Vec<Vec<(usize, usize)>>,
    /// Defined symbol names per file (normalized lower-case).
    defined_symbols: Vec<BTreeSet<String>>,
    /// Raw file contents for cross-reference scanning.
    contents: Vec<String>,
}

impl ReferenceGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a file with its defined symbol names and text content.
    pub fn add_file(&mut self, path: &str, defined_symbols: Vec<String>, content: &str) {
        let idx = match self.nodes_to_idx.get(path) {
            Some(&i) => i,
            None => {
                let i = self.nodes.len();
                self.nodes.push(path.to_string());
                self.nodes_to_idx.insert(path.to_string(), i);
                self.out_edges.push(Vec::new());
                self.in_edges.push(Vec::new());
                self.defined_symbols.push(BTreeSet::new());
                self.contents.push(String::new());
                i
            }
        };

        let sym_set = &mut self.defined_symbols[idx];
        for s in defined_symbols {
            sym_set.insert(s.to_lowercase());
        }
        self.contents[idx] = content.to_string();
    }

    /// Recompute all directed edges across files.
    pub fn rebuild_edges(&mut self) {
        for edge_list in &mut self.out_edges {
            edge_list.clear();
        }
        for edge_list in &mut self.in_edges {
            edge_list.clear();
        }

        let n = self.nodes.len();
        if n <= 1 {
            return;
        }

        // Global symbol lookup: symbol_name -> Vec<dst_file_idx>
        let mut symbol_to_defs: HashMap<String, Vec<usize>> = HashMap::new();
        for (file_idx, sym_set) in self.defined_symbols.iter().enumerate() {
            for sym in sym_set {
                symbol_to_defs
                    .entry(sym.clone())
                    .or_default()
                    .push(file_idx);
            }
        }

        // For each file, extract identifiers and match against global symbol definitions
        for src_idx in 0..n {
            let identifiers = extract_identifiers(&self.contents[src_idx]);
            let mut id_counts: HashMap<String, usize> = HashMap::new();
            for id in identifiers {
                *id_counts.entry(id).or_default() += 1;
            }

            let mut targets_weight: HashMap<usize, usize> = HashMap::new();
            for (id, count) in id_counts {
                if let Some(target_indices) = symbol_to_defs.get(&id) {
                    for &dst_idx in target_indices {
                        // Skip self-references
                        if dst_idx != src_idx {
                            *targets_weight.entry(dst_idx).or_default() += count;
                        }
                    }
                }
            }

            for (dst_idx, weight) in targets_weight {
                self.out_edges[src_idx].push((dst_idx, weight));
                self.in_edges[dst_idx].push((src_idx, weight));
            }
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn nodes(&self) -> &[String] {
        &self.nodes
    }

    pub fn nodes_to_idx(&self) -> &HashMap<String, usize> {
        &self.nodes_to_idx
    }

    pub fn out_degree(&self, file: &str) -> usize {
        match self.nodes_to_idx.get(file) {
            Some(&i) => self.out_edges[i].len(),
            None => 0,
        }
    }

    pub fn in_degree(&self, file: &str) -> usize {
        match self.nodes_to_idx.get(file) {
            Some(&i) => self.in_edges[i].len(),
            None => 0,
        }
    }

    pub fn in_neighbors(&self, file: &str) -> Vec<(String, usize)> {
        match self.nodes_to_idx.get(file) {
            Some(&i) => {
                let mut res: Vec<(String, usize)> = self.in_edges[i]
                    .iter()
                    .map(|&(src, w)| (self.nodes[src].clone(), w))
                    .collect();
                res.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                res
            }
            None => Vec::new(),
        }
    }

    pub fn out_neighbors(&self, file: &str) -> Vec<(String, usize)> {
        match self.nodes_to_idx.get(file) {
            Some(&i) => {
                let mut res: Vec<(String, usize)> = self.out_edges[i]
                    .iter()
                    .map(|&(dst, w)| (self.nodes[dst].clone(), w))
                    .collect();
                res.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                res
            }
            None => Vec::new(),
        }
    }
}

/// Tokenizer that extracts word-like code identifiers.
fn extract_identifiers(content: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::with_capacity(32);
    for ch in content.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            cur.push(ch);
        } else {
            if cur.len() >= 2 && !cur.chars().all(|c| c.is_ascii_digit()) {
                tokens.push(cur.to_lowercase());
            }
            cur.clear();
        }
    }
    if cur.len() >= 2 && !cur.chars().all(|c| c.is_ascii_digit()) {
        tokens.push(cur.to_lowercase());
    }
    tokens
}

// ---------------------------------------------------------------------------
// 2. PageRank Centrality Solver
// ---------------------------------------------------------------------------

/// Computes architectural centrality PageRank scores across files in `graph`.
///
/// Parameters:
/// - `damping`: probability of following an edge vs random jump (standard 0.85).
/// - `max_iters`: maximum power-iteration cycles.
/// - `tol`: convergence threshold on L1 norm difference.
/// - `personal_bias`: optional weights biasing the random teleportation vector.
pub fn compute_pagerank(
    graph: &ReferenceGraph,
    damping: f64,
    max_iters: usize,
    tol: f64,
    personal_bias: Option<&HashMap<String, f64>>,
) -> HashMap<String, f64> {
    let n = graph.node_count();
    if n == 0 {
        return HashMap::new();
    }
    if n == 1 {
        let mut res = HashMap::new();
        res.insert(graph.nodes()[0].clone(), 1.0);
        return res;
    }

    let damping = damping.clamp(0.0, 1.0);

    // Construct personalized teleportation distribution vector
    let mut p = vec![1.0 / (n as f64); n];
    if let Some(bias) = personal_bias {
        let mut total_bias = 0.0;
        let mut custom = vec![0.0; n];
        for (path, &w) in bias {
            if let Some(&i) = graph.nodes_to_idx.get(path) {
                if w > 0.0 {
                    custom[i] = w;
                    total_bias += w;
                }
            }
        }
        if total_bias > 0.0 {
            for i in 0..n {
                p[i] = custom[i] / total_bias;
            }
        }
    }

    let mut rank = p.clone();
    let mut new_rank = vec![0.0; n];

    for _ in 0..max_iters {
        // Collect rank from dangling/sink nodes (nodes with 0 out-degree)
        let mut sink_rank = 0.0;
        for (i, &r) in rank.iter().enumerate() {
            if graph.out_edges[i].is_empty() {
                sink_rank += r;
            }
        }

        // Compute new rank distribution
        for dst in 0..n {
            // Incoming rank from neighbors
            let mut inbound_sum = 0.0;
            for &(src, _) in &graph.in_edges[dst] {
                let out_len = graph.out_edges[src].len();
                if out_len > 0 {
                    inbound_sum += rank[src] / (out_len as f64);
                }
            }

            new_rank[dst] =
                (1.0 - damping) * p[dst] + damping * (inbound_sum + (sink_rank * p[dst]));
        }

        // L1 convergence delta
        let delta: f64 = rank
            .iter()
            .zip(new_rank.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        rank.copy_from_slice(&new_rank);

        if delta < tol {
            break;
        }
    }

    // Normalize so sum == 1.0
    let sum: f64 = rank.iter().sum();
    let norm = if sum > 0.0 { sum } else { 1.0 };

    let mut result = HashMap::with_capacity(n);
    for (i, path) in graph.nodes().iter().enumerate() {
        result.insert(path.clone(), rank[i] / norm);
    }
    result
}

// ---------------------------------------------------------------------------
// 3. RepoMap Architecture
// ---------------------------------------------------------------------------

/// Scored repository architecture map for LLM context conditioning.
#[derive(Clone, Debug)]
pub struct RepoMap {
    pub root: PathBuf,
    pub outlines: HashMap<String, FileOutline>,
    pub graph: ReferenceGraph,
    pub ranks: HashMap<String, f64>,
}

impl RepoMap {
    /// Construct a `RepoMap` from a collection of relative file paths and raw contents.
    pub fn from_files(root: PathBuf, files: Vec<(String, String)>) -> Self {
        let mut outlines = HashMap::with_capacity(files.len());
        let mut graph = ReferenceGraph::new();

        for (path, content) in &files {
            let outline = parse_structural_outline(path, content);
            let defined_names: Vec<String> =
                outline.symbols.iter().map(|s| s.name.clone()).collect();
            outlines.insert(path.clone(), outline);
            graph.add_file(path, defined_names, content);
        }

        graph.rebuild_edges();
        let ranks = compute_pagerank(&graph, 0.85, 100, 1e-6, None);

        Self {
            root,
            outlines,
            graph,
            ranks,
        }
    }

    /// Discover and build a `RepoMap` from the disk workspace, ignoring build/vcs artifacts.
    pub fn discover(root: &Path) -> Self {
        let mut pairs = Vec::new();
        let ignored: HashSet<&str> = [
            "target",
            ".git",
            "node_modules",
            "dist",
            "build",
            ".gemini",
            ".venv",
            "venv",
            "__pycache__",
        ]
        .into_iter()
        .collect();

        fn walk(dir: &Path, root: &Path, ignored: &HashSet<&str>, out: &mut Vec<(String, String)>) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let file_name = entry.file_name().to_string_lossy().to_string();
                if ignored.contains(file_name.as_str()) {
                    continue;
                }
                if path.is_dir() {
                    walk(&path, root, ignored, out);
                } else if path.is_file() {
                    let ext = path
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("")
                        .to_lowercase();
                    if matches!(
                        ext.as_str(),
                        "rs" | "py" | "ts" | "js" | "go" | "c" | "cpp" | "h" | "hpp" | "sh"
                    ) {
                        if let Ok(content) = std::fs::read_to_string(&path) {
                            if let Ok(rel) = path.strip_prefix(root) {
                                out.push((rel.to_string_lossy().to_string(), content));
                            }
                        }
                    }
                }
            }
        }

        walk(root, root, &ignored, &mut pairs);
        Self::from_files(root.to_path_buf(), pairs)
    }

    /// Render a compact Markdown tree respecting `max_tokens` (estimated ~3.8 chars/tok).
    ///
    /// If `focus` is provided, computes personalized PageRank biased toward those paths.
    pub fn render(&self, max_tokens: usize, focus: Option<&[String]>) -> String {
        let max_chars = ((max_tokens as f32) * 3.8).max(200.0) as usize;
        let effective_ranks = if let Some(focus_paths) = focus {
            if !focus_paths.is_empty() {
                let mut bias = HashMap::new();
                for fp in focus_paths {
                    bias.insert(fp.clone(), 5.0);
                }
                compute_pagerank(&self.graph, 0.85, 100, 1e-6, Some(&bias))
            } else {
                self.ranks.clone()
            }
        } else {
            self.ranks.clone()
        };

        // Sort files by rank descending
        let mut sorted_files: Vec<(&String, f64)> =
            effective_ranks.iter().map(|(k, &v)| (k, v)).collect();
        sorted_files.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut out = String::new();
        out.push_str("### Repository Architecture Map (Ranked by Centrality)\n\n");

        let mut omitted = 0;
        for (path, rank) in sorted_files {
            let outline = self.outlines.get(path);
            let mut file_block = format!("- `{path}` (centrality: {rank:.4})\n");

            if let Some(ol) = outline {
                let mut count = 0;
                for sym in &ol.symbols {
                    if matches!(
                        sym.kind,
                        SymbolKind::Struct
                            | SymbolKind::Trait
                            | SymbolKind::Class
                            | SymbolKind::Interface
                            | SymbolKind::Function
                            | SymbolKind::Enum
                    ) {
                        file_block.push_str(&format!("  * `{}`\n", sym.signature));
                        count += 1;
                        if count >= 6 {
                            break;
                        }
                    }
                }
            }

            if out.chars().count() + file_block.chars().count() <= max_chars {
                out.push_str(&file_block);
            } else {
                omitted += 1;
            }
        }

        if omitted > 0 {
            out.push_str(&format!(
                "\n…[{omitted} additional lower-ranked files omitted to fit token budget]…\n"
            ));
        }

        out.trim_end().to_string()
    }
}

// ---------------------------------------------------------------------------
// 4. Tool & Kernel Plugin
// ---------------------------------------------------------------------------

pub struct RepoMapTool {
    pub map: Arc<RepoMap>,
}

impl Tool for RepoMapTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "repomap".into(),
            description: "Retrieve a compact, PageRank-centrality ranked structural map of repository files and primary symbols.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "max_tokens": {
                        "type": "integer",
                        "description": "Token budget for the generated repository map (default 1024)",
                        "default": 1024
                    },
                    "focus": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional list of target files or keywords to personalize the PageRank focus"
                    }
                }
            }),
        }
    }

    fn mutating(&self) -> bool {
        false
    }

    fn run(&self, args: &Value) -> Result<String> {
        let max_tokens = args
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .map(|u| u as usize)
            .unwrap_or(1024);

        let focus: Option<Vec<String>> = args.get("focus").and_then(|v| {
            v.as_array().map(|arr| {
                arr.iter()
                    .filter_map(|val| val.as_str().map(String::from))
                    .collect()
            })
        });

        Ok(self.map.render(max_tokens, focus.as_deref()))
    }
}

pub struct RepoMapPlugin {
    pub root: PathBuf,
}

impl RepoMapPlugin {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }
}

impl Plugin for RepoMapPlugin {
    fn name(&self) -> &str {
        "repomap"
    }

    fn apply(&self, scope: &mut Scope) {
        let map = Arc::new(RepoMap::discover(&self.root));
        scope.provide(map.clone());
        if let Some(registry) = scope.context().service::<crate::sandbox::ToolRegistry>() {
            registry.register(Arc::new(RepoMapTool { map }));
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_identifiers() {
        let content = "fn compute_pagerank(graph: &ReferenceGraph) -> HashMap<String, f64> { 123 }";
        let tokens = extract_identifiers(content);
        assert!(tokens.contains(&"compute_pagerank".to_string()));
        assert!(tokens.contains(&"graph".to_string()));
        assert!(tokens.contains(&"referencegraph".to_string()));
        assert!(tokens.contains(&"hashmap".to_string()));
        // Pure digits are filtered out
        assert!(!tokens.contains(&"123".to_string()));
    }

    #[test]
    fn test_reference_graph_construction() {
        let mut g = ReferenceGraph::new();
        g.add_file(
            "src/utils.rs",
            vec!["format_date".into(), "parse_int".into()],
            "pub fn format_date() {} pub fn parse_int() {}",
        );
        g.add_file(
            "src/main.rs",
            vec!["main".into()],
            "fn main() { format_date(); }",
        );

        g.rebuild_edges();

        assert_eq!(g.node_count(), 2);
        // main.rs references format_date in utils.rs -> edge main -> utils
        assert_eq!(g.out_degree("src/main.rs"), 1);
        assert_eq!(g.in_degree("src/utils.rs"), 1);

        let neighbors = g.in_neighbors("src/utils.rs");
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0, "src/main.rs");
    }

    #[test]
    fn test_pagerank_star_network() {
        // Star network: Node A, B, C all point to Central Node D.
        let mut g = ReferenceGraph::new();
        g.add_file(
            "d.rs",
            vec!["core_service".into()],
            "pub fn core_service() {}",
        );
        g.add_file(
            "a.rs",
            vec!["caller_a".into()],
            "fn a() { core_service(); }",
        );
        g.add_file(
            "b.rs",
            vec!["caller_b".into()],
            "fn b() { core_service(); }",
        );
        g.add_file(
            "c.rs",
            vec!["caller_c".into()],
            "fn c() { core_service(); }",
        );

        g.rebuild_edges();

        let ranks = compute_pagerank(&g, 0.85, 100, 1e-7, None);

        // Core node d.rs must have the highest centrality score
        let d_rank = *ranks.get("d.rs").expect("d.rs rank exists");
        let a_rank = *ranks.get("a.rs").expect("a.rs rank exists");

        assert!(
            d_rank > a_rank,
            "d.rs ({d_rank}) should have higher rank than a.rs ({a_rank})"
        );

        // Sum of all ranks should equal 1.0
        let total: f64 = ranks.values().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_pagerank_personalized_bias() {
        let mut g = ReferenceGraph::new();
        g.add_file("x.rs", vec!["x_fn".into()], "fn x() {}");
        g.add_file("y.rs", vec!["y_fn".into()], "fn y() {}");

        g.rebuild_edges();

        let mut bias = HashMap::new();
        bias.insert("y.rs".into(), 10.0);

        let ranks = compute_pagerank(&g, 0.85, 50, 1e-6, Some(&bias));
        let y_rank = *ranks.get("y.rs").unwrap();
        let x_rank = *ranks.get("x.rs").unwrap();

        assert!(y_rank > x_rank, "Personalized y.rs should outrank x.rs");
    }

    #[test]
    fn test_repomap_render_token_budget() {
        let files = vec![
            (
                "crates/agent/src/lib.rs".into(),
                "pub struct Agent; pub trait Llm; pub fn run_turn() {}".into(),
            ),
            (
                "crates/agent/src/daemon.rs".into(),
                "pub struct DaemonServer; pub fn serve_acp() { Agent; }".into(),
            ),
        ];

        let map = RepoMap::from_files(PathBuf::from("."), files);
        let rendered = map.render(500, None);

        assert!(rendered.contains("Repository Architecture Map"));
        assert!(rendered.contains("crates/agent/src/lib.rs"));
        assert!(rendered.contains("crates/agent/src/daemon.rs"));
    }
}
