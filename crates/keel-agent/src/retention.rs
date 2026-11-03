//! Disk retention helpers: bound how much history keel accumulates on disk.

use std::path::Path;
use std::time::SystemTime;

/// Delete the oldest regular files in `dir` with extension `ext` until at most `keep`
/// files and `max_total_bytes` remain. The `protect`ed file (e.g. the one currently
/// being written) is never removed. Symlinks and other entries are ignored.
/// Returns the number of files removed. All errors are non-fatal.
pub fn prune_oldest(
    dir: &Path,
    ext: &str,
    keep: usize,
    max_total_bytes: u64,
    protect: Option<&Path>,
) -> usize {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut files: Vec<(SystemTime, u64, std::path::PathBuf)> = rd
        .flatten()
        .filter_map(|e| {
            let meta = std::fs::symlink_metadata(e.path()).ok()?;
            if !meta.file_type().is_file() {
                return None;
            }
            let path = e.path();
            if path.extension().and_then(|s| s.to_str()) != Some(ext) {
                return None;
            }
            Some((
                meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                meta.len(),
                path,
            ))
        })
        .collect();
    // Newest first; ties broken by name for determinism.
    files.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.2.cmp(&a.2)));

    let (mut kept, mut bytes, mut removed) = (0usize, 0u64, 0usize);
    // Reserve budget for the protected file first so it can never be evicted nor
    // cause the budget to be exceeded by newer files.
    if let Some(p) = protect {
        if let Some((_, len, _)) = files.iter().find(|(_, _, path)| path == p) {
            kept = 1;
            bytes = *len;
        }
    }
    for (_, len, path) in files {
        if protect.is_some_and(|p| p == path) {
            continue;
        }
        if kept < keep && bytes.saturating_add(len) <= max_total_bytes {
            kept += 1;
            bytes = bytes.saturating_add(len);
        } else if std::fs::remove_file(&path).is_ok() {
            removed += 1;
        }
    }
    removed
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("keel-ret-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn touch(dir: &Path, name: &str, size: usize, age_secs: u64) {
        let p = dir.join(name);
        std::fs::write(&p, vec![b'x'; size]).unwrap();
        let t = SystemTime::now() - Duration::from_secs(age_secs);
        std::fs::File::options()
            .write(true)
            .open(&p)
            .unwrap()
            .set_modified(t)
            .unwrap();
    }

    #[test]
    fn keeps_newest_by_count() {
        let d = tmp("count");
        for i in 0..5 {
            touch(&d, &format!("s{i}.keelrec"), 10, 100 - i * 10);
        }
        touch(&d, "other.txt", 10, 1000);
        assert_eq!(prune_oldest(&d, "keelrec", 2, u64::MAX, None), 3);
        assert!(d.join("s4.keelrec").exists() && d.join("s3.keelrec").exists());
        assert!(!d.join("s0.keelrec").exists());
        assert!(d.join("other.txt").exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn enforces_byte_budget_and_protects_current() {
        let d = tmp("bytes");
        touch(&d, "old.keelrec", 100, 500);
        touch(&d, "mid.keelrec", 100, 300);
        touch(&d, "new.keelrec", 100, 10);
        let protect = d.join("old.keelrec");
        let removed = prune_oldest(&d, "keelrec", 10, 250, Some(&protect));
        assert_eq!(removed, 1);
        assert!(protect.exists() && d.join("new.keelrec").exists());
        assert!(!d.join("mid.keelrec").exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn missing_dir_is_noop() {
        assert_eq!(
            prune_oldest(Path::new("/nonexistent-keel-dir"), "x", 1, 1, None),
            0
        );
    }
}
