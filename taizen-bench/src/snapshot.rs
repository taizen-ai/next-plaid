use anyhow::Result;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FileState {
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Default, Debug)]
pub struct Diff {
    pub created: Vec<(String, u64)>,
    pub modified: Vec<(String, u64)>,
    pub deleted: Vec<String>,
}

/// Files next-plaid regenerates on load; they never need uploading.
pub fn is_derived(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.starts_with("merged_") || name.ends_with(".lock")
}

pub fn snapshot(dir: &Path) -> Result<BTreeMap<String, FileState>> {
    let mut out = BTreeMap::new();
    visit(dir, dir, &mut out)?;
    Ok(out)
}

fn visit(root: &Path, dir: &Path, out: &mut BTreeMap<String, FileState>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            visit(root, &path, out)?;
        } else {
            let bytes = std::fs::read(&path)?;
            let rel = path.strip_prefix(root)?.to_string_lossy().replace('\\', "/");
            out.insert(rel, FileState { bytes: bytes.len() as u64, sha256: format!("{:x}", Sha256::digest(&bytes)) });
        }
    }
    Ok(())
}

pub fn diff(before: &BTreeMap<String, FileState>, after: &BTreeMap<String, FileState>) -> Diff {
    let mut d = Diff::default();
    for (path, state) in after {
        match before.get(path) {
            None => d.created.push((path.clone(), state.bytes)),
            Some(old) if old != state => d.modified.push((path.clone(), state.bytes)),
            _ => {}
        }
    }
    d.deleted = before.keys().filter(|p| !after.contains_key(*p)).cloned().collect();
    d
}

impl Diff {
    pub fn to_json(&self) -> Value {
        let changed = self.created.iter().chain(self.modified.iter());
        let (derived, source): (Vec<_>, Vec<_>) = changed.partition(|(p, _)| is_derived(p));
        json!({
            "created": self.created, "modified": self.modified, "deleted": self.deleted,
            "upload_bytes": source.iter().map(|(_, b)| b).sum::<u64>(),
            "derived_bytes": derived.iter().map(|(_, b)| b).sum::<u64>(),
            "largest_upload": source.iter().max_by_key(|(_, b)| *b).map(|(p, b)| json!([p, b])),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_changes_and_excludes_derived() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        std::fs::write(p.join("ivf.npy"), b"one").unwrap();
        std::fs::write(p.join("old.json"), b"x").unwrap();
        let before = snapshot(p).unwrap();
        std::fs::write(p.join("ivf.npy"), b"two!").unwrap();
        std::fs::remove_file(p.join("old.json")).unwrap();
        std::fs::write(p.join("0.codes.npy"), b"new").unwrap();
        std::fs::write(p.join("merged_codes.npy"), b"derived").unwrap();
        let d = diff(&before, &snapshot(p).unwrap());
        assert_eq!(d.modified, vec![("ivf.npy".to_string(), 4)]);
        assert_eq!(d.deleted, vec!["old.json".to_string()]);
        let v = d.to_json();
        assert_eq!(v["upload_bytes"], 7); // 0.codes.npy (3) + ivf.npy (4)
        assert_eq!(v["derived_bytes"], 7); // merged_codes.npy
    }

    #[test]
    fn derived_files() {
        assert!(is_derived("merged_residuals.npy"));
        assert!(is_derived("merged_codes.npy.manifest.json"));
        assert!(is_derived("merged_codes.lock"));
        assert!(!is_derived("ivf.npy"));
    }
}
