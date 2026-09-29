use anyhow::{bail, ensure, Context, Result};
use ndarray::{concatenate, s, Array1, Array2, Axis};
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use serde_json::Value;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Multi-vector items; on disk item i is rows offsets[i]..offsets[i+1] of vectors.npy.
pub struct EmbeddingSet {
    pub ids: Vec<String>,
    pub arrays: Vec<Array2<f32>>,
    pub meta: Value,
}

impl EmbeddingSet {
    pub fn tokens(&self) -> usize {
        self.arrays.iter().map(|a| a.nrows()).sum()
    }

    pub fn load(dir: &Path) -> Result<Self> {
        let vectors = Array2::<f32>::read_npy(BufReader::new(File::open(dir.join("vectors.npy"))?))
            .context("vectors.npy")?;
        let lengths = Array1::<i64>::read_npy(BufReader::new(File::open(dir.join("lengths.npy"))?))
            .context("lengths.npy")?;
        let ids: Vec<String> = serde_json::from_reader(BufReader::new(File::open(dir.join("ids.json"))?))?;
        let meta: Value = serde_json::from_reader(BufReader::new(File::open(dir.join("meta.json"))?))?;
        ensure!(ids.len() == lengths.len(), "ids ({}) and lengths ({}) differ", ids.len(), lengths.len());
        ensure!(lengths.iter().all(|&l| l > 0), "zero-length item");
        ensure!(lengths.sum() as usize == vectors.nrows(), "lengths do not sum to vector rows");
        let mut arrays = Vec::with_capacity(ids.len());
        let mut start = 0usize;
        for &len in lengths.iter() {
            let end = start + len as usize;
            arrays.push(vectors.slice(s![start..end, ..]).to_owned());
            start = end;
        }
        Ok(Self { ids, arrays, meta })
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        if self.ids.len() != self.arrays.len() {
            bail!("ids and arrays differ in size");
        }
        std::fs::create_dir_all(dir)?;
        let views: Vec<_> = self.arrays.iter().map(|a| a.view()).collect();
        let vectors = concatenate(Axis(0), &views)?;
        let lengths: Array1<i64> = self.arrays.iter().map(|a| a.nrows() as i64).collect();
        vectors.write_npy(BufWriter::new(File::create(dir.join("vectors.npy"))?))?;
        lengths.write_npy(BufWriter::new(File::create(dir.join("lengths.npy"))?))?;
        serde_json::to_writer(BufWriter::new(File::create(dir.join("ids.json"))?), &self.ids)?;
        serde_json::to_writer_pretty(BufWriter::new(File::create(dir.join("meta.json"))?), &self.meta)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn sample() -> EmbeddingSet {
        EmbeddingSet {
            ids: vec!["a".into(), "b".into()],
            arrays: vec![array![[1.0f32, 0.0], [0.0, 1.0]], array![[0.5f32, 0.5]]],
            meta: serde_json::json!({"kind": "document"}),
        }
    }

    #[test]
    fn round_trip() {
        let dir = tempfile::tempdir().unwrap();
        sample().save(dir.path()).unwrap();
        let loaded = EmbeddingSet::load(dir.path()).unwrap();
        assert_eq!(loaded.ids, vec!["a", "b"]);
        assert_eq!(loaded.arrays[1], sample().arrays[1]);
        assert_eq!(loaded.tokens(), 3);
    }

    #[test]
    fn rejects_length_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        sample().save(dir.path()).unwrap();
        std::fs::write(dir.path().join("ids.json"), r#"["a"]"#).unwrap();
        assert!(EmbeddingSet::load(dir.path()).is_err());
    }
}
