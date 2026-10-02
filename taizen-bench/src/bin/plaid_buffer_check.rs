//! Whether next-plaid 1.7.0 finds documents still in its update buffer
//! (retrieval part 1c). Each sampled document's reconstruction, at most 128
//! rows, is the query, top 10. Samples: 1,000 random documents of the index
//! (the control); 50 off-domain documents appended with `update`, which stay
//! in the buffer; the same 50 plus 100 more, which flush it. `update` changes
//! the index, so `--index` must be a scratch copy.
use anyhow::{Context, Result};
use ndarray::{s, Array2};
use next_plaid::search::SearchParameters;
use next_plaid::{MmapIndex, UpdateConfig};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
use taizen_bench::embset::EmbeddingSet;

const K: usize = 10;
const QUERY_ROWS: usize = 128;
const RANDOM: usize = 1_000;
const BUFFERED: usize = 50;
const APPENDED: usize = 150;

fn find_rate(index: &MmapIndex, docs: &[i64]) -> Result<Value> {
    let mut found = 0;
    let mut ms = Vec::with_capacity(docs.len());
    for &doc in docs {
        let vectors = index.reconstruct_single(doc)?;
        let rows = vectors.nrows().min(QUERY_ROWS);
        let query: Array2<f32> = vectors.slice(s![..rows, ..]).to_owned();
        let started = Instant::now();
        let result = index.search(&query, &SearchParameters { top_k: K, ..Default::default() }, None)?;
        ms.push(started.elapsed().as_secs_f64() * 1000.0);
        found += usize::from(result.passage_ids.contains(&doc));
    }
    ms.sort_by(f64::total_cmp);
    Ok(json!({
        "sampled": docs.len(),
        "found": found,
        "rate": found as f64 / docs.len() as f64,
        "p50_ms": ms[ms.len() / 2],
    }))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let a: HashMap<String, String> = args
        .chunks(2)
        .map(|c| (c[0].trim_start_matches("--").to_string(), c.get(1).cloned().unwrap_or_default()))
        .collect();
    let path = a.get("index").context("--index SCRATCH_COPY")?.clone();
    let extra = EmbeddingSet::load(Path::new(a.get("extra").context("--extra EMBSET")?))?;
    anyhow::ensure!(extra.arrays.len() >= APPENDED, "--extra needs {APPENDED} documents");
    let mut index = MmapIndex::load(&path)?;
    let n = index.metadata.num_documents;
    let centroids_before = index.metadata.num_partitions;
    let mut order: Vec<i64> = (0..n as i64).collect();
    order.shuffle(&mut ChaCha8Rng::seed_from_u64(42));
    let control = find_rate(&index, &order[..RANDOM])?;

    let config = UpdateConfig { start_from_scratch: 0, ..Default::default() };
    let first = index.update(&extra.arrays[..BUFFERED], &config)?;
    index.reload()?;
    let buffered = find_rate(&index, &first)?;

    let rest = index.update(&extra.arrays[BUFFERED..APPENDED], &config)?;
    index.reload()?;
    let all: Vec<i64> = first.iter().chain(&rest).copied().collect();
    let flushed = find_rate(&index, &all)?;

    println!(
        "{}",
        json!({
            "index_documents": n,
            "k": K,
            "query_rows": QUERY_ROWS,
            "control": control,
            "buffered_off_domain": buffered,
            "after_flush_off_domain": flushed,
            "centroids_before": centroids_before,
            "centroids_after": index.metadata.num_partitions,
        })
    );
    Ok(())
}
