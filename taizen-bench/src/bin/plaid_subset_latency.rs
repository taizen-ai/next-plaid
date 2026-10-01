//! next-plaid 1.7.0 search latency with include subsets of growing size,
//! against leaving the excluded documents out of a larger unrestricted
//! result (retrieval part 1b). k = 10; one pass per query and setting.
use anyhow::Result;
use next_plaid::search::SearchParameters;
use next_plaid::{IndexConfig, MmapIndex};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;
use taizen_bench::embset::EmbeddingSet;

const K: usize = 10;

fn percentiles(mut ms: Vec<f64>) -> Value {
    ms.sort_by(f64::total_cmp);
    let at = |p: f64| ms[((ms.len() - 1) as f64 * p).round() as usize];
    json!({ "p50": at(0.5), "p95": at(0.95) })
}

fn main() -> Result<()> {
    let v: Vec<String> = std::env::args().skip(1).collect();
    let a: HashMap<String, String> = v
        .chunks(2)
        .map(|c| (c[0].trim_start_matches("--").to_string(), c.get(1).cloned().unwrap_or_default()))
        .collect();
    let queries = EmbeddingSet::load(Path::new(&a["queries"]))?;
    let path = a["index"].clone();
    let index = if Path::new(&path).join("metadata.json").exists() {
        MmapIndex::load(&path)?
    } else {
        let docs = EmbeddingSet::load(Path::new(&a["docs"]))?;
        MmapIndex::create_with_kmeans(&docs.arrays, &path, &IndexConfig { nbits: 4, seed: Some(42), ..Default::default() })?
    };
    let n = index.metadata.num_documents;
    let mut order: Vec<i64> = (0..n as i64).collect();
    order.shuffle(&mut ChaCha8Rng::seed_from_u64(42));
    let first = |share: f64| -> Vec<i64> {
        let mut ids = order[..((n as f64 * share).round() as usize).max(1)].to_vec();
        ids.sort_unstable();
        ids
    };
    let time = |run: &dyn Fn(&ndarray::Array2<f32>) -> Result<()>| -> Result<Value> {
        let mut ms = Vec::with_capacity(queries.arrays.len());
        for q in &queries.arrays {
            let t = Instant::now();
            run(q)?;
            ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(percentiles(ms))
    };
    let plain = SearchParameters { top_k: K, ..Default::default() };
    let none = time(&|q| {
        index.search(q, &plain, None)?;
        Ok(())
    })?;
    let mut include = Vec::new();
    for share in [0.01, 0.1, 0.5, 0.9, 0.99] {
        let subset = first(share);
        let mut row = time(&|q| {
            index.search(q, &plain, Some(&subset))?;
            Ok(())
        })?;
        row["share"] = json!(share);
        include.push(row);
    }
    let mut exclude_filter = Vec::new();
    for share in [0.001, 0.01, 0.1, 0.5] {
        let excluded: HashSet<i64> = first(share).into_iter().collect();
        let top_k = K + excluded.len();
        let params = SearchParameters {
            top_k,
            n_full_scores: (4 * top_k).max(SearchParameters::default().n_full_scores),
            ..Default::default()
        };
        let mut row = time(&|q| {
            let found = index.search(q, &params, None)?;
            let kept = found.passage_ids.iter().filter(|id| !excluded.contains(id)).take(K).count();
            anyhow::ensure!(kept == K.min(n - excluded.len()), "filtering left {kept} results");
            Ok(())
        })?;
        row["share"] = json!(share);
        exclude_filter.push(row);
    }
    let out = json!({
        "documents": n,
        "k": K,
        "queries": queries.arrays.len(),
        "none": none,
        "include": include,
        "exclude_filter": exclude_filter,
    });
    std::fs::write(&a["out"], serde_json::to_string_pretty(&out)?)?;
    println!("{out}");
    Ok(())
}
