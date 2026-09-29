//! Full and subset search rankings from next-plaid 1.7.0, for scoring against exact MaxSim.
//! Run once per final-stage mode: default (residual LUT) and NEXT_PLAID_FLOAT_RESCORE=1.
use anyhow::{Context, Result};
use next_plaid::search::SearchParameters;
use next_plaid::{IndexConfig, MmapIndex};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
use taizen_bench::embset::EmbeddingSet;

fn main() -> Result<()> {
    let v: Vec<String> = std::env::args().skip(1).collect();
    let a: HashMap<String, String> = v.chunks(2)
        .map(|c| (c[0].trim_start_matches("--").to_string(), c.get(1).cloned().unwrap_or_default())).collect();
    let docs = EmbeddingSet::load(Path::new(&a["docs"]))?;
    let queries = EmbeddingSet::load(Path::new(&a["queries"]))?;
    let path = a["index"].clone();

    let index = if Path::new(&path).join("metadata.json").exists() {
        MmapIndex::load(&path)?
    } else {
        MmapIndex::create_with_kmeans(&docs.arrays, &path, &IndexConfig { nbits: 4, seed: Some(42), ..Default::default() })?
    };

    let full_params = SearchParameters { top_k: 100, ..Default::default() };
    let mut full = Map::new();
    let mut latency = Vec::new();
    for (qid, q) in queries.ids.iter().zip(&queries.arrays) {
        let t = Instant::now();
        let r = index.search(q, &full_params, None)?;
        latency.push(t.elapsed().as_secs_f64() * 1000.0);
        full.insert(qid.clone(), json!(r.passage_ids.iter().map(|&i| &docs.ids[i as usize]).collect::<Vec<_>>()));
    }

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let subset_queries: usize = a["subset-queries"].parse()?;
    let sub_params = SearchParameters { top_k: 10, ..Default::default() };
    let mut subsets = Vec::new();
    for size in a["subset-sizes"].split(',').map(|s| s.parse::<usize>()) {
        let size = size?;
        let mut all: Vec<i64> = (0..docs.ids.len() as i64).collect();
        all.shuffle(&mut rng);
        let mut subset: Vec<i64> = all[..size].to_vec();
        subset.sort_unstable();
        let mut rankings = Map::new();
        for (qid, q) in queries.ids.iter().zip(&queries.arrays).take(subset_queries) {
            let r = index.search(q, &sub_params, Some(&subset))?;
            rankings.insert(qid.clone(), json!(r.passage_ids.iter().map(|&i| &docs.ids[i as usize]).collect::<Vec<_>>()));
        }
        subsets.push(json!({"size": size, "subset": subset.iter().map(|&i| &docs.ids[i as usize]).collect::<Vec<_>>(),
                            "rankings": rankings}));
    }

    let out: Value = json!({"label": a["label"],
        "float_rescore_env": std::env::var("NEXT_PLAID_FLOAT_RESCORE").ok(),
        "full": full, "latency_ms": latency, "subsets": subsets});
    std::fs::write(a.get("out").context("--out")?, serde_json::to_string(&out)?)?;
    Ok(())
}
