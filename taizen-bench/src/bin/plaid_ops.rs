//! Measures what next-plaid 1.7.0 update/delete cost and rewrite on disk, and
//! checks the id rules the retrieval service's id map relies on, through both the
//! buffer path (updates below buffer_size) and centroid expansion.
use anyhow::{ensure, Context, Result};
use ndarray::Array2;
use next_plaid::search::SearchParameters;
use next_plaid::update::UpdateConfig;
use next_plaid::{IndexConfig, MmapIndex};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
use taizen_bench::embset::EmbeddingSet;
use taizen_bench::ids::{renumber, validate_delete_ids};
use taizen_bench::snapshot::{diff, snapshot};

fn args() -> Result<HashMap<String, String>> {
    let v: Vec<String> = std::env::args().skip(1).collect();
    v.chunks(2)
        .map(|c| Ok((c[0].trim_start_matches("--").to_string(), c.get(1).context("missing value")?.clone())))
        .collect()
}

/// Top-1 of each probe (its own vectors as the query) must be the expected id.
fn self_search_mismatches(index: &MmapIndex, probes: &[(i64, &Array2<f32>)]) -> Result<usize> {
    let params = SearchParameters { top_k: 1, ..Default::default() };
    let mut bad = 0;
    for (expected, query) in probes {
        let r = index.search(query, &params, None)?;
        if r.passage_ids.first() != Some(expected) {
            bad += 1;
        }
    }
    Ok(bad)
}

fn timed<T>(f: impl FnOnce() -> Result<T>) -> Result<(T, f64)> {
    let t = Instant::now();
    let v = f()?;
    Ok((v, t.elapsed().as_secs_f64()))
}

fn main() -> Result<()> {
    let a = args()?;
    let num = |k: &str| -> Result<usize> { Ok(a.get(k).with_context(|| format!("--{k}"))?.parse()?) };
    let docs = EmbeddingSet::load(Path::new(&a["docs"]))?;
    let path = a["index"].clone();
    let (initial, per_update, rounds) = (num("initial")?, num("update-docs")?, num("update-rounds")?);
    let (buffered, buffered_rounds) = (num("buffered-docs")?, num("buffered-rounds")?);
    let buffer_size = UpdateConfig::default().buffer_size;
    ensure!(buffered * buffered_rounds < buffer_size, "buffered rounds must stay below buffer_size ({buffer_size})");
    ensure!(per_update >= buffer_size, "--update-docs must reach buffer_size ({buffer_size}) to expand centroids");
    ensure!(initial + buffered * buffered_rounds + per_update * (rounds + 1) <= docs.arrays.len(),
        "not enough documents in --docs");
    let _ = std::fs::remove_dir_all(&path);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // position[i] = embedding-set row that index document i holds.
    let mut position: Vec<usize> = (0..initial).collect();
    let config = IndexConfig { nbits: 4, seed: Some(42), ..Default::default() };
    let (mut index, create_seconds) = timed(|| Ok(MmapIndex::create_with_kmeans(&docs.arrays[..initial], &path, &config)?))?;
    let mut id_checks = json!({"update_ids_contiguous": true, "update_self_search_mismatches": 0,
        "delete_self_search_mismatches": 0, "buffer_crossings": 0});

    // Small rounds accumulate in next-plaid's buffer; the first large round crosses
    // buffer_size, which deletes the buffered documents and re-adds them after expansion.
    let schedule: Vec<usize> = std::iter::repeat(buffered).take(buffered_rounds)
        .chain(std::iter::repeat(per_update).take(rounds)).collect();
    let buffer_file = Path::new(&path).join("buffer.npy");
    let mut updates = Vec::new();
    let mut next = initial;
    for (round, &size) in schedule.iter().enumerate() {
        let batch = docs.arrays[next..next + size].to_vec();
        let before = snapshot(Path::new(&path))?;
        let buffered_before = buffer_file.exists();
        let n_before = index.num_documents() as i64;
        let (assigned, seconds) = timed(|| Ok(index.update(&batch, &UpdateConfig::default())?))?;
        let expected: Vec<i64> = (n_before..n_before + size as i64).collect();
        if assigned != expected {
            id_checks["update_ids_contiguous"] = json!(false);
        }
        let update_path = if buffer_file.exists() { "buffer" } else { "expand" };
        if buffered_before && update_path == "expand" {
            id_checks["buffer_crossings"] = json!(id_checks["buffer_crossings"].as_u64().unwrap() + 1);
        }
        position.extend(next..next + size);
        next += size;
        // The last 300 documents cover the whole buffer (default buffer_size 100).
        let tail: Vec<(i64, &Array2<f32>)> = (position.len().saturating_sub(300)..position.len())
            .map(|i| (i as i64, &docs.arrays[position[i]])).collect();
        let bad = self_search_mismatches(&index, &tail)?;
        id_checks["update_self_search_mismatches"] = json!(id_checks["update_self_search_mismatches"].as_u64().unwrap() + bad as u64);
        updates.push(json!({"round": round, "documents": size, "path": update_path,
            "tokens": batch.iter().map(|b| b.nrows()).sum::<usize>(), "seconds": seconds,
            "files": diff(&before, &snapshot(Path::new(&path))?).to_json()}));
    }

    let batch = docs.arrays[next..next + per_update].to_vec();
    let before = snapshot(Path::new(&path))?;
    let (_, append_seconds) = timed(|| Ok(MmapIndex::update_append(&batch, &path, &UpdateConfig::default())?))?;
    let mid = snapshot(Path::new(&path))?;
    let (_, reload_seconds) = timed(|| Ok(index.reload()?))?;
    position.extend(next..next + per_update);
    let update_append = json!({"seconds": append_seconds, "files": diff(&before, &mid).to_json()});
    let reload_after_append = json!({"seconds": reload_seconds, "files": diff(&mid, &snapshot(Path::new(&path))?).to_json()});

    let mut deletes = Vec::new();
    for fraction in a["delete-fractions"].split(',').map(|f| f.parse::<f64>()) {
        let fraction = fraction?;
        let n = index.num_documents();
        let count = ((n as f64 * fraction).round() as usize).max(1);
        let mut all: Vec<i64> = (0..n as i64).collect();
        all.shuffle(&mut rng);
        let ids = validate_delete_ids(&all[..count], n)?;
        let survivors: Vec<i64> = all[count..].iter().take(200).copied().collect();
        let probes_before: Vec<(i64, &Array2<f32>)> = survivors.iter().map(|&id| (id, &docs.arrays[position[id as usize]])).collect();
        let stable: Vec<i64> = survivors.iter().zip(probes_before.iter())
            .filter(|(_, p)| self_search_mismatches(&index, &[**p]).map(|b| b == 0).unwrap_or(false))
            .map(|(id, _)| *id).collect();

        let before = snapshot(Path::new(&path))?;
        let (removed, delete_seconds) = timed(|| Ok(index.delete(&ids)?))?;
        let mid = snapshot(Path::new(&path))?;
        let (_, reload_seconds) = timed(|| Ok(index.reload()?))?;

        let probes_after: Vec<(i64, &Array2<f32>)> = stable.iter()
            .map(|&old| (renumber(old, &ids).unwrap(), &docs.arrays[position[old as usize]])).collect();
        let bad = self_search_mismatches(&index, &probes_after)?;
        id_checks["delete_self_search_mismatches"] = json!(id_checks["delete_self_search_mismatches"].as_u64().unwrap() + bad as u64);
        let deleted: std::collections::HashSet<i64> = ids.iter().copied().collect();
        position = position.iter().enumerate().filter(|(i, _)| !deleted.contains(&(*i as i64))).map(|(_, p)| *p).collect();

        deletes.push(json!({"fraction": fraction, "requested": count, "removed": removed,
            "index_documents_before": n, "delete_seconds": delete_seconds, "reload_seconds": reload_seconds,
            "files_delete": diff(&before, &mid).to_json(),
            "files_reload": diff(&mid, &snapshot(Path::new(&path))?).to_json(), "probes": probes_after.len()}));
    }

    let index_bytes: u64 = snapshot(Path::new(&path))?.values().map(|f| f.bytes).sum();
    let out: Value = json!({"create": {"documents": initial, "seconds": create_seconds},
        "index_bytes": index_bytes, "updates": updates, "update_append": update_append,
        "reload_after_append": reload_after_append, "deletes": deletes, "id_checks": id_checks});
    std::fs::write(&a["out"], serde_json::to_string_pretty(&out)?)?;
    println!("{}", serde_json::to_string_pretty(&out["id_checks"])?);

    let ok = id_checks["update_ids_contiguous"] == json!(true)
        && id_checks["buffer_crossings"].as_u64().unwrap() > 0
        && id_checks["update_self_search_mismatches"] == json!(0)
        && id_checks["delete_self_search_mismatches"] == json!(0);
    ensure!(ok, "id checks failed: {id_checks}");
    Ok(())
}
