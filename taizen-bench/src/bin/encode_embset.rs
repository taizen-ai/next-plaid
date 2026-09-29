//! Encode a JSONL of {"id","text"} with next-plaid-onnx and write an embedding set.
use anyhow::{bail, Context, Result};
use next_plaid_onnx::Colbert;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::BufRead;
use std::path::PathBuf;
use std::time::Instant;
use taizen_bench::embset::EmbeddingSet;

fn args() -> Result<HashMap<String, String>> {
    let mut out = HashMap::new();
    let mut it = std::env::args().skip(1);
    while let Some(key) = it.next() {
        let key = key.strip_prefix("--").context("expected --key")?.to_string();
        if key == "quantized" {
            out.insert(key, "true".into());
        } else {
            out.insert(key.clone(), it.next().with_context(|| format!("missing value for --{key}"))?);
        }
    }
    Ok(out)
}

fn main() -> Result<()> {
    let a = args()?;
    let get = |k: &str| a.get(k).with_context(|| format!("--{k} is required"));
    let num = |k: &str| -> Result<usize> { Ok(get(k)?.parse()?) };
    let kind = get("kind")?.clone();
    let quantized = a.contains_key("quantized");

    let file = std::fs::File::open(get("input")?)?;
    let rows: Vec<Value> = std::io::BufReader::new(file)
        .lines()
        .filter_map(|l| l.ok().filter(|l| !l.trim().is_empty()))
        .map(|l| serde_json::from_str(&l))
        .collect::<Result<_, _>>()?;
    let ids: Vec<String> = rows.iter().map(|r| r["id"].as_str().unwrap().to_string()).collect();
    let texts: Vec<&str> = rows.iter().map(|r| r["text"].as_str().unwrap()).collect();

    let started = Instant::now();
    let model = Colbert::builder(get("model")?.as_str())
        .with_threads(num("threads")?)
        .with_batch_size(num("batch-size")?)
        .with_quantized(quantized)
        .with_query_length(num("query-length")?)
        .with_document_length(num("document-length")?)
        .build()?;
    let load_seconds = started.elapsed().as_secs_f64();

    let begin = Instant::now();
    let arrays = match kind.as_str() {
        "query" => model.encode_queries(&texts)?,
        "document" => model.encode_documents(&texts, None)?,
        other => bail!("--kind must be query or document, got {other}"),
    };
    let encode_seconds = begin.elapsed().as_secs_f64();

    let set = EmbeddingSet {
        ids,
        arrays,
        meta: json!({
            "model": "lightonai/mLateOn", "revision": "edd378f99593c0ac8a15518b97ad89786b02685e",
            "runtime": if quantized { "onnx-int8" } else { "onnx-fp32" }, "dtype": "float32",
            "kind": kind, "query_length": num("query-length")?, "document_length": num("document-length")?,
        }),
    };
    set.save(&PathBuf::from(get("out")?))?;
    let tokens = set.tokens();
    println!("{}", json!({"items": set.ids.len(), "tokens": tokens, "load_seconds": load_seconds,
        "encode_seconds": encode_seconds, "tokens_per_second": tokens as f64 / encode_seconds}));
    Ok(())
}
