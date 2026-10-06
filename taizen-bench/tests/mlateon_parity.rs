//! Queries run on ONNX (CPU) and documents on PyTorch (GPU, packed-encoders); both
//! must land in the same space. Run with the model exported by
//! scripts/export_mlateon.py at $MLATEON_DIR and ORT_DYLIB_PATH set
//! (scripts/ort-dylib.sh):
//!   cargo test --release -p taizen-bench --test mlateon_parity -- --ignored --nocapture
use ndarray::Array2;
use next_plaid_onnx::Colbert;
use serde_json::{json, Value};
use std::io::BufRead;
use std::path::{Path, PathBuf};
use taizen_bench::embset::EmbeddingSet;
use taizen_bench::parity::{compare, ParityStats};

const QUERY_LENGTH: usize = 128;
const DOCUMENT_LENGTH: usize = 8192;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mlateon")
}

fn texts() -> Vec<String> {
    let file = std::fs::File::open(fixtures().join("texts.jsonl")).unwrap();
    std::io::BufReader::new(file)
        .lines()
        .map(|l| serde_json::from_str::<Value>(&l.unwrap()).unwrap()["text"].as_str().unwrap().to_string())
        .collect()
}

fn onnx(kind: &str, quantized: bool) -> Vec<Array2<f32>> {
    let dir = std::env::var("MLATEON_DIR").expect("MLATEON_DIR must point at the export from scripts/export_mlateon.py");
    let model = Colbert::builder(dir.as_str())
        .with_threads(4)
        // One text per batch: a batch pads to its longest text, and attention over
        // DOCUMENT_LENGTH tokens costs heads × DOCUMENT_LENGTH² floats per text.
        .with_batch_size(1)
        .with_quantized(quantized)
        .with_query_length(QUERY_LENGTH)
        .with_document_length(DOCUMENT_LENGTH)
        .build()
        .unwrap();
    let texts = texts();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    match kind {
        "queries" => model.encode_queries(&refs).unwrap(),
        _ => model.encode_documents(&refs, None).unwrap(),
    }
}

fn check(pair: &str, ours: &[Array2<f32>], golden: &str, kind: &str) -> ParityStats {
    let reference = EmbeddingSet::load(&fixtures().join(golden).join(kind)).unwrap();
    let stats = compare(ours, &reference.arrays);
    println!("{}", json!({"pair": format!("{pair}/{kind}"), "stats": stats.to_json()}));
    stats
}

#[test]
#[ignore]
fn onnx_fp32_matches_pytorch_fp32() {
    for kind in ["queries", "documents"] {
        let s = check("onnx-fp32~pytorch-fp32", &onnx(kind, false), "pytorch-fp32", kind);
        assert_eq!(s.length_mismatches, 0, "tokenization differs");
        assert!(s.min_token_cosine >= 0.999, "min cosine {}", s.min_token_cosine);
        assert!(s.max_score_abs_delta <= 0.001, "score delta {}", s.max_score_abs_delta);
    }
}

#[test]
#[ignore]
fn onnx_fp32_matches_pytorch_packed_bf16() {
    for kind in ["queries", "documents"] {
        let s = check("onnx-fp32~pytorch-packed-bf16", &onnx(kind, false), "pytorch-packed-bf16", kind);
        assert_eq!(s.length_mismatches, 0, "tokenization differs");
        assert!(s.min_token_cosine >= 0.983, "min cosine {}", s.min_token_cosine);
        assert!(s.mean_token_cosine >= 0.998, "mean cosine {}", s.mean_token_cosine);
        assert!(s.max_score_abs_delta <= 0.004, "score delta {}", s.max_score_abs_delta);
    }
}

#[test]
#[ignore]
fn onnx_int8_against_pytorch_fp32_is_reported() {
    for kind in ["queries", "documents"] {
        let s = check("onnx-int8~pytorch-fp32", &onnx(kind, true), "pytorch-fp32", kind);
        assert_eq!(s.length_mismatches, 0, "tokenization differs");
    }
}
