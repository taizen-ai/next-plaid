use ndarray::Array2;
use serde_json::{json, Value};

pub struct ParityStats {
    pub items: usize,
    pub length_mismatches: usize,
    pub min_token_cosine: f32,
    pub mean_token_cosine: f32,
    pub max_score_abs_delta: f32,
}

impl ParityStats {
    pub fn to_json(&self) -> Value {
        json!({"items": self.items, "length_mismatches": self.length_mismatches,
               "min_token_cosine": self.min_token_cosine, "mean_token_cosine": self.mean_token_cosine,
               "max_score_abs_delta": self.max_score_abs_delta})
    }
}

fn cosine(a: ndarray::ArrayView1<f32>, b: ndarray::ArrayView1<f32>) -> f32 {
    a.dot(&b) / (a.dot(&a).sqrt() * b.dot(&b).sqrt())
}

/// MaxSim normalised by query tokens.
fn maxsim(q: &Array2<f32>, d: &Array2<f32>) -> f32 {
    let sims = q.dot(&d.t());
    sims.rows().into_iter().map(|r| r.iter().cloned().fold(f32::MIN, f32::max)).sum::<f32>() / q.nrows() as f32
}

/// Token cosines over items with equal lengths, and the largest MaxSim change over
/// every (i, j) pair of such items — the quantity rankings depend on.
pub fn compare(a: &[Array2<f32>], b: &[Array2<f32>]) -> ParityStats {
    assert_eq!(a.len(), b.len(), "runtimes encoded a different number of items");
    let same: Vec<usize> = (0..a.len()).filter(|&i| a[i].nrows() == b[i].nrows()).collect();
    let (mut min_cos, mut sum_cos, mut n_cos) = (f32::MAX, 0f32, 0usize);
    for &i in &same {
        for (ra, rb) in a[i].rows().into_iter().zip(b[i].rows()) {
            let c = cosine(ra, rb);
            min_cos = min_cos.min(c);
            sum_cos += c;
            n_cos += 1;
        }
    }
    let mut max_delta = 0f32;
    for &i in &same {
        for &j in &same {
            max_delta = max_delta.max((maxsim(&a[i], &a[j]) - maxsim(&b[i], &b[j])).abs());
        }
    }
    ParityStats {
        items: a.len(),
        length_mismatches: a.len() - same.len(),
        min_token_cosine: if n_cos == 0 { 0.0 } else { min_cos },
        mean_token_cosine: if n_cos == 0 { 0.0 } else { sum_cos / n_cos as f32 },
        max_score_abs_delta: max_delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn identical_runtimes_are_perfect() {
        let a = vec![array![[1.0f32, 0.0], [0.0, 1.0]], array![[0.6f32, 0.8]]];
        let s = compare(&a, &a);
        assert_eq!(s.length_mismatches, 0);
        assert!((s.min_token_cosine - 1.0).abs() < 1e-6);
        assert!(s.max_score_abs_delta < 1e-6);
    }

    #[test]
    fn counts_length_mismatch_and_perturbation() {
        let a = vec![array![[1.0f32, 0.0]], array![[0.0f32, 1.0]]];
        let b = vec![array![[0.8f32, 0.6]], array![[0.0f32, 1.0], [1.0, 0.0]]];
        let s = compare(&a, &b);
        assert_eq!(s.length_mismatches, 1);
        assert!((s.min_token_cosine - 0.8).abs() < 1e-6);
    }
}
