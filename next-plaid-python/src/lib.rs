use ::next_plaid::index::create_index_files;
use ::next_plaid::IndexConfig;
use ndarray::Array2;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

/// Create a PLAID index from document embeddings and pre-computed centroids.
///
/// K-means runs outside (fastkmeans on GPU); this call quantizes, computes
/// residuals and writes the index files. It writes no metadata database.
#[pyfunction]
#[pyo3(signature = (embeddings, centroids, index_path, nbits=4, batch_size=50_000, seed=None, force_cpu=false))]
fn create_index(
    embeddings: Vec<PyReadonlyArray2<f32>>,
    centroids: PyReadonlyArray2<f32>,
    index_path: &str,
    nbits: usize,
    batch_size: usize,
    seed: Option<u64>,
    force_cpu: bool,
) -> PyResult<()> {
    let arrays: Vec<Array2<f32>> = embeddings.iter().map(|e| e.as_array().to_owned()).collect();
    let config = IndexConfig { nbits, batch_size, seed, force_cpu, ..Default::default() };
    create_index_files(&arrays, centroids.as_array().to_owned(), index_path, &config)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(())
}

#[pymodule(name = "next_plaid")]
fn next_plaid_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_index, m)?)?;
    Ok(())
}
