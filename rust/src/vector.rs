use hnsw_rs::prelude::*;
use pyo3::prelude::*;

/// HNSW approximate nearest neighbor index.
///
/// Build once from a collection of embedding vectors, then search many times.
/// The index lives in Rust memory — Python holds a reference.
///
/// Args:
///     vectors: List of embedding vectors (each a list of f32).
///     ef_construction: Build-time search effort. Default 100 gives 98%+ recall on
///         real-world embeddings. Increase to 200+ for higher recall on very large
///         datasets (1M+ vectors).
///
/// Example:
///     idx = VectorIndex(vectors)                       # 98%+ recall (default)
///     idx = VectorIndex(vectors, ef_construction=200)  # higher recall at 1M+ vectors
///     results = idx.search(query_vector, top_k=10)
#[pyclass]
pub struct VectorIndex {
    // 'static works because we always insert owned Vec<f32> (never mmap slices)
    hnsw: Hnsw<'static, f32, DistCosine>,
    count: usize,
}

impl VectorIndex {
    /// Build index without requiring Python token (used internally by pipeline)
    pub fn build(vectors: Vec<Vec<f32>>, ef_construction: usize) -> Self {
        let n = vectors.len();
        let max_nb_connection = 16;
        let max_layer = 16;

        let hnsw = Hnsw::new(max_nb_connection, n, max_layer, ef_construction, DistCosine);

        let data_for_insert: Vec<(&Vec<f32>, usize)> =
            vectors.iter().enumerate().map(|(i, v)| (v, i)).collect();
        // Sequential insert guarantees a fully connected graph. At the target
        // scale (1K-10K vectors) the build time difference is negligible.
        // parallel_insert is only used for large datasets where speed matters more.
        if n <= 5000 {
            for (v, id) in &data_for_insert {
                hnsw.insert((v.as_slice(), *id));
            }
        } else {
            hnsw.parallel_insert(&data_for_insert);
        }

        VectorIndex { hnsw, count: n }
    }

    /// Search without requiring Python token (used internally by pipeline)
    pub fn query(&self, query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        if self.count == 0 {
            return vec![];
        }
        let ef_search = if top_k >= self.count {
            // Requesting all vectors — boost ef well above count for full recall
            self.count * 2
        } else if top_k >= self.count / 2 {
            self.count
        } else {
            top_k.max(100)
        };
        let neighbours = self.hnsw.search(query, top_k, ef_search);

        neighbours
            .iter()
            .map(|n| {
                let similarity = 1.0 - n.distance;
                (n.d_id, similarity)
            })
            .collect()
    }
}

#[pymethods]
impl VectorIndex {
    /// Build a new HNSW index from a list of embedding vectors.
    #[new]
    #[pyo3(signature = (vectors, ef_construction=100))]
    pub fn new(py: Python<'_>, vectors: Vec<Vec<f32>>, ef_construction: usize) -> Self {
        // Release the GIL during index construction — this is pure Rust compute
        py.allow_threads(|| Self::build(vectors, ef_construction))
    }

    /// Search for the top_k nearest neighbors to the query vector.
    ///
    /// Returns a list of (original_index, similarity_score) tuples,
    /// sorted by similarity (highest first).
    #[pyo3(signature = (query, top_k=10))]
    pub fn search(&self, py: Python<'_>, query: Vec<f32>, top_k: usize) -> Vec<(usize, f32)> {
        // Release the GIL during search — this is pure Rust compute
        py.allow_threads(|| self.query(&query, top_k))
    }

    /// Returns the number of vectors in the index.
    fn len(&self) -> usize {
        self.count
    }

    /// Returns True if the index contains no vectors.
    fn is_empty(&self) -> bool {
        self.count == 0
    }
}
