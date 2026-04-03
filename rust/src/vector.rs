use instant_distance::{Builder, HnswMap, Search};
use pyo3::prelude::*;

/// A point in the embedding space, wrapping a Vec<f32>.
/// Distance is computed as 1.0 - dot_product (assumes unit-normalized vectors).
#[derive(Clone)]
struct Embedding(Vec<f32>);

impl instant_distance::Point for Embedding {
    fn distance(&self, other: &Self) -> f32 {
        let dot: f32 = self
            .0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| a * b)
            .sum();
        1.0 - dot
    }
}

/// HNSW approximate nearest neighbor index.
///
/// Build once from a collection of embedding vectors, then search many times.
/// The index lives in Rust memory — Python holds a reference.
///
/// Example:
///     idx = VectorIndex(vectors, ef_construction=200)
///     results = idx.search(query_vector, top_k=10)
#[pyclass]
pub struct VectorIndex {
    map: HnswMap<Embedding, usize>,
}

#[pymethods]
impl VectorIndex {
    /// Build a new HNSW index from a list of embedding vectors.
    ///
    /// Args:
    ///     vectors: List of embedding vectors (each a list of f32).
    ///     ef_construction: Build-time search effort (higher = better quality, slower build).
    #[new]
    #[pyo3(signature = (vectors, ef_construction=200))]
    pub fn new(vectors: Vec<Vec<f32>>, ef_construction: usize) -> Self {
        let n = vectors.len();
        let points: Vec<Embedding> = vectors.into_iter().map(Embedding).collect();
        let values: Vec<usize> = (0..n).collect();

        let map = Builder::default()
            .ef_construction(ef_construction)
            .seed(42)
            .build(points, values);

        VectorIndex { map }
    }

    /// Search for the top_k nearest neighbors to the query vector.
    ///
    /// Returns a list of (original_index, similarity_score) tuples,
    /// sorted by similarity (highest first).
    #[pyo3(signature = (query, top_k=10))]
    pub fn search(&self, query: Vec<f32>, top_k: usize) -> Vec<(usize, f32)> {
        let query_point = Embedding(query);
        let mut search = Search::default();

        self.map
            .search(&query_point, &mut search)
            .take(top_k)
            .map(|item| {
                let similarity = 1.0 - item.distance;
                (*item.value, similarity)
            })
            .collect()
    }

    /// Returns the number of vectors in the index.
    fn len(&self) -> usize {
        self.map.iter().count()
    }

    /// Returns True if the index contains no vectors.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
