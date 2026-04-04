"""Tests for ironrace._core.VectorIndex (HNSW approximate nearest neighbor)."""

import math
import random

from ironrace._core import VectorIndex


def _unit_vector(dim=768, seed=None):
    """Generate a random unit vector."""
    if seed is not None:
        random.seed(seed)
    v = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _known_vectors():
    """Generate a small set of known vectors for deterministic tests."""
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.707, 0.707, 0.0],  # ~45 degrees between x and y
        [0.577, 0.577, 0.577],  # ~equidistant from all axes
    ]


class TestVectorIndexBuild:
    def test_build_basic(self):
        vecs = [_unit_vector(dim=64, seed=i) for i in range(100)]
        idx = VectorIndex(vecs)
        assert idx.len() == 100

    def test_build_single_vector(self):
        idx = VectorIndex([[1.0, 0.0, 0.0]])
        assert idx.len() == 1

    def test_build_large(self):
        vecs = [_unit_vector(dim=768, seed=i) for i in range(1000)]
        idx = VectorIndex(vecs)
        assert idx.len() == 1000

    def test_build_custom_ef(self):
        vecs = [_unit_vector(dim=64, seed=i) for i in range(50)]
        idx = VectorIndex(vecs, ef_construction=50)
        assert idx.len() == 50

    def test_is_empty(self):
        vecs = [_unit_vector(dim=64, seed=0)]
        idx = VectorIndex(vecs)
        assert not idx.is_empty()


class TestVectorIndexSearch:
    def test_self_search(self):
        """Searching for a vector in the index should return itself as the top result."""
        vecs = [_unit_vector(dim=128, seed=i) for i in range(500)]
        idx = VectorIndex(vecs)
        results = idx.search(vecs[42], 5)
        assert results[0][0] == 42
        assert results[0][1] > 0.99  # near-perfect similarity

    def test_known_vectors(self):
        """Test with known geometric vectors."""
        vecs = _known_vectors()
        idx = VectorIndex(vecs, ef_construction=50)
        # Search for [1, 0, 0] — closest should be itself, then [0.707, 0.707, 0]
        results = idx.search([1.0, 0.0, 0.0], 3)
        assert results[0][0] == 0  # exact match
        assert results[0][1] > 0.99

    def test_top_k_count(self):
        vecs = [_unit_vector(dim=64, seed=i) for i in range(100)]
        idx = VectorIndex(vecs)
        results = idx.search(vecs[0], 10)
        assert len(results) == 10

    def test_top_k_greater_than_n(self):
        """Requesting more results than vectors should return all vectors."""
        vecs = [_unit_vector(dim=64, seed=i) for i in range(50)]
        idx = VectorIndex(vecs)
        results = idx.search(vecs[0], 200)
        assert len(results) == 50

    def test_results_sorted_by_similarity(self):
        vecs = [_unit_vector(dim=128, seed=i) for i in range(200)]
        idx = VectorIndex(vecs)
        results = idx.search(vecs[0], 20)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_similarity_score_range(self):
        """Similarity scores should be in [-1, 1] for unit vectors."""
        vecs = [_unit_vector(dim=128, seed=i) for i in range(100)]
        idx = VectorIndex(vecs)
        results = idx.search(vecs[0], 10)
        for _, score in results:
            assert -1.0 <= score <= 1.001  # small epsilon for float precision

    def test_high_dimensional(self):
        """Test with 1536-dimensional vectors (OpenAI embedding size)."""
        vecs = [_unit_vector(dim=1536, seed=i) for i in range(100)]
        idx = VectorIndex(vecs)
        results = idx.search(vecs[0], 5)
        assert results[0][0] == 0
        assert len(results) == 5


class TestVectorIndexPerformance:
    def test_search_returns_quickly(self):
        """Search over 1K vectors should be fast (< 10ms)."""
        import time

        vecs = [_unit_vector(dim=256, seed=i) for i in range(1000)]
        idx = VectorIndex(vecs)
        query = _unit_vector(dim=256, seed=99999)

        start = time.perf_counter()
        for _ in range(50):
            idx.search(query, 10)
        elapsed = (time.perf_counter() - start) / 50 * 1000  # ms per search

        assert elapsed < 10, f"Search took {elapsed:.1f}ms (expected < 10ms)"
