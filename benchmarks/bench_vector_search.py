#!/usr/bin/env python3
"""
IronRace — Vector Search Benchmark at Multiple Scales
========================================================
Measures HNSW index build time, query time, and recall at 1K, 10K, 100K scales.
Uses clustered vectors that simulate real embeddings (sentence-transformers, OpenAI, etc.).

Note: Recall on uniformly random vectors is lower due to the curse of dimensionality.
Real embeddings have cluster structure, which is what this benchmark measures.

Run: python benchmarks/bench_vector_search.py
"""

import sys
import time

import numpy as np

from ironrace._core import VectorIndex


def generate_clustered_vectors(n, dim=768, n_clusters=50):
    """Generate clustered unit vectors that simulate real embeddings.

    Real embeddings from models like sentence-transformers have structure —
    similar documents cluster together. This generates vectors with that property.
    """
    rng = np.random.default_rng(42)
    n_per_cluster = n // n_clusters
    remainder = n - (n_per_cluster * n_clusters)

    # Generate cluster centers
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    # Generate points around each center with noise
    chunks = []
    for i, center in enumerate(centers):
        count = n_per_cluster + (1 if i < remainder else 0)
        noise = rng.standard_normal((count, dim)).astype(np.float32) * 0.1
        chunks.append(center + noise)

    vecs = np.vstack(chunks).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def brute_force_top_k(query, vectors_np, top_k=10):
    """Ground truth: brute-force cosine similarity (guaranteed correct)."""
    sims = vectors_np @ query
    top_indices = np.argsort(sims)[::-1][:top_k]
    return set(top_indices.tolist())


print("=" * 64)
print("  IRONRACE — VECTOR SEARCH SCALING BENCHMARK")
print("=" * 64)

scales = [1000, 10000]

if "--full" in sys.argv:
    scales.append(100000)

for n in scales:
    n_clusters = max(10, n // 200)
    print(f"\n{'─' * 64}")
    print(f"  {n:,} vectors × 768 dimensions ({n_clusters} clusters)")
    print(f"{'─' * 64}")

    print(f"  Generating vectors...", end=" ", flush=True)
    t0 = time.perf_counter()
    vectors_np = generate_clustered_vectors(n, n_clusters=n_clusters)
    vectors = vectors_np.tolist()
    gen_time = (time.perf_counter() - t0) * 1000
    print(f"done ({gen_time:.0f}ms)")

    # Build
    print(f"  Building HNSW index...", end=" ", flush=True)
    t0 = time.perf_counter()
    idx = VectorIndex(vectors)
    build_time = (time.perf_counter() - t0) * 1000
    print(f"done ({build_time:.0f}ms)")

    # Query performance
    query = vectors[n // 2]
    search_times = []
    for _ in range(10):  # warmup
        idx.search(query, 10)
    for _ in range(100):
        t0 = time.perf_counter()
        results = idx.search(query, 10)
        search_times.append((time.perf_counter() - t0) * 1000)

    search_times.sort()
    mean_ms = sum(search_times) / len(search_times)
    p99_ms = search_times[int(len(search_times) * 0.99)]

    # Verify correctness: top-1 self-search
    top_idx, top_score = results[0]
    correct = top_idx == n // 2

    # Recall@10: compare HNSW results against brute-force ground truth
    n_recall_queries = min(100, n // 10)
    query_indices = np.linspace(0, n - 1, n_recall_queries, dtype=int)
    total_recall = 0
    for qi in query_indices:
        q = vectors[qi]
        hnsw_ids = set(r[0] for r in idx.search(q, 10))
        gt_ids = brute_force_top_k(vectors_np[qi], vectors_np, 10)
        total_recall += len(hnsw_ids & gt_ids) / 10.0
    avg_recall = total_recall / n_recall_queries

    print(f"\n  {'Build time':<20} {build_time:>10.1f}ms")
    print(f"  {'Search mean':<20} {mean_ms:>10.3f}ms")
    print(f"  {'Search P99':<20} {p99_ms:>10.3f}ms")
    print(f"  {'Top result correct':<20} {'✓' if correct else '✗'} (idx={top_idx}, score={top_score:.4f})")
    print(f"  {'Recall@10':<20} {avg_recall:>9.0%} ({n_recall_queries} queries vs brute-force)")
    print(f"  {'Searches/sec':<20} {1000/mean_ms:>10,.0f}")

print(f"\n{'=' * 64}")
print(f"  Use --full flag to include 100K vector benchmark")
print(f"{'=' * 64}")
