#!/usr/bin/env python3
"""
IronRace — Vector Search Benchmark at Multiple Scales
========================================================
Measures HNSW index build time and query time at 1K, 10K, 100K scales.

Run: python benchmarks/bench_vector_search.py
"""

import math
import random
import time

random.seed(42)

from ironrace._core import VectorIndex


def generate_vectors(n, dim=768):
    """Generate n random unit vectors."""
    vectors = []
    for _ in range(n):
        v = [random.gauss(0, 1) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in v))
        vectors.append([x / norm for x in v])
    return vectors


print("=" * 64)
print("  IRONRACE — VECTOR SEARCH SCALING BENCHMARK")
print("=" * 64)

scales = [1000, 10000]

# Check if user wants to include 100K (slow to generate)
import sys
if "--full" in sys.argv:
    scales.append(100000)

for n in scales:
    print(f"\n{'─' * 64}")
    print(f"  {n:,} vectors × 768 dimensions")
    print(f"{'─' * 64}")

    print(f"  Generating vectors...", end=" ", flush=True)
    t0 = time.perf_counter()
    vectors = generate_vectors(n)
    gen_time = (time.perf_counter() - t0) * 1000
    print(f"done ({gen_time:.0f}ms)")

    # Build
    print(f"  Building HNSW index...", end=" ", flush=True)
    t0 = time.perf_counter()
    idx = VectorIndex(vectors)
    build_time = (time.perf_counter() - t0) * 1000
    print(f"done ({build_time:.0f}ms)")

    # Query
    query = vectors[n // 2]  # search for a known vector
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

    # Verify correctness
    top_idx, top_score = results[0]
    correct = top_idx == n // 2

    print(f"\n  {'Build time':<20} {build_time:>10.1f}ms")
    print(f"  {'Search mean':<20} {mean_ms:>10.3f}ms")
    print(f"  {'Search P99':<20} {p99_ms:>10.3f}ms")
    print(f"  {'Top result correct':<20} {'✓' if correct else '✗'} (idx={top_idx}, score={top_score:.4f})")
    print(f"  {'Searches/sec':<20} {1000/mean_ms:>10,.0f}")

print(f"\n{'=' * 64}")
print(f"  Use --full flag to include 100K vector benchmark")
print(f"{'=' * 64}")
