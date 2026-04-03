#!/usr/bin/env python3
"""
IronRace vs SimpleVectorStore — LlamaIndex Head-to-Head Benchmark
====================================================================
Compares LlamaIndex's default brute-force vector store against
IronRace's Rust HNSW backend, through the LlamaIndex API.

Run: python benchmarks/bench_vs_simple.py
"""

import math
import os
import random
import time

# Use mock embed model (no OpenAI key needed)
os.environ["IS_TESTING"] = "true"

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import QueryBundle, TextNode
from llama_index.core.vector_stores import SimpleVectorStore

from llama_index.vector_stores.ironrace import IronRaceVectorStore

random.seed(42)


def generate_nodes(n, dim=128):
    """Generate TextNodes with random unit-vector embeddings."""
    nodes = []
    for i in range(n):
        vec = [random.gauss(0, 1) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in vec))
        nodes.append(
            TextNode(
                text=f"Document {i}: " + " ".join(
                    random.choices(["AI", "data", "cloud", "platform", "agent", "model", "API"], k=20)
                ),
                id_=f"doc_{i}",
                embedding=[x / norm for x in vec],
                metadata={"index": i, "category": random.choice(["tech", "finance", "health"])},
            )
        )
    return nodes


def benchmark_retrieve(retriever, query_embedding, iterations=50, warmup=5):
    """Benchmark retriever.retrieve() through the LlamaIndex API."""
    query = QueryBundle(query_str="test query", embedding=query_embedding)

    for _ in range(warmup):
        retriever.retrieve(query)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        results = retriever.retrieve(query)
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return {
        "mean_ms": sum(times) / len(times),
        "median_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95)],
        "p99_ms": times[int(len(times) * 0.99)],
        "min_ms": times[0],
        "n_results": len(results),
    }


def compute_recall(simple_results, af_results, k=10):
    """Compute recall@k: fraction of true top-k that HNSW finds."""
    true_ids = {r.node.node_id for r in simple_results[:k]}
    af_ids = {r.node.node_id for r in af_results[:k]}
    return len(true_ids & af_ids) / len(true_ids) if true_ids else 1.0


print("=" * 64)
print("  IRONRACE vs SIMPLEVECTORSTORE — LLAMAINDEX BENCHMARK")
print("=" * 64)

TOP_K = 10

for n_docs in [100, 500, 1000]:
    print(f"\n{'─' * 64}")
    print(f"  {n_docs:,} DOCUMENTS × 128 dimensions, top_k={TOP_K}")
    print(f"{'─' * 64}")

    nodes = generate_nodes(n_docs)
    query_embedding = nodes[n_docs // 2].embedding  # known vector

    # ── Build SimpleVectorStore index ──
    print(f"  Building SimpleVectorStore index...", end=" ", flush=True)
    t0 = time.perf_counter()
    simple_store = SimpleVectorStore()
    simple_ctx = StorageContext.from_defaults(vector_store=simple_store)
    simple_index = VectorStoreIndex(nodes=nodes, storage_context=simple_ctx)
    simple_build = (time.perf_counter() - t0) * 1000
    print(f"done ({simple_build:.0f}ms)")

    # ── Build IronRace index ──
    print(f"  Building IronRace index...", end=" ", flush=True)
    t0 = time.perf_counter()
    af_store = IronRaceVectorStore(ef_construction=40)
    af_ctx = StorageContext.from_defaults(vector_store=af_store)
    af_index = VectorStoreIndex(nodes=nodes, storage_context=af_ctx)
    af_build = (time.perf_counter() - t0) * 1000
    print(f"done ({af_build:.0f}ms)")

    # ── Benchmark retrieval ──
    simple_retriever = simple_index.as_retriever(similarity_top_k=TOP_K)
    af_retriever = af_index.as_retriever(similarity_top_k=TOP_K)

    print(f"  Benchmarking retrieval (50 iterations)...")
    simple_stats = benchmark_retrieve(simple_retriever, query_embedding)
    af_stats = benchmark_retrieve(af_retriever, query_embedding)

    speedup = simple_stats["mean_ms"] / max(af_stats["mean_ms"], 0.001)

    # ── Compute recall ──
    query = QueryBundle(query_str="test", embedding=query_embedding)
    simple_results = simple_retriever.retrieve(query)
    af_results = af_retriever.retrieve(query)
    recall = compute_recall(simple_results, af_results, TOP_K)

    # ── Results ──
    print(f"\n  {'':>16} {'Simple':>14} {'IronRace':>14} {'Speedup':>10}")
    print(f"  {'─' * 56}")
    print(f"  {'Build time':<16} {simple_build:>12.0f}ms {af_build:>12.0f}ms")
    print(f"  {'Query mean':<16} {simple_stats['mean_ms']:>12.3f}ms {af_stats['mean_ms']:>12.3f}ms {speedup:>9.1f}x")
    print(f"  {'Query median':<16} {simple_stats['median_ms']:>12.3f}ms {af_stats['median_ms']:>12.3f}ms")
    print(f"  {'Query P95':<16} {simple_stats['p95_ms']:>12.3f}ms {af_stats['p95_ms']:>12.3f}ms")
    print(f"  {'Query P99':<16} {simple_stats['p99_ms']:>12.3f}ms {af_stats['p99_ms']:>12.3f}ms")
    print(f"  {'Recall@{TOP_K}':<16} {'—':>14} {recall:>13.0%}")
    print(f"  {'Results':<16} {simple_stats['n_results']:>14} {af_stats['n_results']:>14}")

print(f"\n{'=' * 64}")
print(f"  NOTES")
print(f"{'=' * 64}")
print("""
  - SimpleVectorStore uses brute-force cosine similarity (exact)
  - IronRace uses HNSW approximate nearest neighbor (Rust)
  - Build time includes LlamaIndex overhead (node processing, etc.)
  - Query times are through the full LlamaIndex retriever API
  - Recall@k measures HNSW accuracy vs brute-force ground truth
  - Speedup increases with document count (HNSW is O(log n) vs O(n))
""")
