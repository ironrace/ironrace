#!/usr/bin/env python3
"""
IronRace Context Engine — Head-to-Head Benchmark (vs LlamaIndex)
====================================================================
Compares LlamaIndex context preparation vs IronRace Rust-accelerated operations.

Every baseline uses actual LlamaIndex (or its direct dependency, tiktoken)
so the comparison is apples-to-apples against a real production framework.

Run: python benchmarks/bench_context_prep.py
"""

import json
import os
import random
import sys
import time

# Ensure reproducible results
random.seed(42)

# Use mock embed model (no OpenAI key needed — we supply pre-computed embeddings)
os.environ["IS_TESTING"] = "true"

# ── LlamaIndex imports ──
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import QueryBundle, TextNode
from llama_index.core.vector_stores import SimpleVectorStore

import tiktoken

# ── IronRace imports ──
from ironrace._core import (
    VectorIndex,
    assemble_prompt,
    count_tokens,
    execute_pipeline,
    parse_json,
    serialize_json,
)

# ── Data generation (shared) ──
from benchmarks.baseline_python import (
    compute_features_python,
    generate_api_response,
)


# ═══════════════════════════════════════════════════════════
# TEST DATA GENERATION
# ═══════════════════════════════════════════════════════════

import numpy as np

print("Generating test data...")
KNOWLEDGE_BASE_SIZE = 5000
EMBEDDING_DIM = 384  # Matches docs/BENCHMARKS.md (5K vectors × 384d)

# Use clustered vectors that simulate real embeddings (NOT what we're benchmarking)
_rng = np.random.default_rng(42)
_n_clusters = 25
_centers = _rng.standard_normal((_n_clusters, EMBEDDING_DIM)).astype(np.float32)
_centers /= np.linalg.norm(_centers, axis=1, keepdims=True)
_n_per = KNOWLEDGE_BASE_SIZE // _n_clusters
_chunks = []
for _c in _centers:
    _noise = _rng.standard_normal((_n_per, EMBEDDING_DIM)).astype(np.float32) * 0.1
    _chunks.append(_c + _noise)
_kb_np = np.vstack(_chunks).astype(np.float32)
_kb_np /= np.linalg.norm(_kb_np, axis=1, keepdims=True)
KNOWLEDGE_BASE = _kb_np.tolist()

# Use a vector near a cluster center as the query (realistic: searching for similar docs)
_q_np = _kb_np[0] + _rng.standard_normal(EMBEDDING_DIM).astype(np.float32) * 0.05
_q_np /= np.linalg.norm(_q_np)
QUERY_EMBEDDING = _q_np.tolist()

API_RESPONSE = generate_api_response(50)
API_RESPONSE_JSON = json.dumps(API_RESPONSE)
API_RESPONSE_BYTES = API_RESPONSE_JSON.encode()
DOCUMENTS = API_RESPONSE["results"][:20]
FEATURES = compute_features_python(DOCUMENTS)
IDEA = "AI Travel Agent on WhatsApp - Trippr.com. A WhatsApp-based AI travel concierge."

print(f"  Knowledge base: {KNOWLEDGE_BASE_SIZE:,} vectors × {EMBEDDING_DIM}d")
print(f"  API response: {len(API_RESPONSE_JSON):,} bytes")
print(f"  Documents: {len(DOCUMENTS)} retrieved docs")

# ── Build LlamaIndex VectorStoreIndex ──
print("  Building LlamaIndex SimpleVectorStore index...", end=" ", flush=True)
t0 = time.perf_counter()
_nodes = [
    TextNode(
        text=f"Document {i}",
        id_=f"doc_{i}",
        embedding=KNOWLEDGE_BASE[i],
    )
    for i in range(KNOWLEDGE_BASE_SIZE)
]
_simple_store = SimpleVectorStore()
_simple_ctx = StorageContext.from_defaults(vector_store=_simple_store)
LLAMA_INDEX = VectorStoreIndex(nodes=_nodes, storage_context=_simple_ctx)
LLAMA_RETRIEVER = LLAMA_INDEX.as_retriever(similarity_top_k=10)
llama_build_time = (time.perf_counter() - t0) * 1000
print(f"done ({llama_build_time:.0f}ms)")

# ── Build IronRace HNSW index ──
print("  Building IronRace HNSW index...", end=" ", flush=True)
t0 = time.perf_counter()
RUST_INDEX = VectorIndex(KNOWLEDGE_BASE)
rust_build_time = (time.perf_counter() - t0) * 1000
print(f"done ({rust_build_time:.0f}ms)")

# ── Prepare tiktoken encoder ──
TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")

# ── Prepare LlamaIndex prompt objects ──
LLAMA_PROMPT_TEMPLATE = PromptTemplate(
    "System: {system}\nContext: {context_str}\nUser: {query}"
)
LLAMA_PROMPT_HELPER = PromptHelper(context_window=4096, num_output=256)


# ═══════════════════════════════════════════════════════════
# BENCHMARK HARNESS
# ═══════════════════════════════════════════════════════════

def benchmark(name, func, iterations=200, warmup=10):
    """Run a function many times and return timing stats."""
    for _ in range(warmup):
        func()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return {
        "name": name,
        "iterations": iterations,
        "mean_ms": sum(times) / len(times),
        "median_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95)],
        "p99_ms": times[int(len(times) * 0.99)],
        "min_ms": times[0],
        "max_ms": times[-1],
    }


def print_comparison(llama_result, rust_result):
    """Print a side-by-side comparison."""
    speedup = llama_result["median_ms"] / max(rust_result["median_ms"], 0.001)
    print(f"\n  {'Metric':<12} {'LlamaIndex':>12} {'IronRace':>12} {'Speedup':>10}")
    print(f"  {'─' * 48}")
    print(f"  {'Median':<12} {llama_result['median_ms']:>10.3f}ms {rust_result['median_ms']:>10.3f}ms {speedup:>9.1f}x")
    print(f"  {'Mean':<12} {llama_result['mean_ms']:>10.3f}ms {rust_result['mean_ms']:>10.3f}ms")
    print(f"  {'P95':<12} {llama_result['p95_ms']:>10.3f}ms {rust_result['p95_ms']:>10.3f}ms")
    print(f"  {'P99':<12} {llama_result['p99_ms']:>10.3f}ms {rust_result['p99_ms']:>10.3f}ms")
    return speedup


# ═══════════════════════════════════════════════════════════
# BENCHMARKS
# ═══════════════════════════════════════════════════════════

ITERS = 200

print("\n" + "=" * 64)
print("  IRONRACE CONTEXT ENGINE — PERFORMANCE BENCHMARK")
print("  Baseline: LlamaIndex (SimpleVectorStore + tiktoken)")
print("=" * 64)

# 1. Vector Search
print(f"\n{'─' * 64}")
print(f"  1. VECTOR SIMILARITY SEARCH ({KNOWLEDGE_BASE_SIZE:,} vectors × {EMBEDDING_DIM}d)")
print(f"{'─' * 64}")

_llama_query = QueryBundle(query_str="test", embedding=QUERY_EMBEDDING)
li = benchmark("LlamaIndex", lambda: LLAMA_RETRIEVER.retrieve(_llama_query), ITERS)
rs = benchmark("IronRace", lambda: RUST_INDEX.search(QUERY_EMBEDDING, 10), ITERS)
s1 = print_comparison(li, rs)

# Recall@10: compare HNSW results against LlamaIndex brute-force ground truth
_n_recall_queries = 50
_query_indices = np.linspace(0, KNOWLEDGE_BASE_SIZE - 1, _n_recall_queries, dtype=int)
_total_recall = 0
for _qi in _query_indices:
    _q = QueryBundle(query_str="test", embedding=KNOWLEDGE_BASE[_qi])
    _gt = set(r.node.node_id for r in LLAMA_RETRIEVER.retrieve(_q))
    _rs = set(f"doc_{i}" for i, _ in RUST_INDEX.search(KNOWLEDGE_BASE[_qi], 10))
    _total_recall += len(_gt & _rs) / 10.0
_avg_recall = _total_recall / _n_recall_queries
print(f"  Recall@10: {_avg_recall:.0%} ({_n_recall_queries} queries vs LlamaIndex brute-force)")

# 2. JSON Parsing
print(f"\n{'─' * 64}")
print(f"  2. JSON PARSING ({len(API_RESPONSE_JSON):,} byte API response)")
print(f"     LlamaIndex uses stdlib json internally — same baseline")
print(f"{'─' * 64}")
li = benchmark("LlamaIndex json", lambda: json.loads(API_RESPONSE_JSON), ITERS)
rs = benchmark("IronRace", lambda: parse_json(API_RESPONSE_BYTES), ITERS)
s2 = print_comparison(li, rs)

# 3. JSON Serialization
print(f"\n{'─' * 64}")
print(f"  3. JSON SERIALIZATION (API request body)")
print(f"     LlamaIndex uses stdlib json internally — same baseline")
print(f"{'─' * 64}")
test_obj = {"model": "claude-sonnet", "max_tokens": 1500, "messages": [{"role": "user", "content": "x" * 3000}]}
li = benchmark("LlamaIndex json", lambda: json.dumps(test_obj).encode(), ITERS)
rs = benchmark("IronRace", lambda: serialize_json(test_obj), ITERS)
s3 = print_comparison(li, rs)

# 4. Token Counting
print(f"\n{'─' * 64}")
print(f"  4. TOKEN COUNTING (1000-word text)")
print(f"     LlamaIndex uses tiktoken (Rust-based) — strong baseline")
print(f"{'─' * 64}")
long_text = "The quick brown fox jumps over the lazy dog. " * 100

def tiktoken_count():
    return len(TIKTOKEN_ENC.encode(long_text))

li = benchmark("LlamaIndex tiktoken", tiktoken_count, ITERS)
rs = benchmark("IronRace", lambda: count_tokens(long_text), ITERS)
s4 = print_comparison(li, rs)

# 5. Prompt Assembly
print(f"\n{'─' * 64}")
print(f"  5. PROMPT ASSEMBLY (PromptTemplate + PromptHelper truncation)")
print(f"{'─' * 64}")
template = "System: {system}\nContext: {context}\nUser: {query}"
values = {
    "system": "You are a senior analyst on a startup incubator board.",
    "context": "\n".join(f"- Competitor {i}: funding ${random.randint(1,50)}M, stage {random.choice(['seed','A','B'])}" for i in range(20)),
    "query": IDEA,
}
budgets = {"context": 200, "system": 50}

# LlamaIndex: PromptTemplate.format() + PromptHelper.truncate()
_context_chunks = values["context"].split("\n")

def llama_prompt_assembly():
    truncated = LLAMA_PROMPT_HELPER.truncate(LLAMA_PROMPT_TEMPLATE, _context_chunks)
    return LLAMA_PROMPT_TEMPLATE.format(
        system=values["system"],
        context_str="\n".join(truncated),
        query=values["query"],
    )

li = benchmark("LlamaIndex", llama_prompt_assembly, ITERS)
rs = benchmark("IronRace", lambda: assemble_prompt(template, values, budgets), ITERS)
s5 = print_comparison(li, rs)

# 6. Full Pipeline (pre-built index search + parse + assemble)
print(f"\n{'─' * 64}")
print(f"  6. FULL PIPELINE (search + parse + assembly, pre-built index)")
print(f"{'─' * 64}")

dag_json = json.dumps({
    "nodes": [
        {"id": "parse", "op": {"type": "json_parse", "data": API_RESPONSE_JSON}, "depends_on": []},
        {"id": "count", "op": {"type": "count_tokens", "text": long_text}, "depends_on": []},
        {"id": "assemble", "op": {"type": "assemble", "template": template, "values": values, "budgets": budgets}, "depends_on": ["parse", "count"]},
    ]
})

def full_llama_pipeline():
    LLAMA_RETRIEVER.retrieve(_llama_query)
    json.loads(API_RESPONSE_JSON)
    len(TIKTOKEN_ENC.encode(long_text))
    truncated = LLAMA_PROMPT_HELPER.truncate(LLAMA_PROMPT_TEMPLATE, _context_chunks)
    LLAMA_PROMPT_TEMPLATE.format(
        system=values["system"],
        context_str="\n".join(truncated),
        query=values["query"],
    )

def full_rust_pipeline():
    RUST_INDEX.search(QUERY_EMBEDDING, 10)
    execute_pipeline(dag_json)

li = benchmark("LlamaIndex full", full_llama_pipeline, ITERS)
rs = benchmark("IronRace full", full_rust_pipeline, ITERS)
s6 = print_comparison(li, rs)


# ═══════════════════════════════════════════════════════════
# SCALE PROJECTIONS
# ═══════════════════════════════════════════════════════════

li_single = li["median_ms"]
rs_single = rs["median_ms"]

print(f"\n{'=' * 64}")
print(f"  SCALE PROJECTIONS")
print(f"{'=' * 64}")
print(f"\n  Single pipeline context prep:")
print(f"    LlamaIndex:   {li_single:.2f}ms")
print(f"    IronRace:     {rs_single:.2f}ms")
print(f"    Speedup:      {li_single/max(rs_single, 0.001):.1f}x")

print(f"\n  {'Concurrent':<14} {'LlamaIndex':>14} {'IronRace':>14} {'Savings':>10}")
print(f"  {'pipelines':<14} {'(ms)':>14} {'(ms)':>14} {'':>10}")
print(f"  {'─' * 54}")
for n in [10, 100, 1000, 10000]:
    li_cpu = li_single * n
    rs_cpu = rs_single * n
    savings = li_cpu - rs_cpu
    print(f"  {n:<14,} {li_cpu:>12,.1f}ms {rs_cpu:>12,.1f}ms {savings:>9,.0f}ms")


# ═══════════════════════════════════════════════════════════
# COST PROJECTION
# ═══════════════════════════════════════════════════════════

print(f"\n{'=' * 64}")
print(f"  MONTHLY COMPUTE COST PROJECTION")
print(f"{'=' * 64}")
print(f"\n  Assumptions: 1M agent invocations/day, 6 agents each")

daily_pipelines = 1_000_000 * 6
li_per_sec = 1000 / li_single
rs_per_sec = 1000 / rs_single

li_cpu_sec = daily_pipelines / li_per_sec
rs_cpu_sec = daily_pipelines / rs_per_sec

li_vcpus = li_cpu_sec / 86400
rs_vcpus = rs_cpu_sec / 86400

cost_per_vcpu_month = 0.34 / 8 * 730
li_monthly = li_vcpus * cost_per_vcpu_month
rs_monthly = rs_vcpus * cost_per_vcpu_month

print(f"\n  {'':>25} {'LlamaIndex':>14} {'IronRace':>14}")
print(f"  {'─' * 55}")
print(f"  {'vCPUs needed (sustained)':<25} {li_vcpus:>12.1f} {rs_vcpus:>12.1f}")
print(f"  {'Monthly compute cost':<25} ${li_monthly:>11,.0f} ${rs_monthly:>11,.0f}")
print(f"  {'Monthly savings':<25} ${li_monthly - rs_monthly:>11,.0f}")
print(f"  {'Cost reduction':<25} {li_monthly/max(rs_monthly,1):>12.1f}x")


# ═══════════════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════════════

print(f"\n{'=' * 64}")
print(f"  VERDICT")
print(f"{'=' * 64}")
print(f"""
  Operation speedups (IronRace Rust vs LlamaIndex):
    Vector search ({KNOWLEDGE_BASE_SIZE:,} vectors):  {s1:>6.1f}x faster
    JSON parsing:                     {s2:>6.1f}x faster
    JSON serialization:               {s3:>6.1f}x faster
    Token counting (vs tiktoken):     {s4:>6.1f}x faster
    Prompt assembly:                  {s5:>6.1f}x faster
    Full pipeline:                    {s6:>6.1f}x faster

  Index build: LlamaIndex {llama_build_time:.0f}ms vs IronRace {rust_build_time:.0f}ms (one-time cost)

  At 1M invocations/day with 6 agents each:
    LlamaIndex: {li_vcpus:.1f} vCPUs, ${li_monthly:,.0f}/month
    IronRace:   {rs_vcpus:.1f} vCPUs, ${rs_monthly:,.0f}/month
    Savings:    ${li_monthly - rs_monthly:,.0f}/month ({li_monthly/max(rs_monthly,1):.0f}x reduction)

  Note: Token counting baseline is tiktoken (Rust-based), not pure Python.
  JSON baselines are stdlib json, which LlamaIndex uses internally.
""")
