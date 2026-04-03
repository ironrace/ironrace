#!/usr/bin/env python3
"""
IronRace — Concurrent Pipeline Scale Benchmark
=================================================
Simulates 100/1000/10000 concurrent pipeline invocations
to measure CPU time scaling.

Run: python benchmarks/bench_at_scale.py
"""

import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor

random.seed(42)

from benchmarks.baseline_python import (
    cosine_similarity_python,
    full_pipeline_python,
    generate_api_response,
    generate_embedding,
)

from ironrace._core import VectorIndex, execute_pipeline


# ── Test Data ──
print("Generating test data...")
KB_SIZE = 1000  # smaller for concurrent test
KNOWLEDGE_BASE = [generate_embedding() for _ in range(KB_SIZE)]
API_RESPONSE = generate_api_response(20)
API_RESPONSE_JSON = json.dumps(API_RESPONSE)
QUERY = generate_embedding()
IDEA = "AI Travel Agent - Trippr.com"

# Pre-build Rust index
RUST_INDEX = VectorIndex(KNOWLEDGE_BASE)

# Pre-build DAG JSON
TEMPLATE = "System: {system}\nContext: {ctx}\nQuery: {query}"
VALUES = {"system": "You are an analyst.", "ctx": "Competitor data here. " * 10, "query": IDEA}
DAG_JSON = json.dumps({
    "nodes": [
        {"id": "search", "op": {"type": "vector_search", "vectors": KNOWLEDGE_BASE, "query": QUERY, "top_k": 10, "ef_construction": 100}, "depends_on": []},
        {"id": "parse", "op": {"type": "json_parse", "data": API_RESPONSE_JSON}, "depends_on": []},
        {"id": "assemble", "op": {"type": "assemble", "template": TEMPLATE, "values": VALUES, "budgets": {"ctx": 50}}, "depends_on": ["search", "parse"]},
    ]
})
print("Ready.\n")


def run_python_pipeline(_):
    return full_pipeline_python(IDEA, QUERY, KNOWLEDGE_BASE, API_RESPONSE_JSON)


def run_rust_pipeline(_):
    return execute_pipeline(DAG_JSON)


print("=" * 64)
print("  IRONRACE — CONCURRENT PIPELINE SCALING")
print("=" * 64)

for n_concurrent in [10, 100, 1000]:
    print(f"\n{'─' * 64}")
    print(f"  {n_concurrent:,} CONCURRENT PIPELINE INVOCATIONS")
    print(f"{'─' * 64}")

    # Python
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(n_concurrent, 32)) as pool:
        list(pool.map(run_python_pipeline, range(n_concurrent)))
    py_wall = (time.perf_counter() - t0) * 1000

    # IronRace (Rust)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(n_concurrent, 32)) as pool:
        list(pool.map(run_rust_pipeline, range(n_concurrent)))
    rs_wall = (time.perf_counter() - t0) * 1000

    speedup = py_wall / max(rs_wall, 0.01)
    py_per = py_wall / n_concurrent
    rs_per = rs_wall / n_concurrent

    print(f"\n  {'':>20} {'Python':>14} {'IronRace':>14} {'Speedup':>10}")
    print(f"  {'─' * 60}")
    print(f"  {'Total wall-clock':<20} {py_wall:>12.1f}ms {rs_wall:>12.1f}ms {speedup:>9.1f}x")
    print(f"  {'Per pipeline':<20} {py_per:>12.3f}ms {rs_per:>12.3f}ms")
    print(f"  {'Throughput':<20} {n_concurrent*1000/py_wall:>10,.0f}/s {n_concurrent*1000/rs_wall:>10,.0f}/s")


# ── Cost Projection ──
print(f"\n{'=' * 64}")
print(f"  MONTHLY COST AT SCALE")
print(f"{'=' * 64}")

# Use per-pipeline times from the 1000-concurrent run
print(f"\n  {'Daily invocations':>25} {'Python $/mo':>14} {'IronRace $/mo':>16} {'Savings':>12}")
print(f"  {'─' * 70}")
for daily in [100_000, 1_000_000, 10_000_000]:
    daily_pipelines = daily * 6  # 6 agents per invocation
    py_cpu_sec = daily_pipelines * (py_per / 1000)
    rs_cpu_sec = daily_pipelines * (rs_per / 1000)
    cost_per_vcpu_month = 0.34 / 8 * 730  # c5.2xlarge
    py_cost = (py_cpu_sec / 86400) * cost_per_vcpu_month
    rs_cost = (rs_cpu_sec / 86400) * cost_per_vcpu_month
    print(f"  {daily:>25,} ${py_cost:>12,.0f} ${rs_cost:>14,.0f} ${py_cost-rs_cost:>10,.0f}")
