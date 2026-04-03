#!/usr/bin/env python3
"""
IronRace Example — Batch Research Pipelines
===============================================
Run 100 parallel research context preparations to demonstrate throughput.

Usage:
    python examples/batch_research.py
"""

import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor

random.seed(42)

from ironrace import VectorIndex, compile_agents_dag, execute_pipeline


# ═══════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════

RESEARCH_TOPICS = [
    "AI-powered code review tools",
    "Autonomous drone delivery systems",
    "Personalized nutrition AI",
    "Quantum computing cloud platforms",
    "Real-time language translation earbuds",
    "AI financial advisor for Gen Z",
    "Smart agriculture monitoring",
    "AI-powered legal document analysis",
    "Virtual reality fitness platform",
    "AI customer support automation",
    "Blockchain supply chain tracking",
    "Mental health chatbot therapy",
    "Autonomous trucking fleet management",
    "AI music composition platform",
    "Smart home energy optimization",
    "AI-powered recruitment screening",
    "Telemedicine AI triage system",
    "Autonomous parking garage robots",
    "AI food waste reduction platform",
    "Personalized education AI tutor",
]


def generate_knowledge_base(n=200, dim=128):
    """Generate a compact knowledge base for the demo."""
    vectors = []
    for _ in range(n):
        v = [random.gauss(0, 1) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in v))
        vectors.append([x / norm for x in v])
    return vectors


def main():
    print("\nIronRace Batch Research Example")
    print("=" * 50)

    # Build knowledge base
    print("\nBuilding knowledge base...", end=" ", flush=True)
    t0 = time.perf_counter()
    kb = generate_knowledge_base()
    index = VectorIndex(kb, ef_construction=50)
    build_time = (time.perf_counter() - t0) * 1000
    print(f"done ({build_time:.0f}ms, {len(kb)} vectors × 128d)")

    # Prepare batch of research queries
    n_queries = 100
    topics = [random.choice(RESEARCH_TOPICS) for _ in range(n_queries)]

    template = """You are a startup research analyst.

MARKET CONTEXT:
{context}

Research this startup idea thoroughly. Cover:
1. Market size and growth trajectory
2. Key competitors and their funding
3. Technical feasibility assessment
4. Revenue model analysis
5. Risk factors

TOPIC: {topic}

Provide a structured research brief."""

    # Build all DAGs upfront
    print(f"Building {n_queries} pipeline DAGs...", end=" ", flush=True)
    t0 = time.perf_counter()

    context_data = "Startup ecosystem data: Average seed round $2.5M, Series A $15M. " * 5
    dag_jsons = []
    for topic in topics:
        dag_json = compile_agents_dag([{
            "id": "research",
            "template": template,
            "values": {"context": context_data, "topic": topic},
            "budgets": {"context": 200},
        }])
        dag_jsons.append(dag_json)
    dag_build_time = (time.perf_counter() - t0) * 1000
    print(f"done ({dag_build_time:.1f}ms)")

    # Execute sequentially
    print(f"\nRunning {n_queries} pipelines sequentially...", end=" ", flush=True)
    t0 = time.perf_counter()
    for dag_json in dag_jsons:
        execute_pipeline(dag_json)
    seq_time = (time.perf_counter() - t0) * 1000
    print(f"done ({seq_time:.1f}ms, {seq_time/n_queries:.2f}ms/pipeline)")

    # Execute with thread pool (concurrent)
    print(f"Running {n_queries} pipelines concurrently (32 threads)...", end=" ", flush=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(execute_pipeline, dag_jsons))
    conc_time = (time.perf_counter() - t0) * 1000
    print(f"done ({conc_time:.1f}ms, {conc_time/n_queries:.2f}ms/pipeline)")

    # Verify results
    sample = json.loads(results[0])
    sample_result = sample["research"]

    print(f"\n{'─' * 50}")
    print(f"  RESULTS")
    print(f"{'─' * 50}")
    print(f"  Pipelines executed:    {n_queries}")
    print(f"  Sequential time:       {seq_time:.1f}ms ({n_queries*1000/seq_time:,.0f} pipelines/sec)")
    print(f"  Concurrent time:       {conc_time:.1f}ms ({n_queries*1000/conc_time:,.0f} pipelines/sec)")
    print(f"  Speedup:               {seq_time/conc_time:.1f}x")
    print(f"  Sample prompt tokens:  {sample_result['total_tokens']}")

    # Scale projection
    print(f"\n{'─' * 50}")
    print(f"  SCALE PROJECTION")
    print(f"{'─' * 50}")
    per_pipeline_ms = conc_time / n_queries
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        total_ms = per_pipeline_ms * n
        total_sec = total_ms / 1000
        print(f"  {n:>10,} pipelines: {total_sec:>8.1f}s CPU time ({n*1000/total_ms:>10,.0f}/sec)")


if __name__ == "__main__":
    main()
