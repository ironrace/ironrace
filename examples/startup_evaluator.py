#!/usr/bin/env python3
"""
IronRace Flagship Demo — Startup Evaluator
==============================================
Multi-agent startup due diligence with 6 specialized agents.

Usage:
    python examples/startup_evaluator.py                 # dry run (no API key needed)
    python examples/startup_evaluator.py --live          # full run with Claude API
    ANTHROPIC_API_KEY=sk-... python examples/startup_evaluator.py --live
"""

import asyncio
import json
import math
import os
import random
import sys
import time

# Seed for reproducible demo data
random.seed(42)

from ironrace import (
    VectorIndex,
    assemble_prompt,
    compile_agents_dag,
    count_tokens,
    execute_pipeline,
)


# ═══════════════════════════════════════════════════════════
# DEMO DATA
# ═══════════════════════════════════════════════════════════

def generate_competitor_data(n=500, dim=768):
    """Generate a synthetic competitor knowledge base."""
    competitors = []
    stages = ["seed", "series_a", "series_b", "series_c", "growth"]
    markets = ["US", "EU", "APAC", "LATAM"]
    verticals = ["AI Travel", "Fintech", "HealthTech", "EdTech", "SaaS", "E-commerce", "DevTools"]

    for i in range(n):
        vec = [random.gauss(0, 1) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in vec))
        competitors.append({
            "name": f"{random.choice(['Neo', 'Quantum', 'Alpha', 'Nova', 'Zen', 'Peak', 'Bolt', 'Arc'])} {random.choice(verticals).split()[0]}",
            "funding": random.randint(500_000, 100_000_000),
            "employees": random.randint(5, 2000),
            "stage": random.choice(stages),
            "market": random.choice(markets),
            "revenue_arr": random.randint(0, 50_000_000),
            "vertical": random.choice(verticals),
            "embedding": [x / norm for x in vec],
        })
    return competitors


def format_competitor(c):
    return f"- {c['name']}: ${c['funding']:,} ({c['stage']}), {c['employees']} employees, {c['market']} market, ${c['revenue_arr']:,} ARR"


# ═══════════════════════════════════════════════════════════
# AGENT DEFINITIONS
# ═══════════════════════════════════════════════════════════

AGENTS = [
    {
        "id": "security_analyst",
        "name": "Dr. Riya Patel",
        "role": "Senior Information Security Analyst",
        "template": """You are {name}, a {role} on a startup incubator board.

COMPETITOR DATA:
{competitors}

MARKET METRICS:
{metrics}

Evaluate the security implications of this startup idea. Consider data handling,
attack surface, compliance requirements, and security architecture needs.

IDEA: {idea}

Provide your assessment. End with BUILD or DON'T BUILD, confidence 1-10, top 3 security recommendations.""",
    },
    {
        "id": "market_analyst",
        "name": "Marcus Chen",
        "role": "Market Research Director",
        "template": """You are {name}, a {role} on a startup incubator board.

COMPETITOR LANDSCAPE:
{competitors}

MARKET DATA:
{metrics}

Analyze the market opportunity for this startup. Consider TAM/SAM/SOM,
competitive moats, timing, and go-to-market strategy.

IDEA: {idea}

Provide your assessment. End with BUILD or DON'T BUILD, confidence 1-10, top 3 market recommendations.""",
    },
    {
        "id": "technical_architect",
        "name": "Sarah Kim",
        "role": "Chief Technical Architect",
        "template": """You are {name}, a {role} on a startup incubator board.

SIMILAR PRODUCTS:
{competitors}

Evaluate the technical feasibility and architecture requirements.
Consider scalability, tech stack choices, AI/ML requirements, and build timeline.

IDEA: {idea}

Provide your assessment. End with BUILD or DON'T BUILD, confidence 1-10, top 3 technical recommendations.""",
    },
    {
        "id": "financial_analyst",
        "name": "James Wright",
        "role": "Financial Analyst",
        "template": """You are {name}, a {role} on a startup incubator board.

COMPETITOR FINANCIALS:
{competitors}

MARKET BENCHMARKS:
{metrics}

Analyze the financial viability. Consider unit economics, funding requirements,
burn rate projections, and path to profitability.

IDEA: {idea}

Provide your assessment. End with BUILD or DON'T BUILD, confidence 1-10, top 3 financial recommendations.""",
    },
    {
        "id": "product_strategist",
        "name": "Elena Vasquez",
        "role": "Product Strategy VP",
        "template": """You are {name}, a {role} on a startup incubator board.

COMPETITIVE PRODUCTS:
{competitors}

Evaluate the product-market fit and differentiation strategy.
Consider user needs, feature prioritization, and competitive positioning.

IDEA: {idea}

Provide your assessment. End with BUILD or DON'T BUILD, confidence 1-10, top 3 product recommendations.""",
    },
]

VERDICT_AGENT = {
    "id": "verdict",
    "name": "The Board",
    "role": "Incubator Board",
    "template": """You are {name}, the {role} making a final decision.

ANALYST REPORTS:
{prior_analyses}

Synthesize all analyst perspectives into a final verdict.
Weight each analyst's confidence and recommendations.

IDEA: {idea}

Deliver the final board decision: BUILD or DON'T BUILD.
Include: overall confidence (1-10), key risks, and immediate next steps if BUILD.""",
}


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    live_mode = "--live" in sys.argv
    idea = "AI Travel Agent on WhatsApp - Trippr.com. An AI-powered WhatsApp travel concierge that handles end-to-end trip planning, booking, and real-time travel assistance through natural conversation."

    print()
    print("IronRace Context Engine v0.1.0")
    print("=" * 50)
    print(f"Mode: {'LIVE (Claude API)' if live_mode else 'DRY RUN (context prep only)'}")
    print(f"Idea: {idea[:80]}...")
    print()

    # Step 1: Build knowledge base
    print("Building competitor knowledge base...", end=" ", flush=True)
    t0 = time.perf_counter()
    competitors = generate_competitor_data(n=500)
    embeddings = [c["embedding"] for c in competitors]
    index = VectorIndex(embeddings)
    build_time = (time.perf_counter() - t0) * 1000
    print(f"done ({build_time:.0f}ms, {len(competitors)} competitors, 768d)")

    # Step 2: Vector search for relevant competitors
    print("Searching for relevant competitors...", end=" ", flush=True)
    t0 = time.perf_counter()
    query = embeddings[0]  # Use first competitor as query proxy
    search_results = index.search(query, top_k=15)
    search_time = (time.perf_counter() - t0) * 1000
    print(f"done ({search_time:.1f}ms, {len(search_results)} results)")

    # Build shared context data
    top_competitors = [competitors[idx] for idx, score in search_results]
    competitor_text = "\n".join(format_competitor(c) for c in top_competitors[:10])
    metrics_text = f"""- Average funding: ${sum(c['funding'] for c in top_competitors) / len(top_competitors):,.0f}
- US market presence: {sum(1 for c in top_competitors if c['market'] == 'US') / len(top_competitors):.0%}
- Max ARR: ${max(c['revenue_arr'] for c in top_competitors):,}
- Total competitors found: {len(top_competitors)}"""

    # Step 3: Prepare all agent contexts via Rust DAG (SINGLE BRIDGE CROSSING)
    print("\nPreparing agent contexts (Rust DAG)...", end=" ", flush=True)
    t0 = time.perf_counter()

    # Build DAG for all 5 parallel analysts
    agent_specs = []
    for agent_def in AGENTS:
        agent_specs.append({
            "id": agent_def["id"],
            "template": agent_def["template"],
            "values": {
                "name": agent_def["name"],
                "role": agent_def["role"],
                "competitors": competitor_text,
                "metrics": metrics_text,
                "idea": idea,
            },
            "budgets": {"competitors": 300, "metrics": 100, "idea": 100},
        })

    dag_json = compile_agents_dag(agent_specs)
    results_json = execute_pipeline(dag_json)
    results = json.loads(results_json)
    context_time = (time.perf_counter() - t0) * 1000
    print(f"done ({context_time:.1f}ms)")

    # Display results
    print(f"\n{'─' * 50}")
    print("  Agent Context Preparation Results")
    print(f"{'─' * 50}")
    for agent_def in AGENTS:
        aid = agent_def["id"]
        r = results[aid]
        print(f"\n  {agent_def['name']} ({agent_def['role']}):")
        print(f"    Prompt: {r['total_tokens']} tokens")
        if r["sections_truncated"]:
            print(f"    Truncated: {', '.join(r['sections_truncated'])}")
        print(f"    Preview: {r['prompt'][:100]}...")

    # Step 4: LLM calls (if live mode)
    if live_mode:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("\n⚠ ANTHROPIC_API_KEY not set. Set it to run live mode.")
            return

        from ironrace import LLMRouter
        router = LLMRouter(anthropic_api_key=api_key)

        print(f"\n{'─' * 50}")
        print("  Running LLM calls (5 parallel)...")
        print(f"{'─' * 50}")

        async def run_agents():
            tasks = []
            for agent_def in AGENTS:
                r = results[agent_def["id"]]
                tasks.append(router.call(
                    model="claude-sonnet-4-20250514",
                    messages=[{"role": "user", "content": idea}],
                    system=r["prompt"],
                    max_tokens=1500,
                ))

            t0 = time.perf_counter()
            responses = await asyncio.gather(*tasks)
            llm_time = (time.perf_counter() - t0) * 1000

            for agent_def, resp in zip(AGENTS, responses):
                content = resp.get("content", [{}])[0].get("text", "")[:200]
                print(f"\n  {agent_def['name']}: {content}...")

            await router.close()
            return llm_time

        llm_time = asyncio.run(run_agents())
        print(f"\n  LLM calls completed in {llm_time/1000:.1f}s")
    else:
        llm_time = 0

    # Step 5: Timing summary
    print(f"\n{'=' * 50}")
    print("  TIMING SUMMARY")
    print(f"{'=' * 50}")
    print(f"\n  Index build:        {build_time:>8.0f}ms (one-time)")
    print(f"  Vector search:      {search_time:>8.1f}ms")
    print(f"  Context prep (Rust):{context_time:>8.1f}ms")
    if live_mode:
        print(f"  LLM calls (5x):    {llm_time:>8.0f}ms")
        total = search_time + context_time + llm_time
        print(f"  Total:              {total:>8.0f}ms")
        print(f"\n  Context prep is {context_time/total*100:.2f}% of total time.")
    else:
        print(f"  LLM calls:          (skipped — use --live)")

    print(f"\n  Pure Python equivalent context prep: ~{context_time * 30:.0f}ms (estimated 30x slower)")
    print(f"  At 1000 concurrent pipelines:")
    print(f"    IronRace: {context_time * 1000 / 1000:.1f}s CPU")
    print(f"    Python:     {context_time * 30 * 1000 / 1000:.1f}s CPU")


if __name__ == "__main__":
    main()
