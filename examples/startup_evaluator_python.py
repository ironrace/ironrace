#!/usr/bin/env python3
"""
Pure Python Startup Evaluator — Baseline Comparison
=====================================================
Same logic as startup_evaluator.py but using pure Python for context prep.
Run this THEN run startup_evaluator.py to see the speed difference.

Usage:
    python examples/startup_evaluator_python.py
"""

import json
import math
import random
import time

random.seed(42)


# ═══════════════════════════════════════════════════════════
# PURE PYTHON IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════

def cosine_similarity_search(query, vectors, top_k=15):
    """Pure Python cosine similarity — no numpy, no Rust."""
    scores = []
    for i, vec in enumerate(vectors):
        dot = sum(a * b for a, b in zip(query, vec))
        scores.append((i, dot))
    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


def count_tokens_python(text):
    """Pure Python token counting."""
    count = 0
    for word in text.split():
        count += max(1, len(word) // 4 + 1)
    return count


def truncate_to_budget_python(text, max_tokens):
    """Pure Python truncation at sentence boundaries."""
    if count_tokens_python(text) <= max_tokens:
        return text
    sentences = []
    tokens_used = 0
    for part in text.replace('. ', '.|').replace('! ', '!|').replace('? ', '?|').split('|'):
        t = count_tokens_python(part)
        if tokens_used + t > max_tokens:
            break
        sentences.append(part)
        tokens_used += t
    return ' '.join(sentences) if sentences else text[:max_tokens * 4]


def assemble_prompt_python(template, values, budgets):
    """Pure Python prompt assembly with token budgeting."""
    result = template
    sections_truncated = []
    token_breakdown = {}

    for key, value in values.items():
        placeholder = '{' + key + '}'
        if placeholder not in result:
            continue
        if key in budgets:
            truncated = truncate_to_budget_python(value, budgets[key])
            if len(truncated) < len(value):
                sections_truncated.append(key)
            final_value = truncated
        else:
            final_value = value
        token_breakdown[key] = count_tokens_python(final_value)
        result = result.replace(placeholder, final_value)

    return {
        "prompt": result,
        "total_tokens": count_tokens_python(result),
        "sections_truncated": sections_truncated,
        "token_breakdown": token_breakdown,
    }


# ═══════════════════════════════════════════════════════════
# DEMO DATA (same as startup_evaluator.py)
# ═══════════════════════════════════════════════════════════

def generate_competitor_data(n=500, dim=768):
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

Evaluate the security implications of this startup idea.

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

Analyze the market opportunity for this startup.

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

Analyze the financial viability.

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

IDEA: {idea}

Provide your assessment. End with BUILD or DON'T BUILD, confidence 1-10, top 3 product recommendations.""",
    },
]


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    idea = "AI Travel Agent on WhatsApp - Trippr.com. An AI-powered WhatsApp travel concierge that handles end-to-end trip planning, booking, and real-time travel assistance through natural conversation."

    print()
    print("Pure Python Context Engine (BASELINE)")
    print("=" * 50)
    print(f"Idea: {idea[:80]}...")
    print()

    # Step 1: Build knowledge base
    print("Building competitor knowledge base...", end=" ", flush=True)
    t0 = time.perf_counter()
    competitors = generate_competitor_data(n=500)
    embeddings = [c["embedding"] for c in competitors]
    build_time = (time.perf_counter() - t0) * 1000
    print(f"done ({build_time:.0f}ms, {len(competitors)} competitors, 768d)")

    # Step 2: Vector search (PURE PYTHON — this is the slow part)
    print("Searching for relevant competitors...", end=" ", flush=True)
    t0 = time.perf_counter()
    query = embeddings[0]
    search_results = cosine_similarity_search(query, embeddings, top_k=15)
    search_time = (time.perf_counter() - t0) * 1000
    print(f"done ({search_time:.1f}ms, {len(search_results)} results)")

    # Build shared context
    top_competitors = [competitors[idx] for idx, score in search_results]
    competitor_text = "\n".join(format_competitor(c) for c in top_competitors[:10])
    metrics_text = f"""- Average funding: ${sum(c['funding'] for c in top_competitors) / len(top_competitors):,.0f}
- US market presence: {sum(1 for c in top_competitors if c['market'] == 'US') / len(top_competitors):.0%}
- Max ARR: ${max(c['revenue_arr'] for c in top_competitors):,}
- Total competitors found: {len(top_competitors)}"""

    # Step 3: Prepare all agent contexts (PURE PYTHON)
    print("\nPreparing agent contexts (Pure Python)...", end=" ", flush=True)
    t0 = time.perf_counter()

    results = {}
    for agent_def in AGENTS:
        results[agent_def["id"]] = assemble_prompt_python(
            agent_def["template"],
            {
                "name": agent_def["name"],
                "role": agent_def["role"],
                "competitors": competitor_text,
                "metrics": metrics_text,
                "idea": idea,
            },
            {"competitors": 300, "metrics": 100, "idea": 100},
        )
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

    # Timing summary
    total_prep = search_time + context_time
    print(f"\n{'=' * 50}")
    print("  TIMING SUMMARY (Pure Python)")
    print(f"{'=' * 50}")
    print(f"\n  Data generation:    {build_time:>8.0f}ms")
    print(f"  Vector search:      {search_time:>8.1f}ms")
    print(f"  Context assembly:   {context_time:>8.1f}ms")
    print(f"  Total context prep: {total_prep:>8.1f}ms")
    print(f"\n  At 1000 concurrent pipelines:")
    print(f"    Total CPU time: {total_prep * 1000 / 1000:.1f}s")
    print()
    print("  Now run: python examples/startup_evaluator.py")
    print("  to see the IronRace (Rust) version.")
    print()


if __name__ == "__main__":
    main()
