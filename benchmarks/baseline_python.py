"""Pure Python baseline implementations for benchmarking.

These are the operations IronRace accelerates with Rust.
Used as the 'before' comparison in bench_context_prep.py.
"""

import json
import math
import random
import string
import time


def generate_embedding(dim=768):
    """Generate a random unit vector."""
    v = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def generate_document():
    """Simulate a retrieved document with metadata."""
    return {
        "id": "".join(random.choices(string.ascii_lowercase, k=12)),
        "title": " ".join(
            random.choices(
                ["AI", "Travel", "Booking", "Platform", "Agent", "API", "Cloud"],
                k=random.randint(3, 6),
            )
        ),
        "content": " ".join(random.choices(string.ascii_lowercase, k=random.randint(200, 500))),
        "metadata": {
            "funding": random.randint(100000, 50000000),
            "employees": random.randint(5, 500),
            "founded": random.randint(2015, 2025),
            "market": random.choice(["US", "EU", "APAC"]),
            "stage": random.choice(["seed", "series_a", "series_b", "series_c"]),
            "revenue_arr": random.randint(0, 10000000),
        },
        "embedding": generate_embedding(),
        "score": random.random(),
    }


def generate_api_response(n_results=20):
    """Simulate a competitor data API response."""
    return {
        "status": "ok",
        "count": n_results,
        "results": [generate_document() for _ in range(n_results)],
        "pagination": {"page": 1, "total_pages": 5},
        "meta": {"query_time_ms": random.randint(50, 200)},
    }


def cosine_similarity_python(query, vectors, top_k=10):
    """Pure Python cosine similarity search."""
    scores = []
    for i, vec in enumerate(vectors):
        dot = sum(a * b for a, b in zip(query, vec))
        scores.append((i, dot))
    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


def count_tokens_python(text):
    """Approximate token counting in pure Python."""
    words = text.split()
    count = 0
    for word in words:
        count += max(1, len(word) // 4 + 1)
    return count


def compute_features_python(documents):
    """Pure Python feature aggregation."""
    if not documents:
        return {}
    total_funding = 0
    total_employees = 0
    us_count = 0
    max_revenue = 0
    for doc in documents:
        meta = doc.get("metadata", {})
        total_funding += meta.get("funding", 0)
        total_employees += meta.get("employees", 0)
        if meta.get("market") == "US":
            us_count += 1
        rev = meta.get("revenue_arr", 0)
        if rev > max_revenue:
            max_revenue = rev
    n = len(documents)
    return {
        "avg_funding": total_funding / n,
        "avg_employees": total_employees / n,
        "us_market_pct": us_count / n,
        "max_revenue": max_revenue,
        "competitor_count": n,
    }


PROMPT_TEMPLATE = """You are {agent_name}, a {agent_role} on a startup incubator board.

COMPETITOR DATA:
{competitor_section}

MARKET METRICS:
- Average funding: ${avg_funding:,.0f}
- US market: {us_pct:.0%}
- Max revenue: ${max_revenue:,.0f}
- Competitors: {competitor_count}

Evaluate this startup idea:
{idea}

End with BUILD or DON'T BUILD, confidence 1-10."""


def assemble_prompt_python(documents, features, idea, token_budget=3000):
    """Pure Python prompt assembly with token budgeting."""
    competitor_lines = []
    tokens_used = 0
    for doc in documents:
        line = f"- {doc['title']} (${doc['metadata']['funding']:,}, {doc['metadata']['stage']})"
        line_tokens = count_tokens_python(line)
        if tokens_used + line_tokens > token_budget * 0.4:
            break
        competitor_lines.append(line)
        tokens_used += line_tokens

    prompt = PROMPT_TEMPLATE.format(
        agent_name="Dr. Riya Patel",
        agent_role="security analyst",
        competitor_section="\n".join(competitor_lines),
        avg_funding=features.get("avg_funding", 0),
        us_pct=features.get("us_market_pct", 0),
        max_revenue=features.get("max_revenue", 0),
        competitor_count=features.get("competitor_count", 0),
        idea=idea[:500],
    )

    request_body = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1500,
        "system": prompt[:1200],
        "messages": [{"role": "user", "content": idea[:3000]}],
    })
    return request_body


def full_pipeline_python(idea, query_embedding, knowledge_base, api_response_json):
    """Complete context preparation pipeline in pure Python."""
    results = cosine_similarity_python(query_embedding, knowledge_base, top_k=10)
    parsed = json.loads(api_response_json)
    documents = parsed["results"][:20]
    features = compute_features_python(documents)
    request = assemble_prompt_python(documents, features, idea)
    return request
