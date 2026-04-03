#!/usr/bin/env python3
"""
IronRace Example — Simple RAG Chatbot
========================================
Build a vector index, search on user query, assemble prompt with token budget.

Usage:
    python examples/rag_chatbot.py
    ANTHROPIC_API_KEY=sk-... python examples/rag_chatbot.py --live
"""

import asyncio
import json
import math
import os
import random
import sys
import time

random.seed(42)

from ironrace import VectorIndex, assemble_prompt, count_tokens


# ═══════════════════════════════════════════════════════════
# KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════

DOCUMENTS = [
    {"title": "Python Basics", "content": "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python's extensive standard library and package ecosystem make it ideal for web development, data science, AI, and automation."},
    {"title": "Rust Performance", "content": "Rust is a systems programming language that guarantees memory safety without garbage collection. It achieves C/C++ level performance while preventing common bugs like null pointer dereferences and buffer overflows. Rust's ownership model enables fearless concurrency and zero-cost abstractions."},
    {"title": "PyO3 Bridge", "content": "PyO3 is a Rust crate that enables writing Python extensions in Rust. It provides seamless interop between Python and Rust, allowing developers to write performance-critical code in Rust while maintaining a Pythonic API. PyO3 handles type conversions, GIL management, and module registration automatically."},
    {"title": "Vector Databases", "content": "Vector databases store and search high-dimensional vectors for similarity matching. They use algorithms like HNSW (Hierarchical Navigable Small World) for approximate nearest neighbor search. Common use cases include semantic search, recommendation systems, and RAG (Retrieval Augmented Generation) pipelines."},
    {"title": "Token Budgeting", "content": "Token budgeting is the practice of allocating a fixed number of tokens to each section of an LLM prompt. This ensures prompts stay within model context limits while maximizing the information density of each section. Smart truncation at sentence boundaries preserves readability."},
    {"title": "Agent Architectures", "content": "AI agent architectures typically involve multiple specialized agents that collaborate on complex tasks. Each agent has a specific role (e.g., researcher, analyst, writer) and access to tools. The orchestration layer manages agent communication, context sharing, and task delegation."},
    {"title": "HNSW Algorithm", "content": "HNSW builds a multi-layer graph where each layer is a navigable small world network. The top layers have few connections for fast long-range navigation, while bottom layers have dense connections for precise local search. This hierarchical structure enables O(log N) search complexity."},
    {"title": "Maturin Build System", "content": "Maturin is a build tool for creating Python packages with Rust extensions. It handles the compilation of Rust code, packaging into wheels, and installation. Maturin supports both pure Rust packages and mixed Python/Rust projects with automatic module detection."},
]

# Generate simple embeddings (word overlap based, not real embeddings)
def simple_embed(text, dim=64):
    random.seed(hash(text) % 2**32)
    v = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def main():
    live_mode = "--live" in sys.argv

    print("\nIronRace RAG Chatbot Example")
    print("=" * 40)

    # Build index
    print("\nBuilding vector index...", end=" ", flush=True)
    t0 = time.perf_counter()
    embeddings = [simple_embed(doc["content"]) for doc in DOCUMENTS]
    index = VectorIndex(embeddings, ef_construction=100)
    build_time = (time.perf_counter() - t0) * 1000
    print(f"done ({build_time:.1f}ms, {len(DOCUMENTS)} documents)")

    # Query
    query = "How does Rust improve Python performance?"
    print(f"\nQuery: {query}")

    # Search
    t0 = time.perf_counter()
    query_embedding = simple_embed(query)
    results = index.search(query_embedding, top_k=3)
    search_time = (time.perf_counter() - t0) * 1000

    print(f"\nTop {len(results)} results (searched in {search_time:.2f}ms):")
    retrieved_docs = []
    for idx, score in results:
        doc = DOCUMENTS[idx]
        print(f"  [{score:.3f}] {doc['title']}")
        retrieved_docs.append(doc)

    # Assemble prompt with token budget
    context_text = "\n\n".join(
        f"### {doc['title']}\n{doc['content']}" for doc in retrieved_docs
    )

    t0 = time.perf_counter()
    result = assemble_prompt(
        template="You are a helpful AI assistant. Answer the user's question based on the provided context.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nProvide a clear, concise answer.",
        values={"context": context_text, "question": query},
        budgets={"context": 500, "question": 50},
    )
    assemble_time = (time.perf_counter() - t0) * 1000

    print(f"\nPrompt assembled ({assemble_time:.2f}ms):")
    print(f"  Total tokens: {result.total_tokens}")
    print(f"  Sections truncated: {result.sections_truncated or 'none'}")
    print(f"  Token breakdown: {dict(result.token_breakdown)}")

    if live_mode:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("\n⚠ Set ANTHROPIC_API_KEY to use --live mode")
            return

        from ironrace import LLMRouter
        router = LLMRouter(anthropic_api_key=api_key)

        async def call_llm():
            resp = await router.call(
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": query}],
                system=result.prompt,
                max_tokens=500,
            )
            await router.close()
            return resp

        print("\nCalling Claude...")
        resp = asyncio.run(call_llm())
        answer = resp.get("content", [{}])[0].get("text", "No response")
        print(f"\nAnswer:\n{answer}")
    else:
        print(f"\nPrompt preview:\n{result.prompt[:300]}...")
        print("\n(Use --live flag to call Claude API)")


if __name__ == "__main__":
    main()
