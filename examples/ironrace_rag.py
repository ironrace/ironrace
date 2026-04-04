#!/usr/bin/env python3
"""
ironrace-rag — Ask questions about any folder of documents.
Compares IronRace (Rust HNSW) against actual LlamaIndex SimpleVectorStore
side-by-side on every query.

Usage:
    python ironrace_rag.py ./docs
    python ironrace_rag.py ./src --extensions .py .rs .md
    python ironrace_rag.py ./docs --query "What is the auth flow?"
    python ironrace_rag.py ./docs --no-llm  # Just show retrieval comparison

Requirements:
    pip install ironrace
    pip install llama-index-core llama-index-embeddings-huggingface
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

# ── Dependency checks ─────────────────────────────────────

try:
    from ironrace import VectorIndex, assemble_prompt, count_tokens
except ImportError:
    print("Install IronRace: pip install ironrace")
    sys.exit(1)

try:
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
    from llama_index.core.indices.prompt_helper import PromptHelper
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    print("Install LlamaIndex:")
    print("  pip install llama-index-core llama-index-embeddings-huggingface")
    sys.exit(1)

import numpy as np


# ── Setup ─────────────────────────────────────────────────

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def setup_llamaindex():
    """Configure LlamaIndex to use local HuggingFace embeddings (no OpenAI key needed)."""
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.embed_model = embed_model
    # Disable LLM in LlamaIndex — we handle the LLM call ourselves
    Settings.llm = None
    return embed_model


# ── Indexing (both engines) ───────────────────────────────

def build_both_indices(folder: Path, extensions: list[str] = None, chunk_size: int = 512):
    """Load docs, chunk, embed, and build both LlamaIndex and IronRace indices."""

    # Step 1: Load documents using LlamaIndex's reader
    print("  Loading documents (LlamaIndex SimpleDirectoryReader)...", end=" ", flush=True)
    t0 = time.perf_counter()

    required_exts = extensions or [
        ".md", ".txt", ".py", ".rs", ".ts", ".tsx", ".js", ".jsx",
        ".go", ".java", ".scala", ".rb", ".sh", ".sql", ".yaml", ".yml",
        ".toml", ".json", ".html", ".css",
    ]
    reader = SimpleDirectoryReader(
        input_dir=str(folder),
        required_exts=required_exts,
        recursive=True,
        exclude_hidden=True,
        errors="ignore",
    )
    try:
        documents = reader.load_data()
    except Exception as e:
        print(f"\n  Error loading documents: {e}")
        print("  Try using --extensions to limit file types, or check for broken symlinks.")
        sys.exit(1)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"done ({len(documents)} docs, {load_ms:.0f}ms)")

    if not documents:
        print("  No documents found. Check the folder and extensions.")
        sys.exit(1)

    # Step 2: Chunk using LlamaIndex's splitter (same chunks for both engines)
    print(f"  Chunking (SentenceSplitter, {chunk_size} tokens)...", end=" ", flush=True)
    t0 = time.perf_counter()
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=64)
    nodes = splitter.get_nodes_from_documents(documents)
    chunk_ms = (time.perf_counter() - t0) * 1000
    print(f"done ({len(nodes)} chunks, {chunk_ms:.0f}ms)")

    # Step 3: Build LlamaIndex VectorStoreIndex (embeds + builds SimpleVectorStore)
    print("  Embedding + building LlamaIndex VectorStoreIndex...", end=" ", flush=True)
    t0 = time.perf_counter()
    llama_index = VectorStoreIndex(nodes=nodes)
    llama_build_ms = (time.perf_counter() - t0) * 1000
    print(f"done ({llama_build_ms:.0f}ms)")

    # Step 4: Extract pre-computed embeddings and build IronRace HNSW index
    print("  Building IronRace HNSW index (from same embeddings)...", end=" ", flush=True)
    t0 = time.perf_counter()

    # Extract embeddings from the vector store's internal dict
    vector_store = llama_index._vector_store
    vector_store_data = vector_store._data

    # Map node IDs to their embeddings and text
    node_map = {}
    for node in nodes:
        node_map[node.node_id] = node

    embeddings_list = []
    chunk_data = []  # parallel array: metadata for each embedded chunk

    # Access the embedding dict from SimpleVectorStore
    embedding_dict = vector_store_data.embedding_dict
    for node_id, embedding in embedding_dict.items():
        embeddings_list.append(embedding)
        node = node_map.get(node_id)
        source = ""
        if node and node.metadata:
            source = node.metadata.get("file_path", node.metadata.get("file_name", "unknown"))
        chunk_data.append({
            "node_id": node_id,
            "source": source,
            "text": node.get_content() if node else "",
        })

    ir_index = VectorIndex(embeddings_list)
    ir_build_ms = (time.perf_counter() - t0) * 1000
    print(f"done ({ir_build_ms:.0f}ms)")

    dim = len(embeddings_list[0]) if embeddings_list else 0
    print(f"\n  Ready: {len(embeddings_list)} chunks × {dim}d")
    print(f"  LlamaIndex: {llama_build_ms:.0f}ms (embed + SimpleVectorStore)")
    print(f"  IronRace:   {ir_build_ms:.0f}ms (HNSW graph only, reuses embeddings)")

    return llama_index, ir_index, chunk_data, nodes


# ── Retrieval Comparison ──────────────────────────────────

def retrieve_both(
    query: str,
    llama_index: VectorStoreIndex,
    ir_index: VectorIndex,
    chunk_data: list[dict],
    nodes: list,
    top_k: int = 5,
    context_budget: int = 3000,
) -> dict:
    """Run the same query through both engines, measure timing."""

    results = {}

    # ── LlamaIndex Retrieval ─────────────────────────
    retriever = VectorIndexRetriever(index=llama_index, similarity_top_k=top_k)

    t0 = time.perf_counter()
    llama_results = retriever.retrieve(query)
    llama_search_ms = (time.perf_counter() - t0) * 1000

    llama_sources = []
    llama_context_parts = []
    for node_with_score in llama_results:
        node = node_with_score.node
        score = node_with_score.score or 0.0
        source = ""
        if node.metadata:
            source = node.metadata.get("file_path", node.metadata.get("file_name", "unknown"))
        text = node.get_content()
        llama_sources.append({"source": source, "score": score})
        llama_context_parts.append(text)

    # LlamaIndex prompt assembly: PromptHelper.repack() + DEFAULT_TEXT_QA_PROMPT
    t0 = time.perf_counter()
    prompt_helper = PromptHelper(context_window=context_budget + 500, num_output=256)
    compact_chunks = prompt_helper.repack(DEFAULT_TEXT_QA_PROMPT, llama_context_parts)
    llama_prompt = DEFAULT_TEXT_QA_PROMPT.format(
        context_str="\n\n".join(compact_chunks),
        query_str=query,
    )
    llama_assembly_ms = (time.perf_counter() - t0) * 1000
    llama_total_ms = llama_search_ms + llama_assembly_ms

    results["llamaindex"] = {
        "search_ms": llama_search_ms,
        "assembly_ms": llama_assembly_ms,
        "total_ms": llama_total_ms,
        "sources": llama_sources,
        "prompt": llama_prompt,
    }

    # ── IronRace Retrieval ───────────────────────────
    query_embedding = Settings.embed_model.get_query_embedding(query)

    t0 = time.perf_counter()
    ir_results = ir_index.search(query_embedding, top_k)
    ir_search_ms = (time.perf_counter() - t0) * 1000

    ir_sources = []
    ir_context_parts = []
    for idx, score in ir_results:
        if 0 <= idx < len(chunk_data):
            chunk = chunk_data[idx]
            ir_sources.append({"source": chunk["source"], "score": score})
            ir_context_parts.append(f"[Source: {chunk['source']} (score: {score:.3f})]\n{chunk['text']}")

    ir_context = "\n\n---\n\n".join(ir_context_parts)

    t0 = time.perf_counter()
    ir_prompt_result = assemble_prompt(
        "You are a helpful assistant. Answer the user's question based on the provided context. "
        "Cite which source files your answer comes from. If the context doesn't contain enough "
        "information, say so.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:",
        values={"context": ir_context, "question": query},
        budgets={"context": context_budget},
    )
    ir_assembly_ms = (time.perf_counter() - t0) * 1000
    ir_total_ms = ir_search_ms + ir_assembly_ms

    results["ironrace"] = {
        "search_ms": ir_search_ms,
        "assembly_ms": ir_assembly_ms,
        "total_ms": ir_total_ms,
        "tokens": ir_prompt_result.total_tokens,
        "truncated": ir_prompt_result.sections_truncated,
        "sources": ir_sources,
        "prompt": ir_prompt_result.prompt,
    }

    # Check result overlap (by node ID, not source file)
    ll_ids = set()
    for nws in llama_results:
        ll_ids.add(nws.node.node_id)
    ir_ids = set()
    for idx, _ in ir_results:
        if 0 <= idx < len(chunk_data):
            ir_ids.add(chunk_data[idx]["node_id"])
    results["overlap"] = len(ll_ids & ir_ids)
    results["total"] = max(len(ll_ids), len(ir_ids))
    results["same_results"] = ll_ids == ir_ids

    return results


# ── LLM Call ──────────────────────────────────────────────

def ask_llm(prompt: str, use_api: bool = False) -> tuple:
    t0 = time.perf_counter()
    if use_api:
        try:
            import anthropic
        except ImportError:
            print("pip install anthropic && export ANTHROPIC_API_KEY=...")
            sys.exit(1)
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
    else:
        try:
            result = subprocess.run(
                ["claude", "-p", prompt],
                capture_output=True, text=True, timeout=120,
            )
            text = result.stdout.strip() if result.returncode == 0 else f"Error: {(result.stderr or '')[:300]}"
        except FileNotFoundError:
            print("Error: 'claude' CLI not found. Use --api.")
            sys.exit(1)
        except subprocess.TimeoutExpired:
            text = "Timed out."
    return text, (time.perf_counter() - t0) * 1000


# ── Display ───────────────────────────────────────────────

def display_comparison(query: str, results: dict, answer: str, llm_ms: float):
    ll = results["llamaindex"]
    ir = results["ironrace"]

    search_speedup = ll["search_ms"] / max(ir["search_ms"], 0.001)
    total_speedup = ll["total_ms"] / max(ir["total_ms"], 0.001)

    print(f"\n  {'─'*62}")
    print(f"  RETRIEVAL COMPARISON")
    print(f"  {'─'*62}")
    print(f"  {'Operation':<24} {'LlamaIndex':>12} {'IronRace':>12} {'Speedup':>10}")
    print(f"  {'─'*58}")
    print(f"  {'Vector search':<24} {ll['search_ms']:>10.2f}ms {ir['search_ms']:>10.2f}ms {search_speedup:>9.0f}x")
    print(f"  {'Prompt assembly':<24} {ll['assembly_ms']:>10.2f}ms {ir['assembly_ms']:>10.2f}ms")
    print(f"  {'─'*58}")
    print(f"  {'Context prep total':<24} {ll['total_ms']:>10.2f}ms {ir['total_ms']:>10.2f}ms {total_speedup:>9.0f}x")

    if llm_ms > 0:
        print(f"  {'LLM call':<24} {llm_ms:>10.0f}ms {llm_ms:>10.0f}ms {'(same)':>10}")
        print(f"  {'─'*58}")
        ll_e2e = ll["total_ms"] + llm_ms
        ir_e2e = ir["total_ms"] + llm_ms
        print(f"  {'End-to-end':<24} {ll_e2e:>10.0f}ms {ir_e2e:>10.0f}ms")
        ctx_pct_ll = (ll["total_ms"] / ll_e2e) * 100
        ctx_pct_ir = (ir["total_ms"] / ir_e2e) * 100
        print(f"  {'Context prep %':<24} {ctx_pct_ll:>9.1f}% {ctx_pct_ir:>9.2f}%")

    # Result agreement
    overlap = results["overlap"]
    total = results["total"]
    match_str = "identical" if results["same_results"] else f"{overlap}/{total} overlap"
    print(f"\n  Results: {match_str}")

    # Sources from both
    print(f"\n  {'LlamaIndex retrieved:':<30} {'IronRace retrieved:'}")
    for i in range(max(len(ll["sources"]), len(ir["sources"]))):
        ll_src = f"[{ll['sources'][i]['score']:.3f}] {Path(ll['sources'][i]['source']).name}" if i < len(ll["sources"]) else ""
        ir_src = f"[{ir['sources'][i]['score']:.3f}] {Path(ir['sources'][i]['source']).name}" if i < len(ir["sources"]) else ""
        print(f"    {ll_src:<30} {ir_src}")

    # Scale projection
    print(f"\n  At scale (context prep only):")
    for n in [100, 1000]:
        ll_n = ll["total_ms"] * n
        ir_n = ir["total_ms"] * n
        speedup = ll_n / max(ir_n, 0.01)
        print(f"    {n:>5} concurrent: LlamaIndex {ll_n:>8,.0f}ms -> IronRace {ir_n:>8,.1f}ms ({speedup:,.0f}x)")

    # Answer
    if answer:
        print(f"\n  {'─'*62}")
        print(f"  ANSWER")
        print(f"  {'─'*62}\n")
        for line in answer.split("\n"):
            print(f"    {line}")


# ── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RAG comparison: LlamaIndex SimpleVectorStore vs IronRace HNSW.",
        epilog="Example: python ironrace_rag.py ./docs --query 'How does auth work?'",
    )
    parser.add_argument("folder", type=Path, help="Folder of documents to index")
    parser.add_argument("--query", "-q", type=str, help="Single question")
    parser.add_argument("--extensions", nargs="+", help="File extensions (e.g., .py .rs .md)")
    parser.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve (default: 5)")
    parser.add_argument("--context-budget", type=int, default=3000, help="Token budget (default: 3000)")
    parser.add_argument("--chunk-size", type=int, default=512, help="Tokens per chunk (default: 512)")
    parser.add_argument("--api", action="store_true", help="Use Anthropic API instead of Claude CLI")
    parser.add_argument("--no-llm", action="store_true", help="Compare retrieval only, skip LLM")
    args = parser.parse_args()

    if not args.folder.exists():
        print(f"Error: Folder not found: {args.folder}")
        sys.exit(1)

    print(f"\n{'='*64}")
    print(f"  IRONRACE RAG — {args.folder}")
    print(f"  LlamaIndex SimpleVectorStore vs IronRace HNSW")
    print(f"{'='*64}\n")

    # Configure LlamaIndex
    setup_llamaindex()

    # Build both indices from the same documents and embeddings
    llama_idx, ir_idx, chunk_data, nodes = build_both_indices(
        args.folder,
        extensions=args.extensions,
        chunk_size=args.chunk_size,
    )

    # Query handler
    def handle_query(query: str):
        print(f"\n{'='*64}")
        print(f"  Q: {query}")
        print(f"{'='*64}")

        results = retrieve_both(
            query, llama_idx, ir_idx, chunk_data, nodes,
            top_k=args.top_k, context_budget=args.context_budget,
        )

        answer, llm_ms = "", 0
        if not args.no_llm:
            print(f"\n  Asking Claude...", end=" ", flush=True)
            answer, llm_ms = ask_llm(results["ironrace"]["prompt"], use_api=args.api)
            print(f"done ({llm_ms:.0f}ms)")

        display_comparison(query, results, answer, llm_ms)

    if args.query:
        handle_query(args.query)
    else:
        mode = "API" if args.api else "CLI"
        print(f"\n  Interactive mode (Claude {mode}). Type 'quit' to exit.\n")
        while True:
            try:
                query = input("  Ask: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye.")
                break
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue
            handle_query(query)


if __name__ == "__main__":
    main()
