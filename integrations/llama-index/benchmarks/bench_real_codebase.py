#!/usr/bin/env python3
"""
IronRace vs SimpleVectorStore — Real Codebase Benchmark
==========================================================
Loads a real codebase, chunks it, embeds it with a local model,
and compares retrieval speed through the LlamaIndex API.

Usage:
    python benchmarks/bench_real_codebase.py ~/git-repos/python-repos/everything-claude-code
    python benchmarks/bench_real_codebase.py /path/to/any/repo
"""

import os
import sys
import time

os.environ["IS_TESTING"] = "false"  # We want real embeddings

# Check args
if len(sys.argv) < 2:
    print("Usage: python benchmarks/bench_real_codebase.py /path/to/codebase")
    sys.exit(1)

codebase_path = os.path.expanduser(sys.argv[1])
if not os.path.isdir(codebase_path):
    print(f"Error: {codebase_path} is not a directory")
    sys.exit(1)

print("=" * 64)
print("  IRONRACE vs SIMPLEVECTORSTORE — REAL CODEBASE")
print("=" * 64)
print(f"\n  Codebase: {codebase_path}")

# ── Step 1: Install dependencies if needed ──
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    print("\n  Installing HuggingFace embeddings...")
    os.system(f"{sys.executable} -m pip install llama-index-embeddings-huggingface sentence-transformers -q")
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import QueryBundle
from llama_index.core.vector_stores import SimpleVectorStore

from llama_index.vector_stores.ironrace import IronRaceVectorStore

# ── Step 2: Configure embedding model (runs locally, no API key) ──
print("\n  Loading embedding model (BAAI/bge-small-en-v1.5)...", end=" ", flush=True)
t0 = time.perf_counter()
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",  # 384 dimensions, fast
)
print(f"done ({time.perf_counter() - t0:.1f}s)")

# ── Step 3: Load documents ──
print(f"  Loading documents from {codebase_path}...", end=" ", flush=True)
t0 = time.perf_counter()

# Only load text-based files
required_exts = [".py", ".md", ".txt", ".rst", ".ts", ".js", ".tsx", ".jsx", ".yaml", ".yml", ".toml", ".json"]
reader = SimpleDirectoryReader(
    input_dir=codebase_path,
    recursive=True,
    required_exts=required_exts,
    exclude_hidden=True,
    num_files_limit=500,  # Cap to keep benchmark practical
)
documents = reader.load_data()
load_time = time.perf_counter() - t0
print(f"done ({load_time:.1f}s, {len(documents)} files)")

# ── Step 4: Chunk documents ──
print(f"  Chunking documents...", end=" ", flush=True)
t0 = time.perf_counter()
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(documents)
chunk_time = time.perf_counter() - t0
print(f"done ({chunk_time:.1f}s, {len(nodes)} chunks)")

# ── Step 5: Build SimpleVectorStore index (baseline) ──
print(f"\n  Building SimpleVectorStore index...", end=" ", flush=True)
t0 = time.perf_counter()
simple_index = VectorStoreIndex(nodes=nodes, show_progress=True)
simple_build = time.perf_counter() - t0
print(f"done ({simple_build:.1f}s)")

# ── Step 6: Build IronRace index ──
print(f"  Building IronRace index...", end=" ", flush=True)
t0 = time.perf_counter()
af_store = IronRaceVectorStore(ef_construction=40)
af_ctx = StorageContext.from_defaults(vector_store=af_store)
af_index = VectorStoreIndex(nodes=nodes, storage_context=af_ctx, show_progress=True)
af_build = time.perf_counter() - t0
print(f"done ({af_build:.1f}s)")

# ── Step 7: Benchmark queries ──
QUERIES = [
    "How do agents communicate with each other?",
    "What is the authentication flow?",
    "How are MCP servers configured?",
    "What hooks are available for customization?",
    "How does the CLI handle user input?",
    "What is the architecture of the system?",
    "How are errors handled and reported?",
    "What testing patterns are used?",
]

TOP_K = 10
ITERATIONS = 20

print(f"\n  Benchmarking {len(QUERIES)} queries × {ITERATIONS} iterations each...")
print(f"  top_k={TOP_K}")

simple_retriever = simple_index.as_retriever(similarity_top_k=TOP_K)
af_retriever = af_index.as_retriever(similarity_top_k=TOP_K)

simple_times = []
af_times = []
recalls = []

for query_str in QUERIES:
    # Embed the query once
    query_embedding = Settings.embed_model.get_query_embedding(query_str)
    query = QueryBundle(query_str=query_str, embedding=query_embedding)

    # Warmup
    simple_retriever.retrieve(query)
    af_retriever.retrieve(query)

    # Benchmark SimpleVectorStore
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        simple_results = simple_retriever.retrieve(query)
        simple_times.append((time.perf_counter() - t0) * 1000)

    # Benchmark IronRace
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        af_results = af_retriever.retrieve(query)
        af_times.append((time.perf_counter() - t0) * 1000)

    # Recall
    simple_ids = {r.node.node_id for r in simple_results[:TOP_K]}
    af_ids = {r.node.node_id for r in af_results[:TOP_K]}
    recall = len(simple_ids & af_ids) / len(simple_ids) if simple_ids else 1.0
    recalls.append(recall)

simple_times.sort()
af_times.sort()

simple_mean = sum(simple_times) / len(simple_times)
af_mean = sum(af_times) / len(af_times)
speedup = simple_mean / max(af_mean, 0.001)
avg_recall = sum(recalls) / len(recalls)

# ── Results ──
n_chunks = len(nodes)
embed_dim = len(nodes[0].embedding) if nodes[0].embedding else "?"

print(f"\n{'=' * 64}")
print(f"  RESULTS — {n_chunks:,} chunks × {embed_dim}d embeddings")
print(f"{'=' * 64}")

print(f"\n  {'':>18} {'Simple':>14} {'IronRace':>14} {'Speedup':>10}")
print(f"  {'─' * 58}")
print(f"  {'Build time':<18} {simple_build:>12.1f}s {af_build:>12.1f}s")
print(f"  {'Query mean':<18} {simple_mean:>12.3f}ms {af_mean:>12.3f}ms {speedup:>9.1f}x")
print(f"  {'Query median':<18} {simple_times[len(simple_times)//2]:>12.3f}ms {af_times[len(af_times)//2]:>12.3f}ms")
print(f"  {'Query P95':<18} {simple_times[int(len(simple_times)*0.95)]:>12.3f}ms {af_times[int(len(af_times)*0.95)]:>12.3f}ms")
print(f"  {'Query P99':<18} {simple_times[int(len(simple_times)*0.99)]:>12.3f}ms {af_times[int(len(af_times)*0.99)]:>12.3f}ms")
print(f"  {'Avg Recall@{TOP_K}':<18} {'100% (exact)':>14} {avg_recall:>13.0%}")

print(f"\n  Queries tested:")
for i, q in enumerate(QUERIES):
    print(f"    {i+1}. \"{q}\" (recall: {recalls[i]:.0%})")

# Show sample results
print(f"\n  Sample results for \"{QUERIES[0]}\":")
query_embedding = Settings.embed_model.get_query_embedding(QUERIES[0])
query = QueryBundle(query_str=QUERIES[0], embedding=query_embedding)
results = af_retriever.retrieve(query)
for r in results[:5]:
    text_preview = r.node.text[:80].replace("\n", " ")
    print(f"    [{r.score:.3f}] {text_preview}...")

print(f"\n{'=' * 64}")
print(f"  {n_chunks:,} chunks: IronRace is {speedup:.1f}x faster through the LlamaIndex API")
print(f"  Recall@{TOP_K}: {avg_recall:.0%} (HNSW approximate vs brute-force exact)")
print(f"{'=' * 64}")
