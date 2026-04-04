# IronRace

Rust-powered context engine for AI agent pipelines. Accelerates the CPU-intensive context preparation layer (vector search, JSON processing, tokenization, prompt assembly) while keeping LLM calls in Python where they belong.

## Why

In multi-agent AI systems, context preparation is the CPU bottleneck at scale. LlamaIndex's vector search over 5K embeddings takes ~54ms. IronRace's Rust runtime does it in ~0.5ms with 99% recall — verified against LlamaIndex brute-force on every benchmark run. The full pipeline runs **16x faster** single-threaded, scaling to **55x throughput** at 1,000 concurrent pipelines where Python's GIL becomes the bottleneck.

## 30-Second Example

```python
import json
from ironrace import VectorIndex, assemble_prompt, execute_pipeline, compile_agents_dag

# Build a vector index (lives in Rust memory)
index = VectorIndex(embeddings)  # ~10s for 10K docs, 98%+ recall
results = index.search(query, top_k=10)  # < 0.5ms for 10K vectors

# Assemble prompts with token budgets
result = assemble_prompt(
    "System: {role}\nContext: {docs}\nQuery: {q}",
    values={"role": "analyst", "docs": long_text, "q": "evaluate this"},
    budgets={"docs": 2000},
)
# result.prompt, result.total_tokens, result.sections_truncated

# Run a full multi-agent pipeline in ONE Rust call
dag = compile_agents_dag([
    {"id": "analyst", "template": "...", "values": {...}, "budgets": {...}},
    {"id": "engineer", "template": "...", "values": {...}, "budgets": {...}},
])
results = json.loads(execute_pipeline(dag))
```

## Benchmark Results

All numbers are **medians** across 200 iterations with 10 warmup runs. Baseline is actual LlamaIndex code (SimpleVectorStore, tiktoken, PromptHelper). Recall verified against LlamaIndex brute-force on every run.

| Operation | LlamaIndex | IronRace (Rust) | Speedup |
|-----------|------------|-----------------|---------|
| Vector search (5K × 384d) | 54ms | 0.5ms | **96-133x** |
| JSON parsing (900KB) | ~9ms | ~3ms | **3x** |
| Token counting (vs tiktoken) | 0.24ms | 0.005ms | **50x** |
| Prompt assembly | 0.6ms | 0.01ms | **65x** |
| **Full pipeline** | **64ms** | **4ms** | **16x** |

Token counting baseline is tiktoken, which is itself Rust-based. JSON baselines are stdlib json, which LlamaIndex uses internally.

Vector search scaling:

| Scale | Build time | Search (median) | Recall@10 |
|-------|-----------|-----------------|-----------|
| 1K × 768d | 0.6s | 0.58ms | 99% |
| 10K × 768d | 7.8s | 1.33ms | 98% |
| 100K × 768d | 80.6s | 2.12ms | 91%* |

HNSW index build is a one-time cost amortized over all searches. Recall measured against brute-force cosine similarity across 100 queries.

*IronRace is optimized for 1K-10K document collections. For larger datasets, pair IronRace's pipeline acceleration with a dedicated vector database like Pinecone, Milvus, or pgvector.

Real-world RAG (19K chunks from a production codebase):

| Operation | LlamaIndex | IronRace | Speedup |
|---|---|---|---|
| Context prep (search + assembly) | 399ms | 1.6ms | **245x** |
| Result overlap vs brute-force | — | 4/5 | — |

Concurrent throughput (GIL released during Rust execution):

| Concurrent Pipelines | Python | IronRace | Speedup |
|---|---|---|---|
| 100 | 32/sec | 1,823/sec | **58x** |
| 1,000 | 31/sec | 1,715/sec | **55x** |

## Installation

```bash
pip install ironrace
```

Development:
```bash
git clone https://github.com/ironrace/ironrace.git
cd ironrace
pip install maturin
maturin develop --release
```

## Key Design Decisions

1. **Python API, Rust runtime** — Decorators, type hints, dataclasses. No Rust knowledge needed.
2. **Bridge once per pipeline** — The entire context prep DAG executes in Rust as a single call.
3. **LLM calls stay in Python** — I/O-bound, asyncio handles them fine.
4. **Vector index lives in Rust** — Built once, searched many times, zero-copy reference.
5. **Token budgeting is first-class** — Per-section budgets with sentence-boundary truncation.
6. **Accuracy by default** — ef_construction=100 gives 98%+ recall out of the box. Tunable for power users.

## Project Structure

```
ironrace/
├── rust/src/           # Rust core (PyO3)
│   ├── vector.rs       # HNSW approximate nearest neighbor (hnsw_rs)
│   ├── tokenizer.rs    # Token counting + truncation
│   ├── assembler.rs    # Prompt assembly + budgeting
│   ├── json_fast.rs    # Serde JSON bridge
│   ├── pipeline.rs     # Tokio DAG executor
│   └── lib.rs          # Module entry point
├── python/ironrace/  # Python SDK
│   ├── decorators.py   # @agent, @context, @pipeline
│   ├── compiler.py     # DAG compilation
│   ├── types.py        # TokenBudget, Document, VectorSearch
│   └── router.py       # Async LLM caller
├── benchmarks/         # Performance benchmarks (with recall verification)
├── examples/           # Runnable demo apps
├── tests/              # Test suite (87 tests)
└── docs/               # Documentation
```

## Documentation

- [PROBLEM.md](docs/PROBLEM.md) — The business case and benchmark data
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — Technical deep dive
- [QUICKSTART.md](docs/QUICKSTART.md) — 5-minute getting started guide
- [BENCHMARKS.md](docs/BENCHMARKS.md) — Reproducible benchmark methodology

## Running

```bash
# Tests
pytest tests/ -v

# Benchmarks
python -m benchmarks.bench_context_prep
python -m benchmarks.bench_vector_search
python -m benchmarks.bench_at_scale

# Examples
python examples/startup_evaluator.py
python examples/rag_chatbot.py
python examples/batch_research.py
```

## License

Apache 2.0 — see [LICENSE](LICENSE)
