# IronRace

Rust-powered context engine for AI agent pipelines. Accelerates the CPU-intensive context preparation layer (vector search, JSON processing, tokenization, prompt assembly) while keeping LLM calls in Python where they belong.

## Why

In multi-agent AI systems, context preparation is the CPU bottleneck at scale. Vector search over 5K embeddings takes ~70ms in pure Python. IronRace's Rust runtime does it in ~0.15ms with 97%+ recall — verified against brute-force on every benchmark run. The full context preparation pipeline runs ~3x faster single-threaded, and at 1,000 concurrent pipelines IronRace achieves **62x** throughput over Python.

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

All numbers are **medians** across 200 iterations with 10 warmup runs. Recall verified against brute-force ground truth on every run.

| Operation | Pure Python | IronRace (Rust) | Speedup |
|-----------|------------|-----------------|---------|
| Vector search (5K × 384d) | 71ms | 0.15ms | **480x** |
| JSON parsing (900KB) | ~8ms | ~2.2ms | **4x** |
| Token counting | ~0.07ms | ~0.004ms | **16x** |
| Prompt assembly | ~0.03ms | ~0.009ms | **3x** |
| **Full pipeline** | **11ms** | **3.7ms** | **~3x** |

Vector search scaling:

| Scale | Build time | Search (median) | Recall@10 |
|-------|-----------|-----------------|-----------|
| 1K × 768d | 0.15s | 0.28ms | 97% |
| 10K × 768d | 3.4s | 0.30ms | 97% |
| 100K × 768d | 69s | 0.73ms | 72% |

HNSW index build is a one-time cost amortized over all searches. Recall measured against brute-force cosine similarity across 100 queries. Recall at 100K can be improved by increasing `ef_construction`.

Concurrent throughput (GIL released during Rust execution):

| Concurrent Pipelines | Python | IronRace | Speedup |
|---|---|---|---|
| 100 | 25/sec | 1,497/sec | **59x** |
| 1,000 | 30/sec | 1,858/sec | **62x** |

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
