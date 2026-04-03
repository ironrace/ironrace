# IronRace

Rust-powered context engine for AI agent pipelines. Accelerates the CPU-intensive context preparation layer (vector search, JSON processing, tokenization, prompt assembly) while keeping LLM calls in Python where they belong.

## Why

In multi-agent AI systems, context preparation is the CPU bottleneck at scale. Pure Python context prep takes ~185ms per pipeline invocation. IronRace's Rust runtime does it in ~2ms — a **90x improvement**. At 1M invocations/day, that's the difference between 13 vCPUs and 0.14 vCPUs.

## 30-Second Example

```python
import json
from ironrace import VectorIndex, assemble_prompt, execute_pipeline, compile_agents_dag

# Build a vector index (lives in Rust memory)
index = VectorIndex(embeddings, ef_construction=200)
results = index.search(query, top_k=10)  # < 1ms for 5K vectors

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

## Installation

```bash
pip install ironrace
```

Development:
```bash
git clone https://github.com/yourorg/ironrace.git
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

## Project Structure

```
ironrace/
├── rust/src/           # Rust core (PyO3)
│   ├── vector.rs       # HNSW approximate nearest neighbor
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
├── benchmarks/         # Performance benchmarks
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

# Examples
python examples/startup_evaluator.py
python examples/rag_chatbot.py
python examples/batch_research.py
```

## License

Apache 2.0 — see [LICENSE](LICENSE)
