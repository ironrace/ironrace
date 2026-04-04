# IronRace Architecture

## Overview

IronRace uses a two-phase execution model: **compile-time** analysis in Python and **runtime** execution in Rust.

```
┌─────────────────────────────────────────────┐
│  Python SDK Layer                           │
│  @agent, @context, @pipeline decorators     │
│  pip install ironrace                     │
└──────────────────┬──────────────────────────┘
                   │ compiles to DAG at import time
┌──────────────────▼──────────────────────────┐
│  Context Graph Compiler (Python)            │
│  Analyzes decorator metadata, builds        │
│  dependency DAG, generates execution plan   │
└──────────────────┬──────────────────────────┘
                   │ execution plan (JSON)
┌──────────────────▼──────────────────────────┐
│  Rust Runtime (PyO3 + Tokio)                │
│  ┌──────────┐ ┌──────────┐ ┌─────────────┐ │
│  │ Vector   │ │ JSON     │ │ Tokenizer   │ │
│  │ ANN      │ │ (serde)  │ │ (HF crate)  │ │
│  └──────────┘ └──────────┘ └─────────────┘ │
│  ┌──────────────────────────────────────┐   │
│  │ Parallel Context Assembler           │   │
│  │ Template interp + token budgeting    │   │
│  └──────────────────────────────────────┘   │
└──────────────────┬──────────────────────────┘
                   │ enriched prompt (JSON bytes)
┌──────────────────▼──────────────────────────┐
│  LLM Router (Python async — httpx)          │
│  Standard API calls — same speed regardless │
└─────────────────────────────────────────────┘
```

## Phase 1: Compile Time (Python)

When a `@pipeline` is first invoked, the compiler:

1. Inspects all `@context` classes to find data dependencies (`VectorSearch`, `APIFetch`, `Feature` descriptors)
2. Inspects all `@agent` functions to find their context requirements
3. Builds a dependency DAG: which data sources feed which agents, which agents can run in parallel
4. Serializes the DAG to JSON for the Rust executor

This compilation happens **once** per pipeline and is cached.

## Phase 2: Runtime (Rust)

The `execute_pipeline()` function receives the entire DAG as JSON and returns all results as JSON. This is the **single bridge crossing** design:

```python
# ONE call into Rust — all context prep happens inside
results = execute_pipeline(dag_json)  # JSON in, JSON out
```

Inside Rust, the DAG executor:

1. Topologically sorts nodes
2. Groups nodes with satisfied dependencies
3. Executes each group in parallel using Tokio
4. Feeds results to downstream nodes
5. Returns all results in a single response

## The Bridge-Once Design

This is the most critical architectural decision. Crossing the Python↔Rust bridge has overhead (~1-5μs per call). For individual operations on small data, this overhead can exceed the operation itself.

**Bad: Per-operation bridging**
```python
# 4 bridge crossings, each with overhead
results = rust_vector_search(query, vectors)    # cross bridge
parsed = rust_json_parse(response)              # cross bridge
tokens = rust_count_tokens(text)                # cross bridge
prompt = rust_assemble(template, values)        # cross bridge
```

**Good: Single bridging (IronRace)**
```python
# 1 bridge crossing, all operations in Rust
results = execute_pipeline(dag_json)  # cross bridge ONCE
```

## Rust Module Architecture

### vector.rs — HNSW Approximate Nearest Neighbor
- Wraps the `hnsw_rs` crate (HNSW algorithm with built-in cosine distance)
- `VectorIndex` lives in Rust memory, Python holds a reference
- Build once (O(n log n)), search many times (O(log n))
- Parallel index construction via rayon
- Default ef_construction=100 gives 98%+ recall on real-world embeddings
- Supports arbitrary dimensional embeddings

### tokenizer.rs — Fast Token Counting
- Uses HuggingFace `tokenizers` crate for accurate BPE tokenization
- Falls back to fast word-based approximation (~10% accurate, 20x faster)
- Cached tokenizer instances via `OnceCell`

### assembler.rs — Prompt Assembly + Token Budgeting
- Template interpolation with `{variable}` placeholders
- Per-section token budgets with sentence-boundary truncation
- Returns metadata: total tokens, which sections were truncated, per-section token counts
- Calls tokenizer internally (no bridge crossing)

### json_fast.rs — Serde JSON Bridge
- serde_json parsing with recursive conversion to Python objects
- SIMD-accelerated JSON parsing (via serde_json's `--release` optimization)

### pipeline.rs — Parallel DAG Executor
- Receives DAG definition as JSON
- Topological sort → parallel group execution via Tokio
- Supports operations: vector_search, json_parse, count_tokens, truncate, assemble, passthrough

## Token Budgeting Model

Every prompt section gets a token allocation. The assembler enforces budgets with intelligent truncation:

1. Encode full text → get total token count
2. If over budget, estimate character position from token ratio
3. Scan backward to nearest sentence boundary (`. `, `! `, `? `, `\n`)
4. Verify with actual token count
5. Binary search converges in 3-4 iterations

This preserves sentence integrity rather than cutting mid-word.

## Memory Model

- **Rust owns** the HNSW index and tokenizer instances
- **Python holds references** to Rust objects via PyO3
- **Zero-copy where possible**: JSON bytes pass directly to serde_json
- **Results are Python objects**: the Rust runtime converts results to Python dicts/lists before returning

## LLM Calls Stay in Python

LLM API calls are I/O-bound (3-15 seconds each). Python's `asyncio` with `httpx` handles them efficiently. Keeping them in Python means:

- Users can use any HTTP client or LLM SDK
- No complex async Rust-Python interop needed
- The performance bottleneck (context prep) is already solved in Rust
