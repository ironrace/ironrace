# IronRace Benchmarks

## What We Measure

IronRace benchmarks measure **context preparation time only** — not LLM API call latency. Context preparation includes:

1. **Vector similarity search** — HNSW approximate nearest neighbor over a knowledge base
2. **JSON parsing/serialization** — API response handling
3. **Token counting** — for budget management
4. **Prompt assembly** — template interpolation with token budget enforcement
5. **Full pipeline** — all of the above via the DAG executor

All benchmarks report **median** latency (not mean) and verify **recall against brute-force ground truth**.

## Test Data

Benchmarks use **clustered vectors** that simulate real embeddings from models like sentence-transformers or OpenAI. Uniformly random vectors are a pathological worst case due to the curse of dimensionality; real embeddings have cluster structure that HNSW exploits effectively.

- Knowledge base: 5,000 vectors × 384 dimensions (clustered, unit-normalized)
- API response: ~900KB JSON (50 documents with metadata)
- Token counting text: 1,000-word English passage
- Prompt template: multi-section with 3 variable interpolations and budgets

All test data is synthetically generated with a fixed random seed for reproducibility.

## Running Benchmarks

```bash
# Full head-to-head benchmark (with recall verification)
python -m benchmarks.bench_context_prep

# Vector search at multiple scales (with recall verification)
python -m benchmarks.bench_vector_search
python -m benchmarks.bench_vector_search --full  # includes 100K

# Concurrent pipeline scaling
python -m benchmarks.bench_at_scale
```

## Methodology

Each operation is benchmarked with:
- **200 iterations** after 10 warmup iterations
- Statistics: mean, **median** (used for all claims), P95, P99
- Both LlamaIndex and IronRace (Rust) implementations run identical operations
- Timing uses `time.perf_counter()` for high-resolution measurement
- Test data is generated once and reused across all iterations
- **Recall@10** verified against LlamaIndex SimpleVectorStore (brute-force) on every run

## HNSW Configuration

IronRace uses the `hnsw_rs` crate with these defaults:

- `ef_construction=100` — gives 98%+ recall on real-world embeddings
- `max_nb_connection=16` — standard HNSW connectivity parameter
- `ef_search=max(top_k, 16)` — adapts to query size

For very large datasets (1M+ vectors), increase `ef_construction` to 200+ for higher recall:

```python
index = VectorIndex(vectors)                       # 98%+ recall (default)
index = VectorIndex(vectors, ef_construction=200)  # higher recall at scale
```

## GIL Release

All compute-heavy Rust functions release the Python GIL via `py.allow_threads()`. This enables true concurrent throughput from Python threads:

- `VectorIndex.__init__()` — releases GIL during index construction
- `VectorIndex.search()` — releases GIL during HNSW search
- `execute_pipeline()` — releases GIL during entire DAG execution

At 1,000 concurrent pipelines, IronRace achieves **49x** throughput over pure Python (1,530 pipelines/sec vs 31/sec).

## How to Reproduce

```bash
git clone https://github.com/ironrace/ironrace.git
cd ironrace
pip install maturin
maturin develop --release
pip install numpy  # for test data generation
python -m benchmarks.bench_context_prep
python -m benchmarks.bench_vector_search
```
