# IronRace Benchmarks

## What We Measure

IronRace benchmarks measure **context preparation time only** — not LLM API call latency. Context preparation includes:

1. **Vector similarity search** — cosine similarity over a knowledge base
2. **JSON parsing/serialization** — API response handling
3. **Token counting** — for budget management
4. **Prompt assembly** — template interpolation with token budget enforcement
5. **Full pipeline** — all of the above via the DAG executor

## Test Data

- Knowledge base: 5,000 vectors × 384 dimensions (unit-normalized)
- API response: ~900KB JSON (50 documents with metadata)
- Token counting text: 1,000-word English passage
- Prompt template: multi-section with 3 variable interpolations and budgets

All test data is synthetically generated with a fixed random seed for reproducibility.

## Running Benchmarks

```bash
# Full head-to-head benchmark
python -m benchmarks.bench_context_prep

# Vector search at multiple scales
python benchmarks/bench_vector_search.py
python benchmarks/bench_vector_search.py --full  # includes 100K

# Concurrent pipeline scaling
python -m benchmarks.bench_at_scale
```

## Methodology

Each operation is benchmarked with:
- **200 iterations** after 10 warmup iterations
- Statistics: mean, median, P95, P99
- Both pure Python and IronRace (Rust) implementations run identical operations
- Timing uses `time.perf_counter()` for high-resolution measurement
- Test data is generated once and reused across all iterations

## Cost Projection Assumptions

Monthly compute projections assume:
- 1M agent pipeline invocations per day
- 6 agents per invocation (6M context preparations/day)
- AWS c5.2xlarge instances ($0.34/hr, 8 vCPU)
- Sustained utilization (24/7 operation)

## How to Reproduce

```bash
git clone https://github.com/yourorg/ironrace.git
cd ironrace
pip install maturin
maturin develop --release
pip install numpy  # for test data generation
python -m benchmarks.bench_context_prep
```
