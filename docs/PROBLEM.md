# The Problem IronRace Solves

## Multi-Agent AI Systems Are Moving to Autonomous Infrastructure

The AI industry is shifting from single request-response chatbots to autonomous multi-agent systems. These systems deploy 5-20 specialized agents per task — each with its own data requirements, context windows, and tool access. At enterprise scale, that means millions of agent pipeline invocations per day.

## The Hidden CPU Bottleneck

In any agent pipeline, two things happen:

1. **Context preparation** — vector search, JSON parsing, tokenization, feature computation, prompt assembly
2. **LLM API calls** — the actual model inference

The LLM call dominates **wall-clock time** (3-15 seconds per call). But context preparation dominates **CPU time**. At single-pipeline scale, you don't notice — a few hundred milliseconds of context prep disappears next to a 10-second LLM call. But at 1,000 concurrent pipelines, that's hundreds of seconds of CPU time per batch.

## The Numbers

We benchmarked the full context preparation pipeline: vector similarity search over a knowledge base, JSON parsing of API responses, token counting, feature aggregation, and prompt assembly with budget enforcement. All numbers are **medians** with recall verified against brute-force ground truth.

| Operation | Pure Python | Rust-Accelerated | Speedup |
|-----------|------------|-----------------|---------|
| Vector search (5K × 384d) | 255ms | 0.11ms | **2,200x** |
| JSON parsing (900KB) | ~24ms | ~2.4ms | 10x |
| Token counting | ~0.07ms | ~0.005ms | 14x |
| Prompt assembly | ~0.03ms | ~0.01ms | 3x |
| **Full pipeline** | **~23ms** | **~4.8ms** | **~5x** |

Vector search recall: 98-99% at 5K-10K vectors, verified against brute-force on every benchmark run.

## Concurrent Scaling

The speedup compounds at scale. IronRace releases the Python GIL during Rust execution, enabling true parallel throughput:

| Concurrent Pipelines | Python | IronRace | Speedup |
|---|---|---|---|
| 10 | 32ms/pipeline, 32/sec | 0.7ms/pipeline, 1,460/sec | **46x** |
| 100 | 32ms/pipeline, 31/sec | 0.4ms/pipeline, 2,259/sec | **72x** |
| 1,000 | 32ms/pipeline, 31/sec | 0.7ms/pipeline, 1,530/sec | **49x** |

Python's throughput is flat because the GIL serializes all CPU work. IronRace's throughput scales with available cores.

## Prior Art

This pattern — Rust acceleration under a Python API — is proven:

- **Pydantic V2**: Rewrote validation in Rust, achieved 17x speedup
- **Polars**: Rust DataFrame library, 5-10x faster than pandas
- **Ruff**: Rust Python linter, 100x faster than flake8

IronRace applies the same architecture to AI agent context preparation: the developer writes pure Python (decorators, type hints, dataclasses), and the Rust runtime handles the CPU-intensive work.

## Why Not Just Use numpy/orjson?

Individual Rust-backed Python libraries (numpy for vectors, orjson for JSON) help, but each call crosses the Python↔Rust bridge separately. We measured this: numpy's vector similarity was actually **5x slower** than pure Python for 20 documents because the per-call bridge overhead dominated.

IronRace crosses the bridge **once per pipeline**. The entire context preparation DAG — vector search, JSON parsing, token counting, and prompt assembly — executes in Rust as a single call. This amortizes the bridge cost and lets the speedup scale with the number of operations.
