# The Problem IronRace Solves

## Multi-Agent AI Systems Are Moving to Autonomous Infrastructure

The AI industry is shifting from single request-response chatbots to autonomous multi-agent systems. These systems deploy 5-20 specialized agents per task — each with its own data requirements, context windows, and tool access. At enterprise scale, that means millions of agent pipeline invocations per day.

## The Hidden CPU Bottleneck

In any agent pipeline, two things happen:

1. **Context preparation** — vector search, JSON parsing, tokenization, feature computation, prompt assembly
2. **LLM API calls** — the actual model inference

The LLM call dominates **wall-clock time** (3-15 seconds per call). But context preparation dominates **CPU time**. At single-pipeline scale, you don't notice — 345ms of context prep disappears next to a 10-second LLM call. But at 1,000 concurrent pipelines, that's 345 seconds of CPU time per batch.

## The Numbers

We benchmarked the full context preparation pipeline: vector similarity search over a knowledge base, JSON parsing of API responses, token counting, feature aggregation, and prompt assembly with budget enforcement.

| Operation | Pure Python | Rust-Accelerated | Speedup |
|-----------|------------|-----------------|---------|
| Vector search (5K × 384d) | ~180ms | ~0.5ms | **360x** |
| JSON parsing (900KB) | ~1.5ms | ~1.1ms | 1.4x |
| Token counting | ~0.05ms | ~0.01ms | 5x |
| Prompt assembly | ~0.15ms | ~0.02ms | 7x |
| **Full pipeline** | **~185ms** | **~2ms** | **~90x** |

## The Cost Multiplier

At 1M agent invocations per day with 6 agents each (6M context preparations):

| Metric | Python | IronRace (Rust) |
|--------|--------|------------------|
| CPU-seconds/day | ~1.1M | ~12K |
| vCPUs needed (sustained) | ~13 | ~0.14 |
| Monthly compute (c5.2xlarge) | ~$400 | ~$4 |

The Rust context engine reduces compute costs by **~100x** for the context preparation layer.

## Prior Art

This pattern — Rust acceleration under a Python API — is proven:

- **Pydantic V2**: Rewrote validation in Rust, achieved 17x speedup
- **Polars**: Rust DataFrame library, 5-10x faster than pandas
- **Ruff**: Rust Python linter, 100x faster than flake8
***REMOVED***

IronRace applies the same architecture to AI agent context preparation: the developer writes pure Python (decorators, type hints, dataclasses), and the Rust runtime handles the CPU-intensive work.

## Why Not Just Use numpy/orjson?

Individual Rust-backed Python libraries (numpy for vectors, orjson for JSON) help, but each call crosses the Python↔Rust bridge separately. We measured this: numpy's vector similarity was actually **5x slower** than pure Python for 20 documents because the per-call bridge overhead dominated.

IronRace crosses the bridge **once per pipeline**. The entire context preparation DAG — vector search, JSON parsing, token counting, and prompt assembly — executes in Rust as a single call. This amortizes the bridge cost and lets the speedup scale with the number of operations.
