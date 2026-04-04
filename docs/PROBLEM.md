# The Problem IronRace Solves

## Multi-Agent AI Systems Are Moving to Autonomous Infrastructure

The AI industry is shifting from single request-response chatbots to autonomous multi-agent systems. These systems deploy 5-20 specialized agents per task — each with its own data requirements, context windows, and tool access. At enterprise scale, that means millions of agent pipeline invocations per day.

## The Hidden CPU Bottleneck

In any agent pipeline, two things happen:

1. **Context preparation** — vector search, JSON parsing, tokenization, feature computation, prompt assembly
2. **LLM API calls** — the actual model inference

The LLM call dominates **wall-clock time** (3-15 seconds per call). But context preparation dominates **CPU time**. At single-pipeline scale, you don't notice — a few hundred milliseconds of context prep disappears next to a 10-second LLM call. But at 1,000 concurrent pipelines, that's hundreds of seconds of CPU time per batch.

## The Numbers

We benchmarked the full context preparation pipeline against actual LlamaIndex code: vector similarity search (SimpleVectorStore), JSON parsing, token counting (tiktoken), and prompt assembly (PromptHelper + PromptTemplate). All numbers are **medians** with recall verified against LlamaIndex brute-force ground truth.

| Operation | LlamaIndex | IronRace (Rust) | Speedup |
|-----------|------------|-----------------|---------|
| Vector search (5K × 384d) | 54ms | 0.5ms | **96-133x** |
| JSON parsing (900KB) | ~9ms | ~3ms | 3x |
| Token counting (vs tiktoken) | 0.24ms | 0.005ms | 50x |
| Prompt assembly | 0.6ms | 0.01ms | 65x |
| **Full pipeline** | **64ms** | **4ms** | **16x** |

Vector search recall: 99% at 5K vectors, 98% at 10K vectors, verified against LlamaIndex brute-force on every benchmark run.

## Concurrent Scaling

The single-threaded 16x speedup compounds dramatically at scale. IronRace releases the Python GIL during Rust execution, enabling true parallel throughput:

| Concurrent Pipelines | Python | IronRace | Speedup |
|---|---|---|---|
| 10 | 32ms/pipeline, 31/sec | 0.8ms/pipeline, 1,267/sec | **41x** |
| 100 | 31ms/pipeline, 32/sec | 0.5ms/pipeline, 1,823/sec | **58x** |
| 1,000 | 32ms/pipeline, 31/sec | 0.6ms/pipeline, 1,715/sec | **55x** |

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
