# llama-index-vector-stores-ironrace

Rust-accelerated vector store for LlamaIndex, powered by IronRace's HNSW implementation.

## When to use

- Fast similarity search over 1K-100K documents (sub-millisecond queries)
- RAG pipelines where retrieval latency matters
- Drop-in replacement for `SimpleVectorStore` with no infrastructure dependencies

For billion-scale vectors or managed infrastructure, use a dedicated vector database instead.

## Installation

```bash
pip install llama-index-vector-stores-ironrace
```

## Quick Start

```python
from llama_index.vector_stores.ironrace import IronRaceVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

vector_store = IronRaceVectorStore()
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes=documents, storage_context=storage_context)

results = index.as_retriever(similarity_top_k=10).retrieve("my query")
```

## Configuration

```python
vector_store = IronRaceVectorStore(
    ef_construction=40,               # HNSW build quality (higher = better recall, slower build)
    similarity_top_k_multiplier=3.0,  # Over-fetch factor when metadata filters are applied
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ef_construction` | 40 | Controls HNSW graph quality. Higher values improve recall at the cost of slower index builds. 40 is good for most RAG use cases. |
| `similarity_top_k_multiplier` | 3.0 | When metadata filters are active, fetches this many times `top_k` results before filtering. Increase if filters are very selective. |

## Features

- **HNSW search** via Rust's instant-distance crate (approximate nearest neighbor)
- **Metadata filtering** with all LlamaIndex filter operators (EQ, GT, IN, CONTAINS, TEXT_MATCH, etc.)
- **AND/OR filter conditions**
- **Persistence** via JSON (save/load with `persist()` / `from_persist_path()`)
- **Namespace support** for multi-tenant storage
- **Lazy index rebuild** -- index is only rebuilt when nodes are added or deleted

## Scaling

| Vectors | Build time | Query (top-10) | Notes |
|---------|-----------|----------------|-------|
| 1K | ~5ms | <0.1ms | Instant for small collections |
| 10K | ~50ms | ~0.5ms | Typical RAG application |
| 100K | ~500ms | ~1ms | Large knowledge base |

Build is a one-time cost (or on add/delete). Queries stay fast as the collection grows.

## Compatibility

- LlamaIndex >= 0.10
- Python >= 3.11
- Supported platforms: Linux (x86_64, aarch64), macOS (x86_64, arm64), Windows (x86_64)
