# llama-index-vector-stores-ironrace

Rust-accelerated vector store for LlamaIndex, powered by IronRace's HNSW implementation.

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
