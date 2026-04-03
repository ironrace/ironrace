# IronRace Quickstart

Get up and running in 5 minutes.

## Installation

```bash
pip install ironrace
```

For development:
```bash
git clone https://github.com/ironrace/ironrace.git
cd ironrace
pip install maturin
maturin develop --release
pip install -e ".[dev]"
```

## 1. Vector Search

Build an index and search it:

```python
from ironrace import VectorIndex

# Build index from embeddings (list of float lists)
vectors = [...]  # your document embeddings
index = VectorIndex(vectors, ef_construction=200)

# Search
results = index.search(query_vector, top_k=10)
for idx, score in results:
    print(f"Document {idx}: similarity {score:.3f}")
```

## 2. Prompt Assembly with Token Budgets

```python
from ironrace import assemble_prompt

result = assemble_prompt(
    template="System: {system}\nContext: {context}\nQuestion: {question}",
    values={
        "system": "You are a helpful assistant.",
        "context": very_long_document_text,
        "question": user_query,
    },
    budgets={"context": 2000, "system": 200, "question": 100},
)

print(f"Prompt: {result.total_tokens} tokens")
print(f"Truncated sections: {result.sections_truncated}")
```

## 3. Full Pipeline (Single Rust Call)

```python
import json
from ironrace import execute_pipeline, compile_agents_dag

# Define agents with templates and budgets
agents = [
    {
        "id": "analyst",
        "template": "You are {role}. Context: {docs}. Question: {query}",
        "values": {"role": "a market analyst", "docs": document_text, "query": idea},
        "budgets": {"docs": 500},
    },
    {
        "id": "engineer",
        "template": "You are {role}. Specs: {specs}. Evaluate: {query}",
        "values": {"role": "a tech lead", "specs": tech_specs, "query": idea},
        "budgets": {"specs": 500},
    },
]

# Compile and execute — ONE Rust call for all agents
dag_json = compile_agents_dag(agents)
results = json.loads(execute_pipeline(dag_json))

for agent_id, result in results.items():
    print(f"{agent_id}: {result['total_tokens']} tokens")
```

## 4. Decorators (Pythonic API)

```python
from ironrace import agent, context, pipeline, VectorSearch, Document

@context
class StartupContext:
    idea: str
    competitors: list[Document] = VectorSearch(collection="kb", top_k=10)

@agent(model="claude-sonnet-4-20250514", context=StartupContext)
def analyst(ctx: StartupContext, idea: str = "") -> str:
    return f"Analyze competitors: {ctx.competitors}. Idea: {idea}"

# Call the agent
result = analyst(idea="AI travel concierge")
print(result["prompt"])
```

## 5. Run the Examples

```bash
# Startup evaluator (dry run — no API key needed)
python examples/startup_evaluator.py

# RAG chatbot
python examples/rag_chatbot.py

# Batch research (100 parallel pipelines)
python examples/batch_research.py

# Full benchmark suite
python benchmarks/bench_context_prep.py
```

## Key Concepts

- **Context preparation** (vector search, JSON, tokenization, assembly) runs in **Rust**
- **LLM API calls** run in **Python** (async, I/O-bound)
- The Rust runtime crosses the Python bridge **once per pipeline**, not per operation
- Token budgets are enforced with **sentence-boundary truncation**
- Vector indices live in **Rust memory** — built once, searched many times
