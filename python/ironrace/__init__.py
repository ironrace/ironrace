"""IronRace — Rust-powered context engine for AI agent pipelines."""

from ironrace._core import (
    AssemblyResult,
    VectorIndex,
    assemble_prompt,
    count_tokens,
    execute_pipeline,
    parse_json,
    serialize_json,
    truncate_to_budget,
    version,
)
from ironrace.compiler import compile_agents_dag, compile_pipeline
from ironrace.decorators import agent, context, get_registry, pipeline
from ironrace.router import LLMRouter
from ironrace.types import (
    APIFetch,
    ContextConfig,
    Document,
    Feature,
    TokenBudget,
    VectorSearch,
)

__version__ = version()

__all__ = [
    # Rust core
    "VectorIndex",
    "parse_json",
    "serialize_json",
    "count_tokens",
    "truncate_to_budget",
    "assemble_prompt",
    "AssemblyResult",
    "execute_pipeline",
    "version",
    # Python SDK
    "agent",
    "context",
    "pipeline",
    "compile_pipeline",
    "compile_agents_dag",
    "get_registry",
    "LLMRouter",
    # Types
    "TokenBudget",
    "Document",
    "ContextConfig",
    "VectorSearch",
    "APIFetch",
    "Feature",
]
