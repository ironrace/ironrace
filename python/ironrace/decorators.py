"""Decorators for defining agents, contexts, and pipelines."""

import asyncio
import functools
from collections.abc import Callable
from typing import Any

from ironrace.types import APIFetch, Feature, VectorSearch

# Global registry for all decorated components
_registry: dict[str, dict[str, Any]] = {
    "agents": {},
    "contexts": {},
    "pipelines": {},
}

# Compiled DAG cache
_compiled_dags: dict[str, dict] = {}


def get_registry() -> dict[str, dict[str, Any]]:
    """Access the global registry (for compiler and testing)."""
    return _registry


def context(cls: type) -> type:
    """Decorator for context classes.

    Inspects class annotations to identify data source descriptors
    (VectorSearch, APIFetch, Feature) and registers the context class.

    Example:
        @context
        class MyContext:
            query: str  # input field
            docs: list[Document] = VectorSearch(collection="kb", top_k=10)
            data: dict = APIFetch(url="https://api.example.com")
    """
    fields = {}
    descriptors = {}

    # Get type annotations
    annotations = getattr(cls, "__annotations__", {})

    for field_name, field_type in annotations.items():
        # Check if there's a default value that's a descriptor
        default = getattr(cls, field_name, None)
        if isinstance(default, (VectorSearch, APIFetch, Feature)):
            descriptors[field_name] = default
        else:
            fields[field_name] = {"type": field_type, "default": default}

    # Store metadata on the class
    cls._af_fields = fields
    cls._af_descriptors = descriptors
    cls._af_type = "context"

    # Register
    _registry["contexts"][cls.__name__] = {
        "cls": cls,
        "fields": fields,
        "descriptors": descriptors,
    }

    return cls


def agent(
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1500,
    context: type | None = None,
    token_budget: Any | None = None,
) -> Callable:
    """Decorator for agent functions.

    At runtime, the decorated function:
    1. Prepares context via the Rust runtime
    2. Calls the wrapped function to build the prompt
    3. Sends the prompt to the LLM via the router

    Example:
        @agent(model="claude-sonnet-4-20250514", context=MyContext)
        def analyst(ctx: MyContext, idea: str) -> str:
            return f"Analyze this: {ctx.docs}. Idea: {idea}"
    """

    def decorator(func: Callable) -> Callable:
        agent_name = func.__name__

        # Store metadata
        func._af_type = "agent"
        func._af_model = model
        func._af_max_tokens = max_tokens
        func._af_context = context
        func._af_token_budget = token_budget

        # Register
        _registry["agents"][agent_name] = {
            "func": func,
            "model": model,
            "max_tokens": max_tokens,
            "context": context,
            "token_budget": token_budget,
        }

        @functools.wraps(func)
        async def async_wrapper(**kwargs):
            # Build context if context class is specified
            ctx_instance = None
            if context is not None:
                ctx_instance = context()
                # Populate input fields from kwargs
                for key, val in kwargs.items():
                    if hasattr(ctx_instance, key) or key in getattr(
                        context, "__annotations__", {}
                    ):
                        setattr(ctx_instance, key, val)

            # Call the agent function to get the prompt
            if ctx_instance is not None:
                prompt = func(ctx_instance, **kwargs)
            else:
                prompt = func(**kwargs)

            # Return prompt and metadata (LLM call handled by pipeline/router)
            return {
                "agent": agent_name,
                "prompt": prompt,
                "model": model,
                "max_tokens": max_tokens,
            }

        @functools.wraps(func)
        def sync_wrapper(**kwargs):
            try:
                asyncio.get_running_loop()
                # Already in async context — return coroutine
                return async_wrapper(**kwargs)
            except RuntimeError:
                # No event loop — run synchronously
                return asyncio.run(async_wrapper(**kwargs))

        # Allow both sync and async calling
        sync_wrapper._af_async = async_wrapper
        sync_wrapper._af_type = "agent"
        sync_wrapper._af_model = model
        sync_wrapper._af_max_tokens = max_tokens
        sync_wrapper._af_context = context

        return sync_wrapper

    return decorator


def pipeline(concurrency: int = 10) -> Callable:
    """Decorator for pipeline functions.

    A pipeline orchestrates multiple agents, managing their execution order
    and parallelism. The Rust runtime handles context preparation for all
    agents in a single bridge crossing.

    Example:
        @pipeline(concurrency=50)
        def evaluate(idea: str):
            sec = security_analyst(idea=idea)
            mkt = market_analyst(idea=idea)
            return verdict(results={"security": sec, "market": mkt})
    """

    def decorator(func: Callable) -> Callable:
        pipeline_name = func.__name__

        func._af_type = "pipeline"
        func._af_concurrency = concurrency

        _registry["pipelines"][pipeline_name] = {
            "func": func,
            "concurrency": concurrency,
        }

        @functools.wraps(func)
        async def async_wrapper(**kwargs):
            return func(**kwargs)

        @functools.wraps(func)
        def sync_wrapper(**kwargs):
            try:
                asyncio.get_running_loop()
                return async_wrapper(**kwargs)
            except RuntimeError:
                return asyncio.run(async_wrapper(**kwargs))

        sync_wrapper._af_async = async_wrapper
        sync_wrapper._af_type = "pipeline"
        sync_wrapper._af_concurrency = concurrency

        return sync_wrapper

    return decorator
