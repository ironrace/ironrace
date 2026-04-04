"""Core types for IronRace context definitions."""

__all__ = ["TokenBudget", "Document", "ContextConfig", "VectorSearch", "APIFetch", "Feature"]

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenBudget:
    """Token allocation per prompt section."""

    system: int = 400
    context: int = 3000
    user: int = 600

    @property
    def total(self) -> int:
        return self.system + self.context + self.user


@dataclass
class Document:
    """A retrieved document with content and metadata."""

    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


@dataclass
class ContextConfig:
    """Configuration for context preparation."""

    cache_ttl: int = 3600
    vector_index: str = "default"
    token_model: str = "approximate"


class VectorSearch:
    """Descriptor declaring a vector search dependency.

    Used as a type annotation in @context classes to declare that
    a field should be populated by searching a vector index.
    """

    def __init__(
        self,
        collection: str = "default",
        top_k: int = 10,
        query_field: str = "query",
    ):
        self.collection = collection
        self.top_k = top_k
        self.query_field = query_field

    def __repr__(self) -> str:
        return f"VectorSearch(collection={self.collection!r}, top_k={self.top_k})"


class APIFetch:
    """Descriptor declaring an API data fetch dependency.

    Used as a type annotation in @context classes to declare that
    a field should be populated from an external API call.
    """

    def __init__(self, url: str = "", params: dict | None = None):
        self.url = url
        self.params = params or {}

    def __repr__(self) -> str:
        return f"APIFetch(url={self.url!r})"


class Feature:
    """Descriptor declaring a computed feature.

    Wraps a callable that computes a derived value from other context fields.
    """

    def __init__(self, func: Callable[..., Any]):
        self.func = func

    def __repr__(self) -> str:
        return f"Feature({self.func.__name__ if hasattr(self.func, '__name__') else '...'})"
