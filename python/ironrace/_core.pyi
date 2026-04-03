"""Type stubs for the ironrace._core Rust module."""

from typing import Any

def version() -> str: ...

class VectorIndex:
    def __init__(
        self, vectors: list[list[float]], ef_construction: int = 200
    ) -> None: ...
    def search(
        self, query: list[float], top_k: int = 10
    ) -> list[tuple[int, float]]: ...
    def len(self) -> int: ...
    def is_empty(self) -> bool: ...

def parse_json(data: bytes) -> Any: ...
def serialize_json(obj: Any) -> bytes: ...
def count_tokens(text: str, model: str = "approximate") -> int: ...
def truncate_to_budget(
    text: str, max_tokens: int, model: str = "approximate"
) -> str: ...

class AssemblyResult:
    prompt: str
    total_tokens: int
    sections_truncated: list[str]
    token_breakdown: dict[str, int]

def assemble_prompt(
    template: str,
    values: dict[str, str],
    budgets: dict[str, int],
    model: str = "approximate",
) -> AssemblyResult: ...

def execute_pipeline(dag_json: str) -> str: ...
