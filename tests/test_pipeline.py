"""Tests for ironrace._core.execute_pipeline (DAG executor)."""

import json
import time

import pytest
from ironrace._core import execute_pipeline


def _make_dag(nodes):
    return json.dumps({"nodes": nodes})


class TestPipelineExecution:
    def test_single_node(self):
        dag = _make_dag(
            [
                {
                    "id": "n1",
                    "op": {"type": "count_tokens", "text": "hello world"},
                    "depends_on": [],
                }
            ]
        )
        result = json.loads(execute_pipeline(dag))
        assert "n1" in result
        assert result["n1"]["count"] > 0

    def test_parallel_nodes(self):
        dag = _make_dag(
            [
                {
                    "id": "a",
                    "op": {"type": "count_tokens", "text": "first"},
                    "depends_on": [],
                },
                {
                    "id": "b",
                    "op": {"type": "count_tokens", "text": "second"},
                    "depends_on": [],
                },
                {
                    "id": "c",
                    "op": {"type": "count_tokens", "text": "third"},
                    "depends_on": [],
                },
            ]
        )
        result = json.loads(execute_pipeline(dag))
        assert len(result) == 3
        assert all(k in result for k in ["a", "b", "c"])

    def test_dependency_ordering(self):
        dag = _make_dag(
            [
                {
                    "id": "first",
                    "op": {"type": "count_tokens", "text": "step one"},
                    "depends_on": [],
                },
                {
                    "id": "second",
                    "op": {
                        "type": "assemble",
                        "template": "Result: {val}",
                        "values": {"val": "done"},
                        "budgets": {},
                    },
                    "depends_on": ["first"],
                },
            ]
        )
        result = json.loads(execute_pipeline(dag))
        assert "first" in result
        assert "second" in result
        assert "Result: done" in result["second"]["prompt"]

    def test_diamond_dependency(self):
        """A -> B, A -> C, B+C -> D."""
        dag = _make_dag(
            [
                {
                    "id": "A",
                    "op": {"type": "count_tokens", "text": "start"},
                    "depends_on": [],
                },
                {
                    "id": "B",
                    "op": {"type": "count_tokens", "text": "branch1"},
                    "depends_on": ["A"],
                },
                {
                    "id": "C",
                    "op": {"type": "count_tokens", "text": "branch2"},
                    "depends_on": ["A"],
                },
                {
                    "id": "D",
                    "op": {
                        "type": "assemble",
                        "template": "Final: {r}",
                        "values": {"r": "merged"},
                        "budgets": {},
                    },
                    "depends_on": ["B", "C"],
                },
            ]
        )
        result = json.loads(execute_pipeline(dag))
        assert len(result) == 4

    def test_assemble_operation(self):
        dag = _make_dag(
            [
                {
                    "id": "prompt",
                    "op": {
                        "type": "assemble",
                        "template": "You are {role}. {context}",
                        "values": {
                            "role": "an analyst",
                            "context": "Important data here. " * 20,
                        },
                        "budgets": {"context": 10},
                    },
                    "depends_on": [],
                }
            ]
        )
        result = json.loads(execute_pipeline(dag))
        assert "prompt" in result
        assert result["prompt"]["total_tokens"] > 0
        assert "context" in result["prompt"]["sections_truncated"]

    def test_truncate_operation(self):
        dag = _make_dag(
            [
                {
                    "id": "trunc",
                    "op": {
                        "type": "truncate",
                        "text": "A long text. " * 100,
                        "max_tokens": 10,
                    },
                    "depends_on": [],
                }
            ]
        )
        result = json.loads(execute_pipeline(dag))
        assert len(result["trunc"]["text"]) < len("A long text. " * 100)

    def test_passthrough_operation(self):
        dag = _make_dag(
            [
                {
                    "id": "data",
                    "op": {"type": "passthrough", "data": {"key": "value", "num": 42}},
                    "depends_on": [],
                }
            ]
        )
        result = json.loads(execute_pipeline(dag))
        assert result["data"] == {"key": "value", "num": 42}

    def test_invalid_dag_json(self):
        with pytest.raises(ValueError, match="Invalid DAG JSON"):
            execute_pipeline("not json")

    def test_empty_dag(self):
        result = json.loads(execute_pipeline(_make_dag([])))
        assert result == {}

    def test_performance_many_nodes(self):
        """50 parallel nodes should execute quickly."""
        nodes = [
            {
                "id": f"n{i}",
                "op": {"type": "count_tokens", "text": f"text number {i}"},
                "depends_on": [],
            }
            for i in range(50)
        ]
        dag = _make_dag(nodes)

        start = time.perf_counter()
        result = json.loads(execute_pipeline(dag))
        elapsed = (time.perf_counter() - start) * 1000

        assert len(result) == 50
        assert elapsed < 100, f"50-node DAG took {elapsed:.1f}ms (expected < 100ms)"
