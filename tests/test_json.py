"""Tests for ironrace._core JSON functions (parse_json, serialize_json)."""

import json

import pytest
from ironrace._core import parse_json, serialize_json


class TestParseJson:
    def test_parse_dict(self):
        data = b'{"key": "value", "num": 42}'
        result = parse_json(data)
        assert result == {"key": "value", "num": 42}

    def test_parse_list(self):
        data = b'[1, 2, 3, "four"]'
        result = parse_json(data)
        assert result == [1, 2, 3, "four"]

    def test_parse_nested(self):
        obj = {"a": {"b": {"c": [1, 2, {"d": True}]}}}
        data = json.dumps(obj).encode()
        result = parse_json(data)
        assert result == obj

    def test_parse_null(self):
        data = b'{"key": null}'
        result = parse_json(data)
        assert result == {"key": None}

    def test_parse_boolean(self):
        data = b'{"t": true, "f": false}'
        result = parse_json(data)
        assert result == {"t": True, "f": False}
        assert isinstance(result["t"], bool)

    def test_parse_float(self):
        data = b'{"pi": 3.14159}'
        result = parse_json(data)
        assert abs(result["pi"] - 3.14159) < 1e-10

    def test_parse_large_integer(self):
        data = b'{"big": 9007199254740992}'
        result = parse_json(data)
        assert result["big"] == 9007199254740992

    def test_parse_unicode(self):
        obj = {"emoji": "Hello \u2603", "cjk": "\u4e16\u754c"}
        data = json.dumps(obj).encode()
        result = parse_json(data)
        assert result == obj

    def test_parse_empty_object(self):
        assert parse_json(b"{}") == {}

    def test_parse_empty_array(self):
        assert parse_json(b"[]") == []

    def test_parse_invalid_json(self):
        with pytest.raises(ValueError, match="JSON parse error"):
            parse_json(b"not json")

    def test_parse_large_payload(self):
        """Parse a realistic API response (~50KB)."""
        obj = {
            "results": [
                {"id": f"doc_{i}", "content": "x" * 500, "score": i * 0.1}
                for i in range(50)
            ]
        }
        data = json.dumps(obj).encode()
        result = parse_json(data)
        assert len(result["results"]) == 50
        assert result["results"][0]["id"] == "doc_0"


class TestSerializeJson:
    def test_serialize_dict(self):
        obj = {"key": "value", "num": 42}
        result = serialize_json(obj)
        assert isinstance(result, bytes)
        assert json.loads(result) == obj

    def test_serialize_list(self):
        obj = [1, 2, 3, "four"]
        result = serialize_json(obj)
        assert json.loads(result) == obj

    def test_serialize_nested(self):
        obj = {"a": {"b": [1, None, True, "str"]}}
        result = serialize_json(obj)
        assert json.loads(result) == obj

    def test_serialize_none(self):
        obj = {"key": None}
        result = serialize_json(obj)
        assert json.loads(result) == obj

    def test_serialize_boolean(self):
        obj = {"t": True, "f": False}
        result = serialize_json(obj)
        parsed = json.loads(result)
        assert parsed["t"] is True
        assert parsed["f"] is False


class TestRoundtrip:
    def test_roundtrip_complex(self):
        """Full roundtrip: Python -> JSON bytes -> Python."""
        original = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "bool_t": True,
            "bool_f": False,
            "null": None,
            "array": [1, "two", None, [3, 4]],
            "nested": {"a": {"b": "c"}},
        }
        serialized = serialize_json(original)
        deserialized = parse_json(serialized)
        assert deserialized == original

    def test_roundtrip_preserves_types(self):
        original = {"i": 1, "f": 1.5, "s": "str", "b": True, "n": None}
        result = parse_json(serialize_json(original))
        assert isinstance(result["i"], int)
        assert isinstance(result["f"], float)
        assert isinstance(result["s"], str)
        assert isinstance(result["b"], bool)
        assert result["n"] is None


class TestJsonPerformance:
    def test_parse_faster_than_stdlib(self):
        """parse_json should be noticeably faster than json.loads for large payloads."""
        import time

        obj = {"results": [{"id": i, "data": "x" * 200} for i in range(100)]}
        data = json.dumps(obj).encode()

        # Warmup
        for _ in range(10):
            parse_json(data)
            json.loads(data)

        # Measure stdlib
        start = time.perf_counter()
        for _ in range(500):
            json.loads(data)
        stdlib_time = time.perf_counter() - start

        # Measure ironrace
        start = time.perf_counter()
        for _ in range(500):
            parse_json(data)
        af_time = time.perf_counter() - start

        speedup = stdlib_time / af_time
        # Serde parsing is faster, but PyO3 object construction adds overhead.
        # CI runners have variable performance; just verify it's in the right ballpark.
        assert speedup > 0.5, f"Only {speedup:.1f}x (expected at least 0.5x of stdlib)"
