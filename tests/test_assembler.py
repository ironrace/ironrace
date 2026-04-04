"""Tests for ironrace._core.assemble_prompt."""


from ironrace._core import assemble_prompt, count_tokens


class TestAssemblePrompt:
    def test_basic_interpolation(self):
        result = assemble_prompt(
            "Hello {name}, you are a {role}.",
            {"name": "Alice", "role": "developer"},
            {},
        )
        assert result.prompt == "Hello Alice, you are a developer."

    def test_no_placeholders(self):
        result = assemble_prompt("Plain text with no variables.", {}, {})
        assert result.prompt == "Plain text with no variables."

    def test_missing_value_preserved(self):
        """Placeholders without values remain in the template."""
        result = assemble_prompt("Hello {name}, {greeting}.", {"name": "Bob"}, {})
        assert result.prompt == "Hello Bob, {greeting}."

    def test_token_count(self):
        result = assemble_prompt("Hello world.", {}, {})
        assert result.total_tokens > 0
        assert result.total_tokens == count_tokens("Hello world.")

    def test_token_breakdown(self):
        result = assemble_prompt(
            "{a} and {b}",
            {"a": "first part", "b": "second part"},
            {},
        )
        assert "a" in result.token_breakdown
        assert "b" in result.token_breakdown

    def test_budget_enforcement(self):
        long_text = "Word " * 100
        result = assemble_prompt(
            "Context: {context}",
            {"context": long_text},
            {"context": 10},
        )
        context_tokens = result.token_breakdown["context"]
        assert context_tokens <= 10

    def test_sections_truncated_tracking(self):
        long_text = "A sentence here. " * 50
        result = assemble_prompt(
            "{text}",
            {"text": long_text},
            {"text": 5},
        )
        assert "text" in result.sections_truncated

    def test_no_truncation_not_reported(self):
        result = assemble_prompt(
            "{text}",
            {"text": "Short."},
            {"text": 100},
        )
        assert "text" not in result.sections_truncated

    def test_multiple_sections_with_budgets(self):
        result = assemble_prompt(
            "System: {system}\nContext: {context}\nUser: {user}",
            {
                "system": "You are helpful. " * 5,
                "context": "Long context data here. " * 50,
                "user": "What is the meaning of life?",
            },
            {"system": 10, "context": 20, "user": 50},
        )
        assert result.token_breakdown["system"] <= 10
        assert result.token_breakdown["context"] <= 20

    def test_repr(self):
        result = assemble_prompt("test", {}, {})
        assert "AssemblyResult" in repr(result)
        assert "total_tokens" in repr(result)

    def test_empty_values(self):
        result = assemble_prompt("{a}{b}", {"a": "", "b": ""}, {})
        assert result.prompt == ""

    def test_performance(self):
        """Assembly should be fast even with multiple sections."""
        import time

        template = "System: {system}\nDocs: {docs}\nQuery: {query}"
        values = {
            "system": "You are a helpful assistant. " * 10,
            "docs": "Document content here. " * 100,
            "query": "What should I do?",
        }
        budgets = {"system": 30, "docs": 100, "query": 50}

        start = time.perf_counter()
        for _ in range(100):
            assemble_prompt(template, values, budgets)
        elapsed = (time.perf_counter() - start) / 100 * 1000  # ms

        assert elapsed < 5, f"Assembly took {elapsed:.1f}ms (expected < 5ms)"
