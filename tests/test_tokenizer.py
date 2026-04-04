"""Tests for ironrace._core tokenizer functions."""


from ironrace._core import count_tokens, truncate_to_budget


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_single_word(self):
        assert count_tokens("hello") >= 1

    def test_short_sentence(self):
        tokens = count_tokens("The quick brown fox jumps over the lazy dog.")
        assert 9 <= tokens <= 15  # ~9 words, approximate counting

    def test_long_text(self):
        text = "word " * 1000
        tokens = count_tokens(text)
        assert tokens >= 900  # should be close to 1000

    def test_approximate_model(self):
        text = "Hello world"
        t1 = count_tokens(text, "approximate")
        t2 = count_tokens(text)  # default is approximate
        assert t1 == t2

    def test_nonexistent_model_falls_back(self):
        """Non-existent tokenizer file should fall back to approximate."""
        tokens = count_tokens("hello world", "/nonexistent/tokenizer.json")
        assert tokens > 0  # falls back to approximate

    def test_longer_words_more_tokens(self):
        short_count = count_tokens("a b c d e")
        long_count = count_tokens("internationalization implementation")
        # Longer words should produce more tokens per word
        assert long_count >= short_count


class TestTruncateToBudget:
    def test_no_truncation_needed(self):
        text = "Short text."
        result = truncate_to_budget(text, 100)
        assert result == text

    def test_truncation_at_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = truncate_to_budget(text, 5)
        # Should truncate at a sentence boundary
        assert result.endswith(". ") or result.endswith(".")
        assert len(result) < len(text)

    def test_respects_budget(self):
        text = "Word " * 100
        result = truncate_to_budget(text, 10)
        tokens = count_tokens(result)
        assert tokens <= 10

    def test_empty_string(self):
        assert truncate_to_budget("", 10) == ""

    def test_budget_of_zero(self):
        result = truncate_to_budget("Hello world.", 0)
        assert result == ""

    def test_preserves_sentence_structure(self):
        text = "The AI system works well. It processes data quickly. Results are accurate. Performance is good."
        result = truncate_to_budget(text, 8)
        # Result should be complete sentences
        if result:
            assert "." in result

    def test_single_long_sentence(self):
        text = "This is one very long sentence without any breaks " * 20
        result = truncate_to_budget(text, 5)
        assert len(result) < len(text)

    def test_performance(self):
        """Truncation should be fast even for large texts."""
        import time

        text = "A sentence here. " * 1000
        start = time.perf_counter()
        for _ in range(100):
            truncate_to_budget(text, 50)
        elapsed = (time.perf_counter() - start) / 100 * 1000  # ms

        assert elapsed < 10, f"Truncation took {elapsed:.1f}ms (expected < 10ms)"
