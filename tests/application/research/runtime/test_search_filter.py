"""Unit tests for search relevance filter (_filter_relevant)."""

from __future__ import annotations

from houyi.application.research.runtime.search_sufficiency import (
    _extract_keywords,
    _filter_relevant,
)
from houyi.application.research.types import SourceReference


def _src(title: str, snippet: str = "", url: str = "https://x.com") -> SourceReference:
    return SourceReference(url=url, title=title, snippet=snippet, source_type="web")


class TestFilterRelevant:
    def test_keeps_relevant_sources(self):
        sources = [
            _src("AI Agent Framework Comparison", "Compares leading agent frameworks"),
            _src("Weather forecast today", "Sunny with clouds"),
        ]
        filtered = _filter_relevant(sources, "AI agent frameworks", "Compare AI agents")
        assert len(filtered) == 1
        assert filtered[0].title == "AI Agent Framework Comparison"

    def test_keeps_all_relevant(self):
        sources = [
            _src("Python programming guide", "Learn Python basics"),
            _src("Python tutorial for beginners", "Step by step Python"),
        ]
        filtered = _filter_relevant(sources, "Python programming", "Learn Python")
        assert len(filtered) == 2

    def test_returns_all_empty_filter(self):
        sources = [_src("Completely unrelated", "Nothing matches")]
        filtered = _filter_relevant(sources, "quantum physics", "quantum computing")
        assert len(filtered) == 1

    def test_keeps_sources_without_text(self):
        sources = [_src("", "")]
        filtered = _filter_relevant(sources, "AI agent", "AI")
        assert len(filtered) == 1

    def test_empty_sources(self):
        filtered = _filter_relevant([], "test", "test")
        assert filtered == []

    def test_min_overlap_parameter(self):
        sources = [
            _src("AI overview", "Brief intro to AI"),
            _src("Deep dive into AI agent architecture", "Detailed analysis of agent patterns"),
        ]
        filtered = _filter_relevant(
            sources, "AI agent architecture patterns", "AI agents", min_overlap=3
        )
        assert len(filtered) >= 1


class TestExtractKeywords:
    def test_extracts_meaningful_words(self):
        kw = _extract_keywords("What are the best AI agent frameworks?")
        assert "best" in kw
        assert "agent" in kw
        assert "frameworks" in kw

    def test_filters_stop_words(self):
        kw = _extract_keywords("What are the and for with")
        assert "the" not in kw
        assert "and" not in kw

    def test_filters_short_words(self):
        kw = _extract_keywords("I am an AI")
        assert "am" not in kw
        assert "an" not in kw

    def test_lowercased(self):
        kw = _extract_keywords("Python FRAMEWORK")
        assert "python" in kw
        assert "framework" in kw

    def test_empty(self):
        assert _extract_keywords("") == set()
