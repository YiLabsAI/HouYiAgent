"""Tests for agentic policy helpers."""

from __future__ import annotations

import pytest

from houyi.rag.agentic.mode import SearchRoundType
from houyi.rag.agentic.policies import (
    build_answer_simple,
    deduplicate_results,
    extract_entities,
    extract_keywords_simple,
    get_top_files,
    refine_keywords,
    should_terminate,
)
from houyi.rag.types import SearchResult, Source


class TestSearchRoundType:
    def test_round_type_values(self) -> None:
        assert SearchRoundType.BROAD.value == "broad"
        assert SearchRoundType.FOCUSED.value == "focused"
        assert SearchRoundType.SEMANTIC.value == "semantic"
        assert SearchRoundType.CROSS_REF.value == "cross_ref"
        assert SearchRoundType.VERIFY.value == "verify"

    def test_round_type_count(self) -> None:
        assert len(SearchRoundType) == 5


class TestAgenticPolicies:
    def test_extract_keywords_simple(self) -> None:
        keywords = extract_keywords_simple("What is machine learning?")

        assert "machine" in keywords
        assert "learning" in keywords
        assert "what" not in keywords
        assert "is" not in keywords

    def test_extract_filters_short(self) -> None:
        keywords = extract_keywords_simple("A B C test")

        assert "test" in keywords
        assert "A" not in keywords
        assert "B" not in keywords

    def test_should_terminate_early(self) -> None:
        results = [SearchResult(chunk_id=f"c{i}", content="test", score=0.8) for i in range(5)]
        assert not should_terminate(results, SearchRoundType.BROAD)
        assert should_terminate(results, SearchRoundType.FOCUSED)

    def test_not_enough_results(self) -> None:
        results = [SearchResult(chunk_id="c1", content="test", score=0.8)]
        assert not should_terminate(results, SearchRoundType.FOCUSED)

    def test_get_top_files(self) -> None:
        results = [
            SearchResult(chunk_id="c1", content="a", score=0.9, source=Source(file_path="/a.md")),
            SearchResult(chunk_id="c2", content="b", score=0.7, source=Source(file_path="/b.md")),
            SearchResult(chunk_id="c3", content="a2", score=0.8, source=Source(file_path="/a.md")),
            SearchResult(chunk_id="c4", content="c", score=0.6, source=Source(file_path="/c.md")),
        ]

        top_files = get_top_files(results, limit=2)

        assert len(top_files) == 2
        assert top_files[0] == "/a.md"
        assert top_files[1] == "/b.md"

    def test_extract_entities(self) -> None:
        results = [
            SearchResult(
                chunk_id="c1",
                content='Python and TensorFlow are tools. "machine learning" is important.',
                score=0.9,
            ),
            SearchResult(
                chunk_id="c2", content="JavaScript runs in browsers. React is popular.", score=0.8
            ),
        ]

        entities = extract_entities(results)

        assert "Python" in entities
        assert "TensorFlow" in entities
        assert "JavaScript" in entities
        assert "React" in entities
        assert "machine learning" in entities

    def test_refine_keywords(self) -> None:
        results = [
            SearchResult(
                chunk_id="c1", content="Python is great for machine learning tasks", score=0.9
            ),
        ]

        refined = refine_keywords("What is Python machine learning?", results)

        assert "python" in refined or "machine" in refined or "learning" in refined

    def test_deduplicate_results(self) -> None:
        results = [
            SearchResult(chunk_id="c1", content="Same content here", score=0.9),
            SearchResult(chunk_id="c2", content="Same content here", score=0.8),
            SearchResult(chunk_id="c3", content="Different content", score=0.7),
        ]

        deduped = deduplicate_results(results)

        assert len(deduped) == 2
        assert deduped[0].score == 0.9
        assert deduped[1].content == "Different content"

    def test_deduplicate_preserves_order(self) -> None:
        results = [
            SearchResult(chunk_id="c1", content="Low score", score=0.5),
            SearchResult(chunk_id="c2", content="High score", score=0.9),
            SearchResult(chunk_id="c3", content="Medium score", score=0.7),
        ]

        deduped = deduplicate_results(results)

        assert deduped[0].score == 0.9
        assert deduped[1].score == 0.7
        assert deduped[2].score == 0.5

    @pytest.mark.asyncio
    async def test_build_answer_simple(self) -> None:
        results = [
            SearchResult(chunk_id="c1", content="First result content", score=0.9),
            SearchResult(chunk_id="c2", content="Second result content", score=0.8),
        ]

        answer = build_answer_simple(results)

        assert "First result content" in answer
        assert "Second result content" in answer
        assert "---" in answer

    @pytest.mark.asyncio
    async def test_build_answer_simple_empty(self) -> None:
        answer = build_answer_simple([])
        assert "No relevant" in answer
