from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from houyi.rag.generation.crag import CRAGResult, CRAGValidator, RetrievalQuality
from houyi.rag.types import SearchResult, Source


class TestCRAGValidator:
    def test_validate_empty_results(self) -> None:
        validator = CRAGValidator()

        async def run():
            return await validator.validate("test query", [])

        result = asyncio.run(run())

        assert result.quality == RetrievalQuality.INCORRECT
        assert result.needs_refinement is True
        assert len(result.suggested_queries) > 0

    def test_validate_heuristic_relevant(self) -> None:
        validator = CRAGValidator(relevance_threshold=0.3)
        results = [
            SearchResult(
                file_path="/doc.md",
                content="This is about machine learning and AI",
                score=0.8,
            ),
        ]

        async def run():
            return await validator.validate("machine learning", results)

        result = asyncio.run(run())

        assert result.quality in [RetrievalQuality.CORRECT, RetrievalQuality.AMBIGUOUS]
        assert len(result.relevant_results) > 0

    def test_validate_heuristic_irrelevant(self) -> None:
        validator = CRAGValidator(relevance_threshold=0.5)
        results = [
            SearchResult(
                file_path="/doc.md",
                content="This is about cooking recipes",
                score=0.1,
            ),
        ]

        async def run():
            return await validator.validate("machine learning algorithms", results)

        result = asyncio.run(run())

        assert result.quality in [RetrievalQuality.INCORRECT, RetrievalQuality.AMBIGUOUS]

    def test_crag_result_dataclass(self) -> None:
        result = CRAGResult(
            quality=RetrievalQuality.CORRECT,
            confidence=0.9,
            relevant_results=[],
            needs_refinement=False,
            suggested_queries=[],
        )
        assert result.quality == RetrievalQuality.CORRECT
        assert result.confidence == 0.9

    def test_retrieval_quality_enum(self) -> None:
        assert RetrievalQuality.CORRECT.value == "correct"
        assert RetrievalQuality.INCORRECT.value == "incorrect"
        assert RetrievalQuality.AMBIGUOUS.value == "ambiguous"

    def test_validate_heuristic_ambiguous(self) -> None:
        validator = CRAGValidator(relevance_threshold=0.5)
        results = [
            SearchResult(
                file_path="/doc1.md",
                content="This is about machine learning algorithms",
                score=0.8,
            ),
            SearchResult(
                file_path="/doc2.md",
                content="Weather forecast for today is sunny",
                score=0.1,
            ),
            SearchResult(
                file_path="/doc3.md",
                content="Cooking recipes with pasta",
                score=0.1,
            ),
        ]

        async def run():
            return await validator.validate("machine learning", results)

        result = asyncio.run(run())

        assert result.quality == RetrievalQuality.AMBIGUOUS
        assert result.confidence == 0.5
        assert result.needs_refinement is True

    @pytest.mark.asyncio
    async def test_validate_with_llm(self) -> None:
        mock_adapter = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"quality": "correct", "confidence": 0.95, "relevant_indices": [0, 1], "reasoning": "Both relevant", "suggested_queries": []}'
        mock_adapter.chat = AsyncMock(return_value=mock_response)

        validator = CRAGValidator(adapter=mock_adapter)
        results = [
            SearchResult(source=Source(file_path="/doc1.md"), content="ML content", score=0.9),
            SearchResult(source=Source(file_path="/doc2.md"), content="AI content", score=0.85),
        ]

        result = await validator.validate("machine learning", results)

        assert result.quality == RetrievalQuality.CORRECT
        assert result.confidence == 0.95
        assert len(result.relevant_results) == 2
        assert result.needs_refinement is False
        mock_adapter.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_ignores_range(self) -> None:
        mock_adapter = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"quality": "correct", "confidence": 0.9, "relevant_indices": [0, 9], "reasoning": "Only first result exists", "suggested_queries": []}'
        mock_adapter.chat = AsyncMock(return_value=mock_response)

        validator = CRAGValidator(adapter=mock_adapter)
        results = [
            SearchResult(source=Source(file_path="/doc1.md"), content="ML content", score=0.9),
        ]

        result = await validator.validate("machine learning", results)

        assert result.quality == RetrievalQuality.CORRECT
        assert len(result.relevant_results) == 1
        assert result.relevant_results[0].file_path == "/doc1.md"

    @pytest.mark.asyncio
    async def test_validate_fallback_on_error(self) -> None:
        mock_adapter = MagicMock()
        mock_adapter.chat = AsyncMock(side_effect=Exception("API error"))

        validator = CRAGValidator(adapter=mock_adapter)
        results = [
            SearchResult(
                source=Source(file_path="/doc1.md"), content="machine learning", score=0.9
            ),
        ]

        result = await validator.validate("machine learning", results)

        assert result.quality in [RetrievalQuality.CORRECT, RetrievalQuality.AMBIGUOUS]
        assert "Heuristic" in result.reasoning

    @pytest.mark.asyncio
    async def test_refine_query_with_llm(self) -> None:
        mock_adapter = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            '{"queries": ["what is ML", "ML algorithms", "machine learning basics"]}'
        )
        mock_adapter.chat = AsyncMock(return_value=mock_response)

        validator = CRAGValidator(adapter=mock_adapter)
        queries = await validator.refine_query("ML", [])

        assert len(queries) == 3
        assert "what is ML" in queries

    @pytest.mark.asyncio
    async def test_refine_query_fallback(self) -> None:
        validator = CRAGValidator()
        queries = await validator.refine_query("test query", [])

        assert len(queries) == 3
        assert "test query" in queries
        assert "what is test query" in queries

    @pytest.mark.asyncio
    async def test_refine_on_error_fallback(self) -> None:
        mock_adapter = MagicMock()
        mock_adapter.chat = AsyncMock(side_effect=Exception("API error"))

        validator = CRAGValidator(adapter=mock_adapter)
        queries = await validator.refine_query("test", [])

        assert len(queries) == 3
        assert "test" in queries
