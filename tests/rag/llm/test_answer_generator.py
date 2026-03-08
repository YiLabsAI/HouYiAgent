"""Tests for AnswerGenerator."""

from __future__ import annotations

import pytest

from houyi.rag.llm import AnswerGenerator
from houyi.rag.types import SearchResult, Source

from ._fakes import FakeAdapter


class TestAnswerGenerator:
    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        adapter = FakeAdapter(
            [
                "RAG (Retrieval-Augmented Generation) combines retrieval with generation. [1] It uses vector search to find relevant documents. [2]\n\nSources:\n[1] document.md\n[2] overview.md"
            ]
        )
        generator = AnswerGenerator(adapter)

        results = [
            SearchResult(
                chunk_id="chunk1",
                content="RAG is a technique for combining retrieval and generation.",
                score=0.9,
                source=Source(file_path="document.md"),
            ),
            SearchResult(
                chunk_id="chunk2",
                content="Vector search is used to find relevant documents.",
                score=0.8,
                source=Source(file_path="overview.md"),
            ),
        ]

        answer, confidence = await generator.generate("What is RAG?", results)

        assert "RAG" in answer
        assert "[1]" in answer
        assert confidence > 0.5

    @pytest.mark.asyncio
    async def test_generate_empty_results(self) -> None:
        adapter = FakeAdapter(["Some answer"])
        generator = AnswerGenerator(adapter)

        answer, confidence = await generator.generate("What is RAG?", [])

        assert "No relevant" in answer or "not found" in answer.lower()
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_confidence_estimation(self) -> None:
        adapter = FakeAdapter(["The answer is clear [1][2][3] based on the documents."])
        generator = AnswerGenerator(adapter)

        results = [
            SearchResult(chunk_id="1", content="content", score=0.9),
            SearchResult(chunk_id="2", content="content", score=0.85),
            SearchResult(chunk_id="3", content="content", score=0.8),
        ]

        _, confidence = await generator.generate("query", results)

        assert confidence >= 0.7
