"""Tests for LLMReranker."""

from __future__ import annotations

import pytest

from houyi.rag.llm import LLMReranker
from houyi.rag.types import SearchResult

from ._fakes import FakeAdapter


class TestLLMReranker:
    @pytest.mark.asyncio
    async def test_rerank_success(self) -> None:
        adapter = FakeAdapter(['{"scores": [9, 6, 3]}'])
        reranker = LLMReranker(adapter)

        results = [
            SearchResult(chunk_id="c1", content="Low relevance content", score=0.3),
            SearchResult(chunk_id="c2", content="Medium relevance content", score=0.5),
            SearchResult(chunk_id="c3", content="High relevance content", score=0.9),
        ]

        reranked = await reranker.rerank("query", results, top_k=3)

        assert len(reranked) == 3
        assert reranked[0].score == 0.9
        assert reranked[0].metadata.get("llm_score") == 9

    @pytest.mark.asyncio
    async def test_rerank_single_result(self) -> None:
        adapter = FakeAdapter(['{"scores": [8]}'])
        reranker = LLMReranker(adapter)

        results = [SearchResult(chunk_id="c1", content="content", score=0.5)]

        reranked = await reranker.rerank("query", results, top_k=1)

        assert len(reranked) == 1
        assert reranked[0].chunk_id == "c1"

    @pytest.mark.asyncio
    async def test_rerank_invalid_json_fallback(self) -> None:
        adapter = FakeAdapter(["invalid json"])
        reranker = LLMReranker(adapter)

        results = [
            SearchResult(chunk_id="c1", content="content", score=0.5),
            SearchResult(chunk_id="c2", content="content", score=0.6),
        ]

        reranked = await reranker.rerank("query", results, top_k=2)

        assert len(reranked) == 2

    @pytest.mark.asyncio
    async def test_rerank_embedded_json_response(self) -> None:
        adapter = FakeAdapter(['Scores:\n```json\n{"scores": [9, 8, 1]}\n```'])
        reranker = LLMReranker(adapter)

        results = [
            SearchResult(chunk_id="c1", content="Low relevance content", score=0.3),
            SearchResult(chunk_id="c2", content="Medium relevance content", score=0.5),
            SearchResult(chunk_id="c3", content="High relevance content", score=0.9),
        ]

        reranked = await reranker.rerank("query", results, top_k=3)

        assert len(reranked) == 3
        assert reranked[0].metadata.get("llm_score") == 9
        assert reranked[-1].metadata.get("llm_score") == 1
