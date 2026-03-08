"""Tests for KeywordExtractor."""

from __future__ import annotations

import pytest

from houyi.rag.llm import KeywordExtractor

from ._fakes import FakeAdapter


class TestKeywordExtractor:
    @pytest.mark.asyncio
    async def test_extract_success(self) -> None:
        adapter = FakeAdapter(
            [
                '{"keywords": ["RAG", "retrieval", "generation"], "synonyms": {"RAG": ["Retrieval-Augmented Generation"]}}'
            ]
        )
        extractor = KeywordExtractor(adapter)

        result = await extractor.extract("What is RAG?")

        assert "keywords" in result
        assert "RAG" in result["keywords"]
        assert len(result["keywords"]) == 3
        assert "synonyms" in result

    @pytest.mark.asyncio
    async def test_extract_fallback_on_error(self) -> None:
        adapter = FakeAdapter(["invalid json"])
        extractor = KeywordExtractor(adapter)

        result = await extractor.extract("What is RAG system?")

        assert "keywords" in result
        assert isinstance(result["keywords"], list)
        assert "RAG" in result["keywords"]

    @pytest.mark.asyncio
    async def test_extract_embedded_json_response(self) -> None:
        adapter = FakeAdapter(
            [
                'Here is the result:\n```json\n{"keywords": ["RAG", "retrieval", "generation"], "synonyms": {"RAG": ["Retrieval-Augmented Generation"]}}\n```'
            ]
        )
        extractor = KeywordExtractor(adapter)

        result = await extractor.extract("What is RAG?")

        assert result["keywords"] == ["RAG", "retrieval", "generation"]
        assert result["synonyms"]["RAG"] == ["Retrieval-Augmented Generation"]

    @pytest.mark.asyncio
    async def test_expand_keywords(self) -> None:
        adapter = FakeAdapter(['["Retrieval-Augmented Generation", "vector search", "embedding"]'])
        extractor = KeywordExtractor(adapter)

        result = await extractor.expand(["RAG"], context="AI knowledge base")

        assert "RAG" in result
        assert len(result) > 1

    @pytest.mark.asyncio
    async def test_expand_keywords_embedded_array_response(self) -> None:
        adapter = FakeAdapter(['Suggested terms: ["vector search", "embeddings"]'])
        extractor = KeywordExtractor(adapter)

        result = await extractor.expand(["RAG"], context="AI knowledge base")

        assert "RAG" in result
        assert "vector search" in result
