"""Tests for LLMEntityExtractor."""

from __future__ import annotations

import pytest

from houyi.rag.llm import LLMEntityExtractor
from houyi.rag.types import Chunk

from ._fakes import FakeAdapter


class TestLLMEntityExtractor:
    @pytest.mark.asyncio
    async def test_extract_success(self) -> None:
        adapter = FakeAdapter(
            [
                '{"entities": [{"name": "RAG", "type": "concept", "description": "Retrieval-Augmented Generation"}, {"name": "Vector Search", "type": "technology", "description": "Search using embeddings"}], "relations": [{"source": "RAG", "target": "Vector Search", "type": "uses", "description": "RAG uses vector search for retrieval"}]}'
            ]
        )
        extractor = LLMEntityExtractor(adapter)

        chunk = Chunk(
            chunk_id="test-chunk",
            doc_id="doc1",
            content="RAG uses vector search for retrieval.",
        )

        entities, relations = await extractor.extract(chunk)

        assert len(entities) == 2
        assert entities[0].name == "RAG"
        assert entities[0].entity_type == "concept"
        assert len(relations) == 1
        assert relations[0].rel_type == "uses"

    @pytest.mark.asyncio
    async def test_extract_batch_deduplication(self) -> None:
        adapter = FakeAdapter(
            [
                '{"entities": [{"name": "RAG", "type": "concept", "description": "First"}], "relations": []}',
                '{"entities": [{"name": "RAG", "type": "concept", "description": "Duplicate"}, {"name": "LLM", "type": "technology", "description": "Large Language Model"}], "relations": []}',
            ]
        )
        extractor = LLMEntityExtractor(adapter)

        chunks = [
            Chunk(chunk_id="c1", doc_id="doc1", content="RAG is a technique"),
            Chunk(chunk_id="c2", doc_id="doc1", content="RAG uses LLM"),
        ]

        entities, relations = await extractor.extract_batch(chunks)

        assert len(entities) == 2
        entity_names = [e.name for e in entities]
        assert "RAG" in entity_names
        assert "LLM" in entity_names

    @pytest.mark.asyncio
    async def test_extract_fallback(self) -> None:
        adapter = FakeAdapter(["invalid json"])
        extractor = LLMEntityExtractor(adapter)

        chunk = Chunk(
            chunk_id="test-chunk",
            doc_id="doc1",
            content="The Machine Learning model processes Natural Language",
        )

        entities, relations = await extractor.extract(chunk)

        assert len(entities) >= 1
        entity_names = [entity.name for entity in entities]
        assert any(
            "Machine Learning" in name or "Natural Language" in name for name in entity_names
        )

    @pytest.mark.asyncio
    async def test_extract_embedded_json_response(self) -> None:
        adapter = FakeAdapter(
            [
                'Extraction result:\n```json\n{"entities": [{"name": "OpenAI", "type": "org", "description": "AI research company"}, {"name": "GPT-4", "type": "technology", "description": "Large language model"}], "relations": [{"source": "OpenAI", "target": "GPT-4", "type": "develops", "description": "OpenAI developed GPT-4"}]}\n```'
            ]
        )
        extractor = LLMEntityExtractor(adapter)

        chunk = Chunk(
            chunk_id="test-chunk",
            doc_id="doc1",
            content="OpenAI developed GPT-4.",
        )

        entities, relations = await extractor.extract(chunk)

        assert len(entities) == 2
        assert {entity.name for entity in entities} == {"OpenAI", "GPT-4"}
        assert len(relations) == 1
        assert relations[0].metadata.get("source_name") == "OpenAI"
        assert relations[0].metadata.get("target_name") == "GPT-4"
