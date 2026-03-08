"""Tests for indexed graph extractor."""

from __future__ import annotations

import pytest

from houyi.rag.indexed.graph.extractor import (
    _simple_entity_extraction,
    extract_entities,
)
from houyi.rag.types import Chunk


class TestGraphExtractor:
    @pytest.mark.asyncio
    async def test_simple_entity_extraction(self) -> None:
        text = "Apple Inc released the new MacBook Pro in January."
        entities = _simple_entity_extraction(text)

        assert len(entities) > 0
        entity_names = [name for name, _ in entities]
        assert any("Apple" in name for name in entity_names)

    @pytest.mark.asyncio
    async def test_simple_entity_extraction_empty(self) -> None:
        text = "this is all lowercase text with no proper nouns"
        entities = _simple_entity_extraction(text)
        assert len(entities) == 0

    @pytest.mark.asyncio
    async def test_simple_entity_type_classification(self) -> None:
        text = "January February March are months."
        entities = _simple_entity_extraction(text)
        for name, etype in entities:
            if "January" in name or "February" in name or "March" in name:
                assert etype == "date"

    @pytest.mark.asyncio
    async def test_extract_entities_from_chunks(self) -> None:
        chunks = [
            Chunk(
                chunk_id="c1",
                doc_id="d1",
                content="Google Cloud provides Machine Learning services.",
            ),
            Chunk(
                chunk_id="c2",
                doc_id="d1",
                content="Amazon Web Services competes with Google Cloud.",
            ),
        ]

        entities, relations = await extract_entities(chunks)

        assert len(entities) > 0
        entity_names = [e.name for e in entities]
        assert any("Google" in name for name in entity_names)
        assert relations is not None

    @pytest.mark.asyncio
    async def test_extract_entities_deduplication(self) -> None:
        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", content="Google Cloud Platform is great."),
            Chunk(chunk_id="c2", doc_id="d1", content="Google Cloud Platform services."),
        ]

        entities, _ = await extract_entities(chunks)

        names = [e.name for e in entities]
        assert len(names) == len(set(names))

    @pytest.mark.asyncio
    async def test_extract_entities_relations(self) -> None:
        chunks = [
            Chunk(
                chunk_id="c1",
                doc_id="d1",
                content="Microsoft Azure and Google Cloud are cloud platforms.",
            ),
        ]

        entities, relations = await extract_entities(chunks)

        if len(entities) >= 2:
            assert len(relations) > 0
            assert all(r.rel_type == "co_occurs" for r in relations)
