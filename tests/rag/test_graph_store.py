"""Tests for graph store and graph extractor."""

from __future__ import annotations

import tempfile

import pytest

from houyi.rag.config import GraphConfig
from houyi.rag.indexed.graph.extractor import (
    _simple_entity_extraction,
    extract_entities,
)
from houyi.rag.indexed.graph.store import GraphStore
from houyi.rag.types import Chunk, Entity, Relation


class TestGraphStore:
    """Tests for GraphStore."""

    @pytest.mark.asyncio
    async def test_load_creates_tables(self) -> None:
        """Test that load creates SQLite tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(knowledge_dir=tmpdir)
            await store.load()

            assert store._conn is not None
            assert store.count_entities() == 0
            assert store.count_relations() == 0

    @pytest.mark.asyncio
    async def test_add_entities(self) -> None:
        """Test adding entities to graph store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(knowledge_dir=tmpdir)
            await store.load()

            entities = [
                Entity(entity_id="e1", name="Python", entity_type="language"),
                Entity(entity_id="e2", name="Java", entity_type="language"),
            ]
            await store.add_entities(entities)

            assert store.count_entities() == 2
            assert store.get_entity("e1") is not None
            assert store.get_entity("e1").name == "Python"

    @pytest.mark.asyncio
    async def test_add_entities_with_metadata(self) -> None:
        """Test adding entities with metadata and embedding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(knowledge_dir=tmpdir)
            await store.load()

            entities = [
                Entity(
                    entity_id="e1",
                    name="Python",
                    entity_type="language",
                    embedding=[0.1, 0.2, 0.3],
                    metadata={"version": "3.13"},
                ),
            ]
            await store.add_entities(entities)

            entity = store.get_entity("e1")
            assert entity is not None
            assert entity.metadata == {"version": "3.13"}
            assert entity.embedding == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_add_relations(self) -> None:
        """Test adding relations to graph store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(knowledge_dir=tmpdir)
            await store.load()

            entities = [
                Entity(entity_id="e1", name="Python", entity_type="language"),
                Entity(entity_id="e2", name="Django", entity_type="framework"),
            ]
            await store.add_entities(entities)

            relations = [
                Relation(
                    rel_id="r1",
                    source_id="e1",
                    target_id="e2",
                    rel_type="has_framework",
                    weight=1.0,
                ),
            ]
            await store.add_relations(relations)

            assert store.count_relations() == 1
            neighbors = store.get_neighbors("e1")
            assert len(neighbors) == 1
            assert neighbors[0][0] == "e2"
            assert neighbors[0][1] == "has_framework"

    @pytest.mark.asyncio
    async def test_get_entity_not_found(self) -> None:
        """Test getting non-existent entity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(knowledge_dir=tmpdir)
            await store.load()

            assert store.get_entity("nonexistent") is None

    @pytest.mark.asyncio
    async def test_get_neighbors_empty(self) -> None:
        """Test getting neighbors for entity with no connections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(knowledge_dir=tmpdir)
            await store.load()

            assert store.get_neighbors("nonexistent") == []

    @pytest.mark.asyncio
    async def test_ppr_search(self) -> None:
        """Test PPR-based graph search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(knowledge_dir=tmpdir)
            await store.load()

            # Build a small graph
            entities = [
                Entity(entity_id="e1", name="Machine Learning", entity_type="concept"),
                Entity(entity_id="e2", name="Deep Learning", entity_type="concept"),
                Entity(entity_id="e3", name="Neural Networks", entity_type="concept"),
                Entity(entity_id="e4", name="Weather", entity_type="topic"),
            ]
            await store.add_entities(entities)

            relations = [
                Relation(rel_id="r1", source_id="e1", target_id="e2", rel_type="contains", weight=1.0),
                Relation(rel_id="r2", source_id="e2", target_id="e3", rel_type="uses", weight=1.0),
            ]
            await store.add_relations(relations)

            # Search for "Machine Learning"
            results = await store.search("Machine Learning", k=5)

            assert len(results) > 0
            # Machine Learning should be in results
            result_names = [r.content for r in results]
            assert "Machine Learning" in result_names

    @pytest.mark.asyncio
    async def test_search_no_match(self) -> None:
        """Test search with no matching entities."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(knowledge_dir=tmpdir)
            await store.load()

            entities = [
                Entity(entity_id="e1", name="Python", entity_type="language"),
            ]
            await store.add_entities(entities)

            results = await store.search("nonexistent query xyz", k=5)
            assert results == []

    @pytest.mark.asyncio
    async def test_search_empty_graph(self) -> None:
        """Test search on empty graph."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(knowledge_dir=tmpdir)
            await store.load()

            results = await store.search("test query", k=5)
            assert results == []

    @pytest.mark.asyncio
    async def test_bfs_traverse(self) -> None:
        """Test BFS traversal from a starting entity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(knowledge_dir=tmpdir)
            await store.load()

            entities = [
                Entity(entity_id="e1", name="Root", entity_type="node"),
                Entity(entity_id="e2", name="Child1", entity_type="node"),
                Entity(entity_id="e3", name="Child2", entity_type="node"),
                Entity(entity_id="e4", name="Grandchild", entity_type="node"),
            ]
            await store.add_entities(entities)

            relations = [
                Relation(rel_id="r1", source_id="e1", target_id="e2", rel_type="link", weight=1.0),
                Relation(rel_id="r2", source_id="e1", target_id="e3", rel_type="link", weight=1.0),
                Relation(rel_id="r3", source_id="e2", target_id="e4", rel_type="link", weight=1.0),
            ]
            await store.add_relations(relations)

            result = await store.bfs_traverse("e1", max_depth=2)

            assert len(result) >= 1
            entity_ids = [eid for eid, _ in result]
            assert "e1" in entity_ids
            assert "e2" in entity_ids

    @pytest.mark.asyncio
    async def test_bfs_traverse_no_connection(self) -> None:
        """Test BFS on disconnected store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(knowledge_dir=tmpdir)
            await store.load()

            result = await store.bfs_traverse("nonexistent")
            # Should still return the start node
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_persistence(self) -> None:
        """Test saving and reloading graph store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate store
            store1 = GraphStore(knowledge_dir=tmpdir)
            await store1.load()

            entities = [
                Entity(entity_id="e1", name="Persistent Entity", entity_type="test"),
            ]
            relations = [
                Relation(rel_id="r1", source_id="e1", target_id="e1",
                         rel_type="self_ref", weight=0.5),
            ]
            await store1.add_entities(entities)
            await store1.add_relations(relations)
            await store1.save()

            # Load in new instance
            store2 = GraphStore(knowledge_dir=tmpdir)
            await store2.load()

            assert store2.count_entities() == 1
            assert store2.count_relations() == 1
            entity = store2.get_entity("e1")
            assert entity is not None
            assert entity.name == "Persistent Entity"

    @pytest.mark.asyncio
    async def test_add_entities_without_load(self) -> None:
        """Test adding entities triggers auto-load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(knowledge_dir=tmpdir)
            # Don't call load() explicitly
            entities = [Entity(entity_id="e1", name="Test", entity_type="test")]
            await store.add_entities(entities)
            assert store.count_entities() == 1

    @pytest.mark.asyncio
    async def test_add_relations_without_load(self) -> None:
        """Test adding relations triggers auto-load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(knowledge_dir=tmpdir)
            entities = [
                Entity(entity_id="e1", name="A", entity_type="test"),
                Entity(entity_id="e2", name="B", entity_type="test"),
            ]
            await store.add_entities(entities)
            relations = [
                Relation(rel_id="r1", source_id="e1", target_id="e2",
                         rel_type="link", weight=1.0),
            ]
            await store.add_relations(relations)
            assert store.count_relations() == 1

    @pytest.mark.asyncio
    async def test_custom_config(self) -> None:
        """Test graph store with custom config."""
        config = GraphConfig(enabled=True, ppr_alpha=0.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(knowledge_dir=tmpdir, config=config)
            await store.load()
            assert store.config.ppr_alpha == 0.5


class TestGraphExtractor:
    """Tests for entity extraction."""

    @pytest.mark.asyncio
    async def test_simple_entity_extraction(self) -> None:
        """Test simple rule-based entity extraction."""
        text = "Apple Inc released the new MacBook Pro in January."
        entities = _simple_entity_extraction(text)

        assert len(entities) > 0
        entity_names = [name for name, _ in entities]
        assert any("Apple" in name for name in entity_names)

    @pytest.mark.asyncio
    async def test_simple_entity_extraction_empty(self) -> None:
        """Test extraction from text with no entities."""
        text = "this is all lowercase text with no proper nouns"
        entities = _simple_entity_extraction(text)
        assert len(entities) == 0

    @pytest.mark.asyncio
    async def test_simple_entity_type_classification(self) -> None:
        """Test entity type classification."""
        text = "January February March are months."
        entities = _simple_entity_extraction(text)
        # Months should be classified as date
        for name, etype in entities:
            if "January" in name or "February" in name or "March" in name:
                assert etype == "date"

    @pytest.mark.asyncio
    async def test_extract_entities_from_chunks(self) -> None:
        """Test extracting entities from chunks."""
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
        # Should have relations for co-occurring entities
        entity_names = [e.name for e in entities]
        assert any("Google" in name for name in entity_names)

    @pytest.mark.asyncio
    async def test_extract_entities_deduplication(self) -> None:
        """Test that entities are deduplicated across chunks."""
        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", content="Google Cloud Platform is great."),
            Chunk(chunk_id="c2", doc_id="d1", content="Google Cloud Platform services."),
        ]

        entities, _ = await extract_entities(chunks)

        # "Google Cloud Platform" should appear only once
        names = [e.name for e in entities]
        assert len(names) == len(set(names))

    @pytest.mark.asyncio
    async def test_extract_entities_relations(self) -> None:
        """Test co-occurrence relation extraction."""
        chunks = [
            Chunk(
                chunk_id="c1",
                doc_id="d1",
                content="Microsoft Azure and Google Cloud are cloud platforms.",
            ),
        ]

        entities, relations = await extract_entities(chunks)

        if len(entities) >= 2:
            # Should have co-occurrence relations
            assert len(relations) > 0
            assert all(r.rel_type == "co_occurs" for r in relations)
