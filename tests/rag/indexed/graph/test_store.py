"""Tests for GraphStore."""

from __future__ import annotations

import tempfile

import pytest
import pytest_asyncio

from houyi.rag.config import GraphConfig
from houyi.rag.indexed.graph.store import GraphStore
from houyi.rag.types import Entity, Relation


@pytest_asyncio.fixture
async def graph_store(tmp_path):
    store = GraphStore(knowledge_dir=str(tmp_path))
    await store.load()
    try:
        yield store
    finally:
        store.close()


class TestGraphStore:
    @pytest.mark.asyncio
    async def test_load_creates_tables(self, graph_store) -> None:
        assert graph_store._conn is not None
        assert graph_store.count_entities() == 0
        assert graph_store.count_relations() == 0

    @pytest.mark.asyncio
    async def test_add_entities(self, graph_store) -> None:
        entities = [
            Entity(entity_id="e1", name="Python", entity_type="language"),
            Entity(entity_id="e2", name="Java", entity_type="language"),
        ]
        await graph_store.add_entities(entities)

        assert graph_store.count_entities() == 2
        assert graph_store.get_entity("e1") is not None
        assert graph_store.get_entity("e1").name == "Python"

    @pytest.mark.asyncio
    async def test_add_entities_with_metadata(self, graph_store) -> None:
        entities = [
            Entity(
                entity_id="e1",
                name="Python",
                entity_type="language",
                embedding=[0.1, 0.2, 0.3],
                metadata={"version": "3.13"},
            ),
        ]
        await graph_store.add_entities(entities)

        entity = graph_store.get_entity("e1")
        assert entity is not None
        assert entity.metadata == {"version": "3.13"}
        assert entity.embedding == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_add_relations(self, graph_store) -> None:
        entities = [
            Entity(entity_id="e1", name="Python", entity_type="language"),
            Entity(entity_id="e2", name="Django", entity_type="framework"),
        ]
        await graph_store.add_entities(entities)

        relations = [
            Relation(
                rel_id="r1",
                source_id="e1",
                target_id="e2",
                rel_type="has_framework",
                weight=1.0,
            ),
        ]
        await graph_store.add_relations(relations)

        assert graph_store.count_relations() == 1
        neighbors = graph_store.get_neighbors("e1")
        assert len(neighbors) == 1
        assert neighbors[0][0] == "e2"
        assert neighbors[0][1] == "has_framework"

    @pytest.mark.asyncio
    async def test_get_entity_not_found(self, graph_store) -> None:
        assert graph_store.get_entity("nonexistent") is None

    @pytest.mark.asyncio
    async def test_get_neighbors_empty(self, graph_store) -> None:
        assert graph_store.get_neighbors("nonexistent") == []

    @pytest.mark.asyncio
    async def test_ppr_search(self, graph_store) -> None:
        entities = [
            Entity(entity_id="e1", name="Machine Learning", entity_type="concept"),
            Entity(entity_id="e2", name="Deep Learning", entity_type="concept"),
            Entity(entity_id="e3", name="Neural Networks", entity_type="concept"),
            Entity(entity_id="e4", name="Weather", entity_type="topic"),
        ]
        await graph_store.add_entities(entities)

        relations = [
            Relation(rel_id="r1", source_id="e1", target_id="e2", rel_type="contains", weight=1.0),
            Relation(rel_id="r2", source_id="e2", target_id="e3", rel_type="uses", weight=1.0),
        ]
        await graph_store.add_relations(relations)

        results = await graph_store.search("Machine Learning", k=5)

        assert len(results) > 0
        result_names = [r.content for r in results]
        assert "Machine Learning" in result_names

    @pytest.mark.asyncio
    async def test_search_no_match(self, graph_store) -> None:
        await graph_store.add_entities(
            [Entity(entity_id="e1", name="Python", entity_type="language")]
        )

        results = await graph_store.search("nonexistent query xyz", k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_empty_graph(self, graph_store) -> None:
        results = await graph_store.search("test query", k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_bfs_traverse(self, graph_store) -> None:
        entities = [
            Entity(entity_id="e1", name="Root", entity_type="node"),
            Entity(entity_id="e2", name="Child1", entity_type="node"),
            Entity(entity_id="e3", name="Child2", entity_type="node"),
            Entity(entity_id="e4", name="Grandchild", entity_type="node"),
        ]
        await graph_store.add_entities(entities)

        relations = [
            Relation(rel_id="r1", source_id="e1", target_id="e2", rel_type="link", weight=1.0),
            Relation(rel_id="r2", source_id="e1", target_id="e3", rel_type="link", weight=1.0),
            Relation(rel_id="r3", source_id="e2", target_id="e4", rel_type="link", weight=1.0),
        ]
        await graph_store.add_relations(relations)

        result = await graph_store.bfs_traverse("e1", max_depth=2)

        assert len(result) >= 1
        entity_ids = [eid for eid, _ in result]
        assert "e1" in entity_ids
        assert "e2" in entity_ids

    @pytest.mark.asyncio
    async def test_bfs_traverse_no_connection(self, graph_store) -> None:
        result = await graph_store.bfs_traverse("nonexistent")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            async with GraphStore(knowledge_dir=tmpdir) as store1:
                entities = [Entity(entity_id="e1", name="Persistent Entity", entity_type="test")]
                relations = [
                    Relation(
                        rel_id="r1",
                        source_id="e1",
                        target_id="e1",
                        rel_type="self_ref",
                        weight=0.5,
                    ),
                ]
                await store1.add_entities(entities)
                await store1.add_relations(relations)
                await store1.save()

            async with GraphStore(knowledge_dir=tmpdir) as store2:
                assert store2.count_entities() == 1
                assert store2.count_relations() == 1
                entity = store2.get_entity("e1")
                assert entity is not None
                assert entity.name == "Persistent Entity"

    @pytest.mark.asyncio
    async def test_add_entities_without_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            async with GraphStore(knowledge_dir=tmpdir) as store:
                entities = [Entity(entity_id="e1", name="Test", entity_type="test")]
                await store.add_entities(entities)
                assert store.count_entities() == 1

    @pytest.mark.asyncio
    async def test_add_relations_without_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            async with GraphStore(knowledge_dir=tmpdir) as store:
                entities = [
                    Entity(entity_id="e1", name="A", entity_type="test"),
                    Entity(entity_id="e2", name="B", entity_type="test"),
                ]
                await store.add_entities(entities)
                relations = [
                    Relation(
                        rel_id="r1",
                        source_id="e1",
                        target_id="e2",
                        rel_type="link",
                        weight=1.0,
                    ),
                ]
                await store.add_relations(relations)
                assert store.count_relations() == 1

    @pytest.mark.asyncio
    async def test_custom_config(self) -> None:
        config = GraphConfig(enabled=True, ppr_alpha=0.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            async with GraphStore(knowledge_dir=tmpdir, config=config) as store:
                assert store.config.ppr_alpha == 0.5
