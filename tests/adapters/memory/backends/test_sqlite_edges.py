from __future__ import annotations

from collections.abc import Iterator

import pytest

from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.types import MemoryEdge, MemoryRelation


@pytest.fixture
def backend(tmp_path) -> Iterator[SQLiteMemoryBackend]:
    """Fresh SQLiteMemoryBackend for each test."""
    db = SQLiteMemoryBackend(db_path=tmp_path / "memory.db")
    try:
        yield db
    finally:
        db.close()


class TestSQLiteMemoryEdges:
    """Test suite for memory_edges and community labels CRUD operations."""

    def test_edge_round_trip(self, backend: SQLiteMemoryBackend) -> None:
        edge = MemoryEdge(
            edge_id="edge-1",
            namespace="ws",
            source_unit_id="node-a",
            target_unit_id="node-b",
            source_type="state",
            target_type="fact",
            relation=MemoryRelation.CAUSES,
            weight=1.5,
            valid_from=100.0,
            valid_to=None,
            created_at=100.0,
            provenance="test",
        )
        backend.add_edge(edge)

        retrieved = backend.get_edge("edge-1")
        assert retrieved is not None
        assert retrieved.edge_id == "edge-1"
        assert retrieved.namespace == "ws"
        assert retrieved.source_unit_id == "node-a"
        assert retrieved.target_unit_id == "node-b"
        assert retrieved.source_type == "state"
        assert retrieved.target_type == "fact"
        assert retrieved.relation == MemoryRelation.CAUSES
        assert retrieved.weight == 1.5
        assert retrieved.valid_from == 100.0
        assert retrieved.valid_to is None
        assert retrieved.provenance == "test"

    def test_edge_upsert_on_conflict(self, backend: SQLiteMemoryBackend) -> None:
        edge = MemoryEdge(
            edge_id="edge-1",
            namespace="ws",
            source_unit_id="node-a",
            target_unit_id="node-b",
            source_type="state",
            target_type="fact",
            relation=MemoryRelation.CAUSES,
            weight=1.0,
            valid_from=100.0,
            valid_to=None,
        )
        backend.add_edge(edge)

        # Upsert with same ID to update weight and validity
        updated_edge = MemoryEdge(
            edge_id="edge-1",
            namespace="ws",
            source_unit_id="node-a",
            target_unit_id="node-b",
            source_type="state",
            target_type="fact",
            relation=MemoryRelation.CAUSES,
            weight=2.5,
            valid_from=150.0,
            valid_to=200.0,
        )
        backend.add_edge(updated_edge)

        retrieved = backend.get_edge("edge-1")
        assert retrieved is not None
        assert retrieved.weight == 2.5
        assert retrieved.valid_from == 150.0
        assert retrieved.valid_to == 200.0

    def test_delete_edge(self, backend: SQLiteMemoryBackend) -> None:
        edge = MemoryEdge(
            edge_id="edge-1",
            namespace="ws",
            source_unit_id="node-a",
            target_unit_id="node-b",
            source_type="state",
            target_type="fact",
            relation=MemoryRelation.CAUSES,
        )
        backend.add_edge(edge)
        assert backend.delete_edge("edge-1") is True
        assert backend.get_edge("edge-1") is None
        assert backend.delete_edge("edge-1") is False

    def test_invalidate_edge(self, backend: SQLiteMemoryBackend) -> None:
        edge = MemoryEdge(
            edge_id="edge-1",
            namespace="ws",
            source_unit_id="node-a",
            target_unit_id="node-b",
            source_type="state",
            target_type="fact",
            relation=MemoryRelation.CAUSES,
            valid_to=None,
        )
        backend.add_edge(edge)

        assert backend.invalidate_edge("edge-1", 150.0) is True
        retrieved = backend.get_edge("edge-1")
        assert retrieved is not None
        assert retrieved.valid_to == 150.0

        # Subsequent invalidate fails because valid_to is already set
        assert backend.invalidate_edge("edge-1", 200.0) is False

    def test_community_labels_round_trip(self, backend: SQLiteMemoryBackend) -> None:
        assert backend.get_community_id("ws", "entity", "Caroline") is None

        backend.put_community_label(
            namespace="ws",
            node_type="entity",
            node_id="Caroline",
            community_id="c_education_3",
            weight=1.2,
            updated_at=100.0,
        )

        assert backend.get_community_id("ws", "entity", "Caroline") == "c_education_3"

        # Update label
        backend.put_community_label(
            namespace="ws",
            node_type="entity",
            node_id="Caroline",
            community_id="c_sports_1",
            weight=0.9,
            updated_at=200.0,
        )
        assert backend.get_community_id("ws", "entity", "Caroline") == "c_sports_1"
