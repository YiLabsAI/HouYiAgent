from __future__ import annotations

from collections.abc import Iterator

import pytest

from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.backends.sqlite_entity_state import SQLiteEntityStateView
from houyi.adapters.memory.recall.retrievers.graph import GraphRetriever
from houyi.adapters.memory.recall.types import (
    RecallQuery,
    RetrieverContext,
    RetrieverKind,
)
from houyi.adapters.memory.types import (
    Certainty,
    MemoryEdge,
    MemoryRecord,
    MemoryRelation,
    MemoryScope,
)


@pytest.fixture
def test_env(tmp_path) -> Iterator[tuple[SQLiteMemoryBackend, SQLiteEntityStateView]]:
    """Fresh SQLiteMemoryBackend and SQLiteEntityStateView for each test."""
    backend = SQLiteMemoryBackend(db_path=tmp_path / "memory.db")
    view = SQLiteEntityStateView(backend)
    try:
        yield backend, view
    finally:
        backend.close()


@pytest.mark.asyncio
async def test_graph_retriever_hops(test_env) -> None:
    backend, view = test_env

    # 1. Insert Entity State Records
    # Caroline is active in education
    state_caroline_job = view.upsert(
        namespace="n1",
        entity="Caroline",
        attribute="job",
        value="Teacher",
        certainty=Certainty.CERTAIN,
        valid_from=100.0,
    )
    # Nate is active
    state_nate_hobby = view.upsert(
        namespace="n1",
        entity="Nate",
        attribute="hobby",
        value="Hiking",
        certainty=Certainty.CERTAIN,
        valid_from=100.0,
    )
    # Joanna is active
    state_joanna_friend = view.upsert(
        namespace="n1",
        entity="Joanna",
        attribute="knows",
        value="Nate",
        certainty=Certainty.CERTAIN,
        valid_from=100.0,
    )

    # 2. Insert Facts into memories FTS
    record_bobby = MemoryRecord(
        record_id="rec_bobby",
        scope=MemoryScope.USER,
        key="Bobby",
        content="Bobby likes hiking very much",
    )
    backend.put(record_bobby)

    # 3. Create Edges
    # Caroline knows Joanna (state_caroline_job -> state_joanna_friend)
    edge_1 = MemoryEdge(
        edge_id="edge-1",
        namespace="n1",
        source_unit_id=state_caroline_job.state_id,
        target_unit_id=state_joanna_friend.state_id,
        source_type="state",
        target_type="state",
        relation=MemoryRelation.RELATED_TO,
        valid_from=100.0,
    )
    # Joanna knows Bobby (state_joanna_friend -> rec_bobby)
    edge_2 = MemoryEdge(
        edge_id="edge-2",
        namespace="n1",
        source_unit_id=state_joanna_friend.state_id,
        target_unit_id="rec_bobby",
        source_type="state",
        target_type="fact",
        relation=MemoryRelation.RELATED_TO,
        valid_from=100.0,
    )
    # Bobby is Nate's hiking partner (rec_bobby -> state_nate_hobby)
    edge_3 = MemoryEdge(
        edge_id="edge-3",
        namespace="n1",
        source_unit_id="rec_bobby",
        target_unit_id=state_nate_hobby.state_id,
        source_type="fact",
        target_type="state",
        relation=MemoryRelation.RELATED_TO,
        valid_from=100.0,
    )

    backend.add_edge(edge_1)
    backend.add_edge(edge_2)
    backend.add_edge(edge_3)

    # 4. Run GraphRetriever
    retriever = GraphRetriever(backend, view)

    # Caroline is seed. She connects: Caroline --[1]--> Joanna --[2]--> Bobby --[3]--> Nate
    hits = await retriever.retrieve(
        RecallQuery(text="Who is Caroline?", namespace="n1", as_of=150.0),
        RetrieverContext(),
    )

    # Assert results and depth propagation
    assert len(hits) > 0

    # Ensure all matched candidates are tagged with MATCHED_BY == GRAPH
    for hit in hits:
        assert hit.matched_by == RetrieverKind.GRAPH
        assert hit.signals["bfs_depth"] in {0, 1, 2, 3}

    # Verify we traversed and fetched connected nodes (neighbors of Caroline)
    nodes_found = {hit.fact.subject for hit in hits}
    assert "Joanna" in nodes_found  # Hop 1
    assert "Bobby" in nodes_found  # Hop 2
    assert "Nate" in nodes_found  # Hop 3

    # Verify scoring decays by depth
    joanna_candidates = [h for h in hits if h.fact.subject == "Joanna"]
    nate_candidates = [h for h in hits if h.fact.subject == "Nate"]

    assert len(joanna_candidates) == 1
    assert len(nate_candidates) == 1

    # Joanna is depth 1, Nate is depth 3. Joanna should score higher.
    assert joanna_candidates[0].score > nate_candidates[0].score
    assert joanna_candidates[0].signals["bfs_depth"] == 1
    assert nate_candidates[0].signals["bfs_depth"] == 3


@pytest.mark.asyncio
async def test_graph_temporal(test_env) -> None:
    backend, view = test_env

    # Insert state for Nate
    state_nate_hobby = view.upsert(
        namespace="n1",
        entity="Nate",
        attribute="hobby",
        value="Hiking",
        certainty=Certainty.CERTAIN,
        valid_from=100.0,
    )
    # Insert state for Joanna
    state_joanna_friend = view.upsert(
        namespace="n1",
        entity="Joanna",
        attribute="knows",
        value="Nate",
        certainty=Certainty.CERTAIN,
        valid_from=100.0,
    )

    # Edge active from 100.0 to 200.0 only
    edge = MemoryEdge(
        edge_id="edge-1",
        namespace="n1",
        source_unit_id=state_joanna_friend.state_id,
        target_unit_id=state_nate_hobby.state_id,
        source_type="state",
        target_type="state",
        relation=MemoryRelation.RELATED_TO,
        valid_from=100.0,
        valid_to=200.0,
    )
    backend.add_edge(edge)

    retriever = GraphRetriever(backend, view)

    # Scenario A: Query at as_of=150.0 (active)
    hits_active = await retriever.retrieve(
        RecallQuery(text="Who does Joanna know?", namespace="n1", as_of=150.0),
        RetrieverContext(),
    )
    assert len(hits_active) == 1
    subjects_active = {h.fact.subject for h in hits_active}
    assert "Nate" in subjects_active

    # Scenario B: Query at as_of=250.0 (expired edge)
    hits_expired = await retriever.retrieve(
        RecallQuery(text="Who does Joanna know?", namespace="n1", as_of=250.0),
        RetrieverContext(),
    )
    # Should not traverse over the expired edge to Nate
    subjects_expired = {h.fact.subject for h in hits_expired}
    assert "Nate" not in subjects_expired
