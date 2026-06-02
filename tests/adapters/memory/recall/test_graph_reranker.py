from __future__ import annotations

from collections.abc import Iterator

import pytest

from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.backends.sqlite_entity_state import SQLiteEntityStateView
from houyi.adapters.memory.recall.fusion import WeightedFuser
from houyi.adapters.memory.recall.idk_guard import IDKGuard
from houyi.adapters.memory.recall.orchestrator import RecallOrchestrator
from houyi.adapters.memory.recall.rerank import EvidenceAwareReranker
from houyi.adapters.memory.recall.retrievers.entity_state import EntityStateRetriever
from houyi.adapters.memory.recall.retrievers.graph import GraphRetriever
from houyi.adapters.memory.recall.router import QueryRouter, RouteDecision
from houyi.adapters.memory.recall.types import (
    QueryType,
    RecallQuery,
    RetrieverContext,
)
from houyi.adapters.memory.types import (
    Certainty,
    MemoryEdge,
    MemoryRelation,
)


class FixedRouter(QueryRouter):
    def __init__(self, query_type: QueryType) -> None:
        self.query_type = query_type

    async def classify(self, query: RecallQuery) -> RouteDecision:
        return RouteDecision(query_type=self.query_type, confidence=1.0, tier="rule")


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
async def test_graph_cascade(test_env) -> None:
    backend, view = test_env

    # 1. Setup structured factual data
    state_caroline_job = view.upsert(
        namespace="n1",
        entity="Caroline",
        attribute="job",
        value="Teacher",
        certainty=Certainty.CERTAIN,
        valid_from=100.0,
    )
    state_joanna_friend = view.upsert(
        namespace="n1",
        entity="Joanna",
        attribute="knows",
        value="Nate",
        certainty=Certainty.CERTAIN,
        valid_from=100.0,
    )
    state_nate_hobby = view.upsert(
        namespace="n1",
        entity="Nate",
        attribute="hobby",
        value="Hiking",
        certainty=Certainty.CERTAIN,
        valid_from=100.0,
    )

    # 2. Add edges connecting them
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
    edge_2 = MemoryEdge(
        edge_id="edge-2",
        namespace="n1",
        source_unit_id=state_joanna_friend.state_id,
        target_unit_id=state_nate_hobby.state_id,
        source_type="state",
        target_type="state",
        relation=MemoryRelation.RELATED_TO,
        valid_from=100.0,
    )
    backend.add_edge(edge_1)
    backend.add_edge(edge_2)

    # 3. Assemble full Orchestrator
    router = FixedRouter(QueryType.RELATIONAL_CHAIN)
    retrievers = {
        "entity_state": EntityStateRetriever(view),
        "graph": GraphRetriever(backend, view),
    }
    orchestrator = RecallOrchestrator(
        router=router,
        retrievers=retrievers,
        fuser=WeightedFuser(),
        reranker=EvidenceAwareReranker(),
        guard=IDKGuard(),
    )

    # 4. Trigger Relational Chain Query.
    # Caroline connects to Joanna (depth 1) and Joanna connects to Nate (depth 2).
    # This forms a complete chain of depth >= 2, triggering chain completion!
    query = RecallQuery(
        text="Who is Caroline's friends and their hobbies?",
        namespace="n1",
        entity_hint="Caroline",
        top_k=5,
        as_of=150.0,
    )
    # Force route to RELATIONAL_CHAIN for integration testing
    ctx = RetrieverContext()

    # Retrieve all matched and fused candidates
    result = await orchestrator.recall(query, ctx=ctx)

    assert result.is_sufficient() is True
    assert len(result.candidates) > 0

    # Let's inspect the top candidates and their signals
    top_candidates = result.candidates
    joanna_candidates = [c for c in top_candidates if c.fact.subject == "Joanna"]
    nate_candidates = [c for c in top_candidates if c.fact.subject == "Nate"]

    assert len(joanna_candidates) == 1
    assert len(nate_candidates) == 1

    joanna = joanna_candidates[0]
    nate = nate_candidates[0]

    # Verify that the BFS depth signals were preserved through fusion and reached Reranker
    assert joanna.signals["bfs_depth"] == 1
    assert nate.signals["bfs_depth"] == 2

    # Verify that the complete chain bonus was awarded
    # chain_evidence_complete should be True because nate has depth 2 >= 2
    assert joanna.signals["chain_evidence_complete"] is True
    assert nate.signals["chain_evidence_complete"] is True

    # Joanna's coverage should be:
    # source_anchor_bonus (0.2) + graph_path_bonus (0.8 because depth=1 <= 2) + complete_chain_bonus (0.8)
    # clipped to max coverage = 1.0!
    assert joanna.signals["evidence_coverage"] == 1.0

    # Verify that rerank_score is computed correctly:
    # rerank_score = fused_score + evidence_coverage + iteration_bonus
    expected_rerank_joanna = joanna.signals["fused_score"] + 1.0
    assert joanna.signals["rerank_score"] == pytest.approx(expected_rerank_joanna)
