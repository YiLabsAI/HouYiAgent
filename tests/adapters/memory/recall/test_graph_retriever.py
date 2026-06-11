from __future__ import annotations

from collections.abc import Iterator

import pytest

from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.backends.sqlite_entity_state import SQLiteEntityStateView
from houyi.adapters.memory.recall.retrievers.graph import GraphRetriever
from houyi.adapters.memory.recall.types import (
    QueryType,
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


@pytest.mark.asyncio
async def test_graph_seed_discovery(test_env) -> None:
    backend, view = test_env
    retriever = GraphRetriever(backend, view)

    # Verify query with punctuation, possessives, and noise
    # "Sam's" -> "Sam", trailing question mark, and standard lowercase words or month words ignored
    query = RecallQuery(
        text="When did Sam's partner buy the bonsai_tree in January?",
        namespace="n1",
    )
    seeds = await retriever._discover_seeds(query)
    seed_nodes = {s[0] for s in seeds}

    # "Sam" should be extracted as seed node, but "January", "When", "did", "partner", "buy", "the", "bonsai_tree", "in" should be ignored
    assert "Sam" in seed_nodes
    assert "January" not in seed_nodes
    assert "When" not in seed_nodes
    assert "buy" not in seed_nodes


@pytest.mark.asyncio
async def test_graph_shortest_path(test_env) -> None:
    backend, view = test_env

    # Setup states
    state_a = view.upsert(
        namespace="n1",
        entity="A",
        attribute="attr",
        value="val_a",
        certainty=Certainty.CERTAIN,
        valid_from=100.0,
    )
    state_b = view.upsert(
        namespace="n1",
        entity="B",
        attribute="attr",
        value="val_b",
        certainty=Certainty.CERTAIN,
        valid_from=100.0,
    )
    state_c = view.upsert(
        namespace="n1",
        entity="C",
        attribute="attr",
        value="val_c",
        certainty=Certainty.CERTAIN,
        valid_from=100.0,
    )

    # We create paths:
    # A -> B (depth 1)
    # A -> C -> B (depth 2)
    edge_ab = MemoryEdge(
        edge_id="edge_ab",
        namespace="n1",
        source_unit_id=state_a.state_id,
        target_unit_id=state_b.state_id,
        source_type="state",
        target_type="state",
        relation=MemoryRelation.RELATED_TO,
        valid_from=100.0,
    )
    edge_ac = MemoryEdge(
        edge_id="edge_ac",
        namespace="n1",
        source_unit_id=state_a.state_id,
        target_unit_id=state_c.state_id,
        source_type="state",
        target_type="state",
        relation=MemoryRelation.RELATED_TO,
        valid_from=100.0,
    )
    edge_cb = MemoryEdge(
        edge_id="edge_cb",
        namespace="n1",
        source_unit_id=state_c.state_id,
        target_unit_id=state_b.state_id,
        source_type="state",
        target_type="state",
        relation=MemoryRelation.RELATED_TO,
        valid_from=100.0,
    )

    backend.add_edge(edge_ab)
    backend.add_edge(edge_ac)
    backend.add_edge(edge_cb)

    retriever = GraphRetriever(backend, view)
    hits = await retriever.retrieve(
        RecallQuery(text="Who is A?", namespace="n1", entity_hint="A", as_of=150.0),
        RetrieverContext(),
    )

    # Find candidate for B
    b_cand = [h for h in hits if h.fact.subject == "B"]
    assert len(b_cand) == 1
    # B should be traversed via direct path (depth 1) instead of A->C->B (depth 2)
    assert b_cand[0].signals["bfs_depth"] == 1


@pytest.mark.asyncio
async def test_graph_missing_nodes(test_env) -> None:
    backend, view = test_env

    state_a = view.upsert(
        namespace="n1",
        entity="A",
        attribute="attr",
        value="val_a",
        certainty=Certainty.CERTAIN,
        valid_from=100.0,
    )

    # Edge targeting nonexistent node
    edge_hallucinated = MemoryEdge(
        edge_id="edge_hallucinated",
        namespace="n1",
        source_unit_id=state_a.state_id,
        target_unit_id="nonexistent_state_id",
        source_type="state",
        target_type="state",
        relation=MemoryRelation.RELATED_TO,
        valid_from=100.0,
    )
    backend.add_edge(edge_hallucinated)

    retriever = GraphRetriever(backend, view)
    # This should run perfectly and not crash when trying to resolve the nonexistent target ID
    hits = await retriever.retrieve(
        RecallQuery(text="A information", namespace="n1", entity_hint="A"),
        RetrieverContext(),
    )
    assert len(hits) == 0


@pytest.mark.asyncio
async def test_identity_self_loop_filtered(test_env) -> None:
    backend, view = test_env

    # Evan is the seed (depth 0, not emitted as a candidate itself).
    state_evan = view.upsert(
        namespace="n1",
        entity="Evan",
        attribute="painted",
        value="forest scene",
        certainty=Certainty.CERTAIN,
        valid_from=100.0,
    )
    # A real neighbor that SHOULD surface via traversal.
    state_neighbor = view.upsert(
        namespace="n1",
        entity="Sam",
        attribute="hobby",
        value="hiking",
        certainty=Certainty.CERTAIN,
        valid_from=100.0,
    )
    # Self-loop identity anchor (auto-derived edge endpoint) — pure noise.
    state_identity = view.upsert(
        namespace="n1",
        entity="art class",
        attribute="identity",
        value="art class",
        certainty=Certainty.CERTAIN,
        valid_from=100.0,
    )
    backend.add_edge(
        MemoryEdge(
            edge_id="edge-real",
            namespace="n1",
            source_unit_id=state_evan.state_id,
            target_unit_id=state_neighbor.state_id,
            source_type="state",
            target_type="state",
            relation=MemoryRelation.RELATED_TO,
            valid_from=100.0,
        )
    )
    backend.add_edge(
        MemoryEdge(
            edge_id="edge-id-anchor",
            namespace="n1",
            source_unit_id=state_evan.state_id,
            target_unit_id=state_identity.state_id,
            source_type="state",
            target_type="state",
            relation=MemoryRelation.RELATED_TO,
            valid_from=100.0,
        )
    )

    retriever = GraphRetriever(backend, view)
    hits = await retriever.retrieve(
        RecallQuery(text="What did Evan paint?", namespace="n1", entity_hint="Evan"),
        RetrieverContext(),
    )

    subjects = {hit.fact.subject for hit in hits}
    # The real neighbor surfaces; the 'art class | identity | art class' self-loop never does.
    assert "Sam" in subjects
    assert "art class" not in subjects
    for hit in hits:
        assert not (hit.fact.predicate == "identity" and hit.fact.subject == hit.fact.object)


@pytest.mark.asyncio
async def test_graph_cjk_support(test_env) -> None:
    backend, view = test_env
    retriever = GraphRetriever(backend, view)

    # Chinese CJK seeds extraction
    query = RecallQuery(
        text="\u674e\u96f7\u548c\u97e9\u6845\u6845\u53bb\u54ea\u91cc\u4e86\uff1f",
        namespace="n1",
    )
    seeds = await retriever._discover_seeds(query)
    seed_nodes = {s[0] for s in seeds}

    assert "\u674e\u96f7" in seed_nodes
    assert "\u97e9\u6845\u6845" in seed_nodes


@pytest.mark.asyncio
async def test_factual_lookup_narrows_traversal(test_env) -> None:
    """factual_lookup must keep same_as aliases but drop related_to fan-out.

    Regression for graph flooding: under FACTUAL_LOOKUP the deep
    bidirectional related_to traversal dragged the seed entity's unrelated
    attributes into the fused top-k and crowded out the true evidence. The
    retriever now restricts factual_lookup to depth-1 same_as/supports edges
    so coreference still resolves while the noise is gone. Other query types
    keep the full depth-3 related_to traversal.
    """
    backend, view = test_env

    seed = view.upsert(
        namespace="n1",
        entity="John",
        attribute="job",
        value="banker",
        certainty=Certainty.CERTAIN,
        valid_from=100.0,
    )
    # Alias of the seed reachable via a same_as edge — graph's unique value.
    alias = view.upsert(
        namespace="n1",
        entity="Johnny",
        attribute="job",
        value="banker",
        certainty=Certainty.CERTAIN,
        valid_from=100.0,
    )
    # An unrelated attribute reachable only via related_to (the flooding kind).
    noise = view.upsert(
        namespace="n1",
        entity="apple pie",
        attribute="made_by",
        value="John",
        certainty=Certainty.CERTAIN,
        valid_from=100.0,
    )
    # Depth-2 node behind the noise; must never surface under factual_lookup.
    deep_noise = view.upsert(
        namespace="n1",
        entity="boot camp",
        attribute="kind",
        value="fitness",
        certainty=Certainty.CERTAIN,
        valid_from=100.0,
    )

    backend.add_edge(
        MemoryEdge(
            edge_id="edge-alias",
            namespace="n1",
            source_unit_id=seed.state_id,
            target_unit_id=alias.state_id,
            source_type="state",
            target_type="state",
            relation=MemoryRelation.SAME_AS,
            valid_from=100.0,
        )
    )
    backend.add_edge(
        MemoryEdge(
            edge_id="edge-noise",
            namespace="n1",
            source_unit_id=seed.state_id,
            target_unit_id=noise.state_id,
            source_type="state",
            target_type="state",
            relation=MemoryRelation.RELATED_TO,
            valid_from=100.0,
        )
    )
    backend.add_edge(
        MemoryEdge(
            edge_id="edge-deep",
            namespace="n1",
            source_unit_id=noise.state_id,
            target_unit_id=deep_noise.state_id,
            source_type="state",
            target_type="state",
            relation=MemoryRelation.RELATED_TO,
            valid_from=100.0,
        )
    )

    retriever = GraphRetriever(backend, view)
    query = RecallQuery(text="What is John's job?", namespace="n1", entity_hint="John", as_of=150.0)

    # factual_lookup: same_as alias surfaces, related_to flood is suppressed.
    factual_hits = await retriever.retrieve(
        query, RetrieverContext(query_type=QueryType.FACTUAL_LOOKUP)
    )
    factual_subjects = {h.fact.subject for h in factual_hits}
    assert "Johnny" in factual_subjects  # same_as alias kept
    assert "apple pie" not in factual_subjects  # related_to flood dropped
    assert "boot camp" not in factual_subjects  # depth-2 never reached

    # Default (unrouted) context keeps the full related_to traversal so the
    # narrowing is scoped strictly to factual_lookup.
    default_hits = await retriever.retrieve(query, RetrieverContext())
    default_subjects = {h.fact.subject for h in default_hits}
    assert "apple pie" in default_subjects
    assert "boot camp" in default_subjects
