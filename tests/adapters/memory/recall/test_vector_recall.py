"""VectorRecallRetriever — recall-layer wrapper tests.

 validates the store→recall adapter and verifies the
orchestrator routing table now includes the vector slot for every
query type where expects vector evidence.
"""

from __future__ import annotations

import pytest

from houyi.adapters.memory.recall.orchestrator import (
    _DEFAULT_FUSION_WEIGHTS,
    _DEFAULT_ROUTE_TABLE,
)
from houyi.adapters.memory.recall.retrievers.vector import VectorRecallRetriever
from houyi.adapters.memory.recall.types import (
    QueryType,
    RecallQuery,
    RetrieverContext,
    RetrieverKind,
)
from houyi.adapters.memory.types import MemoryProvenance, MemoryRecord, MemoryScope


class _FakeStoreRetriever:
    """Minimal stand-in for the store-layer VectorRetriever.

    Records the captured arguments so tests can assert that the wrapper
    forwards query text + top_k correctly, and returns canned hits.
    """

    def __init__(self, hits: list[tuple[MemoryRecord, float]]):
        self.hits = hits
        self.calls: list[dict] = []

    async def retrieve(self, text, *, top_k=None, scope=None):
        self.calls.append({"text": text, "top_k": top_k, "scope": scope})
        return self.hits


class TestVectorRecallRetriever:
    async def test_wraps_records_as_candidates(self):
        records = [
            (MemoryRecord(key="py", content="Python notes", scope=MemoryScope.USER), 0.91),
            (MemoryRecord(key="rs", content="Rust notes", scope=MemoryScope.USER), 0.42),
        ]
        wrapper = VectorRecallRetriever(_FakeStoreRetriever(records))
        out = await wrapper.retrieve(
            RecallQuery(text="programming notes", top_k=5),
            RetrieverContext(),
        )
        assert len(out) == 2
        assert out[0].matched_by == RetrieverKind.VECTOR
        assert out[0].fact.subject == "py"
        assert out[0].fact.object == "Python notes"
        assert out[0].score == 0.91
        # Signals carry the underlying scope so downstream fusion can use it.
        assert out[0].signals["scope"] == MemoryScope.USER.value

    async def test_passes_top_k_through(self):
        fake = _FakeStoreRetriever([])
        wrapper = VectorRecallRetriever(fake)
        await wrapper.retrieve(RecallQuery(text="x", top_k=7), RetrieverContext())
        assert fake.calls[0]["top_k"] == 7

    async def test_empty_results_pass_through(self):
        wrapper = VectorRecallRetriever(_FakeStoreRetriever([]))
        out = await wrapper.retrieve(RecallQuery(text="q"), RetrieverContext())
        assert out == []

    def test_requires_inner_retriever(self):
        with pytest.raises(ValueError):
            VectorRecallRetriever(None)  # type: ignore[arg-type]

    async def test_restores_real_triple(self):
        """Vector candidates must carry the real subject/predicate/object
        from the record metadata, not the synthetic record.key /
        "content" / record.content form. Without this restore, RRF
        fusion (which groups by the proposition triple) treats a vector
        candidate and an entity_state candidate for the same fact as two
        unrelated entries, splitting RRF votes."""
        record = MemoryRecord(
            record_id="fact:abc123",
            key="Dave.has_hobby.def",
            content="Dave has hobby photography (time: 2023-11-17)",
            scope=MemoryScope.USER,
            memory_type="fact",
            valid_from=1700000000.0,
            provenance=MemoryProvenance(
                source_type="atomic_fact",
                source_ids=["conv-50:D30:1"],
            ),
            metadata={
                "session_id": "s1",
                "turn_id": "conv-50:session_1:D30:1:1",
                "fact_subject": "Dave",
                "fact_predicate": "has_hobby",
                "fact_object": "photography",
            },
        )
        wrapper = VectorRecallRetriever(_FakeStoreRetriever([(record, 0.95)]))
        out = await wrapper.retrieve(
            RecallQuery(text="hobby", top_k=5),
            RetrieverContext(),
        )
        assert len(out) == 1
        fact = out[0].fact
        assert fact.subject == "Dave"
        assert fact.predicate == "has_hobby"
        assert fact.object == "photography"
        assert fact.source_anchor == "conv-50:D30:1"
        assert fact.valid_from == 1700000000.0

    async def test_fallback_without_metadata(self):
        """Non-fact records without fact_subject/predicate/object in
        metadata fall back to the synthetic form (record.key /
        "content" / record.content)."""
        record = MemoryRecord(
            record_id="note:xyz",
            key="some_key",
            content="free-form note text",
            scope=MemoryScope.USER,
        )
        wrapper = VectorRecallRetriever(_FakeStoreRetriever([(record, 0.5)]))
        out = await wrapper.retrieve(
            RecallQuery(text="note", top_k=5),
            RetrieverContext(),
        )
        assert out[0].fact.subject == "some_key"
        assert out[0].fact.predicate == "content"


class TestRoutingTable:
    """Verify vector-aware routing & fusion weights."""

    @pytest.mark.parametrize(
        "query_type",
        [
            QueryType.FACTUAL_LOOKUP,
            QueryType.TEMPORAL_QUERY,
            QueryType.RELATIONAL_CHAIN,
            QueryType.PROCEDURAL_RECALL,
            QueryType.THEMATIC_SUMMARY,
        ],
    )
    def test_vector_slot_present(self, query_type):
        assert "vector" in _DEFAULT_ROUTE_TABLE[query_type]

    def test_negation_check_includes_vector(self):
        # NEGATION_CHECK now includes vector to rescue semantic mismatch.
        assert "vector" in _DEFAULT_ROUTE_TABLE[QueryType.NEGATION_CHECK]

    def test_thematic_weights_vector(self):
        weights = _DEFAULT_FUSION_WEIGHTS[QueryType.THEMATIC_SUMMARY]
        assert weights[RetrieverKind.VECTOR] >= max(
            weights[k] for k in weights if k is not RetrieverKind.VECTOR
        )

    def test_factual_weights_entity(self):
        weights = _DEFAULT_FUSION_WEIGHTS[QueryType.FACTUAL_LOOKUP]
        assert weights[RetrieverKind.ENTITY_STATE] == weights[RetrieverKind.VECTOR]
