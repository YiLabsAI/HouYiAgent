"""Tests for the L1→L2 fact promoter that writes deferred-embedding MemoryRecords."""

from __future__ import annotations

import pytest

from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.backends.sqlite_candidate_inbox import SQLiteCandidateInbox
from houyi.adapters.memory.backends.sqlite_entity_state import SQLiteEntityStateView
from houyi.adapters.memory.fact_promoter import (
    MemoryRecordPromoter,
    _certainty_to_confidence,
)
from houyi.adapters.memory.triggers import all_of
from houyi.adapters.memory.turn_writer import TurnWriter
from houyi.adapters.memory.types import (
    AtomicFact,
    Certainty,
    MemoryScope,
    MemoryType,
    RawTurn,
)
from houyi.adapters.memory.workers import ExtractorWorker


@pytest.fixture()
def backend(tmp_path):
    b = SQLiteMemoryBackend(db_path=tmp_path / "promoter.db")
    yield b
    b.close()


def _fact(
    subject="alice", predicate="likes", obj="tea", *, certainty=Certainty.CERTAIN, anchor="anchor1"
):
    return AtomicFact(
        subject=subject,
        predicate=predicate,
        object=obj,
        certainty=certainty,
        source_anchor=anchor,
    )


def _turn() -> RawTurn:
    return RawTurn(session_id="s", role="user", content="alice likes tea")


class TestCertaintyMapping:
    def test_certainty_ordering(self):
        c = _certainty_to_confidence(_fact(certainty=Certainty.CERTAIN))
        p = _certainty_to_confidence(_fact(certainty=Certainty.PROBABLE))
        v = _certainty_to_confidence(_fact(certainty=Certainty.VAGUE))
        assert c > p > v


class TestMemoryRecordPromoter:
    def test_writes_pending_embedding_record(self, backend):
        promoter = MemoryRecordPromoter(backend)
        promoter.promote(_turn(), _fact())

        # The backend exposes pending-embedding rows via its backfill
        # surface (). We assert there is exactly one row keyed
        # on subject.predicate.
        pending = backend.list_pending_embeddings(limit=10)
        assert len(pending) == 1
        _, record = pending[0]
        assert record.key == "alice.likes"
        assert record.content == "tea"
        assert record.scope is MemoryScope.USER
        assert record.memory_type is MemoryType.FACT

    def test_provenance_source_anchor(self, backend):
        promoter = MemoryRecordPromoter(backend, provider_label="unit_test")
        promoter.promote(_turn(), _fact(anchor="turn-XYZ"))
        _, record = backend.list_pending_embeddings(limit=1)[0]
        assert record.provenance is not None
        assert record.provenance.source_ids == ["turn-XYZ"]
        assert record.provenance.extracted_by == "unit_test"

    def test_metadata_session_turn(self, backend):
        promoter = MemoryRecordPromoter(backend)
        turn = RawTurn(session_id="sess-A", role="user", content="x")
        backend.append_raw_turn(turn)
        promoter.promote(turn, _fact())
        _, record = backend.list_pending_embeddings(limit=1)[0]
        assert record.metadata["session_id"] == "sess-A"
        assert record.metadata["turn_id"] == turn.turn_id

    def test_promoter_failure_is_swallowed(self):
        class _BoomBackend:
            def put(self, record):
                raise RuntimeError("disk full")

        promoter = MemoryRecordPromoter(_BoomBackend())
        # Must not raise — promotion is best-effort.
        promoter.promote(_turn(), _fact())

    def test_construction_validation(self):
        with pytest.raises(ValueError):
            MemoryRecordPromoter(None)


class _FakeExtractorOne:
    """Returns a single canned ExtractionResult-like object."""

    def __init__(self, fact):
        self._fact = fact

    async def extract(self, text, source_anchor):
        # Build a result on demand so source_anchor ties back to the turn.
        f = self._fact.model_copy(update={"source_anchor": source_anchor})

        class _R:
            facts = [f]
            raw_sourceless = []

        return _R()


class TestExtractorWorkerWithPromoter:
    async def test_promoter_certain_fact(self, backend):
        wp = TurnWriter(backend, extract_trigger=all_of())
        turn = wp.fast_path(RawTurn(session_id="s", role="user", content="alice likes tea")).turn

        worker = ExtractorWorker(
            backend=backend,
            extractor=_FakeExtractorOne(_fact()),
            entity_state=SQLiteEntityStateView(backend),
            candidate_inbox=SQLiteCandidateInbox(backend),
            promoter=MemoryRecordPromoter(backend),
        )
        await worker.process_once()

        # entity_state holds the fact AND a MemoryRecord row was written.
        active = SQLiteEntityStateView(backend).get_active("default", "alice")
        assert len(active) == 1

        pending = backend.list_pending_embeddings(limit=10)
        assert len(pending) == 1
        _, record = pending[0]
        assert record.key == "alice.likes"
        assert record.metadata["turn_id"] == turn.turn_id

    async def test_promoter_vague_fact(self, backend):
        wp = TurnWriter(backend, extract_trigger=all_of())
        wp.fast_path(RawTurn(session_id="s", role="user", content="maybe?"))
        worker = ExtractorWorker(
            backend=backend,
            extractor=_FakeExtractorOne(_fact(certainty=Certainty.VAGUE)),
            entity_state=SQLiteEntityStateView(backend),
            candidate_inbox=SQLiteCandidateInbox(backend),
            promoter=MemoryRecordPromoter(backend),
        )
        await worker.process_once()
        # Vague facts go to the inbox, not entity_state, and must NOT
        # be promoted — the recall vector path should not surface them.
        assert backend.list_pending_embeddings(limit=10) == []
