"""Unit tests for MemoryService."""

from __future__ import annotations

import pytest
from houyi_studio.server.memory.service import MemoryService

from houyi.adapters.memory.engine import MemoryEngine
from houyi.adapters.memory.store import MemoryStore
from houyi.adapters.memory.types import (
    CandidateStatus,
    MemoryCandidate,
    MemoryScope,
)


@pytest.fixture()
def svc(tmp_path):
    store = MemoryStore(data_dir=str(tmp_path))
    engine = MemoryEngine(store)
    return MemoryService(engine)


class TestCandidates:
    def test_add_and_list(self, svc):
        c = MemoryCandidate(content="test candidate", confidence=0.8)
        svc.add_candidates([c])
        assert len(svc.list_candidates()) == 1
        assert svc.list_candidates()[0].content == "test candidate"

    def test_filter_by_status(self, svc):
        c1 = MemoryCandidate(content="c1")
        c2 = MemoryCandidate(content="c2")
        c2.status = CandidateStatus.REJECTED
        svc.add_candidates([c1, c2])
        pending = svc.list_candidates(status=CandidateStatus.PENDING)
        assert len(pending) == 1
        assert pending[0].content == "c1"

    async def test_approve_creates_record(self, svc):
        c = MemoryCandidate(content="approve me", confidence=0.9, suggested_tags=["test"])
        svc.add_candidates([c])
        record = await svc.approve_candidate(c.candidate_id)
        assert record is not None
        assert record.content == "approve me"
        records = svc.list_records()
        assert len(records) >= 1

    async def test_approve_nonexistent(self, svc):
        result = await svc.approve_candidate("nonexistent")
        assert result is None

    async def test_reject(self, svc):
        c = MemoryCandidate(content="reject me")
        svc.add_candidates([c])
        ok = await svc.reject_candidate(c.candidate_id)
        assert ok is True
        assert c.status == CandidateStatus.REJECTED

    async def test_update_content(self, svc):
        c = MemoryCandidate(content="old")
        svc.add_candidates([c])
        updated = await svc.update_candidate(c.candidate_id, content="new")
        assert updated is not None
        assert updated.content == "new"

    async def test_approve_twice_creates(self, svc):
        c = MemoryCandidate(content="double approve", confidence=0.9)
        svc.add_candidates([c])
        r1 = await svc.approve_candidate(c.candidate_id)
        r2 = await svc.approve_candidate(c.candidate_id)
        assert r1 is not None
        assert r2 is not None


class TestRecords:
    def test_list_empty(self, svc):
        assert svc.list_records() == []

    async def test_crud_flow(self, svc):
        c = MemoryCandidate(
            content="memory content",
            confidence=0.9,
            scope=MemoryScope.USER,
        )
        svc.add_candidates([c])
        record = await svc.approve_candidate(c.candidate_id)
        assert record is not None

        found = svc.get_record(record.record_id)
        assert found is not None
        assert found.content == "memory content"

        updated = await svc.update_record(record.record_id, content="updated content")
        assert updated is not None
        assert updated.content == "updated content"

        ok = await svc.delete_record(record.record_id)
        assert ok is True
        assert svc.get_record(record.record_id) is None

    async def test_delete_nonexistent(self, svc):
        ok = await svc.delete_record("missing")
        assert ok is False

    async def test_update_record_missing(self, svc):
        result = await svc.update_record("nonexistent", content="x")
        assert result is None

    async def test_list_records_by_scope(self, svc):
        c = MemoryCandidate(content="scoped mem", scope=MemoryScope.USER)
        svc.add_candidates([c])
        await svc.approve_candidate(c.candidate_id)
        user_records = svc.list_records(scope=MemoryScope.USER)
        session_records = svc.list_records(scope=MemoryScope.SESSION)
        assert len(user_records) >= 1
        assert len(session_records) == 0

    async def test_recall_history_placeholder(self, svc):
        history = await svc.get_recall_history("any_id")
        assert history == []
