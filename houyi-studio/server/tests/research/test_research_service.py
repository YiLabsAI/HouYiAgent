"""Unit tests for ResearchService."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from houyi_studio.server.research.service import (
    ResearchService,
    SessionNotFoundError,
    SessionNotTerminalError,
    VersionConflictError,
    _ArchivedSession,
)

from houyi.adapters.llm.base import LLMAdapter, LLMResponse, StreamChunk
from houyi.application.research.types import (
    PlanEdit,
    PlanEditOperation,
    PlanStatus,
    ResearchSettings,
    ResearchStatus,
)
from houyi.skills.web_search.service import WebSearchService
from houyi.skills.web_search.types import (
    WebSearchMetadata,
    WebSearchResponse,
    WebSearchResult,
)

_QUICK = ResearchSettings(depth="quick")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PLAN_JSON = json.dumps(
    {
        "sub_questions": [
            {"question": "Q1", "priority": 5, "search_strategy": "web", "expected_sources": 3}
        ],
        "outline": [{"title": "Section 1", "objective": "Cover Q1", "related_question_ids": [0]}],
        "estimated_duration_min": 5,
    }
)

_SEARCHER_RESPONSE = json.dumps(
    {
        "sources": [
            {"url": "https://example.com/1", "title": "Source 1", "snippet": "snippet"},
        ],
        "summary": "Found relevant sources",
        "queries_used": ["test query"],
    }
)
_SECTION = json.dumps({"content": "Body text [ref_001].", "citations": []})
_RACE = json.dumps(
    {
        "comprehensiveness": {"score": 80, "reasoning": "ok"},
        "depth": {"score": 75, "reasoning": "ok"},
        "instruction_following": {"score": 85, "reasoning": "ok"},
        "readability": {"score": 90, "reasoning": "ok"},
    }
)
_FACT = json.dumps({"citation_accuracy": 90.0, "effective_citations": 5})

_QUERY_GEN = json.dumps(["test search query", "alternate query"])
_SUFFICIENCY = json.dumps({"sufficient": True, "rationale": "Enough"})


def _all_responses() -> list[str]:
    return [_PLAN_JSON, _QUERY_GEN, _SUFFICIENCY, _SECTION, "Summary.", _RACE, _FACT]


class _MockLLM(LLMAdapter):
    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or [_PLAN_JSON]
        self._idx = 0

    async def chat(self, messages: list, **kwargs: Any) -> LLMResponse:
        content = self._responses[self._idx] if self._idx < len(self._responses) else "{}"
        self._idx += 1
        return LLMResponse(content=content, finish_reason="stop", model="mock")

    async def stream_chat(self, messages: list, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        yield StreamChunk()


def _mock_ws() -> WebSearchService:
    svc = AsyncMock(spec=WebSearchService)
    svc.search = AsyncMock(
        return_value=WebSearchResponse(
            query="q",
            provider="mock",
            results=[WebSearchResult(title="R1", url="https://a.com", snippet="Snip")],
            metadata=WebSearchMetadata(
                cached=False, cache_hit=False, latency_ms=5, provider="mock"
            ),
        )
    )
    return svc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateSession:
    async def test_creates_plan(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        session, plan = await svc.create_session("AI frameworks", settings=_QUICK)
        assert plan is not None
        assert session.status == ResearchStatus.PLAN_READY

    async def test_idempotency_key(self, tmp_path):
        svc = ResearchService(_MockLLM([_PLAN_JSON, _PLAN_JSON]), _mock_ws(), data_dir=tmp_path)
        s1, _ = await svc.create_session("Q1", idempotency_key="key1", settings=_QUICK)
        s2, _ = await svc.create_session("Q2", idempotency_key="key1", settings=_QUICK)
        assert s1.session_id == s2.session_id

    async def test_create_empty_settings(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        session, plan = await svc.create_session("test", settings=_QUICK)
        assert session.status == ResearchStatus.PLAN_READY
        assert plan is not None


class TestEditPlan:
    async def test_edit_bumps_version(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        session, plan = await svc.create_session("test", settings=_QUICK)
        edit = PlanEdit(op=PlanEditOperation.ADD, target_question="Extra?")
        updated = await svc.edit_plan(session.session_id, [edit])
        assert updated.version == 2

    async def test_version_conflict(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        session, _ = await svc.create_session("test", settings=_QUICK)
        with pytest.raises(VersionConflictError):
            await svc.edit_plan(session.session_id, [], client_plan_version=99)


class TestNotFound:
    async def test_require_session(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        with pytest.raises(SessionNotFoundError):
            svc.get_progress("nonexistent")


class TestListSessions:
    async def test_list_after_create(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        await svc.create_session("test", settings=_QUICK)
        items = svc.list_sessions()
        assert len(items) == 1
        assert items[0]["query"] == "test"


class TestCancelSession:
    async def test_cancel(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        session, _ = await svc.create_session("test", settings=_QUICK)
        await svc.cancel_session(session.session_id, "user cancelled")
        assert session.status == ResearchStatus.CANCELLED


class TestDeleteSession:
    async def test_delete_cancelled(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        session, _ = await svc.create_session("test", settings=_QUICK)
        await svc.cancel_session(session.session_id)
        await svc.delete_session(session.session_id)
        assert svc.get_session(session.session_id) is None

    async def test_delete_plan_ready(self, tmp_path):
        """Non-executing sessions (e.g. plan_ready) can be deleted."""
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        session, _ = await svc.create_session("test", settings=_QUICK)
        assert session.status == ResearchStatus.PLAN_READY
        await svc.delete_session(session.session_id)
        assert svc.get_session(session.session_id) is None

    async def test_delete_executing_blocked(self, tmp_path):
        """Executing sessions cannot be deleted."""
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        session, _ = await svc.create_session("test", settings=_QUICK)
        session._status = ResearchStatus.EXECUTING
        with pytest.raises(SessionNotTerminalError):
            await svc.delete_session(session.session_id)


class TestExecuteLifecycle:
    async def test_full_execute_lifecycle(self, tmp_path):
        svc = ResearchService(_MockLLM(_all_responses()), _mock_ws(), data_dir=tmp_path)
        session, _ = await svc.create_session("AI frameworks", settings=_QUICK)
        await svc.confirm_and_execute(session.session_id)
        assert session.status == ResearchStatus.COMPLETED

    async def test_execute_nonexistent(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        with pytest.raises(SessionNotFoundError):
            await svc.confirm_and_execute("nonexistent")


class TestGetters:
    async def test_progress_after_create(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        session, _ = await svc.create_session("test", settings=_QUICK)
        progress = svc.get_progress(session.session_id)
        assert progress.status == ResearchStatus.PLAN_READY
        assert progress.total_steps == 1

    async def test_emitter_after_create(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        session, _ = await svc.create_session("test", settings=_QUICK)
        emitter = svc.get_emitter(session.session_id)
        assert emitter is not None

    async def test_emitter_missing_none(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        assert svc.get_emitter("nonexistent") is None


class TestPersistence:
    async def test_json_written(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        session, _ = await svc.create_session("test", settings=_QUICK)
        path = tmp_path / f"{session.session_id}.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["session_id"] == session.session_id

    async def test_sessions_loaded_on_startup(self, tmp_path):
        """Sessions persisted as JSON are hydrated when a new service starts."""
        svc1 = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        session, _ = await svc1.create_session("persisted query", settings=_QUICK)
        sid = session.session_id

        svc2 = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        items = svc2.list_sessions()
        assert any(s["session_id"] == sid for s in items)

    async def test_archived_session_shows_plan(self, tmp_path):
        """Archived session preserves the plan for display."""
        svc1 = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        session, plan = await svc1.create_session("test plan", settings=_QUICK)
        sid = session.session_id

        svc2 = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        restored = svc2.get_session(sid)
        assert restored is not None
        assert restored.plan is not None
        assert restored.plan.query == "test plan"

    async def test_archived_session_preserves_error(self, tmp_path):
        """If a session had an error, it's visible after reload."""
        svc1 = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        session, _ = await svc1.create_session("will fail", settings=_QUICK)
        session._error = "Research timed out after 420s"
        session._status = ResearchStatus.FAILED
        svc1._persist_session(session)

        svc2 = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        restored = svc2.get_session(session.session_id)
        assert restored is not None
        assert restored.error == "Research timed out after 420s"
        assert restored.status == ResearchStatus.FAILED

    async def test_delete_archived_session(self, tmp_path):
        """Archived (terminal) sessions can be deleted."""
        svc1 = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        session, _ = await svc1.create_session("to delete", settings=_QUICK)
        await svc1.cancel_session(session.session_id)
        sid = session.session_id

        svc2 = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        await svc2.delete_session(sid)
        assert svc2.get_session(sid) is None
        assert not (tmp_path / f"{sid}.json").exists()


class TestConcurrencyLimit:
    async def test_running_count_zero_initially(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        assert svc.running_session_count() == 0

    async def test_running_count_increments(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        session, _ = await svc.create_session("running test", settings=_QUICK)
        assert svc.running_session_count() == 0
        session._status = ResearchStatus.EXECUTING
        assert svc.running_session_count() == 1
        session._status = ResearchStatus.GENERATING_REPORT
        assert svc.running_session_count() == 1
        session._status = ResearchStatus.PLAN_READY
        assert svc.running_session_count() == 0

    async def test_max_concurrent_constant(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        assert ResearchService.MAX_CONCURRENT_SESSIONS == 3
        assert svc.MAX_CONCURRENT_SESSIONS == 3


class TestMemoryIntegration:
    """Research → Memory extraction pipeline."""

    async def test_memory_push_on_complete(self, tmp_path):
        """Completed sessions push memory candidates to MemoryService."""
        from houyi_studio.server.memory.service import MemoryService

        from houyi.adapters.memory.engine import MemoryEngine
        from houyi.adapters.memory.store import MemoryStore

        mem_store = MemoryStore(data_dir=str(tmp_path / "memory"))
        mem_engine = MemoryEngine(mem_store)
        mem_svc = MemoryService(mem_engine)

        svc = ResearchService(
            _MockLLM(_all_responses()),
            _mock_ws(),
            data_dir=tmp_path,
            memory_service=mem_svc,
        )
        session, _ = await svc.create_session("AI research", settings=_QUICK)
        await svc.confirm_and_execute(session.session_id)
        assert session.status == ResearchStatus.COMPLETED
        candidates = mem_svc.list_candidates()
        assert len(candidates) > 0

    async def test_no_push_without_memory_svc(self, tmp_path):
        """Without memory_service, execution still succeeds."""
        svc = ResearchService(
            _MockLLM(_all_responses()),
            _mock_ws(),
            data_dir=tmp_path,
            memory_service=None,
        )
        session, _ = await svc.create_session("AI research", settings=_QUICK)
        await svc.confirm_and_execute(session.session_id)
        assert session.status == ResearchStatus.COMPLETED

    async def test_memory_extraction_failure_nonfatal(self, tmp_path):
        """Memory extraction failure does not crash execution."""
        from unittest.mock import MagicMock

        broken_svc = MagicMock()
        broken_svc.add_candidates.side_effect = RuntimeError("memory broken")

        svc = ResearchService(
            _MockLLM(_all_responses()),
            _mock_ws(),
            data_dir=tmp_path,
            memory_service=broken_svc,
        )
        session, _ = await svc.create_session("test", settings=_QUICK)
        await svc.confirm_and_execute(session.session_id)
        assert session.status == ResearchStatus.COMPLETED


class TestRehydrate:
    async def test_rehydrate_resets_plan_to_draft(self, tmp_path):
        svc1 = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        session, _ = await svc1.create_session("rehydrate me", settings=_QUICK)
        sid = session.session_id
        await svc1.cancel_session(sid, "user cancelled")

        svc2 = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        archived = svc2.get_session(sid)
        assert archived is not None
        assert isinstance(archived, _ArchivedSession)

        live = svc2._require_live_session(sid)
        assert live.plan is not None
        assert live.plan.status == PlanStatus.DRAFT
