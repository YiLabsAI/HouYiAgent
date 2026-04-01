"""Unit tests for ResearchSession lifecycle."""

from __future__ import annotations

import json

import pytest

from houyi.application.research.session import ResearchSession
from houyi.application.research.types import (
    PlanEdit,
    PlanEditOperation,
    PlanStatus,
    ResearchSettings,
    ResearchStatus,
)

from .conftest import MockLLM, make_mock_web_search

_PLAN_JSON = json.dumps(
    {
        "sub_questions": [
            {
                "question": "What are current frameworks?",
                "priority": 5,
                "search_strategy": "web",
                "expected_sources": 3,
            },
            {
                "question": "How do they compare?",
                "priority": 4,
                "search_strategy": "web",
                "expected_sources": 3,
            },
        ],
        "outline": [
            {"title": "Overview", "objective": "Landscape", "related_question_ids": [0]},
            {"title": "Comparison", "objective": "Feature comparison", "related_question_ids": [1]},
        ],
        "estimated_duration_min": 5,
    }
)

_SEARCHER_RESPONSE = json.dumps(
    {
        "sources": [
            {"url": "https://example.com/1", "title": "Mock Result 1", "snippet": "snippet 1"},
            {"url": "https://example.com/2", "title": "Mock Result 2", "snippet": "snippet 2"},
        ],
        "summary": "Found relevant sources on frameworks",
        "queries_used": ["ai agent framework comparison"],
    }
)
_SECTION_JSON = json.dumps(
    {
        "content": "Section content with [ref_001] citation.",
        "citations": [{"reference_id": "ref_001", "text_span": "content", "context": "ctx"}],
    }
)
_RACE_JSON = json.dumps(
    {
        "comprehensiveness": {"score": 80, "reasoning": "good"},
        "depth": {"score": 75, "reasoning": "ok"},
        "instruction_following": {"score": 85, "reasoning": "follows"},
        "readability": {"score": 90, "reasoning": "clear"},
    }
)
_FACT_JSON = json.dumps(
    {
        "citation_accuracy": 90.0,
        "effective_citations": 5,
    }
)


def _session_responses() -> list[str]:
    """Build the full LLM response sequence for a 2-question delegate session.

    Sequence: plan → searcher_q1 → searcher_q2 → section*2 → summary → RACE → FACT.
    Each sub-question is now handled by a SubAgent that returns a single JSON response.
    """
    return [
        _PLAN_JSON,
        _SEARCHER_RESPONSE,
        _SEARCHER_RESPONSE,
        _SECTION_JSON,
        _SECTION_JSON,
        "Summary of the report.",
        _RACE_JSON,
        _FACT_JSON,
    ]


class TestStart:
    async def test_generates_plan(self):
        llm = MockLLM(responses=[_PLAN_JSON])
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws)
        plan = await session.start("AI frameworks")
        assert plan.query == "AI frameworks"
        assert plan.status == PlanStatus.DRAFT
        assert session.status == ResearchStatus.PLAN_READY

    async def test_progress_after_start(self):
        llm = MockLLM(responses=[_PLAN_JSON])
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws)
        await session.start("test")
        prog = session.progress
        assert prog.total_steps == 2
        assert prog.completed_steps == 0


class TestEditPlan:
    async def test_edit_adds_question(self):
        llm = MockLLM(responses=[_PLAN_JSON])
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws)
        await session.start("test")
        edit = PlanEdit(op=PlanEditOperation.ADD, target_question="New Q?")
        plan = await session.edit_plan([edit])
        assert len(plan.sub_questions) == 3

    async def test_edit_before_start_fails(self):
        llm = MockLLM()
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws)
        with pytest.raises(RuntimeError, match="No plan"):
            await session.edit_plan([])


class TestConfirmPlan:
    async def test_confirm(self):
        llm = MockLLM(responses=[_PLAN_JSON])
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws)
        await session.start("test")
        plan = await session.confirm_plan()
        assert plan.status == PlanStatus.CONFIRMED


class TestExecute:
    async def test_full_lifecycle_delegate(self):
        llm = MockLLM(responses=_session_responses())
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws)
        await session.start("AI frameworks")
        await session.confirm_plan()
        await session.execute()
        assert session.status == ResearchStatus.COMPLETED
        report = await session.get_report()
        assert len(report.sections) == 2
        assert session.quality_score is not None
        assert session.quality_score.overall > 0

    async def test_full_lifecycle_autonomous(self):
        llm = MockLLM(responses=_session_responses())
        ws = make_mock_web_search()
        settings = ResearchSettings(orchestration_mode="autonomous")
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=settings)
        await session.start("AI frameworks")
        await session.confirm_plan()
        await session.execute()
        assert session.status == ResearchStatus.COMPLETED

    async def test_execute_before_start_fails(self):
        llm = MockLLM()
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws)
        with pytest.raises(RuntimeError, match="No plan"):
            await session.execute()


class TestCancel:
    async def test_cancel_sets_status(self):
        llm = MockLLM(responses=[_PLAN_JSON])
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws)
        await session.start("test")
        await session.cancel("user cancelled")
        assert session.status == ResearchStatus.CANCELLED

    async def test_cancel_progress_error(self):
        llm = MockLLM(responses=[_PLAN_JSON])
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws)
        await session.start("test")
        await session.cancel()
        assert session.progress.error is not None


class TestGetReport:
    async def test_get_report_before_execute_fails(self):
        llm = MockLLM(responses=[_PLAN_JSON])
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws)
        await session.start("test")
        with pytest.raises(RuntimeError, match="Report not ready"):
            await session.get_report()


class TestExtractMemories:
    async def test_extracts_from_report(self):
        llm = MockLLM(responses=_session_responses())
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws)
        await session.start("AI frameworks")
        await session.confirm_plan()
        await session.execute()
        candidates = await session.extract_memories()
        assert len(candidates) >= 1
        assert all(c.source_context == "deep_research" for c in candidates)

    async def test_empty_before_execute(self):
        llm = MockLLM(responses=[_PLAN_JSON])
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws)
        await session.start("test")
        candidates = await session.extract_memories()
        assert candidates == []

    async def test_candidates_have_tags(self):
        llm = MockLLM(responses=_session_responses())
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws)
        await session.start("AI frameworks")
        await session.confirm_plan()
        await session.execute()
        candidates = await session.extract_memories()
        for c in candidates:
            assert len(c.suggested_tags) >= 1


_PLAN_ONE_Q = json.dumps(
    {
        "sub_questions": [
            {
                "question": "Single question?",
                "priority": 5,
                "search_strategy": "web",
                "expected_sources": 3,
            },
        ],
        "outline": [
            {"title": "Overview", "objective": "Landscape", "related_question_ids": [0]},
        ],
        "estimated_duration_min": 3,
    }
)


def _one_q_responses() -> list[str]:
    """Sequence: plan → searcher → section → summary → RACE → FACT."""
    return [
        _PLAN_ONE_Q,
        _SEARCHER_RESPONSE,
        _SECTION_JSON,
        "Summary.",
        _RACE_JSON,
        _FACT_JSON,
    ]


class TestBoundaryAndInteraction:
    async def test_single_question_plan(self):
        llm = MockLLM(responses=_one_q_responses())
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws)
        await session.start("test")
        assert len(session.plan.sub_questions) == 1
        await session.confirm_plan()
        await session.execute()
        assert session.status == ResearchStatus.COMPLETED

    async def test_search_failure_propagates(self):
        no_sources = json.dumps({"sources": [], "summary": "No results found", "queries_used": []})
        llm = MockLLM(
            responses=[
                _PLAN_ONE_Q,
                no_sources,
                _SECTION_JSON,
                "Summary.",
                _RACE_JSON,
                _FACT_JSON,
            ],
        )
        ws = make_mock_web_search()
        session = ResearchSession(
            llm_adapter=llm,
            web_search=ws,
            settings=ResearchSettings(max_search_rounds=1),
        )
        await session.start("test")
        await session.confirm_plan()
        await session.execute()
        assert session.progress.sources_found == 0

    async def test_events_emitted_in_order(self):
        from houyi.application.runtime.events import AgentEventType, EventEmitter

        captured: list[dict] = []

        async def _handler(event):
            captured.append(event.data)

        emitter = EventEmitter()
        emitter.on(AgentEventType.PROGRESS, _handler)

        llm = MockLLM(responses=_one_q_responses())
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, event_emitter=emitter)
        await session.start("test")
        await session.confirm_plan()
        await session.execute()
        seqs = [e["sequence"] for e in captured]
        assert seqs == sorted(seqs)
        assert len(captured) >= 3
