"""Unit tests for ResearchSession lifecycle."""

from __future__ import annotations

import asyncio
import json

import pytest

from houyi.application.research import session as _session_mod
from houyi.application.research.session import ResearchSession, _parse_search_output
from houyi.application.research.types import (
    PlanEdit,
    PlanEditOperation,
    PlanStatus,
    ResearchSettings,
    ResearchStatus,
)

from .conftest import MockLLM, make_mock_web_search

_QUICK = ResearchSettings(depth="quick")

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

_QUERY_GEN_RESPONSE = json.dumps(["AI agent framework comparison", "best AI agent tools 2026"])
_SUFFICIENCY_TRUE = json.dumps({"sufficient": True, "rationale": "Enough sources found"})
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


_INTERMEDIATE_JSON = json.dumps(
    {
        "analysis": "Analysis of findings with [ref_001] citations.",
        "key_findings": ["Finding 1", "Finding 2"],
        "confidence": 0.8,
        "gaps": [],
    }
)

_STANDARD = ResearchSettings(depth="standard")

_CLARIFICATION_REFINE_JSON = json.dumps(
    {
        "needs_clarification": True,
        "confidence": 0.5,
        "issues": ["Query scope is ambiguous"],
        "suggested_questions": ["What time period?"],
        "refined_query": "AI agent frameworks 2025 comparison",
    }
)

_CLARIFICATION_PASS_JSON = json.dumps(
    {
        "needs_clarification": False,
        "confidence": 0.9,
        "issues": [],
        "suggested_questions": [],
        "refined_query": None,
    }
)

_VALIDATION_JSON = json.dumps(
    {
        "quality_score": 80,
        "has_citations": True,
        "issues": [],
        "needs_rewrite": False,
        "suggested_queries": [],
        "reasoning": "Adequate quality",
    }
)

_SEARCHER_RESPONSE_ALT = json.dumps(
    {
        "sources": [
            {
                "url": "https://alt.com/1",
                "title": "Alternative Source",
                "snippet": "different info",
            },
        ],
        "summary": (
            "Novel autonomous architectures and emerging distributed paradigms "
            "represent a fundamental shift in how systems are designed"
        ),
        "queries_used": ["alternative research query"],
    }
)


def _session_responses() -> list[str]:
    """Build the full LLM response sequence for a 2-question quick-depth session.

    SearchCoordinator flow per question: query_gen → sufficiency.
    Full sequence: plan → (query_gen + sufficiency) × 2 → section*2 → summary → RACE → FACT.
    """
    return [
        _PLAN_JSON,
        _QUERY_GEN_RESPONSE,
        _SUFFICIENCY_TRUE,
        _QUERY_GEN_RESPONSE,
        _SUFFICIENCY_TRUE,
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
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        plan = await session.start("AI frameworks")
        assert plan.query == "AI frameworks"
        assert plan.status == PlanStatus.DRAFT
        assert session.status == ResearchStatus.PLAN_READY

    async def test_progress_after_start(self):
        llm = MockLLM(responses=[_PLAN_JSON])
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        await session.start("test")
        prog = session.progress
        assert prog.total_steps == 2
        assert prog.completed_steps == 0


class TestEditPlan:
    async def test_edit_adds_question(self):
        llm = MockLLM(responses=[_PLAN_JSON])
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        await session.start("test")
        edit = PlanEdit(op=PlanEditOperation.ADD, target_question="New Q?")
        plan = await session.edit_plan([edit])
        assert len(plan.sub_questions) == 3

    async def test_edit_before_start_fails(self):
        llm = MockLLM()
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        with pytest.raises(RuntimeError, match="No plan"):
            await session.edit_plan([])


class TestConfirmPlan:
    async def test_confirm(self):
        llm = MockLLM(responses=[_PLAN_JSON])
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        await session.start("test")
        plan = await session.confirm_plan()
        assert plan.status == PlanStatus.CONFIRMED

    async def test_confirm_before_start_fails(self):
        llm = MockLLM()
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        with pytest.raises(RuntimeError, match="No plan to confirm"):
            await session.confirm_plan()


class TestExecute:
    async def test_full_lifecycle_direct(self):
        """DIRECT mode: SearchCoordinator (default, no Agent SDK in search)."""
        llm = MockLLM(responses=_session_responses())
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        await session.start("AI frameworks")
        await session.confirm_plan()
        await session.execute()
        assert session.status == ResearchStatus.COMPLETED
        report = await session.get_report()
        assert len(report.sections) == 2
        assert session.quality_score is not None
        assert session.quality_score.overall > 0

    async def test_full_lifecycle_delegate(self):
        """DELEGATE mode: AgentTeamManager.spawn() + join() sequentially."""
        responses = [
            _PLAN_JSON,
            _SEARCHER_RESPONSE,
            _SEARCHER_RESPONSE,
            _SECTION_JSON,
            _SECTION_JSON,
            "Summary of the report.",
            _RACE_JSON,
            _FACT_JSON,
        ]
        llm = MockLLM(responses=responses)
        ws = make_mock_web_search()
        settings = ResearchSettings(orchestration_mode="delegate", depth="quick")
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=settings)
        await session.start("AI frameworks")
        await session.confirm_plan()
        await session.execute()
        assert session.status == ResearchStatus.COMPLETED
        report = await session.get_report()
        assert len(report.sections) == 2

    async def test_full_lifecycle_autonomous(self):
        """AUTONOMOUS mode: AgentTeamManager.spawn_parallel() → AgentRunner tool-loop."""
        responses = [
            _PLAN_JSON,
            _SEARCHER_RESPONSE,
            _SEARCHER_RESPONSE,
            _SECTION_JSON,
            _SECTION_JSON,
            "Summary of the report.",
            _RACE_JSON,
            _FACT_JSON,
        ]
        llm = MockLLM(responses=responses)
        ws = make_mock_web_search()
        settings = ResearchSettings(orchestration_mode="autonomous", depth="quick")
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=settings)
        await session.start("AI frameworks")
        await session.confirm_plan()
        await session.execute()
        assert session.status == ResearchStatus.COMPLETED

    async def test_execute_before_start_fails(self):
        llm = MockLLM()
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        with pytest.raises(RuntimeError, match="No plan"):
            await session.execute()

    async def test_execute_timeout(self):
        import asyncio

        llm = MockLLM(responses=[_PLAN_JSON])
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        await session.start("test")
        await session.confirm_plan()

        async def _hang():
            await asyncio.sleep(999)

        session._execute_inner = _hang  # type: ignore[assignment]
        session._session_timeout = lambda: 0.01  # type: ignore[assignment]

        await session.execute()
        assert session.status == ResearchStatus.FAILED
        assert "timed out" in session._error

    async def test_execute_cancelled_error(self):
        import asyncio

        llm = MockLLM(responses=[_PLAN_JSON])
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        await session.start("test")
        await session.confirm_plan()

        async def _cancel():
            raise asyncio.CancelledError("cancelled")

        session._execute_inner = _cancel  # type: ignore[assignment]

        await session.execute()
        assert session.status == ResearchStatus.CANCELLED
        assert session._error == "Cancelled"

    async def test_execute_generic_exception(self):
        llm = MockLLM(responses=[_PLAN_JSON])
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        await session.start("test")
        await session.confirm_plan()

        async def _explode():
            raise ValueError("kaboom")

        session._execute_inner = _explode  # type: ignore[assignment]

        with pytest.raises(ValueError, match="kaboom"):
            await session.execute()
        assert session.status == ResearchStatus.FAILED
        assert session._error == "kaboom"

    async def test_execute_wrong_plan_status(self):
        llm = MockLLM(responses=[_PLAN_JSON])
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        await session.start("test")
        session._plan.status = PlanStatus.COMPLETED
        with pytest.raises(RuntimeError, match="cannot be executed"):
            await session.execute()


class TestCancel:
    async def test_cancel_sets_status(self):
        llm = MockLLM(responses=[_PLAN_JSON])
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        await session.start("test")
        await session.cancel("user cancelled")
        assert session.status == ResearchStatus.CANCELLED

    async def test_cancel_progress_error(self):
        llm = MockLLM(responses=[_PLAN_JSON])
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        await session.start("test")
        await session.cancel()
        assert session.progress.error is not None


class TestGetReport:
    async def test_get_report_before_execute_fails(self):
        llm = MockLLM(responses=[_PLAN_JSON])
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        await session.start("test")
        with pytest.raises(RuntimeError, match="Report not ready"):
            await session.get_report()


class TestExtractMemories:
    async def test_extracts_from_report(self):
        llm = MockLLM(responses=_session_responses())
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        await session.start("AI frameworks")
        await session.confirm_plan()
        await session.execute()
        candidates = await session.extract_memories()
        assert len(candidates) >= 1
        assert all(c.source_context == "deep_research" for c in candidates)

    async def test_empty_before_execute(self):
        llm = MockLLM(responses=[_PLAN_JSON])
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        await session.start("test")
        candidates = await session.extract_memories()
        assert candidates == []

    async def test_candidates_have_tags(self):
        llm = MockLLM(responses=_session_responses())
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
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
    """Sequence: plan → query_gen → sufficiency → section → summary → RACE → FACT."""
    return [
        _PLAN_ONE_Q,
        _QUERY_GEN_RESPONSE,
        _SUFFICIENCY_TRUE,
        _SECTION_JSON,
        "Summary.",
        _RACE_JSON,
        _FACT_JSON,
    ]


class TestBoundaryAndInteraction:
    async def test_single_question_plan(self):
        llm = MockLLM(responses=_one_q_responses())
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
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
        session = ResearchSession(
            llm_adapter=llm, web_search=ws, settings=_QUICK, event_emitter=emitter
        )
        await session.start("test")
        await session.confirm_plan()
        await session.execute()
        seqs = [e["sequence"] for e in captured]
        assert seqs == sorted(seqs)
        assert len(captured) >= 3

    async def test_execute_timeout_sets_failed(self):
        """Timeout during execution sets FAILED status."""
        import asyncio
        from unittest.mock import patch

        llm = MockLLM(responses=[_PLAN_ONE_Q])
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        await session.start("test")
        await session.confirm_plan()

        async def _stall():
            await asyncio.sleep(100)

        with patch.object(session, "_execute_inner", new=_stall):
            with patch.object(session, "_session_timeout", return_value=0.01):
                await session.execute()
        assert session.status == ResearchStatus.FAILED
        assert "timed out" in (session.progress.error or "")

    async def test_execute_cancelled_error(self):
        """CancelledError during execution sets CANCELLED status."""
        import asyncio
        from unittest.mock import patch

        llm = MockLLM(responses=[_PLAN_ONE_Q])
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        await session.start("test")
        await session.confirm_plan()

        async def _cancel():
            raise asyncio.CancelledError("cancelled")

        with patch.object(session, "_execute_inner", new=_cancel):
            await session.execute()
        assert session.status == ResearchStatus.CANCELLED

    async def test_execute_generic_exception(self):
        """Generic exception during execution sets FAILED and re-raises."""
        from unittest.mock import patch

        llm = MockLLM(responses=[_PLAN_ONE_Q])
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        await session.start("test")
        await session.confirm_plan()

        async def _boom():
            raise ValueError("boom")

        with patch.object(session, "_execute_inner", new=_boom):
            with pytest.raises(ValueError, match="boom"):
                await session.execute()
        assert session.status == ResearchStatus.FAILED

    async def test_search_agent_timeout_fallback(self):
        """Per-agent timeout produces empty SearchResult."""
        import asyncio
        from unittest.mock import patch

        responses = [_PLAN_ONE_Q, _SEARCHER_RESPONSE, _SECTION_JSON, "Sum.", _RACE_JSON, _FACT_JSON]
        llm = MockLLM(responses=responses)
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        await session.start("test")
        await session.confirm_plan()

        original_wait_for = asyncio.wait_for

        async def _timeout_search(coro, timeout):
            raise TimeoutError()

        with patch.object(_session_mod.asyncio, "wait_for", side_effect=_timeout_search):
            with patch.object(session, "_run_report"):
                await session._run_search()
        assert session._search_results[0].sources == []

    async def test_search_coordinator_error_falls_back(self):
        """SearchCoordinator failure triggers Agent fallback."""
        llm = MockLLM(
            responses=[
                _PLAN_ONE_Q,
                "not a json array",
                "bad sufficiency",
                _SECTION_JSON,
                "Sum.",
                _RACE_JSON,
                _FACT_JSON,
            ]
        )
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        await session.start("test")
        await session.confirm_plan()
        await session.execute()
        assert session.status == ResearchStatus.COMPLETED

    async def test_clarification_refines_query(self):
        """Standard depth triggers ClarificationAgent; low confidence uses refined query."""
        import json as _json

        clarification_resp = _json.dumps(
            {
                "needs_clarification": True,
                "confidence": 0.5,
                "issues": ["ambiguous"],
                "suggested_questions": ["Which aspect?"],
                "refined_query": "AI agent framework comparison 2026",
            }
        )
        std_settings = ResearchSettings(depth="standard")
        responses = [clarification_resp, _PLAN_ONE_Q]
        llm = MockLLM(responses=responses)
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=std_settings)
        plan = await session.start("AI frameworks")
        assert session._clarification is not None
        assert session._clarification.refined_query == "AI agent framework comparison 2026"

    async def test_coordinator_finds_sources(self):
        """SearchCoordinator returns sources through multi-round search."""
        llm = MockLLM(responses=_one_q_responses())
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        await session.start("test")
        await session.confirm_plan()
        await session.execute()
        assert session.status == ResearchStatus.COMPLETED
        assert len(session._search_results[0].sources) >= 1

    async def test_a2a_bus_receives_messages(self):
        """Verify AgentMessageBus receives mapped A2A messages during execution."""
        import asyncio

        from houyi.application.runtime.message_bus import AgentMessageBus

        bus = AgentMessageBus()

        llm = MockLLM(responses=_one_q_responses())
        ws = make_mock_web_search()
        session = ResearchSession(
            llm_adapter=llm,
            web_search=ws,
            settings=_QUICK,
            message_bus=bus,
        )
        topic = f"research.{session.session_id}"

        sub_queue: asyncio.Queue = asyncio.Queue()
        bus._topic_subscribers[topic]["test_subscriber"] = sub_queue

        await session.start("test")
        await session.confirm_plan()
        await session.execute()

        received = []
        while not sub_queue.empty():
            received.append(sub_queue.get_nowait())

        assert len(received) >= 1
        types = {m.message_type.value for m in received}
        assert "question.covered" in types
        assert all(m.sender_id == session.session_id for m in received)

    async def test_delegate_mode_bus_bridge(self):
        """DELEGATE mode agent events also flow through MessageBus."""
        import asyncio

        from houyi.application.runtime.message_bus import AgentMessageBus

        bus = AgentMessageBus()

        responses = [
            _PLAN_ONE_Q,
            _SEARCHER_RESPONSE,
            _SECTION_JSON,
            "Sum.",
            _RACE_JSON,
            _FACT_JSON,
        ]
        llm = MockLLM(responses=responses)
        ws = make_mock_web_search()
        settings = ResearchSettings(orchestration_mode="delegate", depth="quick")
        session = ResearchSession(
            llm_adapter=llm,
            web_search=ws,
            settings=settings,
            message_bus=bus,
        )
        topic = f"research.{session.session_id}"
        sub_queue: asyncio.Queue = asyncio.Queue()
        bus._topic_subscribers[topic]["test_sub"] = sub_queue

        await session.start("test")
        await session.confirm_plan()
        await session.execute()

        received = []
        while not sub_queue.empty():
            received.append(sub_queue.get_nowait())

        assert len(received) >= 1
        types = {m.message_type.value for m in received}
        assert "task.delegate" in types or "task.result" in types


def _standard_session_responses(
    clarification_json: str = _CLARIFICATION_PASS_JSON,
) -> list[str]:
    """Standard depth: clarification + plan + 2×(query_gen+sufficiency) + 2 intermediates
    + 2 sections + summary + 2 validations + RACE + FACT."""
    return [
        clarification_json,
        _PLAN_JSON,
        _QUERY_GEN_RESPONSE,
        _SUFFICIENCY_TRUE,
        _QUERY_GEN_RESPONSE,
        _SUFFICIENCY_TRUE,
        _INTERMEDIATE_JSON,
        _INTERMEDIATE_JSON,
        _SECTION_JSON,
        _SECTION_JSON,
        "Summary.",
        _VALIDATION_JSON,
        _VALIDATION_JSON,
        _RACE_JSON,
        _FACT_JSON,
    ]


class TestCancelledDuringSearch:
    async def test_cancelled_flag_aborts_search(self):
        llm = MockLLM(responses=[_PLAN_JSON])
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        await session.start("test")
        await session.confirm_plan()
        session._cancelled = True

        await session.execute()
        assert session.status == ResearchStatus.CANCELLED


class TestStandardDepthPaths:
    async def test_clarification_refinement(self):
        llm = MockLLM(
            responses=_standard_session_responses(_CLARIFICATION_REFINE_JSON),
        )
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_STANDARD)
        plan = await session.start("AI frameworks")
        assert session._clarification is not None
        assert session._clarification.refined_query is not None
        await session.confirm_plan()
        await session.execute()
        assert session.status == ResearchStatus.COMPLETED

    async def test_intermediate_report_generation(self):
        llm = MockLLM(responses=_standard_session_responses())
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_STANDARD)
        await session.start("AI frameworks")
        await session.confirm_plan()
        await session.execute()
        assert session.status == ResearchStatus.COMPLETED
        assert len(session._intermediate_reports) == 2
        for ir in session._intermediate_reports:
            assert ir.confidence > 0

    async def test_conflict_detection_and_resolution(self):
        from unittest.mock import AsyncMock, patch

        from houyi.application.runtime.conflict import (
            ConflictRecord,
            ConflictResolution,
        )

        fake_conflict = ConflictRecord(
            agent_a_id="q1",
            agent_b_id="q2",
            output_a="Python dominates ML",
            output_b="R is the standard for statistics",
        )
        fake_resolution = ConflictResolution(
            method="source_voting",
            chosen_output="Python dominates ML",
            confidence=0.8,
            rationale="More sources support Python",
        )

        responses = _standard_session_responses()
        llm = MockLLM(responses=responses)
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_STANDARD)
        await session.start("AI frameworks")
        await session.confirm_plan()

        with (
            patch.object(
                session._report_pipeline._conflict_resolver,
                "detect",
                new=AsyncMock(return_value=[fake_conflict]),
            ),
            patch.object(
                session._report_pipeline._conflict_resolver,
                "resolve",
                new=AsyncMock(return_value=fake_resolution),
            ),
        ):
            await session.execute()

        assert session.status == ResearchStatus.COMPLETED
        assert len(session._conflicts) == 1
        assert session._conflicts[0].resolution is not None
        assert session._conflicts[0].resolution.method == "source_voting"


class TestSearchOneTimeout:
    async def test_search_coordinator_timeout(self):
        import asyncio
        from unittest.mock import patch

        llm = MockLLM(
            responses=[_PLAN_ONE_Q, _SECTION_JSON, "Summary.", _RACE_JSON, _FACT_JSON],
        )
        ws = make_mock_web_search()
        session = ResearchSession(llm_adapter=llm, web_search=ws, settings=_QUICK)
        await session.start("test")
        await session.confirm_plan()

        session._AGENT_TIMEOUT_SECONDS = 0.01

        async def _slow_search(sq, ctx):
            await asyncio.sleep(999)

        with patch.object(session._search_coordinator, "search", _slow_search):
            await session.execute()

        assert session.status == ResearchStatus.COMPLETED
        assert session.progress.sources_found == 0


class TestParseSearchOutput:
    async def test_json_decode_fallback(self):
        from houyi.application.research.types import SubQuestion

        sq = SubQuestion(question="Test?", expected_sources=3)
        result = _parse_search_output(sq, "not json at all")
        assert result.summary == "not json at all"
        assert result.sources == []

    async def test_code_fenced_output_parsed(self):
        from houyi.application.research.types import SubQuestion

        sq = SubQuestion(question="Test?", expected_sources=3)
        fenced = "```json\n" + _SEARCHER_RESPONSE + "\n```"
        result = _parse_search_output(sq, fenced)
        assert len(result.sources) == 2
        assert result.summary != ""
