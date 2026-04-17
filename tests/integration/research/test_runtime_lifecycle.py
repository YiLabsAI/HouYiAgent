"""Integration test: ResearchRuntime end-to-end lifecycle.

Exercises the complete plan → search → aggregate → report → quality → memory
pipeline with mock LLM and WebSearch, verifying cross-component integration
and event emission ordering.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from houyi.adapters.llm.base import LLMAdapter, LLMResponse, StreamChunk
from houyi.application.research.runtime import ResearchRuntime
from houyi.application.research.types import (
    PlanEdit,
    PlanEditOperation,
    PlanStatus,
    ResearchSettings,
    ResearchStatus,
)
from houyi.application.research.url_validator import (
    URLValidationReport,
    URLValidationResult,
    URLValidator,
)
from houyi.application.runtime.events import AgentEvent, EventEmitter
from houyi.skills.web_search.service import WebSearchService
from houyi.skills.web_search.types import (
    WebSearchMetadata,
    WebSearchResponse,
    WebSearchResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PLAN_JSON = json.dumps(
    {
        "sub_questions": [
            {
                "question": "What are the leading AI agent frameworks in 2026?",
                "priority": 5,
                "search_strategy": "web",
                "expected_sources": 5,
            },
            {
                "question": "How do they compare on key dimensions?",
                "priority": 4,
                "search_strategy": "web",
                "expected_sources": 5,
            },
            {
                "question": "What are future trends?",
                "priority": 3,
                "search_strategy": "web",
                "expected_sources": 3,
            },
            {
                "question": "What practical tradeoffs matter in adoption?",
                "priority": 2,
                "search_strategy": "web",
                "expected_sources": 3,
            },
            {
                "question": "What limitations or risks are repeatedly reported?",
                "priority": 1,
                "search_strategy": "web",
                "expected_sources": 3,
            },
        ],
        "outline": [
            {"title": "Overview", "objective": "Landscape", "related_question_ids": [0]},
            {"title": "Comparison", "objective": "Feature comparison", "related_question_ids": [1]},
            {"title": "Future", "objective": "Trends", "related_question_ids": [2]},
            {"title": "Tradeoffs", "objective": "Adoption tradeoffs", "related_question_ids": [3]},
            {
                "title": "Limitations",
                "objective": "Constraints and risks",
                "related_question_ids": [4],
            },
        ],
        "estimated_duration_min": 8,
    }
)

_SEARCHER_RESPONSE = json.dumps(
    {
        "sources": [
            {
                "url": f"https://example.com/s{i}",
                "title": f"AI Framework Source {i}",
                "snippet": f"Snippet about AI frameworks {i}",
            }
            for i in range(5)
        ],
        "summary": "Found relevant sources on AI agent frameworks",
        "queries_used": ["ai agent framework 2026", "compare AI agent tools"],
    }
)
_SECTION_JSON = json.dumps(
    {
        "content": "This section covers AI frameworks [ref_001]. They differ in architecture [ref_002].",
        "citations": [
            {"reference_id": "ref_001", "text_span": "AI frameworks", "context": "overview"},
            {"reference_id": "ref_002", "text_span": "architecture", "context": "comparison"},
        ],
    }
)
_RACE_JSON = json.dumps(
    {
        "comprehensiveness": {"score": 78, "reasoning": "Good breadth"},
        "depth": {"score": 72, "reasoning": "Adequate depth"},
        "instruction_following": {"score": 85, "reasoning": "On topic"},
        "readability": {"score": 88, "reasoning": "Well structured"},
    }
)
_FACT_JSON = json.dumps({"citation_accuracy": 92.0, "effective_citations": 8})

_QUERY_GEN = json.dumps(["AI agent framework 2026", "compare agent tools"])
_SUFFICIENCY = json.dumps({"sufficient": True, "rationale": "Enough"})
_CLARIFICATION_PASS = json.dumps(
    {
        "needs_clarification": False,
        "confidence": 0.9,
        "issues": [],
        "suggested_questions": [],
        "refined_query": None,
    }
)
_INTERMEDIATE_JSON = json.dumps(
    {
        "analysis": "Analysis of findings.",
        "key_findings": ["Finding 1"],
        "confidence": 0.8,
        "gaps": [],
    }
)
_VALIDATION_JSON = json.dumps(
    {
        "quality_score": 80,
        "has_citations": True,
        "is_relevant": True,
        "is_substantive": True,
        "needs_rewrite": False,
        "issues": [],
    }
)


def _plan_with_clarification(plan_json: str, clarification_json: str) -> str:
    plan_data = json.loads(plan_json)
    plan_data["clarification"] = json.loads(clarification_json)
    return json.dumps(plan_data)


def _build_responses() -> list[str]:
    """LLM response sequence for a 5-question standard-depth session.

    Standard depth: planner draft (+ clarification metadata) → 5×(query_gen+sufficiency)
    → 5×intermediate → 5×section → summary → 5×validation → RACE → FACT.
    """
    return [
        _plan_with_clarification(_PLAN_JSON, _CLARIFICATION_PASS),
        _QUERY_GEN,
        _SUFFICIENCY,
        _QUERY_GEN,
        _SUFFICIENCY,
        _QUERY_GEN,
        _SUFFICIENCY,
        _QUERY_GEN,
        _SUFFICIENCY,
        _QUERY_GEN,
        _SUFFICIENCY,
        _INTERMEDIATE_JSON,
        _INTERMEDIATE_JSON,
        _INTERMEDIATE_JSON,
        _INTERMEDIATE_JSON,
        _INTERMEDIATE_JSON,
        _SECTION_JSON,
        _SECTION_JSON,
        _SECTION_JSON,
        _SECTION_JSON,
        _SECTION_JSON,
        "Summary of the research findings on AI agent frameworks.",
        _VALIDATION_JSON,
        _VALIDATION_JSON,
        _VALIDATION_JSON,
        _VALIDATION_JSON,
        _VALIDATION_JSON,
        _RACE_JSON,
        _FACT_JSON,
    ]


class _MockLLM(LLMAdapter):
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._idx = 0
        self._query_idx = 0
        self._sufficiency_idx = 0
        self._intermediate_idx = 0
        self._section_idx = 0
        self._validation_idx = 0

    def _next_linear_content(self) -> str:
        content = self._responses[self._idx] if self._idx < len(self._responses) else "{}"
        self._idx += 1
        return content

    def _route_content(self, prompt: str) -> str:
        text = prompt.strip()
        if '"sub_questions"' in text and '"outline"' in text:
            return _plan_with_clarification(_PLAN_JSON, _CLARIFICATION_PASS)
        if "Respond ONLY with a JSON array of query strings" in text:
            self._query_idx += 1
            return _QUERY_GEN
        if "Structured evidence:" in text and '"sufficient"' in text:
            self._sufficiency_idx += 1
            return _SUFFICIENCY
        if '"key_findings"' in text and '"confidence"' in text and '"gaps"' in text:
            self._intermediate_idx += 1
            return _INTERMEDIATE_JSON
        if '"citations"' in text and '"content"' in text:
            self._section_idx += 1
            return _SECTION_JSON
        if '"quality_score"' in text and '"needs_rewrite"' in text:
            self._validation_idx += 1
            return _VALIDATION_JSON
        if "RACE framework" in text:
            return _RACE_JSON
        if '"citation_accuracy"' in text and '"effective_citations"' in text:
            return _FACT_JSON
        if "Summary of the report" in text or "summary" in text.lower():
            return "Summary of the research findings on AI agent frameworks."
        return self._next_linear_content()

    async def chat(self, messages: list, **kwargs: Any) -> LLMResponse:
        prompt = str(messages[0].get("content", "")) if messages else ""
        return LLMResponse(content=self._route_content(prompt), finish_reason="stop", model="mock")

    async def stream_chat(self, messages: list, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        prompt = str(messages[0].get("content", "")) if messages else ""
        yield StreamChunk(content_delta=self._route_content(prompt))


def _mock_web_search() -> WebSearchService:
    svc = AsyncMock(spec=WebSearchService)
    svc.search = AsyncMock(
        return_value=WebSearchResponse(
            query="test",
            provider="mock",
            results=[
                WebSearchResult(
                    title=f"Result {i}",
                    url=f"https://example.com/{i}",
                    snippet=f"Snippet about AI frameworks {i}",
                    content=f"Full content about AI frameworks {i}",
                )
                for i in range(5)
            ],
            metadata=WebSearchMetadata(
                cached=False, cache_hit=False, latency_ms=10, provider="mock"
            ),
        )
    )
    return svc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fast_url_validation(monkeypatch: pytest.MonkeyPatch):
    """Keep runtime lifecycle tests deterministic and network-independent.

    These tests focus on lifecycle / event ordering. URL reachability is covered
    in URL validator tests, so we stub validation here to avoid 10s network
    timeout tails from external URL probes.
    """

    async def _validate(_self, urls: list[str]) -> URLValidationReport:
        unique_urls = list(dict.fromkeys(urls))
        return URLValidationReport(
            total=len(unique_urls),
            reachable=len(unique_urls),
            unreachable=0,
            error_rate=0.0,
            results=[
                URLValidationResult(url=url, reachable=True, status_code=200) for url in unique_urls
            ],
        )

    monkeypatch.setattr(URLValidator, "validate", _validate)


class TestRuntimeLifecycle:
    """Full plan → execute → report → memory extraction lifecycle."""

    async def test_standard_full_run(self):
        llm = _MockLLM(_build_responses())
        ws = _mock_web_search()
        session = ResearchRuntime(
            llm_adapter=llm,
            web_search=ws,
            settings=ResearchSettings(depth="standard", max_agents=1),
        )

        plan = await session.start("Compare AI agent frameworks in 2026")
        assert plan.status == PlanStatus.DRAFT
        assert 3 <= len(plan.sub_questions) <= 8
        assert session.status == ResearchStatus.PLAN_READY

        plan = await session.confirm_plan()
        assert plan.status == PlanStatus.CONFIRMED

        await session.execute()
        assert session.status == ResearchStatus.COMPLETED

        report = await session.get_report()
        assert len(report.sections) == 5
        assert report.summary
        assert report.metadata.section_count == 5
        assert report.metadata.source_count > 0
        assert report.metadata.quality_overall is not None

        score = session.quality_score
        assert score is not None
        assert score.race.overall > 0
        assert score.fact.citation_accuracy > 0

    async def test_autonomous_full_run(self):
        llm = _MockLLM(_build_responses())
        ws = _mock_web_search()
        settings = ResearchSettings(orchestration_mode="autonomous", max_agents=3)
        session = ResearchRuntime(llm_adapter=llm, web_search=ws, settings=settings)

        await session.start("AI frameworks")
        await session.confirm_plan()
        await session.execute()

        assert session.status == ResearchStatus.COMPLETED
        report = await session.get_report()
        assert len(report.sections) >= 1


class TestPlanEditing:
    async def test_edit_and_confirm(self):
        llm = _MockLLM(_build_responses())
        ws = _mock_web_search()
        session = ResearchRuntime(llm_adapter=llm, web_search=ws)

        plan = await session.start("test")
        assert plan.version == 1

        edit = PlanEdit(op=PlanEditOperation.ADD, target_question="Extra question?")
        plan = await session.edit_plan([edit])
        assert plan.version == 2
        assert len(plan.sub_questions) == 6

        edit2 = PlanEdit(
            op=PlanEditOperation.DELETE,
            question_id=plan.sub_questions[-1].question_id,
        )
        plan = await session.edit_plan([edit2])
        assert plan.version == 3
        assert len(plan.sub_questions) == 5


class TestEventSequence:
    async def test_events_monotonic(self):
        events: list[AgentEvent] = []
        emitter = EventEmitter()
        emitter.on_any(lambda e: _capture(events, e))

        llm = _MockLLM(_build_responses())
        ws = _mock_web_search()
        session = ResearchRuntime(
            llm_adapter=llm,
            web_search=ws,
            event_emitter=emitter,
        )

        await session.start("test")
        await session.confirm_plan()
        await session.execute()

        research_events = [e for e in events if e.data.get("research_event")]
        sequences = [e.data["sequence"] for e in research_events]
        assert sequences == sorted(sequences)
        assert len(set(sequences)) == len(sequences)

        event_types = [e.data.get("research_event") for e in research_events]
        assert "research.plan_generated" in event_types
        assert "research.plan_confirmed" in event_types
        assert "research.step_started" in event_types
        assert "research.source_found" in event_types
        assert "research.step_completed" in event_types
        assert "research.report_section" in event_types
        assert "research.completed" in event_types

    async def test_autonomous_emits_agent_events(self):
        events: list[AgentEvent] = []
        emitter = EventEmitter()
        emitter.on_any(lambda e: _capture(events, e))

        llm = _MockLLM(_build_responses())
        ws = _mock_web_search()
        settings = ResearchSettings(orchestration_mode="autonomous", max_agents=3)
        session = ResearchRuntime(
            llm_adapter=llm,
            web_search=ws,
            settings=settings,
            event_emitter=emitter,
        )

        await session.start("test")
        await session.confirm_plan()
        await session.execute()

        event_types = [e.data.get("research_event") for e in events]
        assert "research.agent_spawned" in event_types
        assert "research.agent_completed" in event_types


class TestMemoryExtraction:
    async def test_extract_after_completion(self):
        llm = _MockLLM(_build_responses())
        ws = _mock_web_search()
        session = ResearchRuntime(llm_adapter=llm, web_search=ws)

        await session.start("AI frameworks")
        await session.confirm_plan()
        await session.execute()

        candidates = await session.extract_memories()
        assert len(candidates) >= 3
        assert all(c.source_context == "deep_research" for c in candidates)
        assert any("research" in c.suggested_tags for c in candidates)


class TestProgress:
    async def test_progress_updates(self):
        llm = _MockLLM(_build_responses())
        ws = _mock_web_search()
        session = ResearchRuntime(llm_adapter=llm, web_search=ws)

        await session.start("test")
        prog = session.progress
        assert prog.status == ResearchStatus.PLAN_READY
        assert prog.total_steps == 5
        assert prog.completed_steps == 0

        await session.confirm_plan()
        await session.execute()

        prog = session.progress
        assert prog.status == ResearchStatus.COMPLETED
        assert prog.completed_steps == 5
        assert prog.sources_found > 0
        assert prog.elapsed_seconds >= 0
        assert prog.last_event_sequence > 0


class TestCancelFlow:
    async def test_cancel_after_start(self):
        llm = _MockLLM(_build_responses())
        ws = _mock_web_search()
        session = ResearchRuntime(llm_adapter=llm, web_search=ws)

        await session.start("test")
        await session.cancel("changed mind")
        assert session.status == ResearchStatus.CANCELLED
        assert session.progress.error == "changed mind"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _capture(lst: list, event: AgentEvent) -> None:
    lst.append(event)
