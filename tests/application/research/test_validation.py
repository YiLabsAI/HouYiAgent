"""Unit tests for ValidationAgent — post-report quality inspection."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from houyi.adapters.llm.base import LLMAdapter, LLMResponse, StreamChunk
from houyi.application.research.types import ReportSection, ResearchReport
from houyi.application.research.validation import (
    SectionValidation,
    ValidationAgent,
    ValidationReport,
    _parse_validation,
)


class _MockLLM(LLMAdapter):
    def __init__(self, response: str) -> None:
        self._response = response

    async def chat(self, messages: list, **kw: Any) -> LLMResponse:
        return LLMResponse(content=self._response, finish_reason="stop", model="mock")

    async def stream_chat(self, messages: list, **kw: Any) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(content_delta=self._response)


class _FailingLLM(LLMAdapter):
    async def chat(self, messages: list, **kw: Any) -> LLMResponse:
        raise RuntimeError("LLM failed")

    async def stream_chat(self, messages: list, **kw: Any) -> AsyncIterator[StreamChunk]:
        raise RuntimeError("LLM failed")
        yield  # unreachable — satisfies async generator type


_GOOD_SECTION = json.dumps(
    {
        "quality_score": 85,
        "has_citations": True,
        "issues": [],
        "needs_rewrite": False,
        "suggested_queries": [],
        "reasoning": "Well-cited and comprehensive section.",
    }
)

_POOR_SECTION = json.dumps(
    {
        "quality_score": 25,
        "has_citations": False,
        "issues": ["No citations", "Thin content"],
        "needs_rewrite": True,
        "suggested_queries": ["more details on topic X"],
        "reasoning": "Section lacks substance and citations.",
    }
)


def _report(n_sections: int = 2) -> ResearchReport:
    sections = [
        ReportSection(
            section_id=f"sec_{i}",
            title=f"Section {i}",
            content=f"Content of section {i} with [ref_001] citation.",
        )
        for i in range(n_sections)
    ]
    return ResearchReport(
        report_id="rpt_test",
        sections=sections,
        summary="Test summary",
    )


class TestValidate:
    async def test_all_sections_pass(self):
        agent = ValidationAgent(_MockLLM(_GOOD_SECTION))
        report = _report(3)
        result = await agent.validate(report, "test query")
        assert len(result.sections) == 3
        assert result.sections_needing_rewrite == 0
        assert result.overall_score > 80

    async def test_poor_sections_flagged(self):
        agent = ValidationAgent(_MockLLM(_POOR_SECTION))
        report = _report(2)
        result = await agent.validate(report, "test query")
        assert result.sections_needing_rewrite == 2
        assert all(s.needs_rewrite for s in result.sections)

    async def test_threshold_controls_rewrite(self):
        agent = ValidationAgent(_MockLLM(_POOR_SECTION), quality_threshold=20)
        report = _report(1)
        result = await agent.validate(report, "test query")
        assert result.sections[0].needs_rewrite is False

    async def test_llm_failure_graceful(self):
        agent = ValidationAgent(_FailingLLM())
        report = _report(1)
        result = await agent.validate(report, "test query")
        assert len(result.sections) == 1
        assert result.sections[0].quality_score == 50

    async def test_empty_report(self):
        agent = ValidationAgent(_MockLLM(_GOOD_SECTION))
        report = ResearchReport(report_id="rpt_empty", sections=[], summary="")
        result = await agent.validate(report, "test query")
        assert result.sections == []
        assert result.overall_score == 0.0


class TestParseValidation:
    def test_valid_json(self):
        result = _parse_validation(_GOOD_SECTION)
        assert result.quality_score == 85
        assert result.has_citations is True

    def test_code_fence(self):
        fenced = f"```json\n{_POOR_SECTION}\n```"
        result = _parse_validation(fenced)
        assert result.quality_score == 25
        assert result.needs_rewrite is True

    def test_malformed(self):
        result = _parse_validation("not valid json")
        assert result.quality_score == 50

    def test_empty(self):
        result = _parse_validation("")
        assert result.quality_score == 50


class TestModels:
    def test_section_validation_defaults(self):
        sv = SectionValidation()
        assert sv.quality_score == 0
        assert sv.issues == []

    def test_validation_report_defaults(self):
        vr = ValidationReport()
        assert vr.overall_score == 0.0
        assert vr.sections == []
