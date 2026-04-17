"""Unit tests for ValidationAgent — post-report quality inspection."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from houyi.adapters.llm.base import LLMAdapter, LLMResponse, StreamChunk
from houyi.application.research.types import (
    OutlineSection,
    ReportSection,
    ResearchPlan,
    ResearchReport,
    SubQuestion,
)
from houyi.application.research.validation import (
    SectionValidation,
    ValidationAgent,
    ValidationReport,
    ValidationSectionContext,
    _needs_identity_repair,
    _parse_validation,
    _section_disambiguation_needed,
)


class _MockLLM(LLMAdapter):
    def __init__(self, response: str) -> None:
        self._response = response
        self.messages: list[list] = []

    async def chat(self, messages: list, **kw: Any) -> LLMResponse:
        self.messages.append(messages)
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

    async def test_validate_selected_sections(self):
        llm = _MockLLM(_GOOD_SECTION)
        agent = ValidationAgent(llm)
        report = _report(3)
        result = await agent.validate(report, "test query", section_titles={"Section 1"})
        assert len(result.sections) == 1
        assert result.sections[0].title == "Section 1"
        assert len(llm.messages) == 1

    async def test_validate_caps_content(self):
        llm = _MockLLM(_GOOD_SECTION)
        agent = ValidationAgent(llm)
        report = ResearchReport(
            report_id="rpt_long",
            sections=[ReportSection(section_id="sec_long", title="Long", content="A" * 5000)],
            summary="Test summary",
        )
        await agent.validate(report, "test query", content_char_limit=120)
        prompt = llm.messages[0][0]["content"]
        assert "A" * 121 not in prompt

    async def test_validate_uses_plan_context(self):
        llm = _MockLLM(_GOOD_SECTION)
        agent = ValidationAgent(llm)
        report = ResearchReport(
            report_id="rpt_ctx",
            sections=[
                ReportSection(
                    section_id="sec_1", title="Overview", content="Coverage text [ref_001]."
                ),
                ReportSection(section_id="sec_2", title="Risks", content="Risk text [ref_002]."),
            ],
            summary="Summary",
        )
        plan = ResearchPlan(
            query="Test query",
            outline=[
                OutlineSection(
                    title="Overview",
                    objective="Explain the landscape",
                    coverage_contract={
                        "must_cover_facets": [
                            {"name": "market size", "intent": "summarize current scale"}
                        ],
                        "required_caveats": ["note data limitations"],
                    },
                ),
                OutlineSection(title="Risks", objective="Summarize key risks"),
            ],
        )
        await agent.validate(report, "test query", plan=plan, section_titles={"Overview"})
        prompt = llm.messages[0][0]["content"]
        assert "Section objective: Explain the landscape" in prompt
        assert "Section position in report: 1 of 2" in prompt
        assert "Next section: Risks" in prompt
        assert "market size" in prompt
        assert "note data limitations" in prompt

    async def test_validate_adds_fallback_queries(self):
        response = json.dumps(
            {
                "quality_score": 20,
                "has_citations": False,
                "issues": ["Missing evidence"],
                "needs_rewrite": True,
                "suggested_queries": [],
                "reasoning": "Thin support.",
            }
        )
        llm = _MockLLM(response)
        agent = ValidationAgent(llm)
        report = ResearchReport(
            report_id="rpt_fallback",
            sections=[ReportSection(section_id="sec_1", title="Overview", content="Thin content")],
            summary="Summary",
        )
        plan = ResearchPlan(
            query="Energy transition",
            outline=[
                OutlineSection(
                    title="Overview",
                    objective="Compare growth and cost trends",
                    coverage_contract={
                        "must_cover_facets": [
                            {"name": "growth", "intent": "quantify adoption"},
                            {"name": "cost", "intent": "track cost decline"},
                        ]
                    },
                )
            ],
        )
        result = await agent.validate(report, "Energy transition", plan=plan)
        assert result.sections[0].needs_rewrite is True
        assert len(result.sections[0].suggested_queries) == 2
        assert "Energy transition" in result.sections[0].suggested_queries[0]

    async def test_validate_adds_identity_fallback(self):
        response = json.dumps(
            {
                "quality_score": 20,
                "has_citations": False,
                "issues": ["Ambiguous identity support"],
                "needs_rewrite": True,
                "suggested_queries": [],
                "reasoning": "Identity evidence is thin.",
            }
        )
        llm = _MockLLM(response)
        agent = ValidationAgent(llm)
        report = ResearchReport(
            report_id="rpt_identity",
            sections=[ReportSection(section_id="sec_1", title="Profile", content="Thin content")],
            summary="Summary",
        )
        plan = ResearchPlan(
            query="Who is Sample Person?",
            outline=[
                OutlineSection(
                    title="Profile",
                    objective="Confirm the intended entity and current role",
                    coverage_contract={
                        "must_cover_facets": [
                            {"name": "identity", "intent": "disambiguate same-name candidates"}
                        ],
                        "required_caveats": [
                            "disambiguate same-name entities before making claims"
                        ],
                    },
                )
            ],
        )
        result = await agent.validate(report, "Who is Sample Person?", plan=plan)
        assert result.sections[0].needs_rewrite is True
        assert "official profile" in result.sections[0].suggested_queries[0]
        assert "disambiguation" in result.sections[0].suggested_queries[1]


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


class TestIdentityRepair:
    def test_uses_metadata_flag(self):
        context = ValidationSectionContext(
            objective="General overview",
            section_position="1 of 3",
            previous_section="(none)",
            next_section="Analysis",
            coverage_facets=["market analysis"],
            required_caveats=[],
            disambiguation_needed=True,
        )
        section = ReportSection(title="Overview", content="Some content")
        assert _needs_identity_repair(context, section) is True

    def test_falls_back_to_hints(self):
        context = ValidationSectionContext(
            objective="Confirm identity of person X",
            section_position="1 of 3",
            previous_section="(none)",
            next_section="Analysis",
            coverage_facets=["identity"],
            required_caveats=[],
            disambiguation_needed=False,
        )
        section = ReportSection(title="Identity", content="Some content")
        assert _needs_identity_repair(context, section) is True

    def test_skips_non_identity(self):
        context = ValidationSectionContext(
            objective="Market analysis",
            section_position="1 of 3",
            previous_section="(none)",
            next_section="Conclusion",
            coverage_facets=["market size"],
            required_caveats=[],
            disambiguation_needed=False,
        )
        section = ReportSection(title="Market", content="Some content")
        assert _needs_identity_repair(context, section) is False

    def test_reads_plan_disambiguation(self):
        question = SubQuestion(
            question="Who is X?",
            query_type="entity",
            disambiguation_needed=True,
        )
        outline = OutlineSection(
            title="Identity",
            objective="Confirm",
            related_question_ids=[question.question_id],
        )
        plan = ResearchPlan(query="Who is X?", sub_questions=[question], outline=[outline])
        assert _section_disambiguation_needed(outline, plan) is True

    def test_skips_clean_plan(self):
        question = SubQuestion(
            question="What is framework Y?",
            query_type="factual",
            disambiguation_needed=False,
        )
        outline = OutlineSection(
            title="Overview",
            objective="Survey",
            related_question_ids=[question.question_id],
        )
        plan = ResearchPlan(query="Framework Y", sub_questions=[question], outline=[outline])
        assert _section_disambiguation_needed(outline, plan) is False
