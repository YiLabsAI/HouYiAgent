"""Tests for intermediate reports: generator, parser, model, and pipeline orchestration."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from houyi.adapters.llm.base import LLMAdapter, LLMResponse, StreamChunk
from houyi.application.research.quality import QualityScore
from houyi.application.research.runtime.intermediate import (
    IntermediateReport,
    IntermediateReportGenerator,
    _parse_response,
)
from houyi.application.research.runtime.report_pipeline import ReportPipeline
from houyi.application.research.types import (
    AggregatedSources,
    Citation,
    OutlineSection,
    ReportSection,
    ResearchPlan,
    ResearchReport,
    ResearchSettings,
    SearchResult,
    SourceReference,
    SubQuestion,
)

from .conftest import MockLLM

# --- shared JSON / fixtures (same style as test_report._SECTION_JSON) ---------

_IR_JSON = json.dumps(
    {
        "analysis": "The analysis with [ref_001] citation.",
        "key_findings": ["Finding 1", "Finding 2"],
        "confidence": 0.85,
        "gaps": ["More data needed on X"],
    }
)

_IR_JSON_B = json.dumps({"analysis": "fresh", "key_findings": ["f"], "confidence": 0.6, "gaps": []})


def _refs(n: int = 3) -> list[SourceReference]:
    return [
        SourceReference(
            url=f"https://example.com/{i}",
            title=f"Source {i}",
            snippet=f"Snippet about topic {i}",
        )
        for i in range(n)
    ]


def _result(qid: str = "q1", n_sources: int = 3) -> SearchResult:
    return SearchResult(
        question_id=qid,
        sources=_refs(n_sources),
        summary="Found relevant information about the topic.",
        coverage_score=0.8,
    )


class _FailingLLM(LLMAdapter):
    async def chat(self, messages: list, **kw: Any) -> LLMResponse:
        raise RuntimeError("LLM failed")

    async def stream_chat(self, messages: list, **kw: Any) -> AsyncIterator[StreamChunk]:
        raise RuntimeError("LLM failed")
        yield


class TestGenerate:
    async def test_happy_path(self) -> None:
        gen = IntermediateReportGenerator(llm_adapter=MockLLM(responses=[_IR_JSON]))
        ir = await gen.generate(_result(), "What is AI?", "AI Research")
        assert ir.question_id == "q1"
        assert ir.question == "What is AI?"
        assert "ref_001" in ir.analysis
        assert len(ir.key_findings) == 2
        assert ir.confidence == 0.85
        assert len(ir.gaps) == 1

    async def test_preserves_sources(self) -> None:
        gen = IntermediateReportGenerator(llm_adapter=MockLLM(responses=[_IR_JSON]))
        ir = await gen.generate(_result(n_sources=5), "Q", "Topic")
        assert len(ir.sources) == 5

    async def test_llm_failure_returns_fallback(self) -> None:
        gen = IntermediateReportGenerator(llm_adapter=_FailingLLM())
        sr = _result()
        ir = await gen.generate(sr, "Q", "Topic")
        assert ir.question_id == "q1"
        assert ir.confidence == 0.3
        assert ir.analysis == sr.summary

    async def test_empty_sources(self) -> None:
        gen = IntermediateReportGenerator(llm_adapter=MockLLM(responses=[_IR_JSON]))
        sr = SearchResult(question_id="q1", sources=[], summary="No sources found")
        ir = await gen.generate(sr, "Q", "Topic")
        assert ir.question_id == "q1"
        assert len(ir.sources) == 0


class TestGenerateBatch:
    async def test_all_questions(self) -> None:
        llm = MockLLM(responses=[_IR_JSON, _IR_JSON, _IR_JSON])
        gen = IntermediateReportGenerator(llm_adapter=llm)
        results = [_result("q1"), _result("q2"), _result("q3")]
        questions = {"q1": "What is X?", "q2": "What is Y?", "q3": "What is Z?"}
        reports = await gen.generate_batch(results, questions, "Research")
        assert len(reports) == 3
        assert llm._call_count == 3

    async def test_empty_input(self) -> None:
        gen = IntermediateReportGenerator(llm_adapter=MockLLM(responses=[_IR_JSON]))
        assert await gen.generate_batch([], {}, "Topic") == []

    async def test_missing_question_text(self) -> None:
        gen = IntermediateReportGenerator(llm_adapter=MockLLM(responses=[_IR_JSON]))
        reports = await gen.generate_batch([_result("q1")], {}, "Topic")
        assert len(reports) == 1
        assert reports[0].question == ""


class TestParseResponse:
    def test_valid_json(self) -> None:
        ir = _parse_response(_IR_JSON, "q1", "Q?", _refs())
        assert ir.question_id == "q1"
        assert ir.confidence == 0.85
        assert len(ir.key_findings) == 2

    def test_json_in_fence(self) -> None:
        fenced = f"```json\n{_IR_JSON}\n```"
        ir = _parse_response(fenced, "q1", "Q?", _refs())
        assert ir.confidence == 0.85

    def test_malformed_fallback(self) -> None:
        ir = _parse_response("Not valid JSON at all", "q1", "Q?", _refs())
        assert ir.question_id == "q1"
        assert ir.confidence == 0.3
        assert ir.analysis.startswith("Not valid")

    def test_partial_json(self) -> None:
        partial = json.dumps({"analysis": "Some analysis"})
        ir = _parse_response(partial, "q1", "Q?", _refs())
        assert ir.analysis == "Some analysis"
        assert ir.confidence == 0.5
        assert ir.key_findings == []

    def test_empty_content(self) -> None:
        ir = _parse_response("", "q1", "Q?", _refs())
        assert ir.confidence == 0.3

    def test_confidence_clamping(self) -> None:
        resp = json.dumps({"analysis": "A", "confidence": "not a number"})
        ir = _parse_response(resp, "q1", "Q?", _refs())
        assert ir.confidence == 0.3


class TestIntermediateReport:
    def test_defaults(self) -> None:
        ir = IntermediateReport()
        assert ir.question_id == ""
        assert ir.confidence == 0.5
        assert ir.key_findings == []
        assert ir.gaps == []

    def test_roundtrip(self) -> None:
        ir = IntermediateReport(
            question_id="q1",
            question="What?",
            analysis="Analysis text",
            key_findings=["f1"],
            confidence=0.9,
        )
        data = ir.model_dump()
        assert data["question_id"] == "q1"
        assert data["confidence"] == 0.9
        assert IntermediateReport.model_validate(data) == ir


def _pipeline(ir_gen: IntermediateReportGenerator) -> ReportPipeline:
    reporter = MagicMock()
    reporter.complete_summary = AsyncMock(return_value=0.0)
    return ReportPipeline(
        reporter=reporter,
        evaluator=MagicMock(),
        url_validator=MagicMock(),
        conflict_resolver=MagicMock(),
        intermediate_gen=ir_gen,
        emit=AsyncMock(),
    )


def _plan() -> ResearchPlan:
    return ResearchPlan(
        query="topic",
        outline=[OutlineSection(title="Overview", objective="Summarize the topic")],
        sub_questions=[SubQuestion(question_id="q1", question="What about AI?")],
        settings=ResearchSettings(depth="standard"),
    )


def _report() -> ResearchReport:
    source = SourceReference(reference_id="ref_keep", url="https://example.com/keep", title="Keep")
    return ResearchReport(
        title="Demo",
        sections=[
            ReportSection(
                title="Overview",
                content="Supported body text with cited analysis [ref_keep]. " * 4,
                citations=[Citation(reference_id=source.reference_id)],
            )
        ],
        references=[source],
    )


class TestPipelineIntermediates:
    async def test_checkpoint_reuses_without_llm(self) -> None:
        """Retry: cached intermediate + checkpoint qid must not call generate again."""
        llm = MockLLM(responses=[_IR_JSON])
        ir_gen = IntermediateReportGenerator(llm)
        spy = AsyncMock(wraps=ir_gen.generate)
        ir_gen.generate = spy  # type: ignore[method-assign]

        cached = IntermediateReport(
            question_id="q1",
            question="sub",
            analysis="from previous run",
            confidence=0.88,
        )
        sr = SearchResult(
            question_id="q1",
            sources=[SourceReference(url="https://a.example", title="T", snippet="s")],
        )
        out = await _pipeline(ir_gen).generate_intermediates(
            [sr],
            [SubQuestion(question_id="q1", question="What about AI?")],
            "topic",
            reuse_by_question_id={"q1": cached},
            checkpoint_question_ids=frozenset({"q1"}),
        )

        assert len(out) == 1
        assert out[0].analysis == "from previous run"
        spy.assert_not_called()

    async def test_generates_when_not_checkpointed(self) -> None:
        llm = MockLLM(responses=[_IR_JSON_B])
        ir_gen = IntermediateReportGenerator(llm)
        sr = SearchResult(
            question_id="q2",
            sources=[SourceReference(url="https://b.example", title="T2", snippet="s2")],
        )
        stale = IntermediateReport(question_id="q1", question="old", analysis="old")

        out = await _pipeline(ir_gen).generate_intermediates(
            [sr],
            [SubQuestion(question_id="q2", question="New Q?")],
            "topic",
            reuse_by_question_id={"q1": stale},
            checkpoint_question_ids=frozenset({"q1"}),
        )

        assert len(out) == 1
        assert out[0].question_id == "q2"
        assert out[0].analysis == "fresh"

    async def test_limits_intermediate_targets(self) -> None:
        llm = MockLLM(responses=[_IR_JSON, _IR_JSON, _IR_JSON])
        ir_gen = IntermediateReportGenerator(llm)
        spy = AsyncMock(wraps=ir_gen.generate)
        ir_gen.generate = spy  # type: ignore[method-assign]
        search_results = [
            SearchResult(
                question_id=f"q{i}", sources=_refs(3), summary=f"summary {i}", coverage_score=0.8
            )
            for i in range(5)
        ]
        questions = [
            SubQuestion(question_id=f"q{i}", question=f"Question {i}", priority=5 - i)
            for i in range(5)
        ]

        out = await _pipeline(ir_gen).generate_intermediates(
            search_results,
            questions,
            "topic",
            depth="standard",
        )

        assert len(out) == 3
        assert spy.await_count == 3


class TestPipelineRuntime:
    async def test_url_degrades_broken(self) -> None:
        """URL validation degrades reliability but keeps citations/references intact."""
        pipe = _pipeline(IntermediateReportGenerator(MockLLM(responses=[_IR_JSON])))
        broken = SourceReference(
            reference_id="ref_dead",
            url="https://example.com/dead",
            title="Dead",
            reliability_score=0.9,
        )
        cit = Citation(reference_id="ref_dead")
        report = ResearchReport(
            title="Demo",
            sections=[
                ReportSection(
                    title="Overview",
                    content="Body",
                    citations=[cit],
                )
            ],
            references=[broken],
        )
        aggregated = AggregatedSources(sources=[broken])
        pipe._url_validator.validate = AsyncMock(
            return_value=MagicMock(
                results=[MagicMock(url=broken.url, reachable=False)],
                total=1,
                reachable=0,
                unreachable=1,
                error_rate=1.0,
            )
        )

        await pipe._validate_urls(aggregated, report)

        assert aggregated.sources[0].reliability_score == pytest.approx(0.6)
        # Citations and references are preserved for RACE scoring;
        # FACT evaluation handles URL validity independently.
        assert len(report.sections[0].citations) == 1
        assert len(report.references) == 1

    async def test_run_skip_failures(self) -> None:
        pipe = _pipeline(IntermediateReportGenerator(MockLLM(responses=[_IR_JSON])))
        report = _report()
        pipe._reporter.generate = AsyncMock(
            return_value=(report, {"report_sections_ms": 100.0, "report_summary_ms": 20.0})
        )
        pipe._conflict_resolver.detect = AsyncMock(return_value=[])
        pipe._url_validator.validate = AsyncMock(side_effect=RuntimeError("url fail"))
        pipe._evaluator.evaluate_with_breakdown = AsyncMock(
            side_effect=RuntimeError("quality fail")
        )

        result = await pipe.run(
            _plan(),
            AggregatedSources(sources=report.references),
            [_result()],
            intermediate_reports=None,
            settings=ResearchSettings(depth="standard"),
        )

        assert result.report is report
        assert result.quality is None

    async def test_disables_inline_quality(self) -> None:
        pipe = _pipeline(IntermediateReportGenerator(MockLLM(responses=[_IR_JSON])))
        report = _report()
        pipe._reporter.generate = AsyncMock(
            return_value=(
                report,
                {
                    "report_sections_ms": 80.0,
                    "report_summary_ms": 10.0,
                    "section_input_metrics": [
                        {
                            "section_id": "sec_1",
                            "title": "Overview",
                            "relevant_source_count": 8,
                            "selected_source_count": 8,
                            "intermediate_context_chars": 400,
                        }
                    ],
                },
            )
        )
        pipe._conflict_resolver.detect = AsyncMock(return_value=[])
        pipe._url_validator.validate = AsyncMock(
            return_value=MagicMock(results=[], total=0, reachable=0, unreachable=0, error_rate=0.0)
        )
        pipe._evaluator.evaluate_with_breakdown = AsyncMock(
            return_value=(QualityScore(overall=88.0), {"quality_eval_total_ms": 1.0})
        )

        result = await pipe.run(
            _plan(),
            AggregatedSources(sources=report.references),
            [_result()],
            intermediate_reports=None,
            settings=ResearchSettings(depth="standard", enable_quality_evaluation=False),
        )

        assert pipe._evaluator.evaluate_with_breakdown.await_count == 0
        assert result.quality is None
        assert result.phase_timings_ms["quality_disabled"] == 1.0
