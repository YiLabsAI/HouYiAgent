"""Tests for intermediate reports: generator, parser, model, and pipeline orchestration."""

from __future__ import annotations

import asyncio
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
from houyi.application.research.validation import SectionValidation, ValidationReport

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
    reporter._generate_section = AsyncMock(
        return_value=ReportSection(
            title="Overview",
            content="Repaired section with supported findings [ref_keep]. " * 4,
            citations=[Citation(reference_id="ref_keep")],
        )
    )
    return ReportPipeline(
        reporter=reporter,
        validator=MagicMock(),
        evaluator=MagicMock(),
        url_validator=MagicMock(),
        conflict_resolver=MagicMock(),
        intermediate_gen=ir_gen,
        web_search=MagicMock(),
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
    async def test_url_remove_broken(self) -> None:
        pipe = _pipeline(IntermediateReportGenerator(MockLLM(responses=[_IR_JSON])))
        broken = SourceReference(
            reference_id="ref_dead",
            url="https://example.com/dead",
            title="Dead",
            reliability_score=0.9,
        )
        report = ResearchReport(
            title="Demo",
            sections=[
                ReportSection(
                    title="Overview",
                    content="Body",
                    citations=[Citation(reference_id="ref_dead")],
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
        assert report.sections[0].citations == []
        assert report.references == []

    async def test_run_skip_failures(self) -> None:
        pipe = _pipeline(IntermediateReportGenerator(MockLLM(responses=[_IR_JSON])))
        report = _report()
        pipe._reporter.generate = AsyncMock(
            return_value=(report, {"report_sections_ms": 100.0, "report_summary_ms": 20.0})
        )
        pipe._conflict_resolver.detect = AsyncMock(return_value=[])
        pipe._url_validator.validate = AsyncMock(side_effect=RuntimeError("url fail"))
        pipe._validator.validate = AsyncMock(side_effect=RuntimeError("validate fail"))
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
        assert result.validation is None
        assert result.quality is None

    async def test_run_repair_sections(self) -> None:
        pipe = _pipeline(IntermediateReportGenerator(MockLLM(responses=[_IR_JSON])))
        report = _report()
        repaired = ReportSection(
            title="Overview",
            content="Repaired section with supported findings [ref_keep]. " * 4,
            citations=[Citation(reference_id="ref_keep")],
        )
        pipe._reporter.generate = AsyncMock(
            return_value=(report, {"report_sections_ms": 100.0, "report_summary_ms": 20.0})
        )
        pipe._reporter._generate_section = AsyncMock(return_value=repaired)
        pipe._conflict_resolver.detect = AsyncMock(return_value=[])
        pipe._url_validator.validate = AsyncMock(
            return_value=MagicMock(results=[], total=0, reachable=0, unreachable=0, error_rate=0.0)
        )
        pipe._validator.validate = AsyncMock(
            return_value=ValidationReport(
                sections=[
                    SectionValidation(
                        title="Overview",
                        quality_score=20,
                        needs_rewrite=True,
                        suggested_queries=[],
                    )
                ],
                overall_score=20.0,
                sections_needing_rewrite=1,
            )
        )
        pre_quality = QualityScore(overall=40.0)
        post_quality = QualityScore(overall=88.0)
        pipe._evaluator.evaluate_with_breakdown = AsyncMock(
            side_effect=[
                (
                    pre_quality,
                    {
                        "quality_race_ms": 11.0,
                        "quality_fact_ms": 12.0,
                        "quality_combine_ms": 0.0,
                        "quality_eval_total_ms": 23.0,
                    },
                ),
                (
                    post_quality,
                    {
                        "quality_race_ms": 21.0,
                        "quality_fact_ms": 22.0,
                        "quality_combine_ms": 0.0,
                        "quality_eval_total_ms": 43.0,
                    },
                ),
            ]
        )

        result = await pipe.run(
            _plan(),
            AggregatedSources(sources=report.references),
            [_result()],
            intermediate_reports=None,
            settings=ResearchSettings(depth="standard"),
        )

        assert result.report.sections[0].content.startswith(
            "Repaired section with supported findings"
        )
        assert result.validation is not None
        assert result.quality is not None
        assert result.report.metadata.quality_overall == 88.0
        assert pipe._evaluator.evaluate_with_breakdown.await_count == 2
        assert result.phase_timings_ms["quality_pre_ms"] >= 0
        assert result.phase_timings_ms["quality_pre_race_ms"] == 11.0
        assert result.phase_timings_ms["quality_pre_fact_ms"] == 12.0
        assert result.phase_timings_ms["quality_pre_eval_total_ms"] == 23.0
        assert result.phase_timings_ms["quality_pre_score"] == 40.0
        assert result.phase_timings_ms["quality_ms"] >= 0
        assert result.phase_timings_ms["quality_race_ms"] == 21.0
        assert result.phase_timings_ms["quality_fact_ms"] == 22.0
        assert result.phase_timings_ms["quality_eval_total_ms"] == 43.0
        assert result.phase_timings_ms["quality_score"] == 88.0
        assert result.phase_timings_ms["repair_ms"] >= 0
        assert result.phase_timings_ms["repair_query_ms"] == 0.0
        assert result.phase_timings_ms["repair_extra_source_count"] == 0.0
        assert result.phase_timings_ms["repair_section_count"] == 1.0

    async def test_pre_post_repair(self) -> None:
        pipe = _pipeline(IntermediateReportGenerator(MockLLM(responses=[_IR_JSON])))
        report = _report()
        pipe._reporter.generate = AsyncMock(
            return_value=(report, {"report_sections_ms": 100.0, "report_summary_ms": 20.0})
        )
        pipe._reporter._generate_section = AsyncMock(
            return_value=ReportSection(
                title="Overview",
                content="Repaired section with supported findings [ref_keep]. " * 4,
                citations=[Citation(reference_id="ref_keep")],
            )
        )
        pipe._conflict_resolver.detect = AsyncMock(return_value=[])
        pipe._url_validator.validate = AsyncMock(
            return_value=MagicMock(results=[], total=0, reachable=0, unreachable=0, error_rate=0.0)
        )
        pipe._validator.validate = AsyncMock(
            return_value=ValidationReport(
                sections=[
                    SectionValidation(
                        title="Overview",
                        quality_score=20,
                        needs_rewrite=True,
                        suggested_queries=[],
                    )
                ],
                overall_score=20.0,
                sections_needing_rewrite=1,
            )
        )
        pipe._evaluator.evaluate_with_breakdown = AsyncMock(
            side_effect=[
                (
                    QualityScore(overall=40.0),
                    {
                        "quality_race_ms": 0.0,
                        "quality_fact_ms": 0.0,
                        "quality_combine_ms": 0.0,
                        "quality_eval_total_ms": 0.0,
                    },
                ),
                (
                    QualityScore(overall=88.0),
                    {
                        "quality_race_ms": 0.0,
                        "quality_fact_ms": 0.0,
                        "quality_combine_ms": 0.0,
                        "quality_eval_total_ms": 0.0,
                    },
                ),
            ]
        )

        await pipe.run(
            _plan(),
            AggregatedSources(sources=report.references),
            [_result()],
            intermediate_reports=None,
            settings=ResearchSettings(depth="standard"),
        )

        emitted_phases = [
            call.kwargs["phase"]
            for call in pipe._emit.await_args_list
            if call.args and call.args[0] == "research.pipeline_phase"
        ]
        assert "quality_evaluation_pre_repair" in emitted_phases
        assert "quality_evaluation_post_repair" in emitted_phases

    async def test_validation_parallel_with_url(self) -> None:
        """URL validation and report validation execute concurrently."""
        pipe = _pipeline(IntermediateReportGenerator(MockLLM(responses=[_IR_JSON])))
        report = _report()
        plan = _plan()

        url_started = asyncio.Event()
        validation_started = asyncio.Event()
        release = asyncio.Event()

        async def _blocking_url_validate(_urls: list[str]) -> Any:
            url_started.set()
            await release.wait()
            return MagicMock(results=[], total=0, reachable=0, unreachable=0, error_rate=0.0)

        async def _blocking_validation(
            _report: ResearchReport,
            _query: str,
            *,
            section_titles: set[str] | None = None,
            content_char_limit: int | None = None,
        ) -> ValidationReport:
            _ = (section_titles, content_char_limit)
            validation_started.set()
            await release.wait()
            return ValidationReport(sections=[], overall_score=100.0, sections_needing_rewrite=0)

        async def _release_when_both_started() -> None:
            await asyncio.wait_for(url_started.wait(), timeout=0.2)
            await asyncio.wait_for(validation_started.wait(), timeout=0.2)
            release.set()

        pipe._reporter.generate = AsyncMock(
            return_value=(report, {"report_sections_ms": 100.0, "report_summary_ms": 20.0})
        )
        pipe._conflict_resolver.detect = AsyncMock(return_value=[])
        pipe._url_validator.validate = AsyncMock(side_effect=_blocking_url_validate)
        pipe._validator.validate = AsyncMock(side_effect=_blocking_validation)
        pipe._evaluator.evaluate_with_breakdown = AsyncMock(
            return_value=(
                QualityScore(overall=88.0),
                {
                    "quality_race_ms": 0.0,
                    "quality_fact_ms": 0.0,
                    "quality_combine_ms": 0.0,
                    "quality_eval_total_ms": 0.0,
                },
            )
        )

        gate = asyncio.create_task(_release_when_both_started())
        result = await asyncio.wait_for(
            pipe.run(
                plan,
                AggregatedSources(sources=report.references),
                [_result()],
                intermediate_reports=None,
                settings=ResearchSettings(depth="standard"),
            ),
            timeout=0.6,
        )
        await gate

        assert url_started.is_set()
        assert validation_started.is_set()
        assert result.validation is not None

    async def test_repair_sections_parallel(self) -> None:
        """Multiple section repairs execute concurrently under semaphore."""
        pipe = _pipeline(IntermediateReportGenerator(MockLLM(responses=[_IR_JSON])))
        report = ResearchReport(
            title="Demo",
            sections=[
                ReportSection(title="Overview", content="Body A"),
                ReportSection(title="Risks", content="Body B"),
            ],
            references=[
                SourceReference(
                    reference_id="ref_keep", url="https://example.com/keep", title="Keep"
                )
            ],
        )
        plan = ResearchPlan(
            query="topic",
            outline=[
                OutlineSection(title="Overview", objective="Summarize"),
                OutlineSection(title="Risks", objective="Assess risks"),
            ],
            sub_questions=[SubQuestion(question_id="q1", question="What about AI?")],
            settings=ResearchSettings(depth="standard"),
        )

        release = asyncio.Event()
        both_started = asyncio.Event()
        started_count = 0

        async def _blocking_generate_section(
            _query: str,
            title: str,
            _objective: str,
            _sources: list[SourceReference],
            intermediate_context: str = "",
        ) -> ReportSection:
            nonlocal started_count
            _ = intermediate_context
            started_count += 1
            if started_count >= 2:
                both_started.set()
            await release.wait()
            return ReportSection(title=title, content=f"Repaired {title}")

        async def _release_when_parallel_started() -> None:
            await asyncio.wait_for(both_started.wait(), timeout=0.2)
            release.set()

        pipe._reporter.generate = AsyncMock(
            return_value=(report, {"report_sections_ms": 100.0, "report_summary_ms": 20.0})
        )
        pipe._reporter._generate_section = AsyncMock(side_effect=_blocking_generate_section)
        pipe._conflict_resolver.detect = AsyncMock(return_value=[])
        pipe._url_validator.validate = AsyncMock(
            return_value=MagicMock(results=[], total=0, reachable=0, unreachable=0, error_rate=0.0)
        )
        pipe._validator.validate = AsyncMock(
            return_value=ValidationReport(
                sections=[
                    SectionValidation(
                        title="Overview",
                        quality_score=20,
                        needs_rewrite=True,
                        suggested_queries=[],
                    ),
                    SectionValidation(
                        title="Risks",
                        quality_score=25,
                        needs_rewrite=True,
                        suggested_queries=[],
                    ),
                ],
                overall_score=22.5,
                sections_needing_rewrite=2,
            )
        )
        pipe._evaluator.evaluate_with_breakdown = AsyncMock(
            return_value=(
                QualityScore(overall=90.0),
                {
                    "quality_race_ms": 0.0,
                    "quality_fact_ms": 0.0,
                    "quality_combine_ms": 0.0,
                    "quality_eval_total_ms": 0.0,
                },
            )
        )

        gate = asyncio.create_task(_release_when_parallel_started())
        result = await asyncio.wait_for(
            pipe.run(
                plan,
                AggregatedSources(sources=report.references),
                [_result()],
                intermediate_reports=None,
                settings=ResearchSettings(depth="standard"),
            ),
            timeout=0.7,
        )
        await gate

        assert both_started.is_set()
        assert result.report.sections[0].content == "Repaired Overview"
        assert result.report.sections[1].content == "Repaired Risks"

    async def test_repair_search_parallel(self) -> None:
        pipe = _pipeline(IntermediateReportGenerator(MockLLM(responses=[_IR_JSON])))
        started: set[str] = set()
        release = asyncio.Event()
        both_started = asyncio.Event()

        async def _blocking_search(query: str, **_: Any) -> Any:
            started.add(query)
            if len(started) >= 2:
                both_started.set()
            await release.wait()
            return MagicMock(
                results=[
                    MagicMock(
                        url=f"https://example.com/{query}",
                        title=f"Title {query}",
                        snippet=f"Snippet {query}",
                    )
                ]
            )

        async def _release_when_parallel_started() -> None:
            await asyncio.wait_for(both_started.wait(), timeout=0.2)
            release.set()

        pipe._web_search.search = AsyncMock(side_effect=_blocking_search)

        gate = asyncio.create_task(_release_when_parallel_started())
        sources, elapsed_ms = await asyncio.wait_for(
            pipe._search_extra_sources_for_repair(["q1", "q2"]),
            timeout=0.5,
        )
        await gate

        assert both_started.is_set()
        assert len(sources) == 2
        assert elapsed_ms >= 0.0
        assert {source.title for source in sources} == {"Title q1", "Title q2"}

    async def test_repair_search_skips_failed(self) -> None:
        pipe = _pipeline(IntermediateReportGenerator(MockLLM(responses=[_IR_JSON])))

        async def _search(query: str, **_: Any) -> Any:
            if query == "bad":
                raise RuntimeError("boom")
            return MagicMock(
                results=[
                    MagicMock(
                        url="https://example.com/good",
                        title="Good",
                        snippet="Good snippet",
                    )
                ]
            )

        pipe._web_search.search = AsyncMock(side_effect=_search)

        sources, elapsed_ms = await pipe._search_extra_sources_for_repair(["bad", "good"])

        assert len(sources) == 1
        assert elapsed_ms >= 0.0
        assert sources[0].title == "Good"

    async def test_run_skips_pre_quality(self) -> None:
        pipe = _pipeline(IntermediateReportGenerator(MockLLM(responses=[_IR_JSON])))
        report = _report()
        pipe._reporter.generate = AsyncMock(
            return_value=(
                report,
                {
                    "report_sections_ms": 100.0,
                    "report_summary_ms": 20.0,
                    "section_input_metrics": [
                        {
                            "section_id": "sec_1",
                            "title": "Overview",
                            "relevant_source_count": 40,
                            "selected_source_count": 5,
                            "intermediate_context_chars": 1600,
                        }
                    ],
                },
            )
        )
        pipe._reporter._generate_section = AsyncMock(
            return_value=ReportSection(title="Overview", content="Repaired")
        )
        pipe._conflict_resolver.detect = AsyncMock(return_value=[])
        pipe._url_validator.validate = AsyncMock(
            return_value=MagicMock(results=[], total=0, reachable=0, unreachable=0, error_rate=0.0)
        )
        pipe._validator.validate = AsyncMock(
            return_value=ValidationReport(
                sections=[
                    SectionValidation(
                        title="Overview", quality_score=20, needs_rewrite=True, suggested_queries=[]
                    )
                ],
                overall_score=20.0,
                sections_needing_rewrite=1,
            )
        )
        pipe._evaluator.evaluate_with_breakdown = AsyncMock(
            return_value=(
                QualityScore(overall=88.0),
                {
                    "quality_race_ms": 21.0,
                    "quality_fact_ms": 22.0,
                    "quality_combine_ms": 0.0,
                    "quality_eval_total_ms": 43.0,
                },
            )
        )

        result = await pipe.run(
            _plan(),
            AggregatedSources(sources=report.references),
            [_result()],
            intermediate_reports=None,
            settings=ResearchSettings(depth="standard"),
        )

        assert pipe._evaluator.evaluate_with_breakdown.await_count == 1
        assert result.phase_timings_ms["quality_pre_skipped"] == 1.0
        assert result.phase_timings_ms["validation_target_sections"] == 1.0

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
        pipe._reporter._generate_section = AsyncMock(
            return_value=ReportSection(title="Overview", content="Repaired")
        )
        pipe._conflict_resolver.detect = AsyncMock(return_value=[])
        pipe._url_validator.validate = AsyncMock(
            return_value=MagicMock(results=[], total=0, reachable=0, unreachable=0, error_rate=0.0)
        )
        pipe._validator.validate = AsyncMock(
            return_value=ValidationReport(
                sections=[
                    SectionValidation(
                        title="Overview", quality_score=20, needs_rewrite=True, suggested_queries=[]
                    )
                ],
                overall_score=20.0,
                sections_needing_rewrite=1,
            )
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
        assert result.phase_timings_ms["quality_pre_disabled"] == 1.0
        assert result.phase_timings_ms["quality_disabled"] == 1.0
