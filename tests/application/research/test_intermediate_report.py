"""Tests for intermediate reports: generator, parser, model, and pipeline orchestration."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from houyi.adapters.llm.base import LLMAdapter, LLMResponse, StreamChunk
from houyi.application.research.intermediate import (
    IntermediateReport,
    IntermediateReportGenerator,
    _parse_response,
)
from houyi.application.research.report_pipeline import ReportPipeline
from houyi.application.research.types import SearchResult, SourceReference, SubQuestion

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

_IR_JSON_B = json.dumps(
    {"analysis": "fresh", "key_findings": ["f"], "confidence": 0.6, "gaps": []}
)


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
    return ReportPipeline(
        reporter=MagicMock(),
        validator=MagicMock(),
        evaluator=MagicMock(),
        url_validator=MagicMock(),
        conflict_resolver=MagicMock(),
        intermediate_gen=ir_gen,
        web_search=MagicMock(),
        emit=AsyncMock(),
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
