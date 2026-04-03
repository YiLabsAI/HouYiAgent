"""Unit tests for IntermediateReportGenerator."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from houyi.adapters.llm.base import LLMAdapter, LLMResponse, StreamChunk
from houyi.application.research.intermediate import (
    IntermediateReport,
    IntermediateReportGenerator,
    _parse_response,
)
from houyi.application.research.types import SearchResult, SourceReference

# -- Helpers -----------------------------------------------------------------


def _sources(n: int = 3) -> list[SourceReference]:
    return [
        SourceReference(
            url=f"https://example.com/{i}",
            title=f"Source {i}",
            snippet=f"Snippet about topic {i}",
        )
        for i in range(n)
    ]


def _search_result(qid: str = "q1", n_sources: int = 3) -> SearchResult:
    return SearchResult(
        question_id=qid,
        sources=_sources(n_sources),
        summary="Found relevant information about the topic.",
        coverage_score=0.8,
    )


class _MockLLM(LLMAdapter):
    def __init__(self, response: str) -> None:
        self._response = response
        self.call_count = 0

    async def chat(self, messages: list, **kw: Any) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(content=self._response, finish_reason="stop", model="mock")

    async def stream_chat(self, messages: list, **kw: Any) -> AsyncIterator[StreamChunk]:
        self.call_count += 1
        yield StreamChunk(content_delta=self._response)


class _FailingLLM(LLMAdapter):
    async def chat(self, messages: list, **kw: Any) -> LLMResponse:
        raise RuntimeError("LLM failed")

    async def stream_chat(self, messages: list, **kw: Any) -> AsyncIterator[StreamChunk]:
        raise RuntimeError("LLM failed")
        yield  # unreachable — satisfies async generator type


_VALID_RESPONSE = json.dumps(
    {
        "analysis": "The analysis with [ref_001] citation.",
        "key_findings": ["Finding 1", "Finding 2"],
        "confidence": 0.85,
        "gaps": ["More data needed on X"],
    }
)


# -- Generator ---------------------------------------------------------------


class TestGenerate:
    async def test_happy_path(self):
        gen = IntermediateReportGenerator(llm_adapter=_MockLLM(_VALID_RESPONSE))
        sr = _search_result()
        ir = await gen.generate(sr, "What is AI?", "AI Research")
        assert ir.question_id == "q1"
        assert ir.question == "What is AI?"
        assert "ref_001" in ir.analysis
        assert len(ir.key_findings) == 2
        assert ir.confidence == 0.85
        assert len(ir.gaps) == 1

    async def test_preserves_sources(self):
        gen = IntermediateReportGenerator(llm_adapter=_MockLLM(_VALID_RESPONSE))
        sr = _search_result(n_sources=5)
        ir = await gen.generate(sr, "Q", "Topic")
        assert len(ir.sources) == 5

    async def test_llm_failure_returns_fallback(self):
        gen = IntermediateReportGenerator(llm_adapter=_FailingLLM())
        sr = _search_result()
        ir = await gen.generate(sr, "Q", "Topic")
        assert ir.question_id == "q1"
        assert ir.confidence == 0.3
        assert ir.analysis == sr.summary

    async def test_empty_sources_handled(self):
        gen = IntermediateReportGenerator(llm_adapter=_MockLLM(_VALID_RESPONSE))
        sr = SearchResult(question_id="q1", sources=[], summary="No sources found")
        ir = await gen.generate(sr, "Q", "Topic")
        assert ir.question_id == "q1"
        assert len(ir.sources) == 0


class TestGenerateBatch:
    async def test_batch_generates_all(self):
        llm = _MockLLM(_VALID_RESPONSE)
        gen = IntermediateReportGenerator(llm_adapter=llm)
        results = [_search_result("q1"), _search_result("q2"), _search_result("q3")]
        questions = {"q1": "What is X?", "q2": "What is Y?", "q3": "What is Z?"}
        reports = await gen.generate_batch(results, questions, "Research")
        assert len(reports) == 3
        assert llm.call_count == 3

    async def test_batch_empty_input(self):
        gen = IntermediateReportGenerator(llm_adapter=_MockLLM(_VALID_RESPONSE))
        reports = await gen.generate_batch([], {}, "Topic")
        assert reports == []

    async def test_batch_missing_question_text(self):
        gen = IntermediateReportGenerator(llm_adapter=_MockLLM(_VALID_RESPONSE))
        results = [_search_result("q1")]
        reports = await gen.generate_batch(results, {}, "Topic")
        assert len(reports) == 1
        assert reports[0].question == ""


# -- Parser ------------------------------------------------------------------


class TestParseResponse:
    def test_valid_json(self):
        ir = _parse_response(_VALID_RESPONSE, "q1", "Q?", _sources())
        assert ir.question_id == "q1"
        assert ir.confidence == 0.85
        assert len(ir.key_findings) == 2

    def test_json_in_code_fence(self):
        fenced = f"```json\n{_VALID_RESPONSE}\n```"
        ir = _parse_response(fenced, "q1", "Q?", _sources())
        assert ir.confidence == 0.85

    def test_malformed_json_fallback(self):
        ir = _parse_response("Not valid JSON at all", "q1", "Q?", _sources())
        assert ir.question_id == "q1"
        assert ir.confidence == 0.3
        assert ir.analysis.startswith("Not valid")

    def test_partial_json(self):
        partial = json.dumps({"analysis": "Some analysis"})
        ir = _parse_response(partial, "q1", "Q?", _sources())
        assert ir.analysis == "Some analysis"
        assert ir.confidence == 0.5
        assert ir.key_findings == []

    def test_empty_content(self):
        ir = _parse_response("", "q1", "Q?", _sources())
        assert ir.confidence == 0.3

    def test_confidence_clamping(self):
        resp = json.dumps({"analysis": "A", "confidence": "not a number"})
        ir = _parse_response(resp, "q1", "Q?", _sources())
        assert ir.confidence == 0.3


# -- Model -------------------------------------------------------------------


class TestIntermediateReportModel:
    def test_default_values(self):
        ir = IntermediateReport()
        assert ir.question_id == ""
        assert ir.confidence == 0.5
        assert ir.key_findings == []
        assert ir.gaps == []

    def test_serialization(self):
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
        restored = IntermediateReport.model_validate(data)
        assert restored == ir
