"""Unit tests for QualityEvaluator."""

from __future__ import annotations

import json

from houyi.application.research.quality import QualityEvaluator
from houyi.application.research.types import (
    AggregatedSources,
    ReportSection,
    ResearchReport,
    SourceReference,
)

from .conftest import MockLLM

_RACE_JSON = json.dumps(
    {
        "comprehensiveness": {"score": 80, "reasoning": "Good coverage"},
        "depth": {"score": 70, "reasoning": "Adequate depth"},
        "instruction_following": {"score": 90, "reasoning": "On topic"},
        "readability": {"score": 85, "reasoning": "Well structured"},
    }
)

_FACT_JSON = json.dumps(
    {
        "citation_accuracy": 95.0,
        "effective_citations": 12,
        "details": [{"reference_id": "ref_001", "accurate": True, "reasoning": "Matches source"}],
    }
)


def _report() -> ResearchReport:
    return ResearchReport(
        title="Test Report",
        summary="A test.",
        sections=[
            ReportSection(title="Intro", content="Content [ref_001]."),
        ],
        references=[SourceReference(reference_id="ref_001", title="Src")],
    )


def _sources() -> AggregatedSources:
    return AggregatedSources(
        sources=[SourceReference(reference_id="ref_001", title="Src", snippet="excerpt")],
    )


class TestRaceEval:
    async def test_race_scores(self):
        llm = MockLLM(responses=[_RACE_JSON])
        ev = QualityEvaluator(llm)
        race = await ev.evaluate_race(_report())
        assert race.comprehensiveness == 80
        assert race.depth == 70
        assert race.readability == 85
        assert race.overall > 0

    async def test_race_malformed_fallback(self):
        llm = MockLLM(responses=["not json"])
        ev = QualityEvaluator(llm)
        race = await ev.evaluate_race(_report())
        assert race.overall == 0


class TestFactEval:
    async def test_fact_scores(self):
        llm = MockLLM(responses=[_FACT_JSON])
        ev = QualityEvaluator(llm)
        fact = await ev.evaluate_fact(_report(), _sources())
        assert fact.citation_accuracy == 95.0
        assert fact.effective_citations == 12


class TestCombinedEval:
    async def test_combined_score(self):
        llm = MockLLM(responses=[_RACE_JSON, _FACT_JSON])
        ev = QualityEvaluator(llm)
        score = await ev.evaluate(_report(), _sources())
        assert score.race.overall > 0
        assert score.fact.citation_accuracy == 95.0
        assert score.overall > 0


class TestBoundaryAndInteraction:
    async def test_zero_scores_all_dims(self):
        zero_json = json.dumps(
            {
                "comprehensiveness": {"score": 0, "reasoning": "empty"},
                "depth": {"score": 0, "reasoning": "empty"},
                "instruction_following": {"score": 0, "reasoning": "empty"},
                "readability": {"score": 0, "reasoning": "empty"},
            }
        )
        llm = MockLLM(responses=[zero_json])
        ev = QualityEvaluator(llm)
        race = await ev.evaluate_race(_report())
        assert race.overall == 0
        assert race.comprehensiveness == 0

    async def test_both_evals_malformed(self):
        llm = MockLLM(responses=["totally broken", "also broken"])
        ev = QualityEvaluator(llm)
        score = await ev.evaluate(_report(), _sources())
        assert score.race.overall == 0
        assert score.fact.citation_accuracy == 0
        assert score.overall == 0

    async def test_race_llm_receives_report(self):
        llm = MockLLM(responses=[_RACE_JSON])
        ev = QualityEvaluator(llm)
        await ev.evaluate_race(_report())
        assert llm._call_count == 1


class TestGracefulDegradation:
    """Quality evaluation must not crash the pipeline on LLM errors."""

    async def test_race_error_returns_zero(self):
        llm = _FailingLLM()
        ev = QualityEvaluator(llm)
        score = await ev.evaluate(_report(), _sources())
        assert score.race.overall == 0.0
        assert score.fact.citation_accuracy == 0.0
        assert score.overall == 0.0

    async def test_race_ok_fact_error(self):
        llm = _OnceFailLLM(fail_on=1, responses=[_RACE_JSON])
        ev = QualityEvaluator(llm)
        score = await ev.evaluate(_report(), _sources())
        assert score.race.overall > 0
        assert score.fact.citation_accuracy == 0.0

    async def test_fact_ok_race_error(self):
        # Call 0 (RACE) fails and consumes a counter slot; FACT at index 1
        llm = _OnceFailLLM(fail_on=0, responses=["", _FACT_JSON])
        ev = QualityEvaluator(llm)
        score = await ev.evaluate(_report(), _sources())
        assert score.race.overall == 0.0
        assert score.fact.citation_accuracy == 95.0


class _FailingLLM(MockLLM):
    async def chat(self, messages, **kwargs):
        raise RuntimeError("billing error")


class _OnceFailLLM(MockLLM):
    """Fails on the Nth call (0-indexed), succeeds otherwise."""

    def __init__(self, fail_on: int, responses: list[str] | None = None):
        super().__init__(responses)
        self._fail_on = fail_on

    async def chat(self, messages, **kwargs):
        idx = self._call_count
        if idx == self._fail_on:
            self._call_count += 1
            raise RuntimeError("billing error")
        return await super().chat(messages, **kwargs)
