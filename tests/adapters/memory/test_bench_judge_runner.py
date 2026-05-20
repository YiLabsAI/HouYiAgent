"""Tests for the deterministic and LLM-based memory judges and bench runner."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from houyi.adapters.memory.answerer import AnswerResult
from houyi.adapters.memory.bench import (
    AdversarialCase,
    AdversarialExpectation,
    AdversarialKind,
    BenchRunner,
    DeterministicJudge,
    LLMMemoryJudge,
    LoCoMoCase,
    LoCoMoSample,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _adv(
    *,
    mode: str = "answer",
    contains=(),
    forbid=(),
    reason: str | None = None,
    case_id: str = "c1",
    kind: AdversarialKind = AdversarialKind.PARAPHRASE_RECALL,
) -> AdversarialCase:
    return AdversarialCase(
        id=case_id,
        kind=kind,
        query="q",
        seed_facts=[],
        expected=AdversarialExpectation(
            mode=mode,
            contains=list(contains),
            forbid=list(forbid),
            reason=reason,
        ),
    )


def _ans(
    text: str = "",
    *,
    abstained: bool = False,
    reason: str = "sufficient",
) -> AnswerResult:
    return AnswerResult(answer=text, abstained=abstained, reason=reason)


@dataclass
class _LLMResp:
    content: str


class _StubJudgeLLM:
    def __init__(self, *, content: str = "MATCH", raise_for: bool = False):
        self.content = content
        self.raise_for = raise_for
        self.calls = 0

    async def chat(self, messages, *, temperature=0.0, max_tokens=64):
        self.calls += 1
        if self.raise_for:
            raise RuntimeError("judge down")
        return _LLMResp(self.content)


# ---------------------------------------------------------------------------
# DeterministicJudge — answer mode
# ---------------------------------------------------------------------------


class TestDeterministicAnswer:
    async def test_contains_match_succeeds(self):
        verdict = await DeterministicJudge().judge(
            _adv(contains=["Boston"]), _ans("I am from boston, MA")
        )
        assert verdict.correct
        assert verdict.reason == "contains_match"

    async def test_contains_miss_fails(self):
        verdict = await DeterministicJudge().judge(_adv(contains=["Boston"]), _ans("I am from NYC"))
        assert not verdict.correct
        assert verdict.reason == "contains_miss"

    async def test_forbid_violation_fails(self):
        verdict = await DeterministicJudge().judge(
            _adv(contains=["Microsoft"], forbid=["Google"]),
            _ans("I work at Microsoft and Google"),
        )
        assert not verdict.correct
        assert verdict.reason == "forbid_violation"

    async def test_forbid_only_passes(self):
        verdict = await DeterministicJudge().judge(_adv(forbid=["mars"]), _ans("I went to Hawaii"))
        assert verdict.correct

    async def test_abstain_when_answer_fails(self):
        verdict = await DeterministicJudge().judge(
            _adv(contains=["Boston"]),
            _ans("idk", abstained=True, reason="no_candidates"),
        )
        assert not verdict.correct
        assert verdict.reason == "abstained_when_answer_expected"


# ---------------------------------------------------------------------------
# DeterministicJudge — abstain mode
# ---------------------------------------------------------------------------


class TestDeterministicAbstain:
    async def test_correct_abstain_reason(self):
        verdict = await DeterministicJudge().judge(
            _adv(mode="abstain", reason="no_candidates"),
            _ans("idk", abstained=True, reason="no_candidates"),
        )
        assert verdict.correct
        assert verdict.reason == "abstain_ok"

    async def test_specific_reason_mismatch_fails(self):
        verdict = await DeterministicJudge().judge(
            _adv(mode="abstain", reason="no_candidates"),
            _ans("idk", abstained=True, reason="llm_idk"),
        )
        assert not verdict.correct
        assert verdict.reason == "abstain_reason_mismatch"

    async def test_any_abstain_accepts(self):
        verdict = await DeterministicJudge().judge(
            _adv(mode="abstain", reason="any_abstain"),
            _ans("idk", abstained=True, reason="llm_idk"),
        )
        assert verdict.correct

    async def test_answered_when_abstain_fails(self):
        verdict = await DeterministicJudge().judge(
            _adv(mode="abstain", reason="no_candidates"),
            _ans("I am from Boston"),
        )
        assert not verdict.correct
        assert verdict.reason == "answered_when_abstain_expected"


# ---------------------------------------------------------------------------
# LLMMemoryJudge
# ---------------------------------------------------------------------------


def _locomo_case(answer: str = "Paris") -> LoCoMoCase:
    sample = LoCoMoSample(sample_id="s-1", speaker_a="A", speaker_b="B", turns=())
    return LoCoMoCase(
        sample_id="s-1",
        question="What is the capital of France?",
        answer=answer,
        evidence=("D1:1",),
        category=1,
        sample=sample,
    )


class TestLLMJudge:
    def test_construction_validation(self):
        with pytest.raises(ValueError):
            LLMMemoryJudge(None)

    async def test_match_parses_to_correct(self):
        llm = _StubJudgeLLM(content="MATCH")
        verdict = await LLMMemoryJudge(llm).judge(_locomo_case(), _ans("Paris."))
        assert verdict.correct
        assert verdict.reason == "llm_match"
        assert llm.calls == 1

    async def test_mismatch_marks_incorrect(self):
        llm = _StubJudgeLLM(content="MISMATCH")
        verdict = await LLMMemoryJudge(llm).judge(_locomo_case(), _ans("Berlin"))
        assert not verdict.correct
        assert verdict.reason == "llm_mismatch"

    async def test_abstain_ok_token(self):
        llm = _StubJudgeLLM(content="ABSTAIN_OK")
        verdict = await LLMMemoryJudge(llm).judge(
            _locomo_case(), _ans("idk", abstained=True, reason="no_candidates")
        )
        assert verdict.correct
        assert verdict.reason == "llm_abstain_ok"

    async def test_trivial_abstain_skips_llm(self):
        llm = _StubJudgeLLM(content="MATCH")
        verdict = await LLMMemoryJudge(llm).judge(
            _locomo_case(answer=""), _ans("idk", abstained=True, reason="no_candidates")
        )
        assert verdict.correct
        assert verdict.reason == "trivial_abstain_ok"
        # LLM never called — structural shortcut wins.
        assert llm.calls == 0

    async def test_parse_failure_no_token(self):
        llm = _StubJudgeLLM(content="hmmm not sure")
        verdict = await LLMMemoryJudge(llm).judge(_locomo_case(), _ans("Paris"))
        assert not verdict.correct
        assert verdict.reason == "judge_parse_failed"

    async def test_llm_failure_marked(self):
        llm = _StubJudgeLLM(raise_for=True)
        verdict = await LLMMemoryJudge(llm).judge(_locomo_case(), _ans("Paris"))
        assert not verdict.correct
        assert verdict.reason == "judge_llm_failed"


# ---------------------------------------------------------------------------
# BenchRunner
# ---------------------------------------------------------------------------


class TestBenchRunner:
    def test_construction_validation(self):
        with pytest.raises(ValueError):
            BenchRunner(None)

    async def test_runs_each_case_aggregates(self):
        async def runner_fn(case):
            return _ans("Boston" if case.id == "good" else "NYC")

        cases = [
            _adv(case_id="good", contains=["Boston"]),
            _adv(case_id="bad", contains=["Boston"]),
        ]
        report = await BenchRunner(DeterministicJudge()).run(cases, runner_fn)
        assert report.total == 2
        assert report.correct == 1
        assert report.errors == 0
        assert report.accuracy == 0.5
        assert report.by_reason["contains_match"] == 1
        assert report.by_reason["contains_miss"] == 1

    async def test_case_runner_exception(self):
        async def boom(case):
            raise RuntimeError("ingestion failed")

        cases = [_adv(case_id="x", mode="abstain", reason="any_abstain")]
        report = await BenchRunner(DeterministicJudge()).run(cases, boom)
        assert report.total == 1
        assert report.errors == 1
        # Synthesized abstain matches any_abstain → verdict correct.
        assert report.correct == 1
        # Per-case error preserved.
        assert report.outcomes[0].error is not None

    async def test_progress_callback_invoked(self):
        async def runner_fn(case):
            return _ans("hi", abstained=False)

        seen = []
        await BenchRunner(DeterministicJudge()).run(
            [_adv(case_id=f"c{i}", contains=["hi"]) for i in range(3)],
            runner_fn,
            on_progress=seen.append,
        )
        assert len(seen) == 3
        assert {o.case_id for o in seen} == {"c0", "c1", "c2"}

    async def test_report_to_dict_shape(self):
        async def runner_fn(case):
            return _ans("ok")

        report = await BenchRunner(DeterministicJudge()).run(
            [_adv(case_id="c1", contains=["ok"])], runner_fn
        )
        d = report.to_dict()
        assert d == {
            "total": 1,
            "correct": 1,
            "errors": 0,
            "accuracy": 1.0,
            "by_reason": {"contains_match": 1},
        }

    async def test_judge_exception_recorded(self):
        class _BoomJudge:
            async def judge(self, case, answer):
                raise RuntimeError("judge broken")

        async def runner_fn(case):
            return _ans("ok")

        report = await BenchRunner(_BoomJudge()).run(
            [_adv(case_id="c1", contains=["ok"])], runner_fn
        )
        assert report.total == 1
        assert report.correct == 0
        assert report.outcomes[0].verdict.reason == "judge_raised"
