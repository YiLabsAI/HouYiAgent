"""LLMAnswerer + IDK abstain budget tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from houyi.adapters.memory.answerer import (
    DEFAULT_IDK_PHRASE,
    AbstainPolicy,
    AnswerBudget,
    LLMAnswerer,
)
from houyi.adapters.memory.recall.types import (
    QueryType,
    RecallCandidate,
    RecallReason,
    RecallResult,
    RetrieverKind,
)
from houyi.adapters.memory.types import AtomicFact, Certainty

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _LLMResponse:
    content: str


class _FakeLLM:
    def __init__(self, *, content: str = "tea", delay: float = 0.0, raise_for: bool = False):
        self.content = content
        self.delay = delay
        self.raise_for = raise_for
        self.calls: list[list[dict]] = []

    async def chat(self, messages, *, temperature=0.0, max_tokens=512):
        self.calls.append(list(messages))
        if self.raise_for:
            raise RuntimeError("provider down")
        if self.delay:
            await asyncio.sleep(self.delay)
        return _LLMResponse(self.content)


def _fact(subject="alice", predicate="likes", obj="tea", *, anchor="anchor1"):
    return AtomicFact(
        subject=subject,
        predicate=predicate,
        object=obj,
        certainty=Certainty.CERTAIN,
        source_anchor=anchor,
    )


def _candidate(fact: AtomicFact, *, score: float = 1.0) -> RecallCandidate:
    return RecallCandidate(
        fact=fact,
        score=score,
        matched_by=RetrieverKind.ENTITY_STATE,
        retriever_name="fake",
    )


def _recall(
    candidates,
    *,
    reason: RecallReason = RecallReason.SUFFICIENT,
    query_type: QueryType = QueryType.FACTUAL_LOOKUP,
) -> RecallResult:
    return RecallResult(
        candidates=list(candidates),
        query_type=query_type,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_requires_llm(self):
        with pytest.raises(ValueError):
            LLMAnswerer(None)


# ---------------------------------------------------------------------------
# Guard-driven abstain (no LLM call)
# ---------------------------------------------------------------------------


class TestGuardAbstain:
    async def test_no_candidates_short_circuits(self):
        llm = _FakeLLM()
        ans = LLMAnswerer(llm)
        result = await ans.answer(
            "What does alice like?",
            _recall([], reason=RecallReason.NO_CANDIDATES),
        )
        assert result.abstained is True
        assert result.answer == DEFAULT_IDK_PHRASE
        assert result.reason == "no_candidates"
        # LLM never called.
        assert llm.calls == []

    async def test_low_evidence_short_circuits(self):
        llm = _FakeLLM()
        ans = LLMAnswerer(llm)
        result = await ans.answer(
            "What does alice like?",
            _recall(
                [_candidate(_fact(), score=0.1)],
                reason=RecallReason.LOW_EVIDENCE,
            ),
        )
        assert result.abstained is True
        assert result.reason == "low_evidence"
        assert llm.calls == []

    async def test_explicit_absence_short_circuits(self):
        llm = _FakeLLM()
        result = await LLMAnswerer(llm).answer(
            "Has bob signed up?",
            _recall([], reason=RecallReason.EXPLICIT_ABSENCE),
        )
        assert result.abstained is True
        assert result.reason == "explicit_absence"


# ---------------------------------------------------------------------------
# Policy abstain
# ---------------------------------------------------------------------------


class TestPolicyAbstain:
    async def test_min_top_score_low(self):
        llm = _FakeLLM()
        ans = LLMAnswerer(llm, policy=AbstainPolicy(min_top_score=0.5, min_facts=1))
        result = await ans.answer(
            "q",
            _recall([_candidate(_fact(), score=0.2)]),
        )
        assert result.abstained is True
        assert result.reason == "low_top_score"
        assert llm.calls == []

    async def test_min_facts_under_threshold(self):
        llm = _FakeLLM()
        ans = LLMAnswerer(llm, policy=AbstainPolicy(min_facts=3))
        result = await ans.answer(
            "q",
            _recall([_candidate(_fact())]),
        )
        assert result.abstained is True
        assert result.reason == "too_few_facts"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    async def test_calls_llm_returns_answer(self):
        llm = _FakeLLM(content="alice likes tea [1]")
        ans = LLMAnswerer(llm)
        result = await ans.answer(
            "what does alice like?",
            _recall([_candidate(_fact(anchor="t-1"))]),
        )
        assert result.abstained is False
        assert result.reason == "sufficient"
        assert result.answer == "alice likes tea [1]"
        assert result.citations == ("t-1",)
        assert result.facts_used == 1
        # Prompt body contains numbered fact + question.
        assert llm.calls
        user_msg = llm.calls[0][1]["content"]
        assert "1. alice likes tea" in user_msg
        assert "what does alice like?" in user_msg

    async def test_max_facts_caps_prompt(self):
        llm = _FakeLLM()
        ans = LLMAnswerer(llm, budget=AnswerBudget(max_facts_in_prompt=2))
        cands = [_candidate(_fact(predicate=f"p{i}"), score=10 - i) for i in range(5)]
        result = await ans.answer("q", _recall(cands))
        assert result.facts_used == 2
        # Only first two facts rendered.
        body = llm.calls[0][1]["content"]
        assert "1. alice p0 tea" in body
        assert "2. alice p1 tea" in body
        assert "3. alice p2 tea" not in body

    async def test_dedup_citations_preserve_order(self):
        llm = _FakeLLM()
        ans = LLMAnswerer(llm)
        cands = [
            _candidate(_fact(anchor="a")),
            _candidate(_fact(predicate="p2", anchor="b")),
            _candidate(_fact(predicate="p3", anchor="a")),  # dup
        ]
        result = await ans.answer("q", _recall(cands))
        assert result.citations == ("a", "b")


# ---------------------------------------------------------------------------
# Answer tag parsing (conv-44 job regression): LLM emits
# <Analysis>...</Analysis><Answer>...</Answer>; the answerer must extract
# the <Answer> tag, not return the whole response. An empty <Answer> with a
# populated <Analysis> falls back to the Analysis so the model is not scored
# wrong when it puts the answer in Analysis but leaves Answer empty.
# ---------------------------------------------------------------------------


class TestAnswerTagParsing:
    async def test_parses_answer_tag(self):
        llm = _FakeLLM(
            content="<Analysis>thinking it through</Analysis><Answer>around 2023-03-20</Answer>"
        )
        result = await LLMAnswerer(llm).answer("q", _recall([_candidate(_fact())]))
        assert result.abstained is False
        assert result.answer == "around 2023-03-20"

    async def test_falls_back_to_analysis(self):
        llm = _FakeLLM(content="<Analysis>around 2023-03-20</Analysis><Answer></Answer>")
        result = await LLMAnswerer(llm).answer("q", _recall([_candidate(_fact())]))
        assert result.abstained is False
        assert result.answer == "around 2023-03-20"

    async def test_no_analysis_abstains(self):
        llm = _FakeLLM(content="<Answer></Answer>")
        result = await LLMAnswerer(llm).answer("q", _recall([_candidate(_fact())]))
        assert result.abstained is True


# ---------------------------------------------------------------------------
# Fact line rendering: qualifiers (original_time) must reach the LLM so
# relative-time expressions like "last week" are not lost. The extractor
# records original_time per TEMPORAL LITERAL PRESERVATION; if the answerer
# drops it the LLM only sees a resolved point date + "uncertain" and
# reasons unstably (conv-44 job: around vs before).
# ---------------------------------------------------------------------------


class TestFormatFactLine:
    def test_original_time_in_prompt(self):
        fact = AtomicFact(
            subject="Andrew",
            predicate="started_job",
            object="Financial Analyst",
            certainty=Certainty.CERTAIN,
            source_anchor="D1:2",
            event_time="2023-03-20",
            qualifiers={"original_time": "last week"},
        )
        cand = _candidate(fact)
        line = LLMAnswerer._format_fact_line(1, cand)
        assert "last week" in line


# ---------------------------------------------------------------------------
# LLM-side abstain detection
# ---------------------------------------------------------------------------


class TestLLMSideAbstain:
    async def test_idk_sentinel_to_abstain(self):
        llm = _FakeLLM(content="[IDK]")
        result = await LLMAnswerer(llm).answer("q", _recall([_candidate(_fact())]))
        assert result.abstained is True
        assert result.reason == "llm_idk"
        assert result.answer == DEFAULT_IDK_PHRASE

    async def test_empty_response_to_abstain(self):
        llm = _FakeLLM(content=" ")
        result = await LLMAnswerer(llm).answer("q", _recall([_candidate(_fact())]))
        assert result.abstained is True
        assert result.reason == "llm_idk"


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    async def test_llm_exception_returns_abstain(self):
        llm = _FakeLLM(raise_for=True)
        result = await LLMAnswerer(llm).answer("q", _recall([_candidate(_fact())]))
        assert result.abstained is True
        assert result.reason == "llm_failed"

    async def test_timeout_returns_abstain(self):
        llm = _FakeLLM(content="late", delay=0.5)
        ans = LLMAnswerer(llm, budget=AnswerBudget(timeout_seconds=0.05))
        result = await ans.answer("q", _recall([_candidate(_fact())]))
        assert result.abstained is True
        assert result.reason == "timeout"


# ---------------------------------------------------------------------------
# Budget — prompt-size cap
# ---------------------------------------------------------------------------


class TestBudgetPromptSize:
    async def test_oversized_prompt_truncates(self):
        # 10 facts × ~40 chars = ~400 char body; cap at 1300 should
        # still keep the first few numbered lines so we DON'T abstain.
        llm = _FakeLLM(content="ok")
        ans = LLMAnswerer(
            llm,
            budget=AnswerBudget(max_input_chars=1300, max_facts_in_prompt=10),
        )
        cands = [_candidate(_fact(predicate=f"pred{i}_long_padding")) for i in range(10)]
        result = await ans.answer("q", _recall(cands))
        assert result.abstained is False
        body = llm.calls[0][1]["content"]
        assert len(body) <= 1300
        assert "1. " in body  # at least one numbered fact survived
