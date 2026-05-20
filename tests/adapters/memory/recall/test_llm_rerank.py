"""Tests for the LLM-based reranker with budget guards and fallback behavior."""

from __future__ import annotations

import pytest

from houyi.adapters.memory.recall.rerank import LLMRerankBudget, LLMReranker
from houyi.adapters.memory.recall.types import QueryType, RecallCandidate, RetrieverKind
from houyi.adapters.memory.types import AtomicFact, Certainty


def _candidate(idx: int, obj: str | None = None) -> RecallCandidate:
    return RecallCandidate(
        fact=AtomicFact(
            subject=f"s{idx}",
            predicate="p",
            object=obj if obj is not None else f"o{idx}",
            certainty=Certainty.CERTAIN,
            source_anchor=f"anchor-{idx}",
        ),
        score=float(idx),
        matched_by=RetrieverKind.VECTOR,
        retriever_name="fake",
    )


class _FakeAdapter:
    """Captures the prompt and returns a canned index list."""

    def __init__(self, reply: str = ""):
        self.reply = reply
        self.last_messages: list[dict] | None = None
        self.timeout: float | None = None

    async def chat(self, messages, **kwargs):
        self.last_messages = list(messages)
        self.timeout = kwargs.get("timeout")
        return {"content": self.reply}


class _RaisingAdapter:
    async def chat(self, messages, **kwargs):
        raise RuntimeError("simulated outage")


class TestLLMRerankerBasics:
    def test_requires_adapter(self):
        with pytest.raises(ValueError):
            LLMReranker(None)

    def test_sync_rerank_raises(self):
        reranker = LLMReranker(_FakeAdapter())
        with pytest.raises(RuntimeError):
            reranker.rerank(
                query_type=QueryType.FACTUAL_LOOKUP,
                candidates=[_candidate(0)],
                top_k=1,
            )


class TestLLMRerankerBudget:
    async def test_below_min_skips_llm(self):
        adapter = _FakeAdapter(reply="0")
        reranker = LLMReranker(
            adapter,
            budget=LLMRerankBudget(min_candidates=2, max_candidates=5),
        )
        out = await reranker.arerank(
            query_type=QueryType.FACTUAL_LOOKUP,
            candidates=[_candidate(0)],
            top_k=1,
        )
        assert out == [_candidate(0)] or len(out) == 1
        assert adapter.last_messages is None  # LLM never called

    async def test_oversized_prompt_skips_llm(self):
        adapter = _FakeAdapter(reply="1,0")
        big = _candidate(0, obj="x" * 500)
        small = _candidate(1, obj="ok")
        reranker = LLMReranker(
            adapter,
            budget=LLMRerankBudget(max_input_chars=100, min_candidates=2),
        )
        out = await reranker.arerank(
            query_type=QueryType.FACTUAL_LOOKUP,
            candidates=[big, small],
            top_k=2,
        )
        # Input order preserved; LLM never called because budget exceeded.
        assert adapter.last_messages is None
        assert [c.fact.subject for c in out] == ["s0", "s1"]

    async def test_top_k_zero_empty(self):
        reranker = LLMReranker(_FakeAdapter(reply="0"))
        out = await reranker.arerank(
            query_type=QueryType.FACTUAL_LOOKUP,
            candidates=[_candidate(0), _candidate(1)],
            top_k=0,
        )
        assert out == []


class TestLLMRerankerReorder:
    async def test_reorders_per_llm_indices(self):
        adapter = _FakeAdapter(reply="2,0,1")
        cands = [_candidate(0), _candidate(1), _candidate(2)]
        reranker = LLMReranker(adapter, budget=LLMRerankBudget(min_candidates=2))
        out = await reranker.arerank(
            query_type=QueryType.THEMATIC_SUMMARY,
            candidates=cands,
            top_k=3,
        )
        assert [c.fact.subject for c in out] == ["s2", "s0", "s1"]
        # Adapter saw a structured prompt with all three candidates.
        assert adapter.last_messages is not None
        assert "[0]" in adapter.last_messages[1]["content"]
        assert "[2]" in adapter.last_messages[1]["content"]

    async def test_partial_indices_append_tail(self):
        # LLM only ranked indices 1 and 0; index 2 must still survive.
        adapter = _FakeAdapter(reply="1,0")
        cands = [_candidate(0), _candidate(1), _candidate(2)]
        reranker = LLMReranker(adapter, budget=LLMRerankBudget(min_candidates=2))
        out = await reranker.arerank(
            query_type=QueryType.FACTUAL_LOOKUP,
            candidates=cands,
            top_k=3,
        )
        subjects = [c.fact.subject for c in out]
        assert subjects[:2] == ["s1", "s0"]
        assert "s2" in subjects  # appended as tail

    async def test_llm_failure_fallback(self):
        cands = [_candidate(0), _candidate(1)]
        reranker = LLMReranker(
            _RaisingAdapter(),
            budget=LLMRerankBudget(min_candidates=2),
        )
        out = await reranker.arerank(
            query_type=QueryType.FACTUAL_LOOKUP,
            candidates=cands,
            top_k=2,
        )
        assert [c.fact.subject for c in out] == ["s0", "s1"]

    async def test_garbage_reply_window(self):
        adapter = _FakeAdapter(reply="not a list")
        cands = [_candidate(0), _candidate(1)]
        reranker = LLMReranker(adapter, budget=LLMRerankBudget(min_candidates=2))
        out = await reranker.arerank(
            query_type=QueryType.FACTUAL_LOOKUP,
            candidates=cands,
            top_k=2,
        )
        assert [c.fact.subject for c in out] == ["s0", "s1"]

    async def test_timeout_is_forwarded(self):
        adapter = _FakeAdapter(reply="0,1")
        budget = LLMRerankBudget(min_candidates=2, timeout_s=1.25)
        reranker = LLMReranker(adapter, budget=budget)
        await reranker.arerank(
            query_type=QueryType.FACTUAL_LOOKUP,
            candidates=[_candidate(0), _candidate(1)],
            top_k=2,
        )
        assert adapter.timeout == 1.25
