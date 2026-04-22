"""Unit tests for ConflictResolver: detect, vote, LLM arbitrate, dual-perspective."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from houyi.adapters.llm.base import LLMAdapter, LLMResponse, StreamChunk
from houyi.application.runtime.conflict import (
    AgentTaskResult,
    ConflictRecord,
    ConflictResolver,
    _source_vote_score,
    _text_similarity,
)

# -- Helpers -----------------------------------------------------------------


class _MockLLM(LLMAdapter):
    """Canned-response LLM for testing LLM arbitration."""

    def __init__(self, response: str) -> None:
        self._response = response

    async def chat(self, messages: list, **kw: Any) -> LLMResponse:
        return LLMResponse(content=self._response, finish_reason="stop", model="mock")

    async def stream_chat(self, messages: list, **kw: Any) -> AsyncIterator[StreamChunk]:
        yield StreamChunk()


class _FailingLLM(LLMAdapter):
    async def chat(self, messages: list, **kw: Any) -> LLMResponse:
        raise RuntimeError("LLM unavailable")

    async def stream_chat(self, messages: list, **kw: Any) -> AsyncIterator[StreamChunk]:
        yield StreamChunk()


def _result(agent_id: str, output: str, *, success: bool = True) -> AgentTaskResult:
    return AgentTaskResult(agent_id=agent_id, output=output, success=success)


# -- Detection ---------------------------------------------------------------


class TestDetect:
    async def test_detects_disagreement(self):
        resolver = ConflictResolver()
        results = [
            _result("a", "The sky is blue because of Rayleigh scattering"),
            _result("b", "The sky is red due to volcanic particles in the atmosphere"),
        ]
        conflicts = await resolver.detect(results)
        assert len(conflicts) == 1
        assert conflicts[0].agent_a_id == "a"
        assert conflicts[0].agent_b_id == "b"

    async def test_skips_identical_outputs(self):
        resolver = ConflictResolver()
        results = [_result("a", "same answer"), _result("b", "same answer")]
        conflicts = await resolver.detect(results)
        assert len(conflicts) == 0

    async def test_skips_near_duplicates(self):
        resolver = ConflictResolver()
        results = [
            _result("a", "The answer is 42 exactly"),
            _result("b", "The answer is 42 exactly."),
        ]
        conflicts = await resolver.detect(results)
        assert len(conflicts) == 0

    async def test_skips_failed_agents(self):
        resolver = ConflictResolver()
        results = [
            _result("a", "good answer"),
            _result("b", "different answer", success=False),
        ]
        conflicts = await resolver.detect(results)
        assert len(conflicts) == 0

    async def test_skips_empty_output(self):
        resolver = ConflictResolver()
        results = [_result("a", "good answer"), _result("b", "")]
        conflicts = await resolver.detect(results)
        assert len(conflicts) == 0

    async def test_multiple_pairwise(self):
        resolver = ConflictResolver()
        results = [
            _result("a", "Python dominates machine learning with TensorFlow and PyTorch"),
            _result("b", "R provides the most robust statistical analysis ecosystem"),
            _result("c", "Julia is clearly superior for high-performance numerical tasks"),
        ]
        conflicts = await resolver.detect(results)
        assert len(conflicts) == 3


# -- Source Voting -----------------------------------------------------------


class TestVoteResolve:
    async def test_longer_answer_wins(self):
        resolver = ConflictResolver()
        conflict = ConflictRecord(
            agent_a_id="short",
            agent_a_conclusion="Brief.",
            agent_b_id="long",
            agent_b_conclusion="Detailed analysis with http://source1.com and http://source2.com evidence.",
        )
        resolution = await resolver.resolve(conflict)
        assert resolution.method == "source_voting"
        assert resolution.winner == "long"
        assert resolution.confidence > 0

    async def test_url_rich_wins(self):
        resolver = ConflictResolver()
        conflict = ConflictRecord(
            agent_a_id="cited",
            agent_a_conclusion="See http://a.com, http://b.com, http://c.com for details.",
            agent_b_id="uncited",
            agent_b_conclusion="I believe this is correct based on my analysis alone.",
        )
        resolution = await resolver.resolve(conflict)
        assert resolution.winner == "cited"

    async def test_equal_score_defaults_first(self):
        resolver = ConflictResolver()
        conflict = ConflictRecord(
            agent_a_id="a",
            agent_a_conclusion="X",
            agent_b_id="b",
            agent_b_conclusion="Y",
        )
        resolution = await resolver.resolve(conflict)
        assert resolution.winner == "a"
        assert resolution.confidence == 0.5


# -- LLM Arbitration --------------------------------------------------------


class TestLLMResolve:
    async def test_agent_a_wins(self):
        resp = json.dumps(
            {
                "winner": "agent_a",
                "reasoning": "Agent A cited more sources",
                "confidence": 0.85,
            }
        )
        resolver = ConflictResolver(llm_adapter=_MockLLM(resp))
        conflict = ConflictRecord(
            agent_a_id="alpha",
            agent_a_conclusion="Claim with evidence",
            agent_b_id="beta",
            agent_b_conclusion="Unsupported claim",
        )
        resolution = await resolver.resolve(conflict)
        assert resolution.method == "llm_arbitration"
        assert resolution.winner == "alpha"
        assert resolution.confidence == 0.85

    async def test_agent_b_wins(self):
        resp = json.dumps(
            {
                "winner": "agent_b",
                "reasoning": "B is more accurate",
                "confidence": 0.7,
            }
        )
        resolver = ConflictResolver(llm_adapter=_MockLLM(resp))
        conflict = ConflictRecord(
            agent_a_id="alpha",
            agent_a_conclusion="Claim A",
            agent_b_id="beta",
            agent_b_conclusion="Claim B",
        )
        resolution = await resolver.resolve(conflict)
        assert resolution.winner == "beta"

    async def test_dual_perspective(self):
        resp = json.dumps(
            {
                "winner": "both",
                "reasoning": "Both valid from different angles",
                "confidence": 0.9,
                "dual_perspective": "A covers theoretical, B covers practical",
            }
        )
        resolver = ConflictResolver(llm_adapter=_MockLLM(resp))
        conflict = ConflictRecord(
            agent_a_id="theory",
            agent_a_conclusion="Theoretical perspective",
            agent_b_id="practice",
            agent_b_conclusion="Practical perspective",
        )
        resolution = await resolver.resolve(conflict)
        assert resolution.winner is None
        assert resolution.dual_perspective is not None
        assert "theoretical" in resolution.dual_perspective.lower()

    async def test_llm_malformed_json_fallback(self):
        resolver = ConflictResolver(llm_adapter=_MockLLM("not json at all"))
        conflict = ConflictRecord(
            agent_a_id="a",
            agent_a_conclusion="Claim A",
            agent_b_id="b",
            agent_b_conclusion="Claim B",
        )
        resolution = await resolver.resolve(conflict)
        assert resolution.method == "llm_arbitration"
        assert resolution.confidence == 0.4

    async def test_failure_falls_back_voting(self):
        resolver = ConflictResolver(llm_adapter=_FailingLLM())
        conflict = ConflictRecord(
            agent_a_id="a",
            agent_a_conclusion="Answer with http://source.com evidence",
            agent_b_id="b",
            agent_b_conclusion="Short",
        )
        resolution = await resolver.resolve(conflict)
        assert resolution.method == "source_voting"
        assert resolution.winner == "a"

    async def test_json_in_code_fence(self):
        resp = '```json\n{"winner":"agent_a","reasoning":"better","confidence":0.8}\n```'
        resolver = ConflictResolver(llm_adapter=_MockLLM(resp))
        conflict = ConflictRecord(
            agent_a_id="a",
            agent_a_conclusion="A",
            agent_b_id="b",
            agent_b_conclusion="B",
        )
        resolution = await resolver.resolve(conflict)
        assert resolution.winner == "a"


# -- Helper Functions --------------------------------------------------------


class TestHelpers:
    def test_vote_score_no_urls(self):
        score = _source_vote_score("short text")
        assert score > 0

    def test_vote_score_with_urls(self):
        no_url = _source_vote_score("text only")
        with_urls = _source_vote_score("see http://a.com and http://b.com")
        assert with_urls > no_url

    def test_similarity_identical(self):
        assert _text_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_similarity_completely_different(self):
        assert _text_similarity("abc", "xyz") == pytest.approx(0.0)

    def test_similarity_partial(self):
        sim = _text_similarity("python programming", "python development")
        assert 0.0 < sim < 1.0

    def test_similarity_empty(self):
        assert _text_similarity("", "hello") == 0.0
        assert _text_similarity("x", "") == 0.0

    def test_conclusion_truncated(self):
        long_text = "x" * 5000
        record = ConflictRecord(
            agent_a_conclusion=long_text,
            agent_b_conclusion="short",
        )
        assert len(record.agent_a_conclusion) <= 5000
