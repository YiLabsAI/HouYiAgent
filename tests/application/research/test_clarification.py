"""Unit tests for ClarificationAgent."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from houyi.adapters.llm.base import LLMAdapter, LLMResponse, StreamChunk
from houyi.application.research.clarification import (
    ClarificationAgent,
    ClarificationResult,
    _parse_clarification,
)


class _MockLLM(LLMAdapter):
    def __init__(self, response: str) -> None:
        self._response = response

    async def chat(self, messages: list, **kw: Any) -> LLMResponse:
        return LLMResponse(content=self._response, finish_reason="stop", model="mock")

    async def stream_chat(self, messages: list, **kw: Any) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(content_delta=self._response)


class _FailingLLM(LLMAdapter):
    async def chat(self, messages: list, **kw: Any) -> LLMResponse:
        raise RuntimeError("LLM down")

    async def stream_chat(self, messages: list, **kw: Any) -> AsyncIterator[StreamChunk]:
        raise RuntimeError("LLM down")
        yield  # unreachable — satisfies async generator type


_CLEAR_QUERY = json.dumps(
    {
        "needs_clarification": False,
        "confidence": 0.95,
        "issues": [],
        "suggested_questions": [],
    }
)

_AMBIGUOUS_QUERY = json.dumps(
    {
        "needs_clarification": True,
        "confidence": 0.3,
        "issues": ["Term 'AI' is too broad", "No time constraint specified"],
        "suggested_questions": [
            "Which specific AI domain? (NLP, CV, Agents?)",
            "What time period are you interested in?",
        ],
        "refined_query": "Compare AI agent frameworks released after 2025",
    }
)


class TestAnalyze:
    async def test_clear_query(self):
        agent = ClarificationAgent(_MockLLM(_CLEAR_QUERY))
        result = await agent.analyze("Compare HouYi vs LangChain in 2026")
        assert result.needs_clarification is False
        assert result.confidence > 0.9

    async def test_ambiguous_query(self):
        agent = ClarificationAgent(_MockLLM(_AMBIGUOUS_QUERY))
        result = await agent.analyze("Tell me about AI")
        assert result.needs_clarification is True
        assert len(result.issues) == 2
        assert len(result.suggested_questions) == 2
        assert result.refined_query is not None

    async def test_llm_failure_graceful(self):
        agent = ClarificationAgent(_FailingLLM())
        result = await agent.analyze("Any query")
        assert result.confidence == 0.5
        assert result.needs_clarification is False


class TestParseClarification:
    def test_valid_json(self):
        result = _parse_clarification(_AMBIGUOUS_QUERY)
        assert result.needs_clarification is True
        assert result.confidence == 0.3

    def test_code_fence(self):
        fenced = f"```json\n{_CLEAR_QUERY}\n```"
        result = _parse_clarification(fenced)
        assert result.needs_clarification is False

    def test_malformed(self):
        result = _parse_clarification("not json")
        assert result.confidence == 0.5

    def test_empty(self):
        result = _parse_clarification("")
        assert result.confidence == 0.5


class TestModel:
    def test_defaults(self):
        r = ClarificationResult()
        assert r.needs_clarification is False
        assert r.confidence == 0.8
        assert r.issues == []

    def test_serialization(self):
        r = ClarificationResult(
            needs_clarification=True,
            confidence=0.4,
            issues=["issue1"],
            suggested_questions=["q1"],
        )
        data = r.model_dump()
        restored = ClarificationResult.model_validate(data)
        assert restored == r
