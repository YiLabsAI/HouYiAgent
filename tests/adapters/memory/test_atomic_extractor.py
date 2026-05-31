from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from houyi.adapters.memory.extractor import (
    _ATOMIC_FACT_SYSTEM_PROMPT,
    AtomicFactExtractor,
    ExtractionResult,
)
from houyi.adapters.memory.types import Certainty


@dataclass
class _StubResponse:
    content: str


class _StubLLM:
    """Minimal stand-in for the llm_adapter.chat() contract.

    Tests preload either a single canned content string (returned for
    every call) or a queue of strings (consumed in order).
    """

    def __init__(self, content: str | list[str] | None = None, raises: Exception | None = None):
        self._content = content
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages, *, temperature: float, max_tokens: int, **kwargs: Any):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "kwargs": kwargs,
            }
        )
        if self._raises is not None:
            raise self._raises
        if isinstance(self._content, list):
            return _StubResponse(self._content.pop(0))
        return _StubResponse(self._content or "[]")


class _PlainLLM:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = 0

    async def chat(self, messages, *, temperature: float, max_tokens: int):
        _ = messages, temperature, max_tokens
        self.calls += 1
        return _StubResponse(self._content)


def _items(*items: dict[str, Any]) -> str:
    return json.dumps(list(items))


class TestExtractorHappyPath:
    """LLM returns well-formed JSON; everything passes through."""

    @pytest.mark.asyncio
    async def test_single_certain_fact(self) -> None:
        llm = _StubLLM(
            _items(
                {
                    "subject": "user",
                    "predicate": "lives_in",
                    "object": "Beijing",
                    "certainty": "certain",
                }
            )
        )
        result = await AtomicFactExtractor(llm).extract("I live in Beijing", source_anchor="msg-1")
        assert isinstance(result, ExtractionResult)
        assert len(result.facts) == 1
        fact = result.facts[0]
        assert (fact.subject, fact.predicate, fact.object) == ("user", "lives_in", "Beijing")
        assert fact.certainty is Certainty.CERTAIN
        assert fact.source_anchor == "msg-1"

    @pytest.mark.asyncio
    async def test_qualifiers_round_trip(self) -> None:
        llm = _StubLLM(
            _items(
                {
                    "subject": "user",
                    "predicate": "lives_in",
                    "object": "Beijing",
                    "certainty": "certain",
                    "qualifiers": {"since": "2022"},
                }
            )
        )
        result = await AtomicFactExtractor(llm).extract("text", source_anchor="a")
        assert result.facts[0].qualifiers == {"since": "2022"}

    @pytest.mark.asyncio
    async def test_multiple_facts(self) -> None:
        llm = _StubLLM(
            _items(
                {"subject": "user", "predicate": "name", "object": "Alice", "certainty": "certain"},
                {
                    "subject": "user",
                    "predicate": "lives_in",
                    "object": "Beijing",
                    "certainty": "certain",
                },
            )
        )
        result = await AtomicFactExtractor(llm).extract("...", source_anchor="a")
        assert len(result.facts) == 2

    @pytest.mark.asyncio
    async def test_strips_json_code_fence(self) -> None:
        llm = _StubLLM(
            "```json\n"
            + _items({"subject": "u", "predicate": "p", "object": "o", "certainty": "certain"})
            + "\n```"
        )
        result = await AtomicFactExtractor(llm).extract("t", source_anchor="a")
        assert len(result.facts) == 1

    @pytest.mark.asyncio
    async def test_facts_wrapper(self) -> None:
        llm = _StubLLM(
            json.dumps(
                {
                    "facts": [
                        {
                            "subject": "u",
                            "predicate": "p",
                            "object": "o",
                            "certainty": "certain",
                        }
                    ]
                }
            )
        )
        result = await AtomicFactExtractor(llm).extract("t", source_anchor="a")
        assert len(result.facts) == 1

    @pytest.mark.asyncio
    async def test_json_mode_requested(self) -> None:
        llm = _StubLLM("[]")
        await AtomicFactExtractor(llm).extract("t", source_anchor="a")
        assert llm.calls[0]["kwargs"]["response_format"]["type"] == "json_schema"

    @pytest.mark.asyncio
    async def test_plain_adapter_fallback(self) -> None:
        llm = _PlainLLM(
            _items({"subject": "u", "predicate": "p", "object": "o", "certainty": "certain"})
        )
        result = await AtomicFactExtractor(llm).extract("t", source_anchor="a")
        assert len(result.facts) == 1
        assert llm.calls == 1


class TestExtractorCertaintyVocab:
    """All three certainty values must round-trip."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("certain", Certainty.CERTAIN),
            ("probable", Certainty.PROBABLE),
            ("vague", Certainty.VAGUE),
        ],
    )
    @pytest.mark.asyncio
    async def test_certainty_value_kept(self, value: str, expected: Certainty) -> None:
        llm = _StubLLM(
            _items({"subject": "u", "predicate": "p", "object": "o", "certainty": value})
        )
        result = await AtomicFactExtractor(llm).extract("t", source_anchor="a")
        assert result.facts[0].certainty is expected

    @pytest.mark.asyncio
    async def test_unknown_certainty_dropped(self) -> None:
        llm = _StubLLM(
            _items({"subject": "u", "predicate": "p", "object": "o", "certainty": "maybe"})
        )
        result = await AtomicFactExtractor(llm).extract("t", source_anchor="a")
        assert result.facts == []
        assert result.invalid_dropped == 1


class TestExtractorFaultTolerance:
    """Bad LLM output must not raise; salvageable items still flow."""

    @pytest.mark.asyncio
    async def test_non_json_response(self) -> None:
        llm = _StubLLM("I don't know how to answer that")
        result = await AtomicFactExtractor(llm).extract("t", source_anchor="a")
        assert result.facts == []
        assert result.invalid_dropped == 0

    @pytest.mark.asyncio
    async def test_retries_json(self) -> None:
        llm = _StubLLM(
            [
                "not json",
                _items(
                    {
                        "subject": "user",
                        "predicate": "lives_in",
                        "object": "Beijing",
                        "certainty": "certain",
                    }
                ),
            ]
        )
        result = await AtomicFactExtractor(llm).extract("I live in Beijing", source_anchor="a")
        assert len(result.facts) == 1
        assert len(llm.calls) == 2
        assert "invalid JSON" in llm.calls[1]["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_repairs_json_span(self) -> None:
        llm = _StubLLM(
            'Here is the JSON:\n{"facts":[{"subject":"u","predicate":"p","object":"o","certainty":"certain"}]}'
        )
        result = await AtomicFactExtractor(llm).extract("t", source_anchor="a")
        assert len(result.facts) == 1
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_repairs_trailing_comma(self) -> None:
        llm = _StubLLM('[{"subject":"u","predicate":"p","object":"o","certainty":"certain",},]')
        result = await AtomicFactExtractor(llm).extract("t", source_anchor="a")
        assert len(result.facts) == 1
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_object_response(self) -> None:
        llm = _StubLLM('{"foo": "bar"}')
        result = await AtomicFactExtractor(llm).extract("t", source_anchor="a")
        assert result.facts == []

    @pytest.mark.asyncio
    async def test_partial_failure(self) -> None:
        llm = _StubLLM(
            _items(
                {"subject": "u", "predicate": "p", "object": "o", "certainty": "certain"},
                {"subject": "", "predicate": "p", "object": "o", "certainty": "certain"},
            )
        )
        result = await AtomicFactExtractor(llm).extract("t", source_anchor="a")
        assert len(result.facts) == 1
        assert result.invalid_dropped == 1

    @pytest.mark.asyncio
    async def test_llm_exception_returns_empty(self) -> None:
        llm = _StubLLM(raises=RuntimeError("boom"))
        result = await AtomicFactExtractor(llm).extract("t", source_anchor="a")
        assert result.facts == []
        assert result.invalid_dropped == 0


class TestExtractorSourceless:
    """When the caller cannot supply an anchor, items must not become facts."""

    @pytest.mark.asyncio
    async def test_none_anchor_to_raw(self) -> None:
        item = {"subject": "u", "predicate": "p", "object": "o", "certainty": "certain"}
        llm = _StubLLM(_items(item))
        result = await AtomicFactExtractor(llm).extract("t", source_anchor=None)
        assert result.facts == []
        assert result.raw_sourceless == [item]

    @pytest.mark.asyncio
    async def test_blank_anchor_to_raw(self) -> None:
        item = {"subject": "u", "predicate": "p", "object": "o", "certainty": "certain"}
        llm = _StubLLM(_items(item))
        result = await AtomicFactExtractor(llm).extract("t", source_anchor=" ")
        assert result.raw_sourceless == [item]

    @pytest.mark.asyncio
    async def test_empty_text_is_noop(self) -> None:
        llm = _StubLLM(
            _items({"subject": "u", "predicate": "p", "object": "o", "certainty": "certain"})
        )
        result = await AtomicFactExtractor(llm).extract(" ", source_anchor="a")
        assert result.facts == []
        assert llm.calls == []  # extractor short-circuits before calling LLM


class TestExtractorConstruction:
    def test_requires_llm(self) -> None:
        with pytest.raises(ValueError, match="llm_adapter"):
            AtomicFactExtractor(None)

    def test_retry_bounds(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            AtomicFactExtractor(_StubLLM(), max_retries=-1)


class TestPromptSpecificityRules:
    """Verify the prompt contains WRONG/RIGHT examples and generic word list."""

    def test_prompt_specificity_rules(self) -> None:
        assert "WRONG" in _ATOMIC_FACT_SYSTEM_PROMPT
        assert "RIGHT" in _ATOMIC_FACT_SYSTEM_PROMPT
        assert "ride" in _ATOMIC_FACT_SYSTEM_PROMPT
        assert "Ferrari 488 GTB" in _ATOMIC_FACT_SYSTEM_PROMPT
