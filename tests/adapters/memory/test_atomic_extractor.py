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


class TestExtractorBatchCoverage:
    """extract_batch must cover every input anchor even when the batch LLM
    response is unreliable at the per-turn level (long-batch attention loss).
    Both an OMITTED anchor and a PRESENT-BUT-EMPTY anchor are re-extracted via
    the reliable single-turn path, since a batch empty is not authoritative.
    """

    @pytest.mark.asyncio
    async def test_dropped_anchor_re_extracted(self) -> None:
        # Batch LLM returns only anchor "a"; "b" is absent (dropped).
        batch_resp = json.dumps(
            {
                "items": [
                    {
                        "source_anchor": "a",
                        "facts": [
                            {
                                "subject": "S",
                                "predicate": "p",
                                "object": "o",
                                "certainty": "certain",
                            }
                        ],
                        "events": [],
                        "edges": [],
                    }
                ]
            }
        )
        # Single-turn fallback for the dropped anchor "b".
        single_resp = _items(
            {"subject": "S2", "predicate": "p2", "object": "o2", "certainty": "certain"}
        )
        llm = _StubLLM([batch_resp, single_resp])
        ext = AtomicFactExtractor(llm, max_retries=0)
        results = await ext.extract_batch([("text-a", "a"), ("text-b", "b")], namespace="ns")
        assert len(results) == 2
        # anchor a: served from the batch response
        assert len(results[0].facts) == 1
        assert results[0].facts[0].object == "o"
        # anchor b: dropped by the batch, re-extracted singly -> has the fact
        assert len(results[1].facts) == 1
        assert results[1].facts[0].object == "o2"
        # one batch call + one single-turn fallback call
        assert len(llm.calls) == 2

    @pytest.mark.asyncio
    async def test_empty_reextracted(self) -> None:
        # Batch returns every anchor but one comes back present-but-empty.
        # A batch empty is NOT authoritative (long-batch attention loss), so
        # the empty anchor is re-extracted via the reliable single-turn path,
        # which recovers its salient fact.
        batch_resp = json.dumps(
            {
                "items": [
                    {
                        "source_anchor": "a",
                        "facts": [
                            {
                                "subject": "S",
                                "predicate": "p",
                                "object": "o",
                                "certainty": "certain",
                            }
                        ],
                        "events": [],
                        "edges": [],
                    },
                    {"source_anchor": "b", "facts": [], "events": [], "edges": []},
                ]
            }
        )
        # Single-turn re-extraction of the present-but-empty anchor "b".
        single_resp = _items(
            {"subject": "S2", "predicate": "p2", "object": "o2", "certainty": "certain"}
        )
        llm = _StubLLM([batch_resp, single_resp])
        ext = AtomicFactExtractor(llm, max_retries=0)
        results = await ext.extract_batch([("text-a", "a"), ("text-b", "b")], namespace="ns")
        assert len(results) == 2
        assert len(results[0].facts) == 1
        # anchor b: batch said empty, re-extracted singly -> recovers the fact
        assert len(results[1].facts) == 1
        assert results[1].facts[0].object == "o2"
        # one batch call + one single-turn re-extraction for the empty anchor
        assert len(llm.calls) == 2

    @pytest.mark.asyncio
    async def test_empty_reextracted_events(self) -> None:
        # A present-but-empty anchor re-extracted singly must recover EVENTS,
        # not just facts. Before the fix the single-turn path read events_raw
        # but never assembled it, so recovered anchors silently lost their
        # time-anchored events -- the regression that cratered the events
        # table for time questions whose gold lives in events.
        batch_resp = json.dumps(
            {
                "items": [
                    {
                        "source_anchor": "a",
                        "facts": [
                            {
                                "subject": "S",
                                "predicate": "p",
                                "object": "o",
                                "certainty": "certain",
                            }
                        ],
                        "events": [],
                        "edges": [],
                    },
                    {"source_anchor": "b", "facts": [], "events": [], "edges": []},
                ]
            }
        )
        # Single-turn response as a dict with facts + events -- the format
        # _parse_json_array extracts events from (a flat array would not).
        single_resp = json.dumps(
            {
                "facts": [
                    {"subject": "S2", "predicate": "p2", "object": "o2", "certainty": "certain"}
                ],
                "events": [
                    {
                        "subject": "Maria",
                        "action": "donated",
                        "object": "car",
                        "timestamp": "2023",
                        "certainty": "certain",
                    }
                ],
                "edges": [],
            }
        )
        llm = _StubLLM([batch_resp, single_resp])
        ext = AtomicFactExtractor(llm, max_retries=0)
        results = await ext.extract_batch([("text-a", "a"), ("text-b", "b")], namespace="ns")
        assert len(results) == 2
        # anchor b: batch empty, re-extracted singly -> recovers the event
        assert len(results[1].events) == 1
        ev = results[1].events[0]
        assert ev.action == "donated"
        assert ev.object == "car"
        # namespace threaded so the EventRetriever (queries by case namespace)
        # can find the recovered event, not just the default namespace.
        assert ev.namespace == "ns"
        assert len(llm.calls) == 2

    @pytest.mark.asyncio
    async def test_invalid_anchor_reextracted(self) -> None:
        # Batch returns anchor "b" with a NON-EMPTY raw facts list, but every
        # fact dict is schema-invalid (here: an unrecognized certainty value),
        # so _build_fact returns None for all of them. The anchor ends up with
        # zero built facts even though the raw list is non-empty.
        #
        # This is the third coverage case (absent / present-but-empty /
        # present-but-all-invalid) and the one the pre-build fallback check
        # (which tests raw-list emptiness, not built-object validity) misses:
        # the raw list is non-empty so no single-turn re-extraction fires, and
        # the anchor is silently stored empty. The fix re-extracts when no raw
        # item survives _build_fact/_build_event.
        batch_resp = json.dumps(
            {
                "items": [
                    {
                        "source_anchor": "a",
                        "facts": [
                            {
                                "subject": "S",
                                "predicate": "p",
                                "object": "o",
                                "certainty": "certain",
                            }
                        ],
                        "events": [],
                        "edges": [],
                    },
                    {
                        "source_anchor": "b",
                        "facts": [
                            # Non-empty list, but certainty is not a valid
                            # Certainty tier -> _build_fact returns None.
                            {
                                "subject": "S",
                                "predicate": "p",
                                "object": "o",
                                "certainty": "maybe",
                            }
                        ],
                        "events": [],
                        "edges": [],
                    },
                ]
            }
        )
        # Single-turn re-extraction of anchor "b" recovers a valid fact.
        single_resp = _items(
            {"subject": "S2", "predicate": "p2", "object": "o2", "certainty": "certain"}
        )
        llm = _StubLLM([batch_resp, single_resp])
        ext = AtomicFactExtractor(llm, max_retries=0)
        results = await ext.extract_batch([("text-a", "a"), ("text-b", "b")], namespace="ns")
        assert len(results) == 2
        # anchor a: served from the batch response
        assert len(results[0].facts) == 1
        assert results[0].facts[0].object == "o"
        # anchor b: batch facts all schema-invalid -> re-extracted singly
        assert len(results[1].facts) == 1
        assert results[1].facts[0].object == "o2"
        # one batch call + one single-turn re-extraction for the invalid anchor
        assert len(llm.calls) == 2

    @pytest.mark.asyncio
    async def test_batch_min_items(self) -> None:
        # The batch call constrains the LLM with a json_schema whose items[]
        # has minItems = number of input turns and requires source_anchor,
        # so turns cannot be silently dropped from the response.
        batch_resp = json.dumps(
            {
                "items": [
                    {"source_anchor": "a", "facts": [], "events": [], "edges": []},
                    {"source_anchor": "b", "facts": [], "events": [], "edges": []},
                ]
            }
        )
        # Both anchors come back batch-empty, so each is re-extracted singly
        # after the batch call; supply empty single-turn responses for them.
        llm = _StubLLM([batch_resp, "[]", "[]"])
        ext = AtomicFactExtractor(llm, max_retries=0)
        await ext.extract_batch([("text-a", "a"), ("text-b", "b")], namespace="ns")
        fmt = llm.calls[0]["kwargs"]["response_format"]
        assert fmt["type"] == "json_schema"
        items_prop = fmt["json_schema"]["schema"]["properties"]["items"]
        assert items_prop["minItems"] == 2  # one item per input turn
        assert "source_anchor" in items_prop["items"]["required"]

    @pytest.mark.asyncio
    async def test_batch_no_schema(self) -> None:
        batch_resp = json.dumps(
            {"items": [{"source_anchor": "a", "facts": [], "events": [], "edges": []}]}
        )
        llm = _StubLLM([batch_resp])
        ext = AtomicFactExtractor(llm, max_retries=0, prefer_json_mode=False)
        await ext.extract_batch([("text-a", "a")], namespace="ns")
        assert "response_format" not in llm.calls[0]["kwargs"]


class TestEventTimestampResolution:
    """The deterministic resolver rewrites relative event timestamps to
    absolute values anchored on the per-turn observation_date embedded in the
    bench/ingest extract-text JSON. This exercises the live _build_event path
    (not just the resolver in isolation) so we prove the wiring catches a
    verbatim relative timestamp the LLM emits non-deterministically.
    """

    @staticmethod
    def _extract_text(observation_date: str, body: str) -> str:
        return json.dumps(
            {
                "observation_date": observation_date,
                "system_date": "2024-01-15",
                "text": body,
                "speaker_name": "Joanna",
            }
        )

    @pytest.mark.asyncio
    async def test_verbatim_relative_resolved(self) -> None:
        llm = _StubLLM(
            json.dumps(
                {
                    "facts": [],
                    "events": [
                        {
                            "subject": "Joanna",
                            "action": "watched",
                            "object": "Eternal Sunshine of the Spotless Mind",
                            "timestamp": "around 3 years ago",
                            "context": "first time watching",
                            "certainty": "certain",
                        }
                    ],
                    "edges": [],
                }
            )
        )
        text = self._extract_text("2022-01-21", "I first watched it around 3 years ago")
        result = await AtomicFactExtractor(llm, max_retries=0).extract(
            text, source_anchor="conv-42:D1:18"
        )
        assert len(result.events) == 1
        assert result.events[0].timestamp == "2019"

    @pytest.mark.asyncio
    async def test_absolute_passthrough(self) -> None:
        llm = _StubLLM(
            json.dumps(
                {
                    "facts": [],
                    "events": [
                        {
                            "subject": "Joanna",
                            "action": "watched",
                            "object": "Eternal Sunshine",
                            "timestamp": "2019",
                            "certainty": "certain",
                        }
                    ],
                    "edges": [],
                }
            )
        )
        text = self._extract_text("2022-01-21", "I first watched it in 2019")
        result = await AtomicFactExtractor(llm, max_retries=0).extract(
            text, source_anchor="conv-42:D1:18"
        )
        assert len(result.events) == 1
        assert result.events[0].timestamp == "2019"

    @pytest.mark.asyncio
    async def test_no_anchor_unchanged(self) -> None:
        # Callers that pass plain text (no observation_date JSON) must be
        # untouched: the resolver is a no-op without an anchor.
        llm = _StubLLM(
            json.dumps(
                {
                    "facts": [],
                    "events": [
                        {
                            "subject": "Joanna",
                            "action": "watched",
                            "object": "Eternal Sunshine",
                            "timestamp": "around 3 years ago",
                            "certainty": "certain",
                        }
                    ],
                    "edges": [],
                }
            )
        )
        result = await AtomicFactExtractor(llm, max_retries=0).extract(
            "I first watched it around 3 years ago", source_anchor="a"
        )
        assert len(result.events) == 1
        assert result.events[0].timestamp == "around 3 years ago"
