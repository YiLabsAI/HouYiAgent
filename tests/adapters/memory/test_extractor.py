"""MemoryCandidateExtractor unit tests.

Covers:
- LLM-based extraction (primary mode, mocked LLM)
- Rule-based pattern extraction (fallback mode, no LLM)
- Confidence filtering, message role filtering, edge cases
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from houyi.adapters.memory.extractor import MemoryCandidateExtractor
from houyi.adapters.memory.types import ExtractionContext, MemoryType


@pytest.fixture()
def extractor() -> MemoryCandidateExtractor:
    """Rule-based extractor (no LLM)."""
    return MemoryCandidateExtractor()


def _make_llm_mock(json_items: list[dict]) -> AsyncMock:
    """Build a mock LLMAdapter that returns a JSON array."""
    mock = AsyncMock()
    resp = AsyncMock()
    resp.content = json.dumps(json_items)
    mock.chat.return_value = resp
    return mock


# ====================================================================
# LLM-based extraction
# ====================================================================


class TestLLMExtraction:
    """Primary extraction mode using LLM structured output."""

    async def test_extracts_preference(self):
        llm = _make_llm_mock(
            [
                {"content": "User prefers Python", "type": "preference", "confidence": 0.85},
            ]
        )
        ext = MemoryCandidateExtractor(llm_adapter=llm)
        cands = await ext.extract([{"role": "user", "content": "I like Python"}])
        assert len(cands) == 1
        assert cands[0].memory_type == MemoryType.PREFERENCE
        assert "Python" in cands[0].content

    async def test_extracts_multiple_types(self):
        llm = _make_llm_mock(
            [
                {"content": "User name: Alice", "type": "profile", "confidence": 0.9},
                {"content": "User prefers dark mode", "type": "preference", "confidence": 0.8},
                {"content": "Never use tabs", "type": "constraint", "confidence": 0.85},
            ]
        )
        ext = MemoryCandidateExtractor(llm_adapter=llm)
        cands = await ext.extract([{"role": "user", "content": "dummy"}])
        types = {c.memory_type for c in cands}
        assert MemoryType.PROFILE in types
        assert MemoryType.PREFERENCE in types
        assert MemoryType.CONSTRAINT in types

    async def test_empty_when_no_user_messages(self):
        llm = _make_llm_mock([])
        ext = MemoryCandidateExtractor(llm_adapter=llm)
        cands = await ext.extract([{"role": "assistant", "content": "Hello"}])
        assert cands == []
        llm.chat.assert_not_called()

    async def test_filters_low_confidence(self):
        llm = _make_llm_mock(
            [
                {"content": "Maybe likes tea", "type": "preference", "confidence": 0.3},
            ]
        )
        ext = MemoryCandidateExtractor(min_confidence=0.6, llm_adapter=llm)
        cands = await ext.extract([{"role": "user", "content": "test"}])
        assert len(cands) == 0

    async def test_handles_empty_json_array(self):
        llm = _make_llm_mock([])
        ext = MemoryCandidateExtractor(llm_adapter=llm)
        cands = await ext.extract([{"role": "user", "content": "hello there"}])
        assert cands == []

    async def test_handles_markdown_code_fence(self):
        mock = AsyncMock()
        resp = AsyncMock()
        resp.content = (
            '```json\n[{"content":"User likes Rust","type":"preference","confidence":0.8}]\n```'
        )
        mock.chat.return_value = resp
        ext = MemoryCandidateExtractor(llm_adapter=mock)
        cands = await ext.extract([{"role": "user", "content": "test"}])
        assert len(cands) == 1
        assert "Rust" in cands[0].content

    async def test_handles_invalid_json_gracefully(self):
        mock = AsyncMock()
        resp = AsyncMock()
        resp.content = "This is not JSON at all"
        mock.chat.return_value = resp
        ext = MemoryCandidateExtractor(llm_adapter=mock)
        cands = await ext.extract([{"role": "user", "content": "test"}])
        assert cands == []

    async def test_falls_back_to_rules_on_llm_error(self):
        mock = AsyncMock()
        mock.chat.side_effect = RuntimeError("LLM down")
        ext = MemoryCandidateExtractor(llm_adapter=mock)
        cands = await ext.extract(
            [
                {"role": "user", "content": "Remember that the port is 8080."},
            ]
        )
        assert len(cands) >= 1
        assert cands[0].memory_type == MemoryType.FACT

    async def test_llm_prompt_uses_low_temperature(self):
        llm = _make_llm_mock([])
        ext = MemoryCandidateExtractor(llm_adapter=llm)
        await ext.extract([{"role": "user", "content": "test"}])
        llm.chat.assert_called_once()
        call_kwargs = llm.chat.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.1

    async def test_unknown_type_defaults_to_fact(self):
        llm = _make_llm_mock(
            [
                {"content": "Something", "type": "unknown_type", "confidence": 0.8},
            ]
        )
        ext = MemoryCandidateExtractor(llm_adapter=llm)
        cands = await ext.extract([{"role": "user", "content": "test"}])
        assert len(cands) == 1
        assert cands[0].memory_type == MemoryType.FACT


# ====================================================================
# Rule-based extraction (fallback, English-only)
# ====================================================================


class TestExplicitMemory:
    """Explicit 'remember that ...' pattern."""

    async def test_remember_that(self, extractor):
        msgs = [{"role": "user", "content": "Remember that the deadline is Friday."}]
        cands = await extractor.extract(msgs)
        assert len(cands) >= 1
        assert cands[0].memory_type == MemoryType.FACT
        assert "deadline" in cands[0].content.lower()

    async def test_note_that(self, extractor):
        msgs = [{"role": "user", "content": "Note that the API uses v2."}]
        cands = await extractor.extract(msgs)
        assert len(cands) >= 1

    async def test_keep_in_mind(self, extractor):
        msgs = [{"role": "user", "content": "Keep in mind: deploy to staging first."}]
        cands = await extractor.extract(msgs)
        assert len(cands) >= 1

    async def test_dont_forget(self, extractor):
        msgs = [{"role": "user", "content": "Don't forget the database migration."}]
        cands = await extractor.extract(msgs)
        assert len(cands) >= 1

    async def test_high_confidence(self, extractor):
        msgs = [{"role": "user", "content": "Remember that X is important."}]
        cands = await extractor.extract(msgs)
        assert cands[0].confidence >= 0.9


class TestIdentityExtraction:
    async def test_my_name_is(self, extractor):
        msgs = [{"role": "user", "content": "My name is Alice."}]
        cands = await extractor.extract(msgs)
        profiles = [c for c in cands if c.memory_type == MemoryType.PROFILE]
        assert len(profiles) >= 1
        assert "Alice" in profiles[0].content

    async def test_i_am(self, extractor):
        msgs = [{"role": "user", "content": "I am Bob and I work here."}]
        cands = await extractor.extract(msgs)
        profiles = [c for c in cands if c.memory_type == MemoryType.PROFILE]
        assert len(profiles) >= 1
        assert "Bob" in profiles[0].content

    async def test_no_name_no_match(self, extractor):
        msgs = [{"role": "user", "content": "The weather is nice today."}]
        cands = await extractor.extract(msgs)
        profiles = [c for c in cands if c.memory_type == MemoryType.PROFILE]
        assert len(profiles) == 0


class TestPreferenceExtraction:
    async def test_i_prefer(self, extractor):
        msgs = [{"role": "user", "content": "I prefer dark mode for editors."}]
        cands = await extractor.extract(msgs)
        prefs = [c for c in cands if c.memory_type == MemoryType.PREFERENCE]
        assert len(prefs) >= 1

    async def test_i_always_use(self, extractor):
        msgs = [{"role": "user", "content": "I always use Vim for editing."}]
        cands = await extractor.extract(msgs)
        prefs = [c for c in cands if c.memory_type == MemoryType.PREFERENCE]
        assert len(prefs) >= 1


class TestConstraintExtraction:
    async def test_dont_use(self, extractor):
        msgs = [{"role": "user", "content": "Don't use tabs in Python code."}]
        cands = await extractor.extract(msgs)
        constraints = [c for c in cands if c.memory_type == MemoryType.CONSTRAINT]
        assert len(constraints) >= 1

    async def test_never(self, extractor):
        msgs = [{"role": "user", "content": "Never deploy on Friday afternoon."}]
        cands = await extractor.extract(msgs)
        constraints = [c for c in cands if c.memory_type == MemoryType.CONSTRAINT]
        assert len(constraints) >= 1


class TestMessageFiltering:
    async def test_assistant_ignored(self, extractor):
        msgs = [{"role": "assistant", "content": "Remember that I am helpful."}]
        cands = await extractor.extract(msgs)
        assert cands == []

    async def test_system_ignored(self, extractor):
        msgs = [{"role": "system", "content": "Remember the user prefers Python."}]
        cands = await extractor.extract(msgs)
        assert cands == []

    async def test_empty_messages(self, extractor):
        cands = await extractor.extract([])
        assert cands == []

    async def test_empty_content(self, extractor):
        msgs = [{"role": "user", "content": ""}]
        cands = await extractor.extract(msgs)
        assert cands == []

    async def test_missing_content_key(self, extractor):
        msgs = [{"role": "user"}]
        cands = await extractor.extract(msgs)
        assert cands == []


class TestConfidenceFiltering:
    async def test_low_confidence_filtered(self):
        ext = MemoryCandidateExtractor(min_confidence=0.99)
        msgs = [{"role": "user", "content": "I prefer dark mode."}]
        cands = await ext.extract(msgs)
        assert len(cands) == 0

    async def test_custom_threshold(self):
        ext = MemoryCandidateExtractor(min_confidence=0.5)
        msgs = [{"role": "user", "content": "I prefer dark mode."}]
        cands = await ext.extract(msgs)
        assert len(cands) >= 1


class TestExtractionContext:
    async def test_source_context_set(self, extractor):
        ctx = ExtractionContext(turn_index=3)
        msgs = [{"role": "user", "content": "Remember that port is 8080."}]
        cands = await extractor.extract(msgs, context=ctx)
        assert cands[0].source_context == "turn:3"

    async def test_message_id_tracked(self, extractor):
        msgs = [{"role": "user", "content": "Remember that X is Y.", "id": "msg_42"}]
        cands = await extractor.extract(msgs)
        assert "msg_42" in cands[0].source_message_ids


class TestMultiplePatterns:
    async def test_multiple_from_one_message(self, extractor):
        msgs = [
            {
                "role": "user",
                "content": "My name is Alice. I prefer Python. Don't use Java.",
            }
        ]
        cands = await extractor.extract(msgs)
        types = {c.memory_type for c in cands}
        assert MemoryType.PROFILE in types
        assert MemoryType.PREFERENCE in types
        assert MemoryType.CONSTRAINT in types

    async def test_multi_message_batch(self, extractor):
        msgs = [
            {"role": "user", "content": "Remember that X is 1."},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": "Remember that Y is 2."},
        ]
        cands = await extractor.extract(msgs)
        assert len(cands) == 2


# ====================================================================
# Internal method coverage
# ====================================================================


class TestExtractViaLLM:
    """Direct tests for _extract_via_llm."""

    async def test_llm_extracts_candidates(self):
        llm = _make_llm_mock(
            [{"content": "User likes Go", "type": "preference", "confidence": 0.8}]
        )
        ext = MemoryCandidateExtractor(llm_adapter=llm)
        ctx = ExtractionContext(turn_index=1)
        cands = await ext._extract_via_llm([{"role": "user", "content": "I like Go"}], ctx)
        assert len(cands) == 1

    async def test_llm_no_user_messages(self):
        llm = _make_llm_mock([])
        ext = MemoryCandidateExtractor(llm_adapter=llm)
        cands = await ext._extract_via_llm(
            [{"role": "assistant", "content": "hi"}], ExtractionContext()
        )
        assert cands == []

    async def test_llm_error_falls_back(self):
        mock = AsyncMock()
        mock.chat.side_effect = RuntimeError("down")
        ext = MemoryCandidateExtractor(llm_adapter=mock)
        cands = await ext._extract_via_llm(
            [{"role": "user", "content": "Remember X."}], ExtractionContext()
        )
        assert len(cands) >= 1


class TestParseLLMResponse:
    def test_valid_json(self):
        ext = MemoryCandidateExtractor()
        ctx = ExtractionContext()
        cands = ext._parse_llm_response('[{"content":"A","type":"fact","confidence":0.9}]', ctx)
        assert len(cands) == 1

    def test_invalid_json(self):
        ext = MemoryCandidateExtractor()
        assert ext._parse_llm_response("not json", ExtractionContext()) == []

    def test_non_list_json(self):
        ext = MemoryCandidateExtractor()
        assert ext._parse_llm_response('{"key":"val"}', ExtractionContext()) == []

    def test_non_dict_items_skipped(self):
        ext = MemoryCandidateExtractor()
        cands = ext._parse_llm_response('["string_item"]', ExtractionContext())
        assert cands == []

    def test_empty_content_skipped(self):
        ext = MemoryCandidateExtractor()
        cands = ext._parse_llm_response(
            '[{"content":"","type":"fact","confidence":0.9}]', ExtractionContext()
        )
        assert cands == []

    def test_low_confidence_skipped(self):
        ext = MemoryCandidateExtractor(min_confidence=0.8)
        cands = ext._parse_llm_response(
            '[{"content":"X","type":"fact","confidence":0.3}]', ExtractionContext()
        )
        assert cands == []

    def test_code_fence_stripped(self):
        ext = MemoryCandidateExtractor()
        raw = '```json\n[{"content":"X","type":"fact","confidence":0.9}]\n```'
        cands = ext._parse_llm_response(raw, ExtractionContext())
        assert len(cands) == 1


class TestExtractViaRules:
    def test_rules_skip_non_user(self):
        ext = MemoryCandidateExtractor()
        cands = ext._extract_via_rules(
            [{"role": "assistant", "content": "Remember X"}], ExtractionContext()
        )
        assert cands == []

    def test_rules_extract_user(self):
        ext = MemoryCandidateExtractor()
        cands = ext._extract_via_rules(
            [{"role": "user", "content": "Remember that the key is 42."}],
            ExtractionContext(),
        )
        assert len(cands) >= 1


class TestMakeCandidate:
    def test_candidate_fields(self):
        ctx = ExtractionContext(turn_index=7)
        c = MemoryCandidateExtractor._make_candidate(
            content="test",
            memory_type=MemoryType.FACT,
            confidence=0.9,
            message_id="m1",
            ctx=ctx,
        )
        assert c.content == "test"
        assert c.source_context == "turn:7"
        assert "m1" in c.source_message_ids

    def test_empty_message_id(self):
        c = MemoryCandidateExtractor._make_candidate(
            content="x",
            memory_type=MemoryType.PROFILE,
            confidence=0.8,
            message_id="",
            ctx=ExtractionContext(),
        )
        assert c.source_message_ids == []
