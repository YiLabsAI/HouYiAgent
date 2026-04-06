from __future__ import annotations

from unittest.mock import AsyncMock

from houyi.adapters.memory.builder import MemoryCandidateBuilder
from houyi.adapters.memory.types import (
    MemoryBuildInput,
    MemoryBuildItem,
    MemoryScope,
    MemorySourceKind,
    MemoryType,
)


class TestBuildConversation:
    async def test_build_conversation_rule(self):
        builder = MemoryCandidateBuilder()
        memory_input = MemoryBuildInput(
            source_type=MemorySourceKind.CONVERSATION,
            scope=MemoryScope.USER,
            items=[MemoryBuildItem(content="Remember that the port is 8080.", role="user")],
        )
        candidates = await builder.build(memory_input)
        assert len(candidates) == 1
        assert candidates[0].memory_type == MemoryType.FACT
        assert candidates[0].source_type == MemorySourceKind.CONVERSATION.value

    async def test_build_conversation_llm(self):
        llm = AsyncMock()
        response = AsyncMock()
        response.content = (
            '[{"content":"User prefers Python","type":"preference","confidence":0.9}]'
        )
        llm.chat.return_value = response
        builder = MemoryCandidateBuilder(llm_adapter=llm)
        memory_input = MemoryBuildInput(
            source_type=MemorySourceKind.CONVERSATION,
            scope=MemoryScope.USER,
            items=[MemoryBuildItem(content="I like Python", role="user")],
        )
        candidates = await builder.build(memory_input)
        assert len(candidates) == 1
        assert candidates[0].memory_type == MemoryType.PREFERENCE


class TestBuildStructured:
    async def test_build_search_items(self):
        builder = MemoryCandidateBuilder()
        memory_input = MemoryBuildInput(
            source_type=MemorySourceKind.SEARCH,
            scope=MemoryScope.USER,
            source_context="deep_research",
            items=[
                MemoryBuildItem(
                    content="Overview: source-backed finding",
                    role="report_section",
                    source_ids=["sec_1"],
                    suggested_tags=["research"],
                    metadata={"kind": "report_section"},
                )
            ],
            metadata={"run_id": "rr_test"},
        )
        candidates = await builder.build(memory_input)
        assert len(candidates) == 1
        assert candidates[0].source_type == MemorySourceKind.SEARCH.value
        assert candidates[0].source_context == "deep_research"
        assert candidates[0].metadata["run_id"] == "rr_test"

    async def test_build_auto_dream(self):
        builder = MemoryCandidateBuilder(min_confidence=0.5)
        memory_input = MemoryBuildInput(
            source_type=MemorySourceKind.AUTO_DREAM,
            scope=MemoryScope.USER,
            items=[MemoryBuildItem(content="Explore a long-term agent roadmap", role="dream_note")],
        )
        candidates = await builder.build(memory_input)
        assert len(candidates) == 1
        assert candidates[0].memory_type == MemoryType.PROJECT
        assert candidates[0].source_type == MemorySourceKind.AUTO_DREAM.value
