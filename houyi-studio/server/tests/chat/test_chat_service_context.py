from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from houyi_studio.server.chat.chat_service import ChatService
from houyi_studio.server.chat.types import Message, MessageRole

from houyi.application.context.context_selection import (
    build_default_context_selection_policy as build_chat_context_selection_policy,
)
from houyi.application.context.context_sources import (
    assemble_context_candidates as build_chat_context_candidates,
)
from houyi.application.context.types import ContextBlockType, ContextSourceKind


class TestBuildChatContextSelectionPolicy:
    def test_defaults(self):
        policy = build_chat_context_selection_policy()

        assert policy.policy_name == "chat_default"
        assert policy.allow_memory is True
        assert policy.allow_tool_summaries is True
        assert policy.allow_pinned is True


class TestBuildChatContextCandidates:
    def test_structured_recent(self):
        candidates = build_chat_context_candidates(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "How are you?"},
            ],
            system_instructions="You are helpful",
            memory_context="User prefers concise replies",
            boundary_id="boundary-chat",
        )

        assert [candidate.source for candidate in candidates] == [
            ContextSourceKind.SYSTEM,
            ContextSourceKind.MEMORY,
            ContextSourceKind.CURRENT_TURN,
            ContextSourceKind.RECENT,
            ContextSourceKind.RECENT,
        ]
        assert [candidate.block_type for candidate in candidates] == [
            ContextBlockType.SYSTEM,
            ContextBlockType.MEMORY,
            ContextBlockType.RECENT,
            ContextBlockType.RECENT,
            ContextBlockType.RECENT,
        ]
        assert candidates[2].content == [{"role": "user", "content": "How are you?"}]
        assert candidates[3].content == [{"role": "user", "content": "Hello"}]
        assert candidates[4].content == [{"role": "assistant", "content": "Hi there"}]
        assert candidates[2].metadata["boundary_id"] == "boundary-chat"
        assert candidates[3].metadata["boundary_id"] == "boundary-chat"
        assert candidates[4].metadata["boundary_id"] == "boundary-chat"

    def test_summary_candidate(
        self,
    ):
        candidates = build_chat_context_candidates(
            messages=[{"role": "user", "content": "How are you?"}],
            system_instructions="You are helpful",
            memory_context=None,
            summary_context="Earlier decisions",
            summary_metadata={"compaction_id": "cmp_1"},
        )

        assert [candidate.source for candidate in candidates] == [
            ContextSourceKind.SYSTEM,
            ContextSourceKind.SUMMARY,
            ContextSourceKind.CURRENT_TURN,
        ]
        assert candidates[1].block_type == ContextBlockType.SUMMARY
        assert candidates[1].content == "Earlier decisions"
        assert candidates[1].metadata["compaction_id"] == "cmp_1"


class TestChatServiceContextMessages:
    def test_drop_reasons(self):
        service = ChatService(json_store=MagicMock())
        conversation = SimpleNamespace(
            messages=[
                Message(role=MessageRole.ASSISTANT, content="older " + ("y" * 20)),
                Message(role=MessageRole.USER, content="current " + ("x" * 400)),
            ]
        )

        class _Span:
            def __init__(self):
                self.attributes = {}

            def set_attribute(self, key, value):
                self.attributes[key] = value

        span = _Span()

        llm_messages, context_usage = service._build_context_messages(
            conversation=conversation,
            model="test-model",
            sys_instructions="Sys",
            span=span,
            input_budget=40,
        )

        assert context_usage["dropped_blocks"]
        assert context_usage["drop_reasons"]
        assert context_usage["dropped_block_details"]
        assert (
            context_usage["dropped_block_details"][0]["candidate_id"]
            in context_usage["drop_reasons"]
        )
        assert all(message["content"] != "older " + ("y" * 20) for message in llm_messages)
        assert span.attributes["chat.context_blocks"]
        assert span.attributes["chat.context_dropped_blocks"]
        assert span.attributes["chat.context_drop_reasons"]
        assert span.attributes["chat.context_dropped_block_details"]

    def test_large_conversation(
        self,
    ):
        service = ChatService(json_store=MagicMock())
        conversation = SimpleNamespace(
            messages=[
                Message(
                    role=MessageRole.USER if index % 2 == 0 else MessageRole.ASSISTANT,
                    content=f"turn-{index} " + ("x" * 120),
                )
                for index in range(520)
            ]
        )

        class _Span:
            def __init__(self):
                self.attributes = {}

            def set_attribute(self, key, value):
                self.attributes[key] = value

        span = _Span()
        llm_messages, context_usage = service._build_context_messages(
            conversation=conversation,
            model="test-model",
            sys_instructions="Sys",
            span=span,
            input_budget=1200,
        )

        assert llm_messages
        assert llm_messages[-1]["content"].startswith("turn-519")
        assert context_usage["used_tokens"] > 0
        assert context_usage["max_context_tokens"] > context_usage["used_tokens"]
        assert context_usage["block_breakdown"]
        assert span.attributes["chat.context_tokens_used"] == context_usage["used_tokens"]
        assert span.attributes["chat.context_tokens_max"] == context_usage["max_context_tokens"]

    def test_preserve_recent_boundary(self):
        service = ChatService(json_store=MagicMock())
        conversation = SimpleNamespace(
            messages=[
                Message(role=MessageRole.USER, content="Investigate old repo issue"),
                Message(role=MessageRole.ASSISTANT, content="I checked the previous context"),
                Message(role=MessageRole.USER, content="Now focus on the current task only"),
            ]
        )

        class _Span:
            def __init__(self):
                self.attributes = {}

            def set_attribute(self, key, value):
                self.attributes[key] = value

        llm_messages, context_usage = service._build_context_messages(
            conversation=conversation,
            model="test-model",
            sys_instructions="Sys",
            span=_Span(),
            input_budget=512,
        )

        recent_messages = [message for message in llm_messages if message["role"] != "system"]
        assert recent_messages == [
            {"role": "user", "content": "Investigate old repo issue"},
            {"role": "assistant", "content": "I checked the previous context"},
            {"role": "user", "content": "Now focus on the current task only"},
        ]
        assert context_usage["drop_reasons"] == {}

    def test_latest_fallback(self):
        service = ChatService(json_store=MagicMock())
        conversation = SimpleNamespace(
            messages=[
                Message(role=MessageRole.USER, content="older"),
                Message(role=MessageRole.USER, content="latest user turn"),
            ]
        )

        class _Span:
            def __init__(self):
                self.attributes = {}

            def set_attribute(self, key, value):
                self.attributes[key] = value

        llm_messages, context_usage = service._build_context_messages(
            conversation=conversation,
            model="test-model",
            sys_instructions="Sys",
            span=_Span(),
            input_budget=0,
        )

        assert llm_messages == [{"role": "user", "content": "latest user turn"}]
        assert context_usage["used_tokens"] > 0

    def test_active_pins(
        self,
    ):
        service = ChatService(json_store=MagicMock())
        conversation = SimpleNamespace(
            messages=[
                Message(role=MessageRole.USER, content="Current task"),
            ],
            metadata={
                "pinned_contexts": [
                    {
                        "pin_id": "pin_active",
                        "conversation_id": "conv_1",
                        "source_message_id": "u1",
                        "title": "Constraint",
                        "content": "Always deploy to staging first.",
                        "role": "user",
                        "scope": "conversation",
                        "status": "active",
                        "priority": 5,
                        "token_count": 8,
                        "metadata": {"origin_message_id": "u1"},
                    },
                    {
                        "pin_id": "pin_archived",
                        "conversation_id": "conv_1",
                        "source_message_id": "u2",
                        "title": "Old constraint",
                        "content": "Ignore this archived pin.",
                        "role": "user",
                        "scope": "conversation",
                        "status": "archived",
                        "priority": 5,
                        "token_count": 6,
                        "metadata": {"origin_message_id": "u2"},
                    },
                ]
            },
        )

        class _Span:
            def __init__(self):
                self.attributes = {}

            def set_attribute(self, key, value):
                self.attributes[key] = value

        llm_messages, context_usage = service._build_context_messages(
            conversation=conversation,
            model="test-model",
            sys_instructions="Sys",
            span=_Span(),
            input_budget=256,
        )

        rendered_text = "\n".join(str(message.get("content", "")) for message in llm_messages)
        assert "Always deploy to staging first." in rendered_text
        assert "Ignore this archived pin." not in rendered_text
        assert "pinned" in context_usage["block_breakdown"]

        usage = service._context_runtime.get_context_usage(
            conversation=conversation,
            model="test-model",
            sys_instructions="Sys",
        )
        assert usage is not None
        assert "pinned" in usage["block_breakdown"]

    def test_pins_over_recent(
        self,
    ):
        service = ChatService(json_store=MagicMock())
        conversation = SimpleNamespace(
            messages=[
                Message(
                    role=MessageRole.ASSISTANT,
                    content="Older assistant fact " + ("x " * 400),
                ),
                Message(role=MessageRole.USER, content="Current task must remain visible"),
            ],
            metadata={
                "pinned_contexts": [
                    {
                        "pin_id": "pin_active",
                        "conversation_id": "conv_1",
                        "source_message_id": "u1",
                        "title": "Constraint",
                        "content": "Always deploy to staging first.",
                        "role": "user",
                        "scope": "conversation",
                        "status": "active",
                        "priority": 500,
                        "token_count": 8,
                        "metadata": {"origin_message_id": "u1"},
                    }
                ]
            },
        )

        class _Span:
            def __init__(self):
                self.attributes = {}

            def set_attribute(self, key, value):
                self.attributes[key] = value

        llm_messages, context_usage = service._build_context_messages(
            conversation=conversation,
            model="test-model",
            sys_instructions="Sys",
            span=_Span(),
            input_budget=40,
        )

        rendered_text = "\n".join(str(message.get("content", "")) for message in llm_messages)
        assert "Always deploy to staging first." in rendered_text
        assert "Current task must remain visible" in rendered_text
        assert "Older assistant fact" not in rendered_text
        assert "pinned" in context_usage["block_breakdown"]

    def test_summary_block(self):
        service = ChatService(json_store=MagicMock())
        conversation = SimpleNamespace(
            messages=[
                Message(role=MessageRole.USER, content="Deploy via staging first for this task")
            ],
            metadata={
                "compaction_history": [
                    {
                        "compaction_id": "cmp_1",
                        "trigger": "manual",
                        "summary": "Earlier discussion decided to deploy via staging first.",
                        "metadata": {"summary_model": "summary-mini"},
                    }
                ]
            },
        )

        class _Span:
            def __init__(self):
                self.attributes = {}

            def set_attribute(self, key, value):
                self.attributes[key] = value

        llm_messages, context_usage = service._build_context_messages(
            conversation=conversation,
            model="test-model",
            sys_instructions="Sys",
            span=_Span(),
            input_budget=256,
        )

        rendered_text = "\n".join(str(message.get("content", "")) for message in llm_messages)
        assert "[Conversation Summary]" in rendered_text
        assert "Earlier discussion decided to deploy via staging first." in rendered_text
        assert "summary" in context_usage["block_breakdown"]

        usage = service._context_runtime.get_context_usage(
            conversation=conversation,
            model="test-model",
            sys_instructions="Sys",
        )
        assert usage is not None
        assert "summary" in usage["block_breakdown"]

    def test_tool_summary_block(self):
        service = ChatService(json_store=MagicMock())
        conversation = SimpleNamespace(
            messages=[
                Message(role=MessageRole.USER, content="Find the config"),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="calling tools",
                    tool_calls=[{"id": "call-1", "type": "function"}],
                ),
                Message(
                    role=MessageRole.TOOL,
                    name="read_file",
                    tool_call_id="call-1",
                    content='{"data":{"content":"RAW TOOL PAYLOAD"}}',
                    metadata={
                        "tool_result_profile": {
                            "compressed": True,
                            "tool_category": "read",
                            "compression_strategy": "excerpt",
                            "tokens_before": 120,
                            "tokens_after": 24,
                            "summary": "Found the config path in settings.py",
                        }
                    },
                ),
                Message(role=MessageRole.USER, content="What changed?"),
            ]
        )

        class _Span:
            def __init__(self):
                self.attributes = {}

            def set_attribute(self, key, value):
                self.attributes[key] = value

        llm_messages, context_usage = service._build_context_messages(
            conversation=conversation,
            model="test-model",
            sys_instructions="Sys",
            span=_Span(),
            input_budget=256,
        )

        rendered_text = "\n".join(str(message.get("content", "")) for message in llm_messages)
        assert "[Tool Results Summary]" in rendered_text
        assert "read_file: Found the config path in settings.py" in rendered_text
        assert "RAW TOOL PAYLOAD" not in rendered_text
        assert "tool_summary" in context_usage["block_breakdown"]

        usage = service._context_runtime.get_context_usage(
            conversation=conversation,
            model="test-model",
            sys_instructions="Sys",
        )
        assert usage is not None
        assert "tool_summary" in usage["block_breakdown"]

    def test_recent_over_summary(self):
        service = ChatService(json_store=MagicMock())
        conversation = SimpleNamespace(
            messages=[
                Message(role=MessageRole.USER, content="Older repo fact"),
                Message(role=MessageRole.ASSISTANT, content="Older assistant fact"),
                Message(role=MessageRole.USER, content="Current task must remain visible"),
            ],
            metadata={
                "compaction_history": [
                    {
                        "compaction_id": "cmp_1",
                        "trigger": "manual",
                        "summary": "Earlier compressed summary " + ("s " * 800),
                        "metadata": {"summary_model": "summary-mini"},
                    }
                ]
            },
        )

        class _Span:
            def __init__(self):
                self.attributes = {}

            def set_attribute(self, key, value):
                self.attributes[key] = value

        llm_messages, context_usage = service._build_context_messages(
            conversation=conversation,
            model="test-model",
            sys_instructions="Sys",
            span=_Span(),
            input_budget=80,
        )

        rendered_text = "\n".join(str(message.get("content", "")) for message in llm_messages)
        assert "Current task must remain visible" in rendered_text
        assert "Older assistant fact" in rendered_text
        assert "Earlier compressed summary" not in rendered_text
        assert "summary" not in context_usage["block_breakdown"]

    def test_current_turn_and_tool_summary_beat_stale_compaction_summary(self):
        service = ChatService(json_store=MagicMock())
        conversation = SimpleNamespace(
            messages=[
                Message(role=MessageRole.USER, content="Old repo context"),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="calling tools",
                    tool_calls=[{"id": "call-1", "type": "function"}],
                ),
                Message(
                    role=MessageRole.TOOL,
                    name="houyi_web_search",
                    tool_call_id="call-1",
                    content='{"data":{"content":"RAW HISTORICAL TOOL PAYLOAD"}}',
                    metadata={
                        "tool_result_profile": {
                            "compressed": True,
                            "tool_category": "search",
                            "compression_strategy": "top_k",
                            "tokens_before": 220,
                            "tokens_after": 28,
                            "summary": "Found the answer in deploy-runbook.md",
                        }
                    },
                ),
                Message(
                    role=MessageRole.ASSISTANT, content="Historical assistant reply " + ("x " * 120)
                ),
                Message(role=MessageRole.USER, content="Current task must remain visible"),
            ],
            metadata={
                "compaction_history": [
                    {
                        "compaction_id": "cmp_1",
                        "trigger": "manual",
                        "summary": "Earlier compressed summary " + ("stale " * 500),
                        "metadata": {"summary_model": "summary-mini"},
                    }
                ]
            },
        )

        class _Span:
            def __init__(self):
                self.attributes = {}

            def set_attribute(self, key, value):
                self.attributes[key] = value

        llm_messages, context_usage = service._build_context_messages(
            conversation=conversation,
            model="test-model",
            sys_instructions="Sys",
            span=_Span(),
            input_budget=110,
        )

        rendered_text = "\n".join(str(message.get("content", "")) for message in llm_messages)
        assert "Current task must remain visible" in rendered_text
        assert "Found the answer in deploy-runbook.md" in rendered_text
        assert "RAW HISTORICAL TOOL PAYLOAD" not in rendered_text
        assert "Earlier compressed summary" not in rendered_text
        assert "Historical assistant reply" not in rendered_text
        assert context_usage["block_breakdown"].get("tool_summary", 0) > 0
        assert "summary" not in context_usage["block_breakdown"]

    def test_memory_over_assistant(self, monkeypatch):
        service = ChatService(json_store=MagicMock())
        conversation = SimpleNamespace(
            messages=[
                Message(role=MessageRole.USER, content="Older repo fact"),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="Older assistant fact " + ("x " * 400),
                ),
                Message(role=MessageRole.USER, content="Current task must remain visible"),
            ]
        )

        class _Span:
            def __init__(self):
                self.attributes = {}

            def set_attribute(self, key, value):
                self.attributes[key] = value

        monkeypatch.setattr(
            service._context_runtime,
            "_get_memory_text",
            lambda *, memory_store, span: "Stable memory rule",
        )

        llm_messages, context_usage = service._build_context_messages(
            conversation=conversation,
            model="test-model",
            sys_instructions="Sys",
            span=_Span(),
            input_budget=50,
        )

        rendered_text = "\n".join(str(message.get("content", "")) for message in llm_messages)
        assert "Current task must remain visible" in rendered_text
        assert "Older repo fact" in rendered_text
        assert "Stable memory rule" in rendered_text
        assert "Older assistant fact" not in rendered_text
        assert "memory" in context_usage["block_breakdown"]


class TestChatServiceCompactionSummary:
    @pytest.mark.asyncio
    async def test_summary_model(self, monkeypatch):
        service = ChatService(json_store=MagicMock())
        monkeypatch.setenv("HOUYI_CHAT_SUMMARY_MODEL", "summary-model")
        captured = {}

        class _Adapter:
            async def chat(self, messages, tools=None, temperature=0.7, max_tokens=None, **kwargs):
                _ = (tools, temperature)
                captured["messages"] = messages
                captured["model"] = kwargs.get("model")
                captured["max_tokens"] = max_tokens
                return SimpleNamespace(content="Condensed carryover summary")

        service._get_adapter_for_model = lambda model: _Adapter()

        result = await service._build_compaction_summary(
            [Message(role=MessageRole.USER, content="Important earlier discussion")],
            model="main-model",
        )

        assert result.text == "Condensed carryover summary"
        assert result.model == "summary-model"
        assert result.mode == "llm"
        assert captured["model"] == "summary-model"
        assert captured["max_tokens"] == 256
        assert len(captured["messages"]) == 2
