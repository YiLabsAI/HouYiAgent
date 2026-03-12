from types import SimpleNamespace
from unittest.mock import MagicMock

from houyi_studio.server.chat.chat_service import (
    ChatService,
    _build_chat_context_candidates,
    _build_chat_context_selection_policy,
    _build_generation_metadata,
    _extract_finish_reason,
    _finalize_stream_result,
    _normalize_usage_payload,
)
from houyi_studio.server.chat.types import Message, MessageRole

from houyi.application.context.types import ContextBlockType, ContextSourceKind, TaskBoundary


class TestNormalizeUsagePayload:
    def test_maps_prompt_tokens(self):
        usage = _normalize_usage_payload(
            {
                "input_tokens": 12,
                "completion_tokens": 8,
                "reasoning_tokens": 3,
            }
        )

        assert usage is not None
        assert usage["prompt_tokens"] == 12
        assert usage["input_tokens"] == 12
        assert usage["answer_tokens"] == 5
        assert usage["total_tokens"] == 20
        assert usage["usage_confidence"] == "reported"

    def test_keeps_timing_metrics(self):
        usage = _normalize_usage_payload(
            {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "first_token_ms": 120,
                "decode_tokens_per_second": 40,
                "end_to_end_tokens_per_second": 25,
            }
        )

        assert usage is not None
        assert usage["first_token_ms"] == 120.0
        assert usage["decode_tokens_per_second"] == 40.0
        assert usage["end_to_end_tokens_per_second"] == 25.0


class TestBuildGenerationMetadata:
    def test_reads_usage_timing(self):
        metadata = _build_generation_metadata(
            usage_payload={
                "first_token_ms": 111,
                "decode_tokens_per_second": 22.2,
                "end_to_end_tokens_per_second": 11.1,
            },
            first_token_ms=115,
            generation_time_ms=800,
        )

        assert metadata["first_token_latency_ms"] == 115
        assert metadata["generation_time_ms"] == 800
        assert metadata["first_token_ms"] == 111
        assert metadata["decode_tokens_per_second"] == 22.2
        assert metadata["end_to_end_tokens_per_second"] == 11.1
        assert metadata["tokens_per_second"] == 11.1


class TestFinalizeStreamResult:
    def test_finalizes_usage(self):
        class _Span:
            def __init__(self):
                self.attributes = {}
                self.tokens = None
                self.status = None
                self.ended = False

            def set_attribute(self, key, value):
                self.attributes[key] = value

            def set_tokens(self, *, input_tokens, output_tokens):
                self.tokens = (input_tokens, output_tokens)

            def set_status(self, status, message=None):
                self.status = (status, message)

            def end(self):
                self.ended = True

        class _Adapter:
            last_usage = {"input_tokens": 12, "completion_tokens": 8, "reasoning_tokens": 3}
            last_finish_reason = "stop"

        span = _Span()
        usage, finish_reason, metadata = _finalize_stream_result(
            llm_adapter=_Adapter(),
            llm_span=span,
            first_token_ms=120,
            generation_time_ms=800,
            chunk_count=4,
        )

        assert usage is not None
        assert usage["prompt_tokens"] == 12
        assert finish_reason == "stop"
        assert metadata["generation_time_ms"] == 800
        assert span.attributes["chat.stream_chunk_count"] == 4
        assert span.tokens == (12, 8)
        assert span.ended is True

    def test_uses_finish_reason_fallback(self):
        class _Span:
            def __init__(self):
                self.attributes = {}
                self.tokens = None
                self.status = None
                self.ended = False

            def set_attribute(self, key, value):
                self.attributes[key] = value

            def set_tokens(self, *, input_tokens, output_tokens):
                self.tokens = (input_tokens, output_tokens)

            def set_status(self, status, message=None):
                self.status = (status, message)

            def end(self):
                self.ended = True

        class _Adapter:
            last_usage = {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
            last_finish_reason = None

        span = _Span()
        usage, finish_reason, metadata = _finalize_stream_result(
            llm_adapter=_Adapter(),
            llm_span=span,
            first_token_ms=80,
            generation_time_ms=500,
            chunk_count=2,
            finish_reason_sources=({"metadata": {"finish_reason": "tool_calls"}},),
        )

        assert usage is not None
        assert finish_reason == "tool_calls"
        assert metadata["generation_time_ms"] == 500
        assert span.ended is True


class TestExtractFinishReason:
    def test_prefers_direct_string_then_nested_metadata(self):
        assert _extract_finish_reason(None, "", "length", {"finish_reason": "stop"}) == "length"
        assert _extract_finish_reason({"metadata": {"finish_reason": "tool_calls"}}) == "tool_calls"


class TestBuildChatContextSelectionPolicy:
    def test_defaults(self):
        policy = _build_chat_context_selection_policy()

        assert policy.policy_name == "chat_default"
        assert policy.allow_memory is True
        assert policy.allow_tool_summaries is True
        assert policy.allow_pinned is True


class TestBuildChatContextCandidates:
    def test_builds_structured_candidates(self):
        task_boundary = TaskBoundary(
            boundary_id="boundary-chat",
            task_kind="chat",
            scope="conversation",
        )
        candidates = _build_chat_context_candidates(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "How are you?"},
            ],
            system_instructions="You are helpful",
            memory_context="User prefers concise replies",
            task_boundary=task_boundary,
        )

        assert [candidate.source for candidate in candidates] == [
            ContextSourceKind.SYSTEM,
            ContextSourceKind.MEMORY,
            ContextSourceKind.CURRENT_TURN,
            ContextSourceKind.RECENT,
        ]
        assert [candidate.block_type for candidate in candidates] == [
            ContextBlockType.SYSTEM,
            ContextBlockType.MEMORY,
            ContextBlockType.RECENT,
            ContextBlockType.RECENT,
        ]
        assert candidates[2].content == [{"role": "user", "content": "How are you?"}]
        assert candidates[3].content == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        assert candidates[2].metadata["boundary_id"] == "boundary-chat"
        assert candidates[2].metadata["task_kind"] == "chat"
        assert candidates[2].metadata["scope"] == "conversation"
        assert candidates[3].metadata["boundary_id"] == "boundary-chat"


class TestBuildContextMessages:
    def test_exposes_drop_reasons(self):
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

        llm_messages, context_usage = service._build_context_messages(
            conversation=conversation,
            model="test-model",
            sys_instructions="Sys",
            chat_span=_Span(),
            input_budget=40,
        )

        assert context_usage["dropped_blocks"]
        assert context_usage["drop_reasons"]
        assert all(message["content"] != "older " + ("y" * 20) for message in llm_messages)

    def test_preserves_task_boundary(self):
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
            chat_span=_Span(),
            input_budget=512,
        )

        recent_messages = [message for message in llm_messages if message["role"] != "system"]
        assert recent_messages == [
            {"role": "user", "content": "Investigate old repo issue"},
            {"role": "assistant", "content": "I checked the previous context"},
            {"role": "user", "content": "Now focus on the current task only"},
        ]
        assert context_usage["drop_reasons"] == {}

    def test_fallback_latest_message(self):
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
            chat_span=_Span(),
            input_budget=0,
        )

        assert llm_messages == [{"role": "user", "content": "latest user turn"}]
        assert context_usage["used_tokens"] > 0
