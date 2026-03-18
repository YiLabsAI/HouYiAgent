from pathlib import Path

import pytest
from houyi_studio.server.chat.context_compressor import (
    ContextCompressor,
    SummaryBuildResult,
)
from houyi_studio.server.chat.json_store import JsonStore
from houyi_studio.server.chat.types import (
    ActiveStreamingState,
    Conversation,
    ConversationContextState,
    Message,
    MessageRole,
)

from houyi.application.context.compaction_summary import build_compaction_summary
from houyi.application.context.context_lifecycle import (
    ContextLifecycleHookService as ChatContextHookService,
)


class _Span:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


@pytest.fixture
def store(tmp_path: Path) -> JsonStore:
    return JsonStore(data_dir=tmp_path)


def test_build_compaction_summary() -> None:
    summary = build_compaction_summary(
        [
            Message(message_id="u1", role=MessageRole.USER, content="搜索文件 skill.md"),
            Message(
                message_id="a1",
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "code_search", "arguments": '{"query":"skill.md"}'},
                    }
                ],
            ),
            Message(
                message_id="t1",
                role=MessageRole.TOOL,
                name="code_search",
                tool_call_id="call_1",
                content='{"data":{"matches":[],"pattern":"skill.md","root_path":"/tmp/repo","truncated":false},"meta":{"ok":true}}',
            ),
        ]
    )

    assert "assistant: [tool loop: code_search]" in summary
    assert "tool: code_search search 'skill.md' returned 0 match(es)" in summary
    assert "root_path" not in summary
    assert '"truncated"' not in summary


@pytest.mark.asyncio
async def test_repo_intent_compacts(store: JsonStore):
    conv = Conversation(title="Repo")
    conv.messages = [
        Message(message_id=f"m{index}", role=MessageRole.USER, content=f"older message {index}")
        for index in range(1, 8)
    ]
    conv.conversation_context_state = ConversationContextState(
        conversation_id=conv.conversation_id,
        used_units=750,
        max_units=1000,
        state="elevated",
        updated_at=1.0,
    )
    conv = store.create(conv)
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="Read README from https://github.com/foo/bar",
        conv_lock=lock,
        chat_span=span,
    )

    assert outcome.record is not None
    assert outcome.compaction_event is not None
    assert len(outcome.conversation_snapshot.messages) == 6
    assert outcome.record.trigger == "pre_request_pressure"
    assert outcome.record.pressure_level == "elevated"
    assert outcome.record.backup_id
    persisted = store.get(conv.conversation_id)
    assert persisted is not None
    history = persisted.metadata.get("compaction_history")
    assert isinstance(history, list) and history
    assert history[-1]["backup_id"] == outcome.record.backup_id
    backup = store.get_backup(outcome.record.backup_id)
    assert backup is not None
    assert backup["record_id"] == outcome.record.compaction_id
    assert span.attributes["chat.compaction.triggered"] is True
    assert span.attributes["chat.compaction.restore_status"] == "ready"
    assert span.attributes["chat.compaction.utilization_source"] == "repo_intent_override"
    assert outcome.record.metadata["utilization_source"] == "repo_intent_override"


@pytest.mark.asyncio
async def test_repo_intent_low_pressure(store: JsonStore):
    conv = Conversation(title="Repo Low")
    conv.messages = [
        Message(message_id=f"m{index}", role=MessageRole.USER, content=f"older message {index}")
        for index in range(1, 8)
    ]
    conv.conversation_context_state = ConversationContextState(
        conversation_id=conv.conversation_id,
        used_units=320,
        max_units=1000,
        state="healthy",
        updated_at=1.0,
    )
    conv = store.create(conv)
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="Read README from https://github.com/foo/bar",
        conv_lock=lock,
        chat_span=span,
    )

    assert outcome.record is None
    assert outcome.compaction_event is None
    assert span.attributes == {}


@pytest.mark.asyncio
async def test_record_fields_populated(store: JsonStore):
    conv = Conversation(title="Record")
    conv.messages = [
        Message(message_id=f"m{index}", role=MessageRole.USER, content=f"older message {index}")
        for index in range(1, 8)
    ]
    conv = store.create(conv)
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="Read README from https://github.com/foo/bar",
        conv_lock=lock,
        chat_span=span,
    )

    assert outcome.record is not None
    assert outcome.record.backup_id
    assert outcome.record.pruned_block_ids == ["m1"]
    assert outcome.record.summarized_block_ids == ["m1"]
    assert outcome.record.protected_block_ids == ["m2", "m3", "m4", "m5", "m6", "m7"]
    assert outcome.record.oversized_block_ids == []
    assert outcome.record.active_turn_protected is True
    assert outcome.record.cooldown_applied is False
    assert outcome.record.restore_status == "ready"
    assert outcome.record.metadata["low_watermark"] == 0.6
    assert outcome.record.metadata["high_watermark"] == 0.7
    assert outcome.record.metadata["critical_watermark"] == 0.9
    assert isinstance(outcome.record.metadata["target_low_watermark_met"], bool)


@pytest.mark.asyncio
async def test_compaction_attrs(store: JsonStore):
    conv = Conversation(title="Attrs")
    conv.messages = [
        Message(message_id=f"m{index}", role=MessageRole.USER, content=f"older message {index}")
        for index in range(1, 8)
    ]
    conv = store.create(conv)
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="Read README from https://github.com/foo/bar",
        conv_lock=lock,
        chat_span=span,
    )

    assert outcome.record is not None
    assert span.attributes["chat.compaction.triggered"] is True
    assert span.attributes["chat.compaction.tokens_before"] > 0
    assert span.attributes["chat.compaction.tokens_after"] > 0
    assert span.attributes["chat.compaction.compression_ratio"] > 0
    assert span.attributes["chat.compaction.pin_violation_count"] == 0
    assert span.attributes["chat.compaction.blocks_dropped_count"] == 1
    assert span.attributes["chat.compaction.blocks_summarized_count"] == 1
    assert span.attributes["chat.compaction.recent_messages_kept"] == 6
    assert span.attributes["chat.compaction.summary_source_messages"] == 1
    assert span.attributes["chat.compaction.low_watermark"] == 0.6
    assert span.attributes["chat.compaction.high_watermark"] == 0.7
    assert span.attributes["chat.compaction.critical_watermark"] == 0.9
    assert isinstance(span.attributes["chat.compaction.target_low_watermark_met"], bool)


@pytest.mark.asyncio
async def test_prune_runs(store: JsonStore):
    conv = Conversation(title="Stage Order")
    conv.messages = [
        Message(message_id=f"m{index}", role=MessageRole.USER, content=f"older message {index}")
        for index in range(1, 8)
    ]
    conv = store.create(conv)
    captured: dict[str, list[str]] = {}

    def record_summary(messages: list[Message]) -> str:
        captured["message_ids"] = [message.message_id for message in messages]
        return "summary ok"

    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
        summary_builder=record_summary,
        repo_recent_window=3,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="Read README from https://github.com/foo/bar",
        conv_lock=lock,
        chat_span=span,
    )

    assert outcome.record is not None
    assert captured["message_ids"] == ["m1", "m2", "m3", "m4"]
    assert outcome.record.source_message_ids == ["m1", "m2", "m3", "m4"]
    assert outcome.record.protected_block_ids == ["m5", "m6", "m7"]
    assert [message.message_id for message in outcome.conversation_snapshot.messages] == [
        "m5",
        "m6",
        "m7",
    ]


@pytest.mark.asyncio
async def test_summary_result(store: JsonStore):
    conv = Conversation(title="Summary Result")
    conv.messages = [
        Message(message_id=f"m{index}", role=MessageRole.USER, content=f"older message {index}")
        for index in range(1, 8)
    ]
    conv = store.create(conv)

    async def build_summary(
        messages: list[Message], *, model: str, chat_span
    ) -> SummaryBuildResult:
        _ = messages
        chat_span.set_attribute("test.summary_builder.model", model)
        return SummaryBuildResult(
            text="summary for compaction",
            model="summary-mini",
            latency_ms=12.5,
            mode="llm",
        )

    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
        summary_builder=build_summary,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="Read README from https://github.com/foo/bar",
        conv_lock=lock,
        chat_span=span,
    )

    assert outcome.record is not None
    assert outcome.record.summary == "summary for compaction"
    assert outcome.record.metadata["summary_model"] == "summary-mini"
    assert outcome.record.metadata["summarization_mode"] == "llm"
    assert span.attributes["chat.compaction.summary_model"] == "summary-mini"
    assert span.attributes["chat.compaction.summary_latency_ms"] == 12.5


@pytest.mark.asyncio
async def test_system_messages_protected(store: JsonStore):
    conv = Conversation(title="System")
    conv.messages = [
        Message(message_id="sys1", role=MessageRole.SYSTEM, content="Guardrail"),
        *[
            Message(message_id=f"m{index}", role=MessageRole.USER, content=f"older message {index}")
            for index in range(1, 8)
        ],
    ]
    conv = store.create(conv)
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="Read README from https://github.com/foo/bar",
        conv_lock=lock,
        chat_span=span,
    )

    assert outcome.record is not None
    assert [message.message_id for message in outcome.conversation_snapshot.messages] == [
        "sys1",
        "m2",
        "m3",
        "m4",
        "m5",
        "m6",
        "m7",
    ]
    assert "sys1" in outcome.record.protected_block_ids
    assert "sys1" not in outcome.record.pruned_block_ids


@pytest.mark.asyncio
async def test_pinned_messages_protected(store: JsonStore):
    conv = Conversation(title="Pinned")
    conv.messages = [
        Message(message_id="m1", role=MessageRole.USER, content="keep me"),
        *[
            Message(message_id=f"m{index}", role=MessageRole.USER, content=f"older message {index}")
            for index in range(2, 9)
        ],
    ]
    conv.metadata["pinned_contexts"] = [
        {
            "pin_id": "pin1",
            "conversation_id": conv.conversation_id,
            "source_message_id": "m1",
            "title": "Pinned",
            "content": "keep me",
            "role": "user",
            "scope": "conversation",
            "status": "active",
            "priority": 5,
            "token_count": 2,
            "metadata": {"origin_message_id": "m1"},
        }
    ]
    conv = store.create(conv)
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="Read README from https://github.com/foo/bar",
        conv_lock=lock,
        chat_span=span,
    )

    assert outcome.record is not None
    assert [message.message_id for message in outcome.conversation_snapshot.messages] == [
        "m1",
        "m3",
        "m4",
        "m5",
        "m6",
        "m7",
        "m8",
    ]
    assert "m1" in outcome.record.protected_block_ids
    assert "m1" not in outcome.record.pruned_block_ids


@pytest.mark.asyncio
async def test_bookmarks_not_protected(store: JsonStore):
    conv = Conversation(title="Bookmark")
    conv.messages = [
        Message(message_id="m1", role=MessageRole.USER, content="bookmark only", bookmarked=True),
        *[
            Message(message_id=f"m{index}", role=MessageRole.USER, content=f"older message {index}")
            for index in range(2, 9)
        ],
    ]
    conv = store.create(conv)
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="Read README from https://github.com/foo/bar",
        conv_lock=lock,
        chat_span=span,
    )

    assert outcome.record is not None
    assert "m1" not in outcome.record.protected_block_ids
    assert "m1" in outcome.record.pruned_block_ids


@pytest.mark.asyncio
async def test_active_turn_protected(store: JsonStore):
    conv = Conversation(title="Active Turn")
    conv.messages = [
        *[
            Message(message_id=f"m{index}", role=MessageRole.USER, content=f"older message {index}")
            for index in range(1, 8)
        ],
        Message(message_id="turn1", role=MessageRole.USER, content="latest user turn"),
    ]
    conv = store.create(conv)
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
        repo_recent_window=3,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="Read README from https://github.com/foo/bar",
        conv_lock=lock,
        chat_span=span,
    )

    assert outcome.record is not None
    assert [message.message_id for message in outcome.conversation_snapshot.messages] == [
        "m6",
        "m7",
        "turn1",
    ]
    assert "turn1" in outcome.record.protected_block_ids
    assert "turn1" not in outcome.record.pruned_block_ids
    assert outcome.record.active_turn_protected is True


@pytest.mark.asyncio
async def test_manual_compacts(store: JsonStore):
    conv = Conversation(title="Manual", model="gpt-4o-mini")
    conv.messages = [
        Message(message_id=f"m{index}", role=MessageRole.USER, content=f"message {index}")
        for index in range(1, 9)
    ]
    conv.conversation_context_state = ConversationContextState(
        conversation_id=conv.conversation_id,
        used_units=750,
        max_units=1000,
        state="elevated",
        updated_at=1.0,
    )
    conv = store.create(conv)
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="",
        conv_lock=lock,
        chat_span=span,
        trigger_kind="manual",
    )

    assert outcome.record is not None
    assert outcome.record.trigger == "manual"
    assert outcome.record.pressure_level == "elevated"
    assert span.attributes["chat.compaction.trigger"] == "manual"


@pytest.mark.asyncio
async def test_post_turn_compacts(store: JsonStore):
    conv = Conversation(title="Post Turn", model="gpt-4o-mini")
    conv.messages = [
        Message(message_id=f"m{index}", role=MessageRole.USER, content=f"message {index}")
        for index in range(1, 9)
    ]
    conv.conversation_context_state = ConversationContextState(
        conversation_id=conv.conversation_id,
        used_units=950,
        max_units=1000,
        state="near_compaction",
        updated_at=1.0,
    )
    conv = store.create(conv)
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="",
        conv_lock=lock,
        chat_span=span,
        trigger_kind="post_turn_background",
    )

    assert outcome.record is not None
    assert outcome.record.trigger == "post_turn_background"
    assert outcome.record.pressure_level == "critical"
    assert span.attributes["chat.compaction.trigger"] == "post_turn_background"


@pytest.mark.asyncio
async def test_post_turn_skips_stream(store: JsonStore):
    conv = Conversation(title="Post Turn Stream")
    conv.messages = [
        Message(message_id=f"m{index}", role=MessageRole.USER, content=f"message {index}")
        for index in range(1, 9)
    ]
    conv.conversation_context_state = ConversationContextState(
        conversation_id=conv.conversation_id,
        used_units=950,
        max_units=1000,
        state="near_compaction",
        updated_at=1.0,
    )
    conv.active_streaming_state = ActiveStreamingState(
        conversation_id=conv.conversation_id,
        message_id="a_stream",
        request_id="req_stream",
        status="streaming",
        started_at=1.0,
        updated_at=1.0,
    )
    conv = store.create(conv)
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="",
        conv_lock=lock,
        chat_span=span,
        trigger_kind="post_turn_background",
    )

    assert outcome.record is None
    assert span.attributes["chat.compaction.trigger"] == "post_turn_background"
    assert span.attributes["chat.compaction.safety_gate"] == "active_streaming"


@pytest.mark.asyncio
async def test_post_turn_skips_toolloop(store: JsonStore):
    conv = Conversation(title="Post Turn Tool")
    conv.messages = [
        Message(message_id=f"m{index}", role=MessageRole.USER, content=f"message {index}")
        for index in range(1, 8)
    ]
    conv.conversation_context_state = ConversationContextState(
        conversation_id=conv.conversation_id,
        used_units=950,
        max_units=1000,
        state="near_compaction",
        updated_at=1.0,
    )
    conv.messages.append(
        Message(
            message_id="a_pending",
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "demo", "arguments": '{"path":"README.md"}'},
                }
            ],
        )
    )
    conv = store.create(conv)
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="",
        conv_lock=lock,
        chat_span=span,
        trigger_kind="post_turn_background",
    )

    assert outcome.record is None
    assert span.attributes["chat.compaction.trigger"] == "post_turn_background"
    assert span.attributes["chat.compaction.safety_gate"] == "active_tool_loop"


@pytest.mark.asyncio
async def test_post_turn_skips(store: JsonStore):
    conv = Conversation(title="Post Turn Split")
    conv.messages = [
        Message(message_id=f"m{index}", role=MessageRole.USER, content=f"message {index}")
        for index in range(1, 8)
    ]
    conv.conversation_context_state = ConversationContextState(
        conversation_id=conv.conversation_id,
        used_units=950,
        max_units=1000,
        state="near_compaction",
        updated_at=1.0,
    )
    conv.messages.append(
        Message(
            message_id="t_orphan",
            role=MessageRole.TOOL,
            content='{"ok":true}',
            tool_call_id="call_missing",
            name="demo",
        )
    )
    conv = store.create(conv)
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="",
        conv_lock=lock,
        chat_span=span,
        trigger_kind="post_turn_background",
    )

    assert outcome.record is None
    assert span.attributes["chat.compaction.trigger"] == "post_turn_background"
    assert span.attributes["chat.compaction.safety_gate"] == "split_incomplete_turn"


@pytest.mark.asyncio
async def test_post_turn_ignores_stale(store: JsonStore):
    conv = Conversation(title="Post Turn Stale Split", model="gpt-4o-mini")
    conv.messages = [
        Message(
            message_id="t_orphan",
            role=MessageRole.TOOL,
            content='{"ok":true}',
            tool_call_id="call_missing",
            name="demo",
        )
    ]
    conv.messages.extend(
        Message(message_id=f"m{index}", role=MessageRole.USER, content=f"message {index}")
        for index in range(1, 9)
    )
    conv.conversation_context_state = ConversationContextState(
        conversation_id=conv.conversation_id,
        used_units=950,
        max_units=1000,
        state="near_compaction",
        updated_at=1.0,
    )
    conv = store.create(conv)
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
        repo_recent_window=6,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="",
        conv_lock=lock,
        chat_span=span,
        trigger_kind="post_turn_background",
    )

    assert outcome.record is not None
    assert outcome.record.trigger == "post_turn_background"
    assert "chat.compaction.safety_gate" not in span.attributes


@pytest.mark.asyncio
async def test_pre_request_cooldown(store: JsonStore):
    conv = Conversation(title="Cooldown")
    conv.messages = [
        Message(message_id=f"m{index}", role=MessageRole.USER, content=f"message {index}")
        for index in range(8)
    ]
    conv.conversation_context_state = ConversationContextState(
        conversation_id=conv.conversation_id,
        used_units=950,
        max_units=1000,
        state="compacted_recently",
        last_compacted_at=9999999999.0,
        updated_at=1.0,
    )
    conv = store.create(conv)
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="Read README from https://github.com/foo/bar",
        conv_lock=lock,
        chat_span=span,
    )

    assert outcome.record is not None
    assert outcome.compaction_event is not None
    assert outcome.record.trigger == "pre_request_pressure"
    assert "chat.compaction.safety_gate" not in span.attributes


@pytest.mark.asyncio
async def test_skips_active_stream(store: JsonStore):
    conv = Conversation(title="Streaming")
    conv.messages = [
        Message(message_id=f"m{index}", role=MessageRole.USER, content=f"message {index}")
        for index in range(8)
    ]
    conv.active_streaming_state = ActiveStreamingState(
        conversation_id=conv.conversation_id,
        message_id="a_stream",
        request_id="req_stream",
        status="streaming",
        started_at=1.0,
        updated_at=1.0,
    )
    conv = store.create(conv)
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="Read README from https://github.com/foo/bar",
        conv_lock=lock,
        chat_span=span,
    )

    assert outcome.record is None
    assert outcome.compaction_event is None
    assert len(outcome.conversation_snapshot.messages) == 8
    assert span.attributes["chat.compaction.triggered"] is False
    assert span.attributes["chat.compaction.safety_gate"] == "active_streaming"


@pytest.mark.asyncio
async def test_block_pre_request(store: JsonStore):
    conv = Conversation(title="Finishing", model="gpt-4o-mini")
    conv.messages = [
        Message(message_id=f"m{index}", role=MessageRole.USER, content=f"message {index}")
        for index in range(8)
    ]
    conv.conversation_context_state = ConversationContextState(
        conversation_id=conv.conversation_id,
        used_units=750,
        max_units=1000,
        state="near_compaction",
        updated_at=1.0,
    )
    conv.active_streaming_state = ActiveStreamingState(
        conversation_id=conv.conversation_id,
        message_id="a_finishing",
        request_id="req_finishing",
        status="finishing",
        started_at=1.0,
        updated_at=1.0,
    )
    conv = store.create(conv)
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="Please continue the analysis",
        conv_lock=lock,
        chat_span=span,
    )

    assert outcome.record is not None
    assert outcome.compaction_event is not None
    assert outcome.record.trigger == "pre_request_pressure"
    assert "chat.compaction.safety_gate" not in span.attributes


@pytest.mark.asyncio
async def test_overrides_cooldown_messages(store: JsonStore):
    conv = Conversation(title="Cooldown Messages")
    conv.messages = [
        Message(message_id=f"m{index}", role=MessageRole.USER, content=f"message {index}")
        for index in range(8)
    ]
    conv.conversation_context_state = ConversationContextState(
        conversation_id=conv.conversation_id,
        used_units=750,
        max_units=1000,
        state="compacted_recently",
        last_compacted_message_count=7,
        updated_at=1.0,
    )
    conv = store.create(conv)
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
        pressure_threshold=0.7,
        overflow_threshold=0.9,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="Read README from https://github.com/foo/bar",
        conv_lock=lock,
        chat_span=span,
        cooldown_messages=2,
        cooldown_seconds=0,
    )

    assert outcome.record is not None
    assert outcome.compaction_event is not None
    assert outcome.record.trigger == "pre_request_pressure"
    assert "chat.compaction.safety_gate" not in span.attributes


@pytest.mark.asyncio
async def test_gnores_cooldown(store: JsonStore):
    conv = Conversation(title="Pressure Cooldown", model="gpt-4o-mini")
    conv.messages = [
        Message(message_id=f"m{index}", role=MessageRole.USER, content=f"message {index}")
        for index in range(10)
    ]
    conv.conversation_context_state = ConversationContextState(
        conversation_id=conv.conversation_id,
        used_units=750,
        max_units=1000,
        state="compacted_recently",
        last_compacted_message_count=9,
        last_compacted_at=9999999999.0,
        updated_at=1.0,
    )
    conv = store.create(conv)
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
        pressure_threshold=0.7,
        overflow_threshold=0.9,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="Please continue the analysis",
        conv_lock=lock,
        chat_span=span,
        cooldown_messages=2,
        cooldown_seconds=30,
    )

    assert outcome.record is not None
    assert outcome.record.trigger == "pre_request_pressure"
    assert "chat.compaction.safety_gate" not in span.attributes


@pytest.mark.asyncio
async def test_skips_toolloop(store: JsonStore):
    conv = Conversation(title="Active Tool Loop")
    conv.messages = [
        Message(message_id=f"m{index}", role=MessageRole.USER, content=f"message {index}")
        for index in range(1, 7)
    ]
    conv.messages.append(
        Message(
            message_id="a_pending",
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "demo", "arguments": '{"path":"README.md"}'},
                }
            ],
        )
    )
    conv = store.create(conv)
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="Read README from https://github.com/foo/bar",
        conv_lock=lock,
        chat_span=span,
    )

    assert outcome.record is None
    assert outcome.compaction_event is None
    assert len(outcome.conversation_snapshot.messages) == 7
    assert span.attributes["chat.compaction.triggered"] is False
    assert span.attributes["chat.compaction.safety_gate"] == "active_tool_loop"


@pytest.mark.asyncio
async def test_safety_gate_skips(store: JsonStore):
    conv = Conversation(title="Split Turn")
    conv.messages = [
        Message(message_id=f"m{index}", role=MessageRole.USER, content=f"message {index}")
        for index in range(1, 7)
    ]
    conv.messages.append(
        Message(
            message_id="t_orphan",
            role=MessageRole.TOOL,
            content='{"ok":true}',
            tool_call_id="call_missing",
            name="demo",
        )
    )
    conv = store.create(conv)
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="Read README from https://github.com/foo/bar",
        conv_lock=lock,
        chat_span=span,
    )

    assert outcome.record is None
    assert outcome.compaction_event is None
    assert len(outcome.conversation_snapshot.messages) == 7
    assert span.attributes["chat.compaction.triggered"] is False
    assert span.attributes["chat.compaction.safety_gate"] == "split_incomplete_turn"


@pytest.mark.asyncio
async def test_uses_overflow(store: JsonStore):
    conv = Conversation(title="Pressure", model="gpt-4o-mini")
    conv.messages = [
        Message(message_id=f"m{index}", role=MessageRole.USER, content="x" * 2000)
        for index in range(10)
    ]
    conv = store.create(conv)
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
        pressure_threshold=0.01,
        overflow_threshold=0.02,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="Please continue the analysis",
        conv_lock=lock,
        chat_span=span,
    )

    assert outcome.record is not None
    assert outcome.record.trigger == "overflow_recovery"
    assert outcome.record.pressure_level == "critical"
    assert span.attributes["chat.compaction.trigger"] == "overflow_recovery"
    assert span.attributes["chat.compaction.low_watermark"] == 0.01
    assert span.attributes["chat.compaction.high_watermark"] == 0.01
    assert span.attributes["chat.compaction.critical_watermark"] == 0.02


@pytest.mark.asyncio
async def test_uses_high_watermark(store: JsonStore):
    conv = Conversation(title="State Pressure", model="gpt-4o-mini")
    conv.messages = [
        Message(message_id=f"m{index}", role=MessageRole.USER, content=f"message {index}")
        for index in range(10)
    ]
    conv.conversation_context_state = ConversationContextState(
        conversation_id=conv.conversation_id,
        used_units=750,
        max_units=1000,
        state="elevated",
        updated_at=1.0,
    )
    conv = store.create(conv)
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
        pressure_threshold=0.7,
        overflow_threshold=0.9,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="Please continue the analysis",
        conv_lock=lock,
        chat_span=span,
    )

    assert outcome.record is not None
    assert outcome.record.trigger == "pre_request_pressure"
    assert outcome.record.pressure_level == "elevated"
    assert span.attributes["chat.compaction.trigger"] == "pre_request_pressure"
    assert span.attributes["chat.compaction.low_watermark"] == 0.6
    assert span.attributes["chat.compaction.high_watermark"] == 0.7
    assert span.attributes["chat.compaction.critical_watermark"] == 0.9


@pytest.mark.asyncio
async def test_commit_failure_restores(store: JsonStore):
    conv = Conversation(title="Restore")
    conv.messages = [
        Message(message_id=f"m{index}", role=MessageRole.USER, content=f"older message {index}")
        for index in range(1, 8)
    ]
    conv = store.create(conv)
    original_update = store.update
    state = {"calls": 0}

    def fail_once(conversation):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("commit failed")
        return original_update(conversation)

    store.update = fail_once  # type: ignore[method-assign]
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="Read README from https://github.com/foo/bar",
        conv_lock=lock,
        chat_span=span,
    )

    assert outcome.record is None
    assert outcome.compaction_event is None
    restored = store.get(conv.conversation_id)
    assert restored is not None
    assert len(restored.messages) == 7
    assert span.attributes["chat.compaction.triggered"] is False
    assert span.attributes["chat.compaction.restore_status"] == "restored_after_commit_failure"


@pytest.mark.asyncio
async def test_backup_failure_aborts(store: JsonStore):
    conv = Conversation(title="Backup Fail")
    conv.messages = [
        Message(message_id=f"m{index}", role=MessageRole.USER, content=f"older message {index}")
        for index in range(1, 8)
    ]
    conv = store.create(conv)

    def fail_backup(*args, **kwargs):
        raise RuntimeError("backup failed")

    store.create_backup = fail_backup  # type: ignore[method-assign]
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="Read README from https://github.com/foo/bar",
        conv_lock=lock,
        chat_span=span,
    )

    assert outcome.record is None
    assert outcome.compaction_event is None
    assert len(outcome.conversation_snapshot.messages) == 7
    assert span.attributes["chat.compaction.abort_reason"] == "backup_failed"


@pytest.mark.asyncio
async def test_summary_failure(store: JsonStore):
    conv = Conversation(title="Summary Fail")
    conv.messages = [
        Message(message_id=f"m{index}", role=MessageRole.USER, content=f"older message {index}")
        for index in range(1, 8)
    ]
    conv = store.create(conv)
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
        summary_builder=lambda _messages: (_ for _ in ()).throw(RuntimeError("summary failed")),
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="Read README from https://github.com/foo/bar",
        conv_lock=lock,
        chat_span=span,
    )

    assert outcome.record is not None
    assert outcome.record.metadata["summarization_mode"] == "fallback_prune_only"
    assert outcome.record.summary.startswith("Pruned ")
    assert span.attributes["chat.compaction.summarize_fallback"] == "prune_only"


@pytest.mark.asyncio
async def test_compressor_runs_hooks(store: JsonStore):
    conv = Conversation(title="Hooks")
    conv.messages = [
        Message(message_id=f"m{index}", role=MessageRole.USER, content=f"older message {index}")
        for index in range(1, 8)
    ]
    conv = store.create(conv)
    captured = {"errors": []}
    original_update = store.update
    state = {"calls": 0}

    def fail_once(conversation):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("commit hook path")
        return original_update(conversation)

    store.update = fail_once  # type: ignore[method-assign]
    hook_service = ChatContextHookService(
        before_compress=lambda payload: {"pressure_level": "critical"},
        after_compress=lambda record: {"restore_status": "hooked"},
        on_compress_error=lambda payload: captured["errors"].append(payload),
    )
    compressor = ContextCompressor(
        json_store=store,
        is_vision_model=lambda _model: False,
        apply_conversation_context_delta=lambda conversation, **kwargs: conversation,
        hook_service=hook_service,
    )
    span = _Span()
    lock = await store.lock(conv.conversation_id)

    outcome = await compressor.compact_for_send(
        conversation_id=conv.conversation_id,
        conversation_snapshot=conv.model_copy(deep=True),
        model="gpt-4o-mini",
        user_content="Read README from https://github.com/foo/bar",
        conv_lock=lock,
        chat_span=span,
    )

    assert outcome.record is None
    assert span.attributes["chat.hooks.before_compress.called"] is True
    assert span.attributes["chat.hooks.after_compress.called"] is True
    assert span.attributes["chat.hooks.on_compress_error.called"] is True
    assert span.attributes["chat.compaction.pressure_level"] == "critical"
    assert span.attributes["chat.compaction.restore_status"] == "restored_after_commit_failure"
    assert captured["errors"][-1]["stage"] == "commit"
