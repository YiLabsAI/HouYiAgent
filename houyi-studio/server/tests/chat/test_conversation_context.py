from pathlib import Path

from houyi_studio.server.chat.conversation_context_adapter import (
    ConversationContextAdapter,
)
from houyi_studio.server.chat.json_store import JsonStore
from houyi_studio.server.chat.types import (
    Conversation,
    ConversationContextState,
    Message,
    MessageRole,
)

CAPACITY = 1000


def _manager(tmp_path: Path) -> ConversationContextAdapter:
    return ConversationContextAdapter(
        json_store=JsonStore(data_dir=tmp_path),
        default_model="gpt-4o-mini",
        rolling_capacity=CAPACITY,
    )


def test_ctx_init(tmp_path: Path) -> None:
    manager = _manager(tmp_path)

    state = manager.build_initial_state("conv-1", now=12.0)

    assert state.conversation_id == "conv-1"
    assert state.used_units == 0
    assert state.max_units == CAPACITY
    assert state.state == "healthy"
    assert state.updated_at == 12.0


def test_ctx_backfill(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    conv = Conversation(conversation_id="conv-1", model="gpt-4o-mini")
    conv.messages = [
        Message(role=MessageRole.USER, content="hello world"),
        Message(role=MessageRole.ASSISTANT, content="reply content"),
    ]

    state = manager.recover_state(conv)

    assert state.conversation_id == "conv-1"
    assert state.used_units > 0
    assert state.max_units == CAPACITY
    assert state.state == "healthy"


def test_ctx_normalize(tmp_path: Path) -> None:
    store = JsonStore(data_dir=tmp_path)
    manager = ConversationContextAdapter(
        json_store=store,
        default_model="gpt-4o-mini",
        rolling_capacity=CAPACITY,
    )
    conv = Conversation(conversation_id="conv-1")
    conv.conversation_context_state = ConversationContextState(
        conversation_id="old-id",
        used_units=5000,
        max_units=5,
        state="healthy",
        last_compaction_delta=3,
        last_compacted_message_count=7,
        updated_at=0,
    )

    state = manager.ensure_state(conv)

    assert state.conversation_id == "conv-1"
    assert state.used_units == CAPACITY
    assert state.max_units == CAPACITY
    assert state.state == "compacted_recently"
    assert state.last_compacted_message_count == 7


def test_ctx_delta_add(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    conv = Conversation(conversation_id="conv-1")
    conv.conversation_context_state = manager.build_initial_state("conv-1")

    state = manager.apply_delta(conv, added_units=720)

    assert state.used_units == 720
    assert state.state == "elevated"


def test_ctx_delta_compact(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    conv = Conversation(conversation_id="conv-1")
    conv.conversation_context_state = ConversationContextState(
        conversation_id="conv-1",
        used_units=920,
        max_units=CAPACITY,
        state="near_compaction",
        updated_at=1,
    )

    state = manager.apply_delta(
        conv,
        released_units=400,
        compacted_at=22.0,
        compaction_delta=400,
        compacted_message_count=12,
    )

    assert state.used_units == 520
    assert state.last_compacted_at == 22.0
    assert state.last_compaction_delta == 400
    assert state.last_compacted_message_count == 12
    assert state.state == "compacted_recently"


def test_ctx_delta_appended_messages(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    conv = Conversation(conversation_id="conv-1", model="gpt-4o-mini")
    conv.conversation_context_state = manager.build_initial_state("conv-1")

    user_msg = Message(role=MessageRole.USER, content="hello")
    system_msg = Message(role=MessageRole.SYSTEM, content="ignore")

    state = manager.apply_appended_messages(
        conv,
        messages=[user_msg, system_msg],
        model="gpt-4o-mini",
    )

    assert state.used_units > 0
    assert state.max_units == CAPACITY
    assert state.state in {"healthy", "elevated"}
