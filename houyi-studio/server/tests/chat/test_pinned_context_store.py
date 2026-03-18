from pathlib import Path

import pytest
from houyi_studio.server.chat.json_store import JsonStore
from houyi_studio.server.chat.pinned_context_store import PinnedContextStore
from houyi_studio.server.chat.types import Conversation, Message, MessageRole, PinStatus


@pytest.fixture
def store(tmp_path: Path) -> JsonStore:
    return JsonStore(data_dir=tmp_path)


@pytest.fixture
def pinned_store(store: JsonStore) -> PinnedContextStore:
    return PinnedContextStore(json_store=store)


@pytest.fixture
def seeded_conversation(store: JsonStore) -> Conversation:
    conversation = Conversation(title="Pinned Context Test")
    conversation.messages = [
        Message(
            message_id="u1",
            role=MessageRole.USER,
            content="Remember that we deploy to staging first.",
        ),
        Message(message_id="a1", role=MessageRole.ASSISTANT, content="Understood, staging first."),
    ]
    store.create(conversation)
    return conversation


@pytest.mark.asyncio
async def test_pin_message_persists_record(
    pinned_store: PinnedContextStore,
    store: JsonStore,
    seeded_conversation: Conversation,
):
    record = await pinned_store.pin_message(
        conversation_id=seeded_conversation.conversation_id,
        message_id="u1",
        title="Deploy rule",
    )

    assert record.conversation_id == seeded_conversation.conversation_id
    assert record.source_message_id == "u1"
    assert record.title == "Deploy rule"
    assert record.status == PinStatus.ACTIVE
    assert record.content == "Remember that we deploy to staging first."

    persisted = store.get(seeded_conversation.conversation_id)
    assert persisted is not None
    pins = persisted.metadata.get("pinned_contexts")
    assert isinstance(pins, list)
    assert len(pins) == 1
    assert pins[0]["source_message_id"] == "u1"
    assert pins[0]["status"] == "active"


@pytest.mark.asyncio
async def test_replace_pin_marks_previous_as_superseded(
    pinned_store: PinnedContextStore,
    store: JsonStore,
    seeded_conversation: Conversation,
):
    first = await pinned_store.pin_message(
        conversation_id=seeded_conversation.conversation_id,
        message_id="u1",
        title="Old rule",
    )

    second = await pinned_store.pin_message(
        conversation_id=seeded_conversation.conversation_id,
        message_id="a1",
        replace_pin_id=first.pin_id,
        title="New rule",
    )

    assert second.status == PinStatus.ACTIVE

    persisted = store.get(seeded_conversation.conversation_id)
    assert persisted is not None
    pins = persisted.metadata.get("pinned_contexts")
    assert isinstance(pins, list)
    by_id = {item["pin_id"]: item for item in pins}
    assert by_id[first.pin_id]["status"] == "superseded"
    assert by_id[second.pin_id]["status"] == "active"


@pytest.mark.asyncio
async def test_update_pin_status_archives_pin(
    pinned_store: PinnedContextStore,
    seeded_conversation: Conversation,
):
    record = await pinned_store.pin_message(
        conversation_id=seeded_conversation.conversation_id,
        message_id="u1",
    )

    updated = await pinned_store.update_pin_status(
        conversation_id=seeded_conversation.conversation_id,
        pin_id=record.pin_id,
        status=PinStatus.ARCHIVED,
    )

    assert updated.pin_id == record.pin_id
    assert updated.status == PinStatus.ARCHIVED


@pytest.mark.asyncio
async def test_list_pins_filters_inactive_by_default(
    pinned_store: PinnedContextStore,
    seeded_conversation: Conversation,
):
    record = await pinned_store.pin_message(
        conversation_id=seeded_conversation.conversation_id,
        message_id="u1",
    )
    await pinned_store.update_pin_status(
        conversation_id=seeded_conversation.conversation_id,
        pin_id=record.pin_id,
        status=PinStatus.REMOVED,
    )

    active = await pinned_store.list_pins(
        conversation_id=seeded_conversation.conversation_id,
    )
    all_pins = await pinned_store.list_pins(
        conversation_id=seeded_conversation.conversation_id,
        include_inactive=True,
    )

    assert active == []
    assert len(all_pins) == 1
    assert all_pins[0].status == PinStatus.REMOVED


@pytest.mark.asyncio
async def test_pin_message_raises_for_missing_message(
    pinned_store: PinnedContextStore,
    seeded_conversation: Conversation,
):
    with pytest.raises(FileNotFoundError, match="Message missing not found"):
        await pinned_store.pin_message(
            conversation_id=seeded_conversation.conversation_id,
            message_id="missing",
        )


@pytest.mark.asyncio
async def test_update_pin_status_raises_for_missing_pin(
    pinned_store: PinnedContextStore,
    seeded_conversation: Conversation,
):
    with pytest.raises(FileNotFoundError, match="Pinned context missing not found"):
        await pinned_store.update_pin_status(
            conversation_id=seeded_conversation.conversation_id,
            pin_id="missing",
            status=PinStatus.REMOVED,
        )
