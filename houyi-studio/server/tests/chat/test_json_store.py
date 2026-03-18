"""Unit tests for houyi_studio.server.chat.json_store."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from houyi_studio.server.chat import json_store as json_store_module
from houyi_studio.server.chat.json_store import JsonStore, resolve_chat_data_dir
from houyi_studio.server.chat.types import (
    Conversation,
    ConversationStatus,
    Message,
    MessageRole,
)


def test_resolve_chat_data_dir_default(monkeypatch, tmp_path):
    project_root = tmp_path / "project-root"
    monkeypatch.setattr(json_store_module, "_project_root", lambda: project_root)
    resolved = resolve_chat_data_dir()
    assert resolved == project_root / "data/conversations"


def test_resolve_chat_data_dir_relative_path(monkeypatch, tmp_path):
    project_root = tmp_path / "project-root"
    monkeypatch.setattr(json_store_module, "_project_root", lambda: project_root)
    resolved = resolve_chat_data_dir("custom/chat-data")
    assert resolved == project_root / "custom/chat-data"


def test_resolve_chat_data_dir_keeps_absolute_path(tmp_path):
    absolute = tmp_path / "chat-data"
    resolved = resolve_chat_data_dir(absolute)
    assert resolved == absolute


@pytest.fixture
def store(tmp_path):
    return JsonStore(data_dir=tmp_path / "conversations")


@pytest.fixture
def sample_conversation():
    return Conversation(
        conversation_id="conv001",
        title="Test Chat",
        model="test-model",
        messages=[
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi there!"),
        ],
    )


class TestJsonStoreCRUD:
    """Test basic CRUD operations."""

    def test_create_and_get(self, store, sample_conversation):
        created = store.create(sample_conversation)
        assert created.conversation_id == "conv001"

        retrieved = store.get("conv001")
        assert retrieved is not None
        assert retrieved.title == "Test Chat"
        assert len(retrieved.messages) == 2
        assert retrieved.messages[0].content == "Hello"

    def test_create_duplicate_raises(self, store, sample_conversation):
        store.create(sample_conversation)
        with pytest.raises(ValueError, match="already exists"):
            store.create(sample_conversation)

    def test_get_nonexistent_returns_none(self, store):
        assert store.get("nonexistent") is None

    def test_update(self, store, sample_conversation):
        store.create(sample_conversation)
        sample_conversation.title = "Updated Title"
        sample_conversation.messages.append(Message(role=MessageRole.USER, content="Follow up"))
        updated = store.update(sample_conversation)
        assert updated.title == "Updated Title"

        retrieved = store.get("conv001")
        assert retrieved is not None
        assert retrieved.title == "Updated Title"
        assert len(retrieved.messages) == 3

    def test_update_nonexistent_raises(self, store):
        conv = Conversation(conversation_id="nope")
        with pytest.raises(FileNotFoundError, match="not found"):
            store.update(conv)

    def test_update_sets_updated_at(self, store, sample_conversation):
        store.create(sample_conversation)
        old_updated = sample_conversation.updated_at
        time.sleep(0.01)
        store.update(sample_conversation)
        retrieved = store.get("conv001")
        assert retrieved.updated_at > old_updated

    def test_delete(self, store, sample_conversation):
        store.create(sample_conversation)
        assert store.delete("conv001") is True
        assert store.get("conv001") is None

    def test_delete_nonexistent(self, store):
        assert store.delete("nonexistent") is False


class TestJsonStoreList:
    """Test listing and filtering."""

    def test_list_empty(self, store):
        result = store.list_conversations()
        assert result == []

    def test_list_multiple(self, store):
        for i in range(3):
            store.create(
                Conversation(
                    conversation_id=f"conv{i:03d}",
                    title=f"Chat {i}",
                )
            )
        result = store.list_conversations()
        assert len(result) == 3

    def test_list_sorted_by_updated_at(self, store):
        for i in range(3):
            conv = Conversation(
                conversation_id=f"conv{i:03d}",
                title=f"Chat {i}",
                updated_at=time.time() + i,
            )
            store.create(conv)
        result = store.list_conversations()
        # Newest first
        assert result[0]["conversation_id"] == "conv002"
        assert result[2]["conversation_id"] == "conv000"

    def test_list_filter_by_status(self, store):
        store.create(Conversation(conversation_id="active1", status=ConversationStatus.ACTIVE))
        store.create(Conversation(conversation_id="archived1", status=ConversationStatus.ARCHIVED))
        store.create(Conversation(conversation_id="active2", status=ConversationStatus.ACTIVE))

        active = store.list_conversations(status="active")
        assert len(active) == 2
        archived = store.list_conversations(status="archived")
        assert len(archived) == 1

    def test_list_pagination(self, store):
        for i in range(10):
            store.create(
                Conversation(
                    conversation_id=f"conv{i:03d}",
                    updated_at=time.time() + i,
                )
            )
        page1 = store.list_conversations(limit=3, offset=0)
        assert len(page1) == 3
        page2 = store.list_conversations(limit=3, offset=3)
        assert len(page2) == 3
        # No overlap
        ids1 = {c["conversation_id"] for c in page1}
        ids2 = {c["conversation_id"] for c in page2}
        assert ids1.isdisjoint(ids2)

    def test_count(self, store):
        store.create(Conversation(conversation_id="a", status=ConversationStatus.ACTIVE))
        store.create(Conversation(conversation_id="b", status=ConversationStatus.ARCHIVED))
        assert store.count() == 2
        assert store.count(status="active") == 1
        assert store.count(status="archived") == 1


class TestJsonStoreAtomicWrite:
    """Test atomic write and file integrity."""

    def test_file_exists_after_create(self, store, sample_conversation, tmp_path):
        store.create(sample_conversation)
        file_path = tmp_path / "conversations" / "conv001.json"
        assert file_path.exists()
        data = json.loads(file_path.read_text())
        assert data["conversation_id"] == "conv001"
        assert data["title"] == "Test Chat"

    def test_no_tmp_files_left(self, store, sample_conversation, tmp_path):
        store.create(sample_conversation)
        tmp_files = list((tmp_path / "conversations").glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_index_file_exists(self, store, sample_conversation, tmp_path):
        store.create(sample_conversation)
        index_path = tmp_path / "conversations" / "index.json"
        assert index_path.exists()
        data = json.loads(index_path.read_text())
        assert len(data["conversations"]) == 1

    def test_index_updated_on_delete(self, store, sample_conversation, tmp_path):
        store.create(sample_conversation)
        store.delete("conv001")
        index_path = tmp_path / "conversations" / "index.json"
        data = json.loads(index_path.read_text())
        assert len(data["conversations"]) == 0

    def test_create_backup_writes_snapshot_and_index(self, store, sample_conversation, tmp_path):
        store.create(sample_conversation)
        backup = store.create_backup("conv001", trigger="manual")
        backup_path = tmp_path / "conversations" / "_backups" / backup["path"]
        assert backup_path.exists()
        payload = json.loads(backup_path.read_text())
        assert payload["conversation_id"] == "conv001"
        backup_index = json.loads(
            (tmp_path / "conversations" / "_backups" / "index.json").read_text()
        )
        assert backup_index["backups"][0]["backup_id"] == backup["backup_id"]
        assert backup_index["backups"][0]["trigger"] == "manual"

    def test_attach_backup_record_updates_backup_index(self, store, sample_conversation):
        store.create(sample_conversation)
        backup = store.create_backup("conv001", trigger="manual")
        updated = store.attach_backup_record(backup["backup_id"], record_id="cmp_123")
        assert updated is not None
        assert updated["record_id"] == "cmp_123"
        assert store.get_backup(backup["backup_id"])["record_id"] == "cmp_123"

    def test_restore_backup_rewrites_conversation_file(self, store, sample_conversation):
        store.create(sample_conversation)
        backup = store.create_backup("conv001", trigger="manual")
        mutated = store.get("conv001")
        assert mutated is not None
        mutated.title = "Mutated"
        mutated.messages.append(Message(role=MessageRole.USER, content="new"))
        store.update(mutated)
        restored = store.restore_backup(backup["backup_id"])
        assert restored.title == "Test Chat"
        assert len(restored.messages) == 2
        assert restored.messages[0].content == "Hello"

    def test_restore_backup_targets_snapshot(self, store, sample_conversation):
        store.create(sample_conversation)

        backup_one = store.create_backup("conv001", trigger="manual")
        first_mutation = store.get("conv001")
        assert first_mutation is not None
        first_mutation.title = "After first mutation"
        store.update(first_mutation)

        backup_two = store.create_backup("conv001", trigger="manual")
        second_mutation = store.get("conv001")
        assert second_mutation is not None
        second_mutation.title = "Latest state"
        store.update(second_mutation)

        assert backup_one["backup_id"] != backup_two["backup_id"]
        assert backup_one["conversation_id"] == "conv001"
        assert backup_two["conversation_id"] == "conv001"
        assert backup_one["created_at"] <= backup_two["created_at"]

        restored_first = store.restore_backup(backup_one["backup_id"])
        assert restored_first.title == "Test Chat"

        restored_second = store.restore_backup(backup_two["backup_id"])
        assert restored_second.title == "After first mutation"

    def test_restore_backup_is_isolated_per_conversation(self, store, sample_conversation):
        store.create(sample_conversation)
        other = Conversation(
            conversation_id="conv002",
            title="Other Chat",
            messages=[Message(role=MessageRole.USER, content="other")],
        )
        store.create(other)

        backup_one = store.create_backup("conv001", trigger="manual")
        backup_two = store.create_backup("conv002", trigger="manual")

        conv_one = store.get("conv001")
        conv_two = store.get("conv002")
        assert conv_one is not None
        assert conv_two is not None
        conv_one.title = "Conv one mutated"
        conv_two.title = "Conv two mutated"
        store.update(conv_one)
        store.update(conv_two)

        restored_two = store.restore_backup(backup_two["backup_id"])
        current_one = store.get("conv001")
        assert current_one is not None

        assert backup_one["conversation_id"] == "conv001"
        assert backup_two["conversation_id"] == "conv002"
        assert restored_two.conversation_id == "conv002"
        assert restored_two.title == "Other Chat"
        assert current_one.title == "Conv one mutated"


class TestJsonStoreIndexRebuild:
    """Test index rebuild from conversation files."""

    def test_rebuild_without_index(self, tmp_path):
        # Manually create conversation files without index
        data_dir = tmp_path / "conversations"
        data_dir.mkdir(parents=True)
        conv = Conversation(conversation_id="manual001", title="Manual")
        file_path = data_dir / "manual001.json"
        file_path.write_text(json.dumps(conv.model_dump(mode="json")))

        # Load store — should rebuild index
        store = JsonStore(data_dir=data_dir)
        result = store.list_conversations()
        assert len(result) == 1
        assert result[0]["conversation_id"] == "manual001"

    def test_corrupted_json_skipped(self, tmp_path):
        data_dir = tmp_path / "conversations"
        data_dir.mkdir(parents=True)
        # Valid file
        conv = Conversation(conversation_id="good", title="Good")
        (data_dir / "good.json").write_text(json.dumps(conv.model_dump(mode="json")))
        # Corrupted file
        (data_dir / "bad.json").write_text("{invalid json")

        store = JsonStore(data_dir=data_dir)
        result = store.list_conversations()
        assert len(result) == 1
        assert result[0]["conversation_id"] == "good"

    def test_corrupted_index_triggers_rebuild(self, tmp_path):
        data_dir = tmp_path / "conversations"
        data_dir.mkdir(parents=True)
        conv = Conversation(conversation_id="rebuild001", title="Rebuild")
        (data_dir / "rebuild001.json").write_text(json.dumps(conv.model_dump(mode="json")))
        # Write corrupted index
        (data_dir / "index.json").write_text("{bad index")

        store = JsonStore(data_dir=data_dir)
        result = store.list_conversations()
        assert len(result) == 1
        assert result[0]["conversation_id"] == "rebuild001"


class TestJsonStoreConcurrencyLock:
    """Test per-conversation asyncio.Lock for concurrent write safety."""

    @pytest.mark.asyncio
    async def test_lock_returns_asyncio_lock(self, store):
        """lock() returns an asyncio.Lock instance."""
        lock = await store.lock("conv001")
        assert isinstance(lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_same_conversation_gets_same_lock(self, store):
        """Same conversation_id always returns the same Lock object."""
        lock1 = await store.lock("conv001")
        lock2 = await store.lock("conv001")
        assert lock1 is lock2

    @pytest.mark.asyncio
    async def test_different_conversations_get_different_locks(self, store):
        """Different conversation_ids get independent locks."""
        lock_a = await store.lock("conv_a")
        lock_b = await store.lock("conv_b")
        assert lock_a is not lock_b

    @pytest.mark.asyncio
    async def test_lock_serializes_writes(self, store):
        """Concurrent read-modify-write on the same conversation is serialized.

        Without locking, two concurrent tasks doing load→append→save would
        lose one message (last-write-wins). With per-conversation lock,
        both messages are preserved.
        """
        conv = Conversation(conversation_id="race", title="Race Test")
        store.create(conv)

        results = []

        async def append_message(msg_content: str, delay: float):
            lock = await store.lock("race")
            async with lock:
                c = store.get("race")
                assert c is not None
                # Simulate some async work between read and write
                await asyncio.sleep(delay)
                c.messages.append(Message(role=MessageRole.USER, content=msg_content))
                store.update(c)
                results.append(msg_content)

        # Launch two concurrent tasks that both modify the same conversation
        await asyncio.gather(
            append_message("msg_A", 0.05),
            append_message("msg_B", 0.01),
        )

        # Both messages must be present (no lost update)
        final = store.get("race")
        assert final is not None
        assert len(final.messages) == 2
        contents = {m.content for m in final.messages}
        assert contents == {"msg_A", "msg_B"}

    @pytest.mark.asyncio
    async def test_without_lock_loses_update(self, store):
        """Demonstrates the race condition that locking prevents.

        Two concurrent read-modify-write WITHOUT locking: one message is lost.
        """
        conv = Conversation(conversation_id="nolock", title="No Lock Test")
        store.create(conv)

        async def append_without_lock(msg_content: str, delay: float):
            c = store.get("nolock")
            assert c is not None
            await asyncio.sleep(delay)
            c.messages.append(Message(role=MessageRole.USER, content=msg_content))
            store.update(c)

        # Both tasks read the same snapshot (0 messages), then each writes 1 message
        await asyncio.gather(
            append_without_lock("lost_A", 0.05),
            append_without_lock("lost_B", 0.01),
        )

        # Last writer wins — only 1 message survives
        final = store.get("nolock")
        assert final is not None
        assert len(final.messages) == 1  # Lost update!

    @pytest.mark.asyncio
    async def test_different_conversations_not_blocked(self, store):
        """Locks on different conversations do not block each other."""
        store.create(Conversation(conversation_id="ind_a", title="A"))
        store.create(Conversation(conversation_id="ind_b", title="B"))

        execution_order = []

        async def modify(conv_id: str, delay: float):
            lock = await store.lock(conv_id)
            async with lock:
                execution_order.append(f"{conv_id}_start")
                await asyncio.sleep(delay)
                c = store.get(conv_id)
                assert c is not None
                c.messages.append(Message(role=MessageRole.USER, content=f"msg_{conv_id}"))
                store.update(c)
                execution_order.append(f"{conv_id}_end")

        # ind_a takes longer but ind_b should not wait for it
        await asyncio.gather(
            modify("ind_a", 0.1),
            modify("ind_b", 0.01),
        )

        # ind_b should finish before ind_a (not blocked)
        assert execution_order.index("ind_b_end") < execution_order.index("ind_a_end")

        # Both conversations updated correctly
        assert len(store.get("ind_a").messages) == 1
        assert len(store.get("ind_b").messages) == 1

    @pytest.mark.asyncio
    async def test_lock_concurrent_creation(self, store):
        """Multiple tasks requesting locks for new conversations simultaneously."""

        async def get_lock(conv_id: str):
            return await store.lock(conv_id)

        locks = await asyncio.gather(
            get_lock("new_a"),
            get_lock("new_b"),
            get_lock("new_a"),  # duplicate
            get_lock("new_c"),
            get_lock("new_b"),  # duplicate
        )

        # Same conv_id → same lock object
        assert locks[0] is locks[2]  # new_a
        assert locks[1] is locks[4]  # new_b
        # Different conv_id → different lock
        assert locks[0] is not locks[1]
        assert locks[0] is not locks[3]
