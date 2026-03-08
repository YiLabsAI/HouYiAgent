"""Unit tests for houyi.adapters.memory.store.MemoryStore."""

from __future__ import annotations

import time

from houyi.adapters.memory.store import MemoryStore
from houyi.adapters.memory.types import MemoryRecord, MemoryScope


class TestMemoryStoreInMemory:
    """Test MemoryStore without file persistence."""

    def test_put_and_get(self):
        store = MemoryStore()
        record = store.put("greeting", "Hello world", scope=MemoryScope.SESSION)
        assert record.key == "greeting"
        assert record.content == "Hello world"
        assert record.scope == MemoryScope.SESSION

        retrieved = store.get("greeting", scope=MemoryScope.SESSION)
        assert retrieved is not None
        assert retrieved.content == "Hello world"

    def test_get_nonexistent(self):
        store = MemoryStore()
        assert store.get("nope") is None

    def test_put_update(self):
        store = MemoryStore()
        store.put("key1", "value1")
        store.put("key1", "value2")
        record = store.get("key1")
        assert record is not None
        assert record.content == "value2"

    def test_put_with_metadata(self):
        store = MemoryStore()
        record = store.put("key1", "val", metadata={"source": "test"})
        assert record.metadata["source"] == "test"

    def test_delete(self):
        store = MemoryStore()
        store.put("key1", "val")
        assert store.delete("key1") is True
        assert store.get("key1") is None

    def test_delete_nonexistent(self):
        store = MemoryStore()
        assert store.delete("nope") is False

    def test_list_by_scope(self):
        store = MemoryStore()
        store.put("a", "val_a", scope=MemoryScope.SESSION)
        store.put("b", "val_b", scope=MemoryScope.SESSION)
        store.put("c", "val_c", scope=MemoryScope.USER)

        session_records = store.list_by_scope(MemoryScope.SESSION)
        assert len(session_records) == 2

        user_records = store.list_by_scope(MemoryScope.USER)
        assert len(user_records) == 1

    def test_list_by_scope_ordering(self):
        store = MemoryStore()
        store.put("old", "old_val", scope=MemoryScope.SESSION)
        time.sleep(0.01)
        store.put("new", "new_val", scope=MemoryScope.SESSION)

        records = store.list_by_scope(MemoryScope.SESSION)
        assert records[0].key == "new"  # newest first
        assert records[1].key == "old"

    def test_clear_all(self):
        store = MemoryStore()
        store.put("a", "1", scope=MemoryScope.SESSION)
        store.put("b", "2", scope=MemoryScope.USER)
        count = store.clear()
        assert count == 2
        assert store.list_by_scope(MemoryScope.SESSION) == []
        assert store.list_by_scope(MemoryScope.USER) == []

    def test_clear_by_scope(self):
        store = MemoryStore()
        store.put("a", "1", scope=MemoryScope.SESSION)
        store.put("b", "2", scope=MemoryScope.USER)
        count = store.clear(scope=MemoryScope.SESSION)
        assert count == 1
        assert store.list_by_scope(MemoryScope.SESSION) == []
        assert len(store.list_by_scope(MemoryScope.USER)) == 1

    def test_scope_isolation(self):
        store = MemoryStore()
        store.put("key1", "session_val", scope=MemoryScope.SESSION)
        store.put("key1", "user_val", scope=MemoryScope.USER)

        session = store.get("key1", scope=MemoryScope.SESSION)
        user = store.get("key1", scope=MemoryScope.USER)
        assert session is not None
        assert user is not None
        assert session.content == "session_val"
        assert user.content == "user_val"


class TestMemoryStoreTTL:
    """Test TTL-based expiration."""

    def test_expired_record_not_returned(self):
        store = MemoryStore()
        store.put("temp", "val", ttl=0.001)
        time.sleep(0.01)
        assert store.get("temp") is None

    def test_non_expired_record_returned(self):
        store = MemoryStore()
        store.put("temp", "val", ttl=60)
        assert store.get("temp") is not None

    def test_expired_cleaned_from_list(self):
        store = MemoryStore()
        store.put("temp", "val", scope=MemoryScope.SESSION, ttl=0.001)
        store.put("perm", "val", scope=MemoryScope.SESSION)
        time.sleep(0.01)
        records = store.list_by_scope(MemoryScope.SESSION)
        assert len(records) == 1
        assert records[0].key == "perm"


class TestMemoryStoreContextText:
    """Test as_context_text rendering."""

    def test_context_text_format(self):
        store = MemoryStore()
        store.put("name", "Alice", scope=MemoryScope.SESSION)
        store.put("lang", "Python", scope=MemoryScope.SESSION)
        text = store.as_context_text(MemoryScope.SESSION)
        assert "- name: Alice" in text
        assert "- lang: Python" in text

    def test_context_text_empty(self):
        store = MemoryStore()
        text = store.as_context_text(MemoryScope.SESSION)
        assert text == ""


class TestMemoryStorePersistence:
    """Test file-based persistence."""

    def test_persist_and_reload(self, tmp_path):
        # Write
        store1 = MemoryStore(data_dir=tmp_path / "mem")
        store1.put("key1", "value1", scope=MemoryScope.SESSION)
        store1.put("key2", "value2", scope=MemoryScope.USER)

        # Reload
        store2 = MemoryStore(data_dir=tmp_path / "mem")
        r1 = store2.get("key1", scope=MemoryScope.SESSION)
        r2 = store2.get("key2", scope=MemoryScope.USER)
        assert r1 is not None
        assert r1.content == "value1"
        assert r2 is not None
        assert r2.content == "value2"

    def test_delete_removes_file(self, tmp_path):
        store = MemoryStore(data_dir=tmp_path / "mem")
        record = store.put("key1", "value1", scope=MemoryScope.SESSION)
        file_path = tmp_path / "mem" / "session" / f"{record.record_id}.json"
        assert file_path.exists()

        store.delete("key1", scope=MemoryScope.SESSION)
        assert not file_path.exists()

    def test_expired_cleaned_on_load(self, tmp_path):
        store1 = MemoryStore(data_dir=tmp_path / "mem")
        store1.put("temp", "val", scope=MemoryScope.SESSION, ttl=0.001)
        time.sleep(0.01)

        store2 = MemoryStore(data_dir=tmp_path / "mem")
        assert store2.get("temp", scope=MemoryScope.SESSION) is None


class TestMemoryRecordModel:
    """Test MemoryRecord data model."""

    def test_is_expired_no_ttl(self):
        record = MemoryRecord(key="k", content="v")
        assert record.is_expired is False

    def test_is_expired_with_ttl(self):
        record = MemoryRecord(key="k", content="v", ttl=0.001, created_at=time.time() - 1)
        assert record.is_expired is True

    def test_serialization(self):
        record = MemoryRecord(
            scope=MemoryScope.USER,
            key="test",
            content="hello",
            metadata={"x": 1},
        )
        data = record.model_dump(mode="json")
        restored = MemoryRecord(**data)
        assert restored.key == "test"
        assert restored.content == "hello"
        assert restored.scope == MemoryScope.USER
