"""Tests for content store."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from houyi.infrastructure.observability.content_store import (
    ContentStoreConfig,
    ContentType,
    FileContentStore,
    get_content_store,
    reset_content_store,
    set_content_store,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def content_store(temp_dir):
    """Create a content store instance with temp directory."""
    config = ContentStoreConfig(base_path=temp_dir / "content")
    store = FileContentStore(config)
    yield store


class TestFileContentStore:
    """Tests for FileContentStore."""

    def test_store_and_retrieve_string(self, content_store):
        """Test storing and retrieving string content."""
        content = "This is a test prompt for the LLM."

        ref = content_store.store(
            content=content,
            content_type=ContentType.LLM_PROMPT,
            span_id="span_001",
            trace_id="trace_001",
        )

        assert ref.content_id is not None
        assert ref.content_type == ContentType.LLM_PROMPT
        assert ref.span_id == "span_001"
        assert ref.trace_id == "trace_001"
        assert ref.size_bytes == len(content.encode("utf-8"))
        assert ref.checksum is not None

        retrieved = content_store.retrieve(ref.content_id)
        assert retrieved == content

    def test_store_and_retrieve_dict(self, content_store):
        """Test storing and retrieving dict content."""
        content = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "model": "gpt-4",
        }

        ref = content_store.store(
            content=content,
            content_type=ContentType.LLM_RESPONSE,
            span_id="span_002",
            trace_id="trace_001",
        )

        retrieved = content_store.retrieve(ref.content_id)
        assert retrieved == content

    def test_store_and_retrieve_bytes(self, content_store):
        """Test storing and retrieving bytes content."""
        content = b"Binary content here"

        ref = content_store.store(
            content=content,
            content_type=ContentType.CUSTOM,
            span_id="span_003",
            trace_id="trace_001",
        )

        retrieved = content_store.retrieve(ref.content_id)
        assert retrieved == content.decode("utf-8")

    def test_retrieve_by_trace(self, content_store):
        """Test faster retrieval with trace_id."""
        content = "Test content"

        ref = content_store.store(
            content=content,
            content_type=ContentType.TOOL_INPUT,
            span_id="span_004",
            trace_id="trace_002",
        )

        retrieved = content_store.retrieve_by_trace("trace_002", ref.content_id)
        assert retrieved == content

    def test_retrieve_nonexistent(self, content_store):
        """Test retrieving nonexistent content."""
        result = content_store.retrieve("nonexistent_id")
        assert result is None

    def test_delete_content(self, content_store):
        """Test deleting content."""
        content = "Content to delete"

        ref = content_store.store(
            content=content,
            content_type=ContentType.LLM_PROMPT,
            span_id="span_005",
            trace_id="trace_003",
        )

        # Verify it exists
        assert content_store.retrieve(ref.content_id) == content

        # Delete it
        result = content_store.delete(ref.content_id)
        assert result is True

        # Verify it's gone
        assert content_store.retrieve(ref.content_id) is None

    def test_delete_nonexistent(self, content_store):
        """Test deleting nonexistent content."""
        result = content_store.delete("nonexistent_id")
        assert result is False

    def test_delete_by_trace(self, content_store):
        """Test deleting all content for a trace."""
        trace_id = "trace_delete"

        # Store multiple contents
        for i in range(5):
            content_store.store(
                content=f"Content {i}",
                content_type=ContentType.LLM_PROMPT,
                span_id=f"span_{i}",
                trace_id=trace_id,
            )

        # Store content for different trace
        other_ref = content_store.store(
            content="Other content",
            content_type=ContentType.LLM_PROMPT,
            span_id="span_other",
            trace_id="other_trace",
        )

        # Delete by trace
        count = content_store.delete_by_trace(trace_id)
        assert count == 5

        # Verify trace content is gone
        refs = content_store.list_refs(trace_id)
        assert len(refs) == 0

        # Verify other trace content still exists
        assert content_store.retrieve(other_ref.content_id) == "Other content"

    def test_get_ref(self, content_store):
        """Test getting content reference."""
        content = "Test content for ref"

        ref = content_store.store(
            content=content,
            content_type=ContentType.TOOL_OUTPUT,
            span_id="span_006",
            trace_id="trace_004",
        )

        retrieved_ref = content_store.get_ref(ref.content_id)
        assert retrieved_ref is not None
        assert retrieved_ref.content_id == ref.content_id
        assert retrieved_ref.content_type == ref.content_type
        assert retrieved_ref.span_id == ref.span_id
        assert retrieved_ref.trace_id == ref.trace_id

    def test_get_ref_nonexistent(self, content_store):
        """Test getting nonexistent reference."""
        result = content_store.get_ref("nonexistent_id")
        assert result is None

    def test_list_refs(self, content_store):
        """Test listing references for a trace."""
        trace_id = "trace_list"

        # Store multiple contents
        stored_refs = []
        for i in range(3):
            ref = content_store.store(
                content=f"Content {i}",
                content_type=ContentType.LLM_PROMPT if i % 2 == 0 else ContentType.LLM_RESPONSE,
                span_id=f"span_{i}",
                trace_id=trace_id,
            )
            stored_refs.append(ref)

        # List refs
        refs = content_store.list_refs(trace_id)
        assert len(refs) == 3

        content_ids = {r.content_id for r in refs}
        for stored_ref in stored_refs:
            assert stored_ref.content_id in content_ids

    def test_list_refs_empty_trace(self, content_store):
        """Test listing refs for empty trace."""
        refs = content_store.list_refs("nonexistent_trace")
        assert len(refs) == 0

    def test_content_size_limit(self, content_store):
        """Test content size limit enforcement."""
        # Create content larger than limit
        large_content = "x" * (content_store.config.max_content_size + 1)

        with pytest.raises(ValueError, match="Content size"):
            content_store.store(
                content=large_content,
                content_type=ContentType.LLM_RESPONSE,
                span_id="span_large",
                trace_id="trace_large",
            )

    def test_get_statistics(self, content_store):
        """Test getting store statistics."""
        # Store some content
        content_store.store(
            content="Prompt 1",
            content_type=ContentType.LLM_PROMPT,
            span_id="span_1",
            trace_id="trace_stats",
        )
        content_store.store(
            content="Response 1",
            content_type=ContentType.LLM_RESPONSE,
            span_id="span_1",
            trace_id="trace_stats",
        )
        content_store.store(
            content="Tool input",
            content_type=ContentType.TOOL_INPUT,
            span_id="span_2",
            trace_id="trace_stats",
        )

        stats = content_store.get_statistics()
        assert stats["total_files"] == 3
        assert stats["total_size_bytes"] > 0
        assert stats["by_type"]["llm_prompt"] == 1
        assert stats["by_type"]["llm_response"] == 1
        assert stats["by_type"]["tool_input"] == 1

    def test_checksum_consistency(self, content_store):
        """Test that same content produces same checksum."""
        content = "Identical content"

        ref1 = content_store.store(
            content=content,
            content_type=ContentType.LLM_PROMPT,
            span_id="span_a",
            trace_id="trace_a",
        )

        ref2 = content_store.store(
            content=content,
            content_type=ContentType.LLM_PROMPT,
            span_id="span_b",
            trace_id="trace_b",
        )

        assert ref1.checksum == ref2.checksum

    def test_unicode_content(self, content_store):
        """Test storing and retrieving unicode content."""
        content = "Hello World - Greetings Earth (special: @#$%)"

        ref = content_store.store(
            content=content,
            content_type=ContentType.LLM_PROMPT,
            span_id="span_unicode",
            trace_id="trace_unicode",
        )

        retrieved = content_store.retrieve(ref.content_id)
        assert retrieved == content


class TestGlobalContentStore:
    """Tests for global content store functions."""

    def test_get_set_reset_store(self, temp_dir):
        """Test global content store management."""
        reset_content_store()

        # Get default store
        store1 = get_content_store()
        assert store1 is not None

        # Set custom store
        config = ContentStoreConfig(base_path=temp_dir / "custom_content")
        custom_store = FileContentStore(config)
        set_content_store(custom_store)

        store2 = get_content_store()
        assert store2 is custom_store

        # Reset
        reset_content_store()


class TestFileContentStoreExtra:
    def test_md5_hash_algorithm(self, temp_dir) -> None:
        config = ContentStoreConfig(base_path=temp_dir / "md5", hash_algorithm="md5")
        store = FileContentStore(config)
        ref = store.store("hello", ContentType.LLM_PROMPT, "s1", "t1")
        assert ref.checksum
        assert len(ref.checksum) == 32  # md5 hex length

    def test_unknown_hash_falls_back(self, temp_dir) -> None:
        config = ContentStoreConfig(base_path=temp_dir / "unk", hash_algorithm="unknown")
        store = FileContentStore(config)
        ref = store.store("hello", ContentType.LLM_PROMPT, "s1", "t1")
        assert len(ref.checksum) == 64  # sha256 hex length

    def test_retrieve_by_trace_miss(self, content_store) -> None:
        result = content_store.retrieve_by_trace("no_trace", "no_id")
        assert result is None

    def test_delete_by_trace_empty(self, content_store) -> None:
        count = content_store.delete_by_trace("nonexistent_trace")
        assert count == 0

    def test_cleanup_old_content(self, temp_dir) -> None:
        import time
        from datetime import UTC, datetime
        from unittest.mock import patch

        config = ContentStoreConfig(base_path=temp_dir / "cleanup")
        store = FileContentStore(config)

        store.store("old data", ContentType.LLM_PROMPT, "s1", "t_old")
        # Use a future UTC timestamp so the stored entry looks old
        future = time.time() + 3600
        # Patch fromtimestamp to produce UTC-aware datetime
        orig = datetime.fromtimestamp
        with patch("houyi.infrastructure.observability.content_store.datetime") as mock_dt:
            mock_dt.now = datetime.now
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.fromtimestamp = lambda ts: orig(ts, tz=UTC)
            count = store.cleanup_old_content(future)
        assert count == 1
        assert store.list_refs("t_old") == []

    def test_cleanup_preserves_recent(self, temp_dir) -> None:
        import time
        from datetime import UTC, datetime
        from unittest.mock import patch

        config = ContentStoreConfig(base_path=temp_dir / "cleanup2")
        store = FileContentStore(config)

        store.store("recent", ContentType.LLM_PROMPT, "s1", "t_recent")
        past = time.time() - 3600
        orig = datetime.fromtimestamp
        with patch("houyi.infrastructure.observability.content_store.datetime") as mock_dt:
            mock_dt.now = datetime.now
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.fromtimestamp = lambda ts: orig(ts, tz=UTC)
            count = store.cleanup_old_content(past)
        assert count == 0
        assert len(store.list_refs("t_recent")) == 1

    def test_short_trace_id_shard(self, temp_dir) -> None:
        config = ContentStoreConfig(base_path=temp_dir / "short")
        store = FileContentStore(config)
        ref = store.store("x", ContentType.CUSTOM, "s1", "a")
        retrieved = store.retrieve_by_trace("a", ref.content_id)
        assert retrieved == "x"
