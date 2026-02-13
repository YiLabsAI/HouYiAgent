"""Tests for bookmark features: JsonStore.get_bookmarks + API endpoint."""

from pathlib import Path

import pytest
from houyi_studio.server.chat.json_store import JsonStore
from houyi_studio.server.chat.types import Conversation, Message, MessageRole


@pytest.fixture
def store(tmp_path: Path) -> JsonStore:
    return JsonStore(data_dir=tmp_path)


def _seed_bookmarks(store: JsonStore) -> tuple[str, str, str]:
    """Create conversations with various bookmark states.

    Returns (conv1_id, conv2_id, conv3_id).
    - conv1: bookmarked conversation, message m2 bookmarked
    - conv2: not bookmarked, message m4 bookmarked
    - conv3: not bookmarked, no bookmarked messages
    """
    c1 = Conversation(title="Bookmarked Chat", bookmarked=True)
    c1.messages = [
        Message(message_id="m1", role=MessageRole.USER, content="Hello world"),
        Message(
            message_id="m2",
            role=MessageRole.ASSISTANT,
            content="Hi! How can I help you today?",
            bookmarked=True,
        ),
    ]
    store.create(c1)

    c2 = Conversation(title="Regular Chat")
    c2.messages = [
        Message(message_id="m3", role=MessageRole.USER, content="What is Python?"),
        Message(
            message_id="m4",
            role=MessageRole.ASSISTANT,
            content="Python is a programming language.",
            bookmarked=True,
        ),
        Message(message_id="m5", role=MessageRole.USER, content="Thanks"),
    ]
    store.create(c2)

    c3 = Conversation(title="No Bookmarks Here")
    c3.messages = [
        Message(message_id="m6", role=MessageRole.USER, content="Just chatting"),
    ]
    store.create(c3)

    return c1.conversation_id, c2.conversation_id, c3.conversation_id


class TestGetBookmarks:
    """Tests for JsonStore.get_bookmarks()."""

    def test_returns_bookmarked_conversation(self, store: JsonStore):
        c1_id, _, _ = _seed_bookmarks(store)
        results = store.get_bookmarks()
        conv_bookmarks = [r for r in results if r["type"] == "conversation"]
        assert len(conv_bookmarks) == 1
        assert conv_bookmarks[0]["conversation_id"] == c1_id
        assert conv_bookmarks[0]["title"] == "Bookmarked Chat"

    def test_returns_bookmarked_messages(self, store: JsonStore):
        _seed_bookmarks(store)
        results = store.get_bookmarks()
        msg_bookmarks = [r for r in results if r["type"] == "message"]
        assert len(msg_bookmarks) == 2
        msg_ids = {r["message_id"] for r in msg_bookmarks}
        assert msg_ids == {"m2", "m4"}

    def test_message_has_snippet(self, store: JsonStore):
        _seed_bookmarks(store)
        results = store.get_bookmarks()
        msg_bookmarks = [r for r in results if r["type"] == "message"]
        for r in msg_bookmarks:
            assert "snippet" in r
            assert len(r["snippet"]) > 0

    def test_message_has_role(self, store: JsonStore):
        _seed_bookmarks(store)
        results = store.get_bookmarks()
        msg_bookmarks = [r for r in results if r["type"] == "message"]
        for r in msg_bookmarks:
            assert r["role"] == "assistant"

    def test_excludes_unbookmarked_conversations(self, store: JsonStore):
        _, c2_id, c3_id = _seed_bookmarks(store)
        results = store.get_bookmarks()
        conv_ids = [r["conversation_id"] for r in results if r["type"] == "conversation"]
        assert c2_id not in conv_ids
        assert c3_id not in conv_ids

    def test_excludes_unbookmarked_messages(self, store: JsonStore):
        _seed_bookmarks(store)
        results = store.get_bookmarks()
        msg_ids = {r["message_id"] for r in results if r["type"] == "message"}
        assert "m1" not in msg_ids
        assert "m3" not in msg_ids
        assert "m5" not in msg_ids
        assert "m6" not in msg_ids

    def test_empty_store_returns_empty(self, store: JsonStore):
        results = store.get_bookmarks()
        assert results == []

    def test_no_bookmarks_returns_empty(self, store: JsonStore):
        c = Conversation(title="Plain Chat")
        c.messages = [Message(role=MessageRole.USER, content="Hi")]
        store.create(c)
        results = store.get_bookmarks()
        assert results == []

    def test_sorted_by_created_at_descending(self, store: JsonStore):
        _seed_bookmarks(store)
        results = store.get_bookmarks()
        timestamps = [r["created_at"] for r in results]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_conversation_entry_has_required_fields(self, store: JsonStore):
        _seed_bookmarks(store)
        results = store.get_bookmarks()
        conv = next(r for r in results if r["type"] == "conversation")
        assert "conversation_id" in conv
        assert "title" in conv
        assert "message_count" in conv
        assert "model" in conv
        assert "created_at" in conv
        assert "updated_at" in conv

    def test_message_entry_has_required_fields(self, store: JsonStore):
        _seed_bookmarks(store)
        results = store.get_bookmarks()
        msg = next(r for r in results if r["type"] == "message")
        assert "conversation_id" in msg
        assert "title" in msg
        assert "message_id" in msg
        assert "role" in msg
        assert "snippet" in msg
        assert "created_at" in msg
        assert "updated_at" in msg

    def test_snippet_truncated_for_long_content(self, store: JsonStore):
        c = Conversation(title="Long Content", bookmarked=False)
        long_text = "A" * 200
        c.messages = [
            Message(
                message_id="mlong", role=MessageRole.ASSISTANT, content=long_text, bookmarked=True
            ),
        ]
        store.create(c)
        results = store.get_bookmarks()
        msg = next(r for r in results if r["message_id"] == "mlong")
        assert len(msg["snippet"]) <= 123  # 120 + "..."
        assert msg["snippet"].endswith("...")
