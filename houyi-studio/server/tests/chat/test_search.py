"""Tests for JsonStore.search: full-text search across conversations."""

from pathlib import Path

import pytest
from houyi_studio.server.chat.json_store import JsonStore
from houyi_studio.server.chat.types import Conversation, Message, MessageRole


@pytest.fixture
def store(tmp_path: Path) -> JsonStore:
    return JsonStore(data_dir=tmp_path)


def _seed(store: JsonStore) -> tuple[str, str]:
    """Create two conversations with searchable content."""
    c1 = Conversation(title="Python Tutorial")
    c1.messages = [
        Message(
            message_id="m1",
            role=MessageRole.USER,
            content="How do I use list comprehensions in Python?",
        ),
        Message(
            message_id="m2",
            role=MessageRole.ASSISTANT,
            content="List comprehensions provide a concise way to create lists.",
        ),
    ]
    store.create(c1)

    c2 = Conversation(title="JavaScript Basics")
    c2.messages = [
        Message(message_id="m3", role=MessageRole.USER, content="What is a closure in JavaScript?"),
        Message(
            message_id="m4",
            role=MessageRole.ASSISTANT,
            content="A closure is a function that has access to its outer scope.",
        ),
    ]
    store.create(c2)
    return c1.conversation_id, c2.conversation_id


class TestSearch:
    def test_search_by_title(self, store: JsonStore):
        _seed(store)
        results = store.search("Python")
        assert len(results) >= 1
        title_hits = [r for r in results if r["match_type"] == "title"]
        assert len(title_hits) == 1
        assert title_hits[0]["title"] == "Python Tutorial"

    def test_search_by_message_content(self, store: JsonStore):
        _seed(store)
        results = store.search("closure")
        msg_hits = [r for r in results if r["match_type"] == "message"]
        assert len(msg_hits) >= 1
        assert "closure" in msg_hits[0]["snippet"].lower()

    def test_search_case_insensitive(self, store: JsonStore):
        _seed(store)
        results = store.search("PYTHON")
        assert len(results) >= 1

    def test_search_empty_query(self, store: JsonStore):
        _seed(store)
        results = store.search("")
        assert results == []

    def test_search_no_match(self, store: JsonStore):
        _seed(store)
        results = store.search("quantum entanglement")
        assert results == []

    def test_search_limit(self, store: JsonStore):
        _seed(store)
        results = store.search("a", limit=2)
        assert len(results) <= 2

    def test_search_snippet_context(self, store: JsonStore):
        _seed(store)
        results = store.search("list comprehensions")
        msg_hits = [r for r in results if r["match_type"] == "message"]
        assert len(msg_hits) >= 1
        assert "list comprehensions" in msg_hits[0]["snippet"].lower()

    def test_search_returns_conversation_metadata(self, store: JsonStore):
        c1_id, _ = _seed(store)
        results = store.search("Python")
        assert results[0]["conversation_id"] == c1_id
        assert results[0]["title"] == "Python Tutorial"
