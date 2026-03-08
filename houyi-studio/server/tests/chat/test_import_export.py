"""Unit tests for houyi_studio.server.chat.import_export."""

from __future__ import annotations

import io
import json
import time
import zipfile

import pytest
from houyi_studio.server.chat.import_export import CherryStudioImporter, ImportResult
from houyi_studio.server.chat.json_store import JsonStore
from houyi_studio.server.chat.types import MessageRole

from houyi.adapters.llm.models import GPT_4O


@pytest.fixture
def store(tmp_path):
    return JsonStore(data_dir=tmp_path / "conversations")


@pytest.fixture
def importer(store):
    return CherryStudioImporter(store)


def _make_cherry_zip(data: dict) -> bytes:
    """Create a CherryStudio backup zip with data.json."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data.json", json.dumps(data, ensure_ascii=False))
    return buf.getvalue()


def _make_cherry_data(
    topics: list[dict] | None = None,
    message_blocks: list[dict] | None = None,
) -> dict:
    """Create a CherryStudio data.json structure."""
    return {
        "indexedDB": {
            "topics": topics or [],
            "message_blocks": message_blocks or [],
        },
        "version": 5,
        "time": int(time.time() * 1000),
    }


class TestCherryStudioImporterZipParsing:
    """Test zip extraction."""

    def test_valid_zip(self, importer):
        data = _make_cherry_data(
            topics=[
                {
                    "id": "t1",
                    "title": "Chat 1",
                    "messages": [],
                }
            ]
        )
        result = importer.import_from_zip(_make_cherry_zip(data))
        assert result.success
        assert result.conversations_imported == 1

    def test_no_data_json_in_zip(self, importer):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("other.txt", "not data.json")
        result = importer.import_from_zip(buf.getvalue())
        assert not result.success
        assert any("data.json" in e for e in result.errors)

    def test_corrupted_zip(self, importer):
        result = importer.import_from_zip(b"not a zip file")
        assert not result.success
        assert len(result.errors) > 0

    def test_empty_topics(self, importer):
        data = _make_cherry_data(topics=[])
        result = importer.import_from_zip(_make_cherry_zip(data))
        assert result.success
        assert result.conversations_imported == 0
        assert any("No topics" in w for w in result.warnings)


class TestCherryStudioImporterTopicMapping:
    """Test topic → Conversation mapping."""

    def test_topic_id_and_title(self, importer, store):
        data = _make_cherry_data(
            topics=[
                {
                    "id": "topic123",
                    "title": "My Chat",
                    "messages": [],
                    "createdAt": 1700000000000,
                    "updatedAt": 1700000001000,
                }
            ]
        )
        result = importer.import_from_zip(_make_cherry_zip(data))
        assert result.conversations_imported == 1

        conv = store.get("topic123")
        assert conv is not None
        assert conv.title == "My Chat"
        # ms → s conversion
        assert conv.created_at == pytest.approx(1700000000.0, abs=1)
        assert conv.updated_at == pytest.approx(1700000001.0, abs=1)

    def test_topic_metadata_preserved(self, importer, store):
        data = _make_cherry_data(
            topics=[
                {
                    "id": "t1",
                    "title": "Test",
                    "messages": [],
                    "model": GPT_4O,
                    "customField": "preserved",
                }
            ]
        )
        importer.import_from_zip(_make_cherry_zip(data))
        conv = store.get("t1")
        assert conv.metadata["import_source"] == "cherrystudio"
        assert conv.metadata["raw_import"]["customField"] == "preserved"

    def test_multiple_topics(self, importer, store):
        data = _make_cherry_data(
            topics=[
                {"id": "t1", "title": "Chat 1", "messages": []},
                {"id": "t2", "title": "Chat 2", "messages": []},
                {"id": "t3", "title": "Chat 3", "messages": []},
            ]
        )
        result = importer.import_from_zip(_make_cherry_zip(data))
        assert result.conversations_imported == 3
        assert store.count() == 3


class TestCherryStudioImporterBlockAggregation:
    """Test message_blocks → Message aggregation."""

    def test_text_block(self, importer, store):
        data = _make_cherry_data(
            topics=[
                {
                    "id": "t1",
                    "title": "Test",
                    "messages": [
                        {"id": "m1", "role": "user", "createdAt": 1000},
                    ],
                }
            ],
            message_blocks=[
                {"messageId": "m1", "type": "text", "content": "Hello world", "createdAt": 1000},
            ],
        )
        importer.import_from_zip(_make_cherry_zip(data))
        conv = store.get("t1")
        assert len(conv.messages) == 1
        assert conv.messages[0].content == "Hello world"

    def test_think_block_becomes_reasoning(self, importer, store):
        data = _make_cherry_data(
            topics=[
                {
                    "id": "t1",
                    "title": "Test",
                    "messages": [
                        {"id": "m1", "role": "assistant", "createdAt": 1000},
                    ],
                }
            ],
            message_blocks=[
                {"messageId": "m1", "type": "text", "content": "Answer", "createdAt": 1000},
                {
                    "messageId": "m1",
                    "type": "think",
                    "content": "Let me think...",
                    "createdAt": 999,
                },
            ],
        )
        importer.import_from_zip(_make_cherry_zip(data))
        conv = store.get("t1")
        msg = conv.messages[0]
        assert msg.content == "Answer"
        assert msg.reasoning_content == "Let me think..."

    def test_image_block_placeholder(self, importer, store):
        data = _make_cherry_data(
            topics=[
                {
                    "id": "t1",
                    "title": "Test",
                    "messages": [
                        {"id": "m1", "role": "user", "createdAt": 1000},
                    ],
                }
            ],
            message_blocks=[
                {
                    "messageId": "m1",
                    "type": "image",
                    "content": "",
                    "fileId": "img001",
                    "createdAt": 1000,
                },
            ],
        )
        importer.import_from_zip(_make_cherry_zip(data))
        conv = store.get("t1")
        assert "[image: img001]" in conv.messages[0].content

    def test_code_block(self, importer, store):
        data = _make_cherry_data(
            topics=[
                {
                    "id": "t1",
                    "title": "Test",
                    "messages": [
                        {"id": "m1", "role": "assistant", "createdAt": 1000},
                    ],
                }
            ],
            message_blocks=[
                {
                    "messageId": "m1",
                    "type": "code",
                    "content": "print('hi')",
                    "language": "python",
                    "createdAt": 1000,
                },
            ],
        )
        importer.import_from_zip(_make_cherry_zip(data))
        conv = store.get("t1")
        assert "```python" in conv.messages[0].content
        assert "print('hi')" in conv.messages[0].content

    def test_tooluse_block_placeholder(self, importer, store):
        data = _make_cherry_data(
            topics=[
                {
                    "id": "t1",
                    "title": "Test",
                    "messages": [
                        {"id": "m1", "role": "assistant", "createdAt": 1000},
                    ],
                }
            ],
            message_blocks=[
                {
                    "messageId": "m1",
                    "type": "tooluse",
                    "content": "",
                    "toolName": "web_search",
                    "createdAt": 1000,
                },
            ],
        )
        importer.import_from_zip(_make_cherry_zip(data))
        conv = store.get("t1")
        assert "[tool_use: web_search]" in conv.messages[0].content

    def test_toolresult_block_placeholder(self, importer, store):
        data = _make_cherry_data(
            topics=[
                {
                    "id": "t1",
                    "title": "Test",
                    "messages": [
                        {"id": "m1", "role": "tool", "createdAt": 1000},
                    ],
                }
            ],
            message_blocks=[
                {
                    "messageId": "m1",
                    "type": "toolresult",
                    "content": "Result data here",
                    "createdAt": 1000,
                },
            ],
        )
        importer.import_from_zip(_make_cherry_zip(data))
        conv = store.get("t1")
        assert "[tool_result:" in conv.messages[0].content

    def test_unknown_block_type_placeholder(self, importer, store):
        data = _make_cherry_data(
            topics=[
                {
                    "id": "t1",
                    "title": "Test",
                    "messages": [
                        {"id": "m1", "role": "user", "createdAt": 1000},
                    ],
                }
            ],
            message_blocks=[
                {
                    "messageId": "m1",
                    "type": "custom_widget",
                    "content": "widget data",
                    "createdAt": 1000,
                },
            ],
        )
        importer.import_from_zip(_make_cherry_zip(data))
        conv = store.get("t1")
        assert "[custom_widget:" in conv.messages[0].content

    def test_raw_import_blocks_preserved(self, importer, store):
        """Content-not-lost invariant: raw blocks preserved in metadata."""
        data = _make_cherry_data(
            topics=[
                {
                    "id": "t1",
                    "title": "Test",
                    "messages": [
                        {"id": "m1", "role": "user", "createdAt": 1000},
                    ],
                }
            ],
            message_blocks=[
                {
                    "messageId": "m1",
                    "type": "text",
                    "content": "Hello",
                    "createdAt": 1000,
                    "extra": "field",
                },
            ],
        )
        importer.import_from_zip(_make_cherry_zip(data))
        conv = store.get("t1")
        raw_blocks = conv.messages[0].metadata.get("raw_import", {}).get("blocks", [])
        assert len(raw_blocks) == 1
        assert raw_blocks[0]["extra"] == "field"

    def test_message_without_blocks_uses_content(self, importer, store):
        """Fallback: message-level content when no blocks exist."""
        data = _make_cherry_data(
            topics=[
                {
                    "id": "t1",
                    "title": "Test",
                    "messages": [
                        {
                            "id": "m1",
                            "role": "user",
                            "content": "Direct content",
                            "createdAt": 1000,
                        },
                    ],
                }
            ],
            message_blocks=[],
        )
        importer.import_from_zip(_make_cherry_zip(data))
        conv = store.get("t1")
        assert conv.messages[0].content == "Direct content"


class TestCherryStudioImporterRoleInference:
    """Test role inference logic."""

    def test_explicit_role(self, importer, store):
        data = _make_cherry_data(
            topics=[
                {
                    "id": "t1",
                    "title": "Test",
                    "messages": [
                        {"id": "m1", "role": "assistant", "createdAt": 1000},
                    ],
                }
            ],
            message_blocks=[
                {"messageId": "m1", "type": "text", "content": "Hi", "createdAt": 1000},
            ],
        )
        importer.import_from_zip(_make_cherry_zip(data))
        assert store.get("t1").messages[0].role == MessageRole.ASSISTANT

    def test_role_from_tool_blocks(self, importer, store):
        data = _make_cherry_data(
            topics=[
                {
                    "id": "t1",
                    "title": "Test",
                    "messages": [
                        {"id": "m1", "createdAt": 1000},  # no explicit role
                    ],
                }
            ],
            message_blocks=[
                {
                    "messageId": "m1",
                    "type": "tool_use",
                    "content": "",
                    "toolName": "search",
                    "createdAt": 1000,
                },
            ],
        )
        importer.import_from_zip(_make_cherry_zip(data))
        assert store.get("t1").messages[0].role == MessageRole.TOOL

    def test_role_from_author_field(self, importer, store):
        data = _make_cherry_data(
            topics=[
                {
                    "id": "t1",
                    "title": "Test",
                    "messages": [
                        {"id": "m1", "author": "ai", "createdAt": 1000},
                    ],
                }
            ],
            message_blocks=[
                {"messageId": "m1", "type": "text", "content": "Hi", "createdAt": 1000},
            ],
        )
        importer.import_from_zip(_make_cherry_zip(data))
        assert store.get("t1").messages[0].role == MessageRole.ASSISTANT

    def test_role_default_user(self, importer, store):
        data = _make_cherry_data(
            topics=[
                {
                    "id": "t1",
                    "title": "Test",
                    "messages": [
                        {"id": "m1", "createdAt": 1000},  # no role, no author, no tool blocks
                    ],
                }
            ],
            message_blocks=[
                {"messageId": "m1", "type": "text", "content": "Hi", "createdAt": 1000},
            ],
        )
        importer.import_from_zip(_make_cherry_zip(data))
        assert store.get("t1").messages[0].role == MessageRole.USER


class TestCherryStudioImporterOrderInvariant:
    """Test order preservation invariant."""

    def test_messages_sorted_by_timestamp(self, importer, store):
        """Messages within a conversation MUST maintain original timestamp order."""
        data = _make_cherry_data(
            topics=[
                {
                    "id": "t1",
                    "title": "Test",
                    "messages": [
                        {"id": "m3", "role": "assistant", "createdAt": 3000},
                        {"id": "m1", "role": "user", "createdAt": 1000},
                        {"id": "m2", "role": "assistant", "createdAt": 2000},
                    ],
                }
            ],
            message_blocks=[
                {"messageId": "m1", "type": "text", "content": "First", "createdAt": 1000},
                {"messageId": "m2", "type": "text", "content": "Second", "createdAt": 2000},
                {"messageId": "m3", "type": "text", "content": "Third", "createdAt": 3000},
            ],
        )
        importer.import_from_zip(_make_cherry_zip(data))
        conv = store.get("t1")
        contents = [m.content for m in conv.messages]
        assert contents == ["First", "Second", "Third"]

    def test_blocks_within_message_sorted(self, importer, store):
        """Multiple blocks within a message sorted by creation time."""
        data = _make_cherry_data(
            topics=[
                {
                    "id": "t1",
                    "title": "Test",
                    "messages": [
                        {"id": "m1", "role": "assistant", "createdAt": 1000},
                    ],
                }
            ],
            message_blocks=[
                {"messageId": "m1", "type": "text", "content": "Part B", "createdAt": 2000},
                {"messageId": "m1", "type": "text", "content": "Part A", "createdAt": 1000},
            ],
        )
        importer.import_from_zip(_make_cherry_zip(data))
        conv = store.get("t1")
        # Blocks sorted by createdAt, so Part A first
        assert "Part A" in conv.messages[0].content
        assert conv.messages[0].content.index("Part A") < conv.messages[0].content.index("Part B")


class TestCherryStudioImporterTimestamps:
    """Test timestamp conversion."""

    def test_ms_to_seconds_conversion(self, importer, store):
        data = _make_cherry_data(
            topics=[
                {
                    "id": "t1",
                    "title": "Test",
                    "messages": [
                        {"id": "m1", "role": "user", "createdAt": 1700000000000},  # ms
                    ],
                }
            ],
            message_blocks=[
                {"messageId": "m1", "type": "text", "content": "Hi", "createdAt": 1700000000000},
            ],
        )
        importer.import_from_zip(_make_cherry_zip(data))
        conv = store.get("t1")
        # Should be ~1.7 billion seconds, not ~1.7 trillion
        assert conv.messages[0].created_at < 2e10


class TestImportResult:
    """Test ImportResult model."""

    def test_success_when_no_errors(self):
        r = ImportResult()
        r.conversations_imported = 5
        assert r.success is True

    def test_failure_when_errors(self):
        r = ImportResult()
        r.errors.append("something broke")
        assert r.success is False

    def test_to_dict(self):
        r = ImportResult()
        r.conversations_imported = 2
        r.messages_imported = 10
        r.warnings.append("warn1")
        d = r.to_dict()
        assert d["success"] is True
        assert d["conversations_imported"] == 2
        assert d["messages_imported"] == 10
        assert d["warnings"] == ["warn1"]
        assert d["errors"] == []
