"""Chat API full endpoint coverage.

Tests all Chat API endpoints via FastAPI TestClient with mocked LLM.
Each endpoint has at least happy path + 1 error path.
"""

from __future__ import annotations

import asyncio
import io
import json
import zipfile
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from houyi_studio.server.chat import chat_api as chat_api_module
from houyi_studio.server.chat import chat_service as chat_service_module
from houyi_studio.server.chat.chat_api import register_chat_routes, router
from houyi_studio.server.chat.chat_service import ChatService
from houyi_studio.server.chat.json_store import JsonStore
from houyi_studio.server.chat.types import (
    Attachment,
    Conversation,
    ConversationContextState,
    Message,
    MessageRole,
)

from houyi.adapters.llm.models import GPT_4O

from .chat_test_utils import (
    cleanup_app_client,
    create_app_client_store,
    create_conversation_id,
    get_registered_chat_service,
    make_mock_llm,
)

_create_conversation_id = create_conversation_id
_get_registered_chat_service = get_registered_chat_service


def _make_trace_span(
    *,
    span_id: str = "root",
    parent_id: str | None = None,
    start_time: float = 10.0,
    end_time: float = 10.2,
    name: str = "chat.send_message",
    span_type: str = "llm",
    status: str = "ok",
    attributes: dict[str, object] | None = None,
    events: list[object] | None = None,
    tokens: object | None = None,
):
    return SimpleNamespace(
        span_id=span_id,
        parent_id=parent_id,
        start_time=start_time,
        end_time=end_time,
        name=name,
        span_type=span_type,
        status=status,
        attributes=attributes or {},
        events=events or [],
        tokens=tokens,
    )


def _install_fake_trace_query(monkeypatch, trace_id: str, trace_view: object) -> None:
    class _FakeObservabilityQuery:
        def get_trace(self, requested_trace_id: str, include_content: bool = False):
            if requested_trace_id == trace_id and include_content is False:
                return trace_view
            return None

    monkeypatch.setattr(chat_api_module, "ObservabilityQuery", _FakeObservabilityQuery)


def _expected_trace_context_contract() -> dict[str, object]:
    return {
        "request_context": {
            "request_id": "msg_req_1",
            "conversation_id": "conv_1",
            "model": "deepseek-chat",
            "max_context_tokens": 8192,
            "llm_messages_count": 14,
        },
        "context_plan": {
            "used_tokens": 1520,
            "planned_prompt_tokens": 1520,
            "reserved_output_tokens": 1024,
            "available_input_tokens": 5648,
            "block_breakdown": {"recent": 1200, "pinned": 320},
        },
        "context_governance": {
            "dropped_blocks": ["memory"],
            "drop_reasons": {"memory": "boundary_excluded"},
            "dropped_block_details": [
                {
                    "candidate_id": "memory",
                    "block_type": "memory",
                    "source": "memory",
                    "token_count": 42,
                    "message_count": 2,
                    "pinned": True,
                }
            ],
            "compaction": {
                "triggered": True,
                "trigger": "pre_request_pressure",
                "messages_compacted": 4,
                "tokens_before": 4800,
                "tokens_after": 2600,
                "saved_tokens": 2200,
                "pin_violation_count": 0,
            },
        },
    }


@pytest.fixture
def app_and_client(tmp_path):
    app, client, store = create_app_client_store(tmp_path, llm=make_mock_llm())
    try:
        yield app, client, store
    finally:
        cleanup_app_client(client)


class TestCreateConversation:
    """POST /api/chat/conversations"""

    def test_create_success(self, app_and_client):
        _, client, _ = app_and_client
        resp = client.post("/api/chat/conversations", json={"title": "Test Chat"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["title"] == "Test Chat"
        assert "conversation_id" in data
        assert data["status"] == "active"
        assert data["message_count"] == 0
        assert data["active_streaming_state"] is None

    def test_create_default_title(self, app_and_client):
        _, client, _ = app_and_client
        resp = client.post("/api/chat/conversations", json={})
        assert resp.status_code == 201
        assert resp.json()["title"] == "New Chat"

    def test_create_with_model(self, app_and_client):
        _, client, _ = app_and_client
        resp = client.post(
            "/api/chat/conversations",
            json={
                "title": "Model Chat",
                "model": GPT_4O,
                "system_instructions": "Be concise",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["model"] == GPT_4O


class TestListConversations:
    """GET /api/chat/conversations"""

    def test_list_empty(self, app_and_client):
        _, client, _ = app_and_client
        resp = client.get("/api/chat/conversations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["conversations"] == []
        assert data["total"] == 0

    def test_list_with_conversations(self, app_and_client):
        _, client, _ = app_and_client
        client.post("/api/chat/conversations", json={"title": "Chat 1"})
        client.post("/api/chat/conversations", json={"title": "Chat 2"})

        resp = client.get("/api/chat/conversations")
        data = resp.json()
        assert data["total"] == 2
        assert len(data["conversations"]) == 2
        assert all(item["active_streaming_state"] is None for item in data["conversations"])

    def test_list_pagination(self, app_and_client):
        _, client, _ = app_and_client
        for i in range(5):
            client.post("/api/chat/conversations", json={"title": f"Chat {i}"})

        resp = client.get("/api/chat/conversations?limit=2&offset=0")
        data = resp.json()
        assert len(data["conversations"]) == 2
        assert data["total"] == 5


class TestGetConversation:
    """GET /api/chat/conversations/{id}"""

    def test_get_success(self, app_and_client):
        _, client, _ = app_and_client
        conv_id = _create_conversation_id(client, title="Get Test")

        resp = client.get(f"/api/chat/conversations/{conv_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["conversation_id"] == conv_id
        assert data["title"] == "Get Test"
        assert "messages" in data
        assert data["conversation_context_state"] == {
            "conversation_id": conv_id,
            "used_units": 0,
            "max_units": 272000,
            "state": "healthy",
            "last_compacted_at": None,
            "last_compaction_delta": None,
            "last_compacted_message_count": None,
            "updated_at": data["conversation_context_state"]["updated_at"],
        }
        assert data["active_streaming_state"] is None

    def test_get_not_found(self, app_and_client):
        _, client, _ = app_and_client
        resp = client.get("/api/chat/conversations/nonexistent")
        assert resp.status_code == 404
        detail = resp.json()["detail"]
        assert detail["error_code"] == "resource_not_found"
        assert detail["public_message"] == "Conversation nonexistent not found"
        assert detail["retryable"] is False


class TestGetTrace:
    """GET /api/chat/trace/{trace_id}"""

    def test_get_trace_success(self, app_and_client, monkeypatch):
        _, client, _ = app_and_client

        root = _make_trace_span(
            attributes={"k": "v"},
            events=[SimpleNamespace(name="started", timestamp=10.0, attributes={"a": 1})],
            tokens=SimpleNamespace(input=100, output=50, total=150),
        )
        child = _make_trace_span(
            span_id="child1",
            parent_id="root",
            start_time=10.05,
            end_time=10.1,
            name="tool.execute",
            span_type="tool",
            tokens=SimpleNamespace(input=20, output=10, total=30),
        )

        fake_trace_view = SimpleNamespace(
            trace_id="trace_123",
            spans=[root, child],
            total_duration_ms=200.0,
        )
        _install_fake_trace_query(monkeypatch, "trace_123", fake_trace_view)

        resp = client.get("/api/chat/trace/trace_123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["trace_id"] == "trace_123"
        assert data["total_duration_ms"] == 200.0
        assert data["total_tokens"] == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "llm_spans": 1,
            "llm_spans_with_usage": 1,
            "is_partial": False,
        }
        assert data["root_span"]["name"] == "chat.send_message"
        assert len(data["root_span"]["children"]) == 1
        assert data["root_span"]["children"][0]["name"] == "tool.execute"

    def test_get_trace_context(self, app_and_client, monkeypatch):
        _, client, _ = app_and_client

        trace_context = _expected_trace_context_contract()
        root = _make_trace_span(
            end_time=10.3,
            span_type="execution",
            attributes={
                "chat.request_id": "msg_req_1",
                "chat.conversation_id": "conv_1",
                "chat.model": "deepseek-chat",
                "chat.context_tokens_max": 8192,
                "chat.llm_messages_count": 14,
                "chat.context_tokens_used": 1520,
                "chat.context_planned_prompt_tokens": 1520,
                "chat.context_reserved_output_tokens": 1024,
                "chat.context_available_input_tokens": 5648,
                "chat.context_blocks": json.dumps(trace_context["context_plan"]["block_breakdown"]),
                "chat.context_dropped_blocks": json.dumps(
                    trace_context["context_governance"]["dropped_blocks"]
                ),
                "chat.context_drop_reasons": json.dumps(
                    trace_context["context_governance"]["drop_reasons"]
                ),
                "chat.context_dropped_block_details": json.dumps(
                    trace_context["context_governance"]["dropped_block_details"]
                ),
                "chat.compaction.triggered": True,
                "chat.compaction.trigger": "pre_request_pressure",
                "chat.compaction.messages_compacted": 4,
                "chat.compaction.tokens_before": 4800,
                "chat.compaction.tokens_after": 2600,
                "chat.compaction.pin_violation_count": 0,
            },
        )

        fake_trace_view = SimpleNamespace(
            trace_id="trace_ctx",
            spans=[root],
            total_duration_ms=300.0,
        )
        _install_fake_trace_query(monkeypatch, "trace_ctx", fake_trace_view)

        resp = client.get("/api/chat/trace/trace_ctx")
        assert resp.status_code == 200
        data = resp.json()
        attrs = data["root_span"]["attributes"]
        assert (
            json.loads(attrs["chat.context_blocks"])
            == trace_context["context_plan"]["block_breakdown"]
        )
        assert (
            json.loads(attrs["chat.context_drop_reasons"])
            == trace_context["context_governance"]["drop_reasons"]
        )
        assert (
            json.loads(attrs["chat.context_dropped_block_details"])
            == trace_context["context_governance"]["dropped_block_details"]
        )
        assert attrs["chat.compaction.triggered"] is True
        assert attrs["chat.compaction.trigger"] == "pre_request_pressure"
        assert attrs["chat.compaction.messages_compacted"] == 4
        assert attrs["chat.compaction.tokens_before"] == 4800
        assert attrs["chat.compaction.tokens_after"] == 2600
        assert attrs["chat.compaction.pin_violation_count"] == 0
        assert data["request_context"] == trace_context["request_context"]
        assert data["context_plan"] == trace_context["context_plan"]
        assert data["context_governance"] == trace_context["context_governance"]

    def test_get_trace_missing(self, app_and_client, monkeypatch):
        _, client, _ = app_and_client

        class _FakeObservabilityQuery:
            def get_trace(self, trace_id: str, include_content: bool = False):
                return None

        monkeypatch.setattr(chat_api_module, "ObservabilityQuery", _FakeObservabilityQuery)

        resp = client.get("/api/chat/trace/missing")
        assert resp.status_code == 404
        detail = resp.json()["detail"]
        assert detail["error_code"] == "trace_not_found"
        assert detail["public_message"] == "Trace missing not found"


class TestUpdateConversation:
    """PATCH /api/chat/conversations/{id}"""

    def test_update_title(self, app_and_client):
        _, client, _ = app_and_client
        conv_id = _create_conversation_id(client, title="Old Title")

        resp = client.patch(f"/api/chat/conversations/{conv_id}", json={"title": "New Title"})
        assert resp.status_code == 200
        assert resp.json()["title"] == "New Title"

    def test_update_not_found(self, app_and_client):
        _, client, _ = app_and_client
        resp = client.patch("/api/chat/conversations/nonexistent", json={"title": "X"})
        assert resp.status_code == 404
        assert resp.json()["detail"]["error_code"] == "resource_not_found"

    def test_update_status(self, app_and_client):
        _, client, _ = app_and_client
        conv_id = _create_conversation_id(client)

        resp = client.patch(f"/api/chat/conversations/{conv_id}", json={"status": "archived"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "archived"

    def test_bookmark_conversation(self, app_and_client):
        _, client, _ = app_and_client
        conv_id = _create_conversation_id(client)

        # Default: not bookmarked
        get_resp = client.get(f"/api/chat/conversations/{conv_id}")
        assert get_resp.json()["bookmarked"] is False

        # Bookmark it
        resp = client.patch(f"/api/chat/conversations/{conv_id}", json={"bookmarked": True})
        assert resp.status_code == 200
        assert resp.json()["bookmarked"] is True

        # Verify persisted
        get_resp2 = client.get(f"/api/chat/conversations/{conv_id}")
        assert get_resp2.json()["bookmarked"] is True

        # Unbookmark
        resp2 = client.patch(f"/api/chat/conversations/{conv_id}", json={"bookmarked": False})
        assert resp2.status_code == 200
        assert resp2.json()["bookmarked"] is False


class TestPinnedContextApi:
    def test_pin_message_to_context(self, app_and_client):
        _, client, store = app_and_client
        conversation = Conversation(title="Pinned API")
        conversation.messages = [
            Message(
                message_id="u1",
                role=MessageRole.USER,
                content="Please remember the deployment order.",
            )
        ]
        store.create(conversation)

        resp = client.post(
            f"/api/chat/conversations/{conversation.conversation_id}/messages/u1/pin-context",
            json={"title": "Deployment order"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["source_message_id"] == "u1"
        assert data["title"] == "Deployment order"
        assert data["status"] == "active"

        persisted = store.get(conversation.conversation_id)
        assert persisted is not None
        pins = persisted.metadata.get("pinned_contexts")
        assert isinstance(pins, list)
        assert len(pins) == 1
        assert pins[0]["source_message_id"] == "u1"

    def test_list_pins_returns_active_and_inactive(self, app_and_client):
        _, client, store = app_and_client
        conversation = Conversation(title="Pinned List")
        conversation.messages = [
            Message(message_id="u1", role=MessageRole.USER, content="Rule A"),
            Message(message_id="u2", role=MessageRole.USER, content="Rule B"),
        ]
        store.create(conversation)

        first = client.post(
            f"/api/chat/conversations/{conversation.conversation_id}/messages/u1/pin-context",
            json={"title": "Rule A"},
        )
        assert first.status_code == 200
        first_pin_id = first.json()["pin_id"]

        second = client.post(
            f"/api/chat/conversations/{conversation.conversation_id}/messages/u2/pin-context",
            json={"title": "Rule B", "replace_pin_id": first_pin_id},
        )
        assert second.status_code == 200

        resp = client.get(f"/api/chat/conversations/{conversation.conversation_id}/pins")
        assert resp.status_code == 200
        pins = resp.json()["pins"]
        assert len(pins) == 2
        by_id = {pin["pin_id"]: pin for pin in pins}
        assert by_id[first_pin_id]["status"] == "superseded"
        second_pin_id = second.json()["pin_id"]
        assert by_id[second_pin_id]["status"] == "active"

        persisted = store.get(conversation.conversation_id)
        assert persisted is not None
        persisted_pins = persisted.metadata.get("pinned_contexts")
        assert isinstance(persisted_pins, list)
        persisted_by_id = {pin["pin_id"]: pin for pin in persisted_pins}
        assert persisted_by_id[first_pin_id]["status"] == "superseded"
        assert persisted_by_id[second_pin_id]["status"] == "active"

    def test_update_pin_status(self, app_and_client):
        _, client, store = app_and_client
        conversation = Conversation(title="Pinned Update")
        conversation.messages = [
            Message(message_id="u1", role=MessageRole.USER, content="Archive me"),
        ]
        store.create(conversation)

        created = client.post(
            f"/api/chat/conversations/{conversation.conversation_id}/messages/u1/pin-context",
            json={},
        )
        assert created.status_code == 200
        pin_id = created.json()["pin_id"]

        resp = client.patch(
            f"/api/chat/conversations/{conversation.conversation_id}/pins/{pin_id}",
            json={"status": "archived"},
        )
        assert resp.status_code == 200
        assert resp.json()["pin_id"] == pin_id
        assert resp.json()["status"] == "archived"

    def test_pin_message_not_found(self, app_and_client):
        _, client, _ = app_and_client
        conv_id = _create_conversation_id(client, title="Missing Message")

        resp = client.post(
            f"/api/chat/conversations/{conv_id}/messages/missing/pin-context",
            json={},
        )
        assert resp.status_code == 404
        assert resp.json()["detail"]["error_code"] == "resource_not_found"

    def test_update_pin_not_found(self, app_and_client):
        _, client, _ = app_and_client
        conv_id = _create_conversation_id(client, title="Missing Pin")

        resp = client.patch(
            f"/api/chat/conversations/{conv_id}/pins/missing",
            json={"status": "removed"},
        )
        assert resp.status_code == 404
        assert resp.json()["detail"]["error_code"] == "resource_not_found"


class TestDeleteConversation:
    """DELETE /api/chat/conversations/{id}"""

    def test_delete_success(self, app_and_client):
        _, client, _ = app_and_client
        conv_id = _create_conversation_id(client)

        resp = client.delete(f"/api/chat/conversations/{conv_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # Verify gone
        resp = client.get(f"/api/chat/conversations/{conv_id}")
        assert resp.status_code == 404

    def test_delete_not_found(self, app_and_client):
        _, client, _ = app_and_client
        resp = client.delete("/api/chat/conversations/nonexistent")
        assert resp.status_code == 404
        assert resp.json()["detail"]["error_code"] == "resource_not_found"


class TestCompactConversation:
    def test_compact_conversation(self, app_and_client):
        _, client, store = app_and_client
        conv_id = _create_conversation_id(client, title="Manual Compact")

        conv = store.get(conv_id)
        assert conv is not None
        conv.messages = [
            chat_service_module.Message(
                message_id=f"u{i}",
                role=chat_service_module.MessageRole.USER,
                content=f"older message {i}",
            )
            for i in range(1, 8)
        ]
        max_units = int(chat_service_module._ROLLING_CONTEXT_CAPACITY)
        conv.conversation_context_state = ConversationContextState(
            conversation_id=conv.conversation_id,
            used_units=max_units - 1000,
            max_units=max_units,
            state="elevated",
            updated_at=1.0,
        )
        store.update(conv)

        resp = client.post(f"/api/chat/conversations/{conv_id}/compact")
        assert resp.status_code == 200
        data = resp.json()
        assert data["conversation_id"] == conv_id
        assert data["applied"] is True
        assert data["compaction"]["trigger"] == "manual"

        persisted = store.get(conv_id)
        assert persisted is not None
        history = persisted.metadata.get("compaction_history")
        assert isinstance(history, list) and history
        assert history[-1]["trigger"] == "manual"

    def test_compact_conversation_missing(self, app_and_client):
        _, client, _ = app_and_client
        resp = client.post("/api/chat/conversations/nonexistent/compact")
        assert resp.status_code == 404
        assert resp.json()["detail"]["error_code"] == "resource_not_found"

    def test_restore_compaction(self, app_and_client):
        _, client, store = app_and_client
        conv_id = _create_conversation_id(client, title="Restore Flow")

        conv = store.get(conv_id)
        assert conv is not None
        conv.messages = [
            Message(message_id="u1", role=MessageRole.USER, content="Before compaction"),
            Message(message_id="a1", role=MessageRole.ASSISTANT, content="Current state"),
        ]
        store.update(conv)

        original_backup = store.create_backup(conv_id, trigger="manual")

        conv = store.get(conv_id)
        assert conv is not None
        conv.messages = [
            Message(message_id="u1", role=MessageRole.USER, content="Older state"),
        ]
        conv.metadata["compaction_history"] = [
            {
                "compaction_id": "cmp_restore_1",
                "trigger": "manual",
                "backup_id": original_backup["backup_id"],
                "summary": "Compacted old history",
                "source_message_ids": ["u1"],
                "pinned_message_ids": [],
                "retained_refs": [],
                "metrics": {"messages_compacted": 1, "tokens_before": 100, "tokens_after": 50},
                "created_at": 1.0,
                "metadata": {},
            }
        ]
        store.update(conv)

        restore_resp = client.post(
            f"/api/chat/conversations/{conv_id}/compactions/cmp_restore_1/restore"
        )
        assert restore_resp.status_code == 200
        restore_data = restore_resp.json()
        assert restore_data["status"] == "restored"
        assert restore_data["restored_compaction_id"] == "cmp_restore_1"
        restore_point_backup_id = restore_data["restore_point_backup_id"]
        assert isinstance(restore_point_backup_id, str) and restore_point_backup_id

        restore_point = store.get_backup(restore_point_backup_id)
        assert restore_point is not None
        assert restore_point["conversation_id"] == conv_id
        assert restore_point["trigger"] == "restore_point"
        assert restore_point["metadata"]["reason"] == "before_restore_compaction"
        assert restore_point["metadata"]["restored_compaction_id"] == "cmp_restore_1"

        restored_conv = store.get(conv_id)
        assert restored_conv is not None
        assert restored_conv.messages[-1].content == "Current state"

        undo_resp = client.post(
            f"/api/chat/conversations/{conv_id}/backups/{restore_point_backup_id}/restore"
        )
        assert undo_resp.status_code == 200
        undo_data = undo_resp.json()
        assert undo_data["status"] == "restored"
        assert undo_data["backup_id"] == restore_point_backup_id

        undone_conv = store.get(conv_id)
        assert undone_conv is not None
        assert undone_conv.messages[-1].content == "Older state"


class TestImportBackup:
    """POST /api/chat/import/cherrystudio — import from compatible backup format."""

    def test_import_success(self, app_and_client):
        _, client, store = app_and_client
        zip_data = _make_backup_zip(
            {
                "indexedDB": {
                    "topics": [
                        {
                            "id": "t1",
                            "title": "Imported Chat",
                            "messages": [
                                {"id": "m1", "role": "user", "content": "Hi", "createdAt": 1000},
                            ],
                        },
                    ],
                    "message_blocks": [
                        {"messageId": "m1", "type": "text", "content": "Hi", "createdAt": 1000},
                    ],
                },
            }
        )

        resp = client.post(
            "/api/chat/import/cherrystudio",
            files={"file": ("backup.zip", io.BytesIO(zip_data), "application/zip")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["conversations_imported"] == 1

    def test_import_invalid_file(self, app_and_client):
        _, client, _ = app_and_client
        resp = client.post(
            "/api/chat/import/cherrystudio",
            files={"file": ("backup.txt", io.BytesIO(b"not a zip"), "text/plain")},
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail["error_code"] == "invalid_import_file"
        assert detail["public_message"] == "Expected a .zip file"

    def test_import_empty_file(self, app_and_client):
        _, client, _ = app_and_client
        resp = client.post(
            "/api/chat/import/cherrystudio",
            files={"file": ("backup.zip", io.BytesIO(b""), "application/zip")},
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail["error_code"] == "empty_import_file"
        assert detail["public_message"] == "Empty file"


class TestExportConversations:
    """GET /api/chat/export"""

    def test_export_empty(self, app_and_client):
        _, client, _ = app_and_client
        resp = client.get("/api/chat/export")
        assert resp.status_code == 200
        assert "application/json" in resp.headers["content-type"]
        data = resp.json()
        assert data["version"] == 1
        assert "exported_at" in data
        assert data["conversations"] == []

    def test_export_with_conversations(self, app_and_client):
        _, client, store = app_and_client
        # Create conversations
        client.post("/api/chat/conversations", json={"title": "Export Chat 1"})
        client.post("/api/chat/conversations", json={"title": "Export Chat 2"})

        resp = client.get("/api/chat/export")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["conversations"]) == 2
        titles = {c["title"] for c in data["conversations"]}
        assert "Export Chat 1" in titles
        assert "Export Chat 2" in titles

    def test_export_includes_messages(self, app_and_client):
        _, client, store = app_and_client
        create_resp = client.post("/api/chat/conversations", json={"title": "Msg Export"})
        conv_id = create_resp.json()["conversation_id"]

        # Send a message to populate
        client.post(
            f"/api/chat/conversations/{conv_id}/messages",
            json={"content": "Hello for export"},
        )

        resp = client.get("/api/chat/export")
        data = resp.json()
        assert len(data["conversations"]) == 1
        conv = data["conversations"][0]
        assert len(conv["messages"]) == 2  # user + assistant
        assert conv["messages"][0]["content"] == "Hello for export"

    def test_export_content_disposition(self, app_and_client):
        _, client, _ = app_and_client
        resp = client.get("/api/chat/export")
        assert "content-disposition" in resp.headers
        assert "houyi-chat-export.json" in resp.headers["content-disposition"]


def _make_backup_zip(data: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data.json", json.dumps(data))
    return buf.getvalue()


class TestTraceTreeBuilder:
    """Trace tree assembly scenarios for root/child span structures."""

    def test_empty_spans_none(self):
        from houyi_studio.server.chat.chat_api import _build_trace_tree

        assert _build_trace_tree([]) is None

    def test_single_root_span(self):
        from houyi_studio.server.chat.chat_api import _build_trace_tree

        root = SimpleNamespace(
            span_id="r",
            parent_id=None,
            start_time=1.0,
            end_time=2.0,
            name="root",
            status="ok",
            attributes={},
            events=[],
        )
        tree = _build_trace_tree([root])
        assert tree is not None
        assert tree["name"] == "root"
        assert tree["children"] == []
        assert tree["duration_ms"] == pytest.approx(1000.0)

    def test_orphan_span_root(self):
        """Span whose parent_id is not in the list should be treated as root."""
        from houyi_studio.server.chat.chat_api import _build_trace_tree

        orphan = SimpleNamespace(
            span_id="o1",
            parent_id="missing_parent",
            start_time=1.0,
            end_time=1.5,
            name="orphan",
            status="ok",
            attributes={},
            events=[],
        )
        tree = _build_trace_tree([orphan])
        assert tree is not None
        assert tree["name"] == "orphan"

    def test_deep_nesting(self):
        from houyi_studio.server.chat.chat_api import _build_trace_tree

        root = SimpleNamespace(
            span_id="a",
            parent_id=None,
            start_time=1.0,
            end_time=3.0,
            name="a",
            status="ok",
            attributes={},
            events=[],
        )
        child = SimpleNamespace(
            span_id="b",
            parent_id="a",
            start_time=1.1,
            end_time=2.0,
            name="b",
            status="ok",
            attributes={},
            events=[],
        )
        grandchild = SimpleNamespace(
            span_id="c",
            parent_id="b",
            start_time=1.2,
            end_time=1.5,
            name="c",
            status="ok",
            attributes={},
            events=[],
        )
        tree = _build_trace_tree([root, child, grandchild])
        assert tree["name"] == "a"
        assert len(tree["children"]) == 1
        assert tree["children"][0]["name"] == "b"
        assert len(tree["children"][0]["children"]) == 1
        assert tree["children"][0]["children"][0]["name"] == "c"

    def test_children_sorted(self):
        from houyi_studio.server.chat.chat_api import _build_trace_tree

        root = SimpleNamespace(
            span_id="r",
            parent_id=None,
            start_time=1.0,
            end_time=3.0,
            name="root",
            status="ok",
            attributes={},
            events=[],
        )
        c2 = SimpleNamespace(
            span_id="c2",
            parent_id="r",
            start_time=2.0,
            end_time=2.5,
            name="second",
            status="ok",
            attributes={},
            events=[],
        )
        c1 = SimpleNamespace(
            span_id="c1",
            parent_id="r",
            start_time=1.5,
            end_time=1.8,
            name="first",
            status="ok",
            attributes={},
            events=[],
        )
        # Pass in reverse order — should still be sorted
        tree = _build_trace_tree([root, c2, c1])
        assert tree["children"][0]["name"] == "first"
        assert tree["children"][1]["name"] == "second"


class TestChatApiHelpers:
    def test_get_service(self, monkeypatch):
        monkeypatch.setattr(chat_api_module, "_chat_service", None)
        with pytest.raises(RuntimeError, match="ChatService not initialized"):
            chat_api_module._get_service()

    def test_get_settings(self, monkeypatch):
        monkeypatch.setattr(chat_api_module, "_settings_store", None)
        with pytest.raises(RuntimeError, match="SettingsStore not initialized"):
            chat_api_module._get_settings_store()

    @pytest.mark.asyncio
    async def test_iter_on_disconnect(self):
        class _Request:
            def __init__(self):
                self.calls = 0

            async def is_disconnected(self):
                self.calls += 1
                return self.calls > 1

        class _SlowStream:
            def __init__(self):
                self.closed = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                await asyncio.sleep(1)
                return "late"

            async def aclose(self):
                self.closed = True

        stream = _SlowStream()
        chunks = []
        async for chunk in chat_api_module._iter_with_disconnect_guard(
            request=_Request(),
            stream=stream,
            disconnect_log="disconnect",
        ):
            chunks.append(chunk)
        assert chunks == []
        assert stream.closed is True

    def test_parse_trace(self):
        assert chat_api_module._parse_trace_json_mapping('{"a":1}') == {"a": 1}
        assert chat_api_module._parse_trace_json_mapping('["x"]') == {}
        assert chat_api_module._parse_trace_json_mapping("{bad") == {}
        assert chat_api_module._parse_trace_json_list('["x"]') == ["x"]
        assert chat_api_module._parse_trace_json_list('{"a":1}') == []
        assert chat_api_module._parse_trace_json_list("[bad") == []
        assert chat_api_module._coerce_int("12") == 12
        assert chat_api_module._coerce_int("") is None
        assert chat_api_module._coerce_int("x") is None

    def test_message_preview(self):
        reasoning = Message(
            message_id="m1",
            role=MessageRole.ASSISTANT,
            content="",
            reasoning_content="think   hard",
        )
        tool_call = Message(
            message_id="m2",
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[{"id": "t1"}],
        )
        tool_result = Message(
            message_id="m3",
            role=MessageRole.TOOL,
            content="",
            tool_call_id="call_1",
        )
        attachment = Message(
            message_id="m4",
            role=MessageRole.USER,
            content="",
            attachments=[
                Attachment(
                    filename="a.txt",
                    mime_type="text/plain",
                    data="data:text/plain;base64,QQ==",
                    size=1,
                )
            ],
        )
        empty = Message(message_id="m5", role=MessageRole.USER, content="")

        previews = chat_api_module._build_message_previews(
            [reasoning, tool_call, tool_result, attachment, empty],
            ordered_ids=["m1", "m2", "m3", "m4", "m5", "missing"],
        )
        assert [item["message_id"] for item in previews] == ["m1", "m2", "m3", "m4", "m5"]
        assert previews[0]["preview"] == "think hard"
        assert previews[1]["preview"] == "[tool calls: 1]"
        assert previews[2]["preview"] == "[tool result: call_1]"
        assert previews[3]["preview"] == "[attachments: 1]"
        assert previews[4]["preview"] == "[empty]"

    def test_compaction_diff_and_history(self, tmp_path):
        store = JsonStore(data_dir=tmp_path / "helper-store")
        service = ChatService(json_store=store, default_model="test-model")
        current = Conversation(
            conversation_id="conv-helper",
            title="Helper",
            messages=[
                Message(message_id="m2", role=MessageRole.ASSISTANT, content="kept"),
                Message(message_id="m3", role=MessageRole.USER, content="added later"),
            ],
        )
        backup = Conversation(
            conversation_id="conv-helper",
            title="Helper",
            messages=[
                Message(message_id="m1", role=MessageRole.USER, content="removed"),
                Message(message_id="m2", role=MessageRole.ASSISTANT, content="kept"),
            ],
        )
        store.create(current)
        backup_entry = store.create_backup("conv-helper", trigger="manual")
        backup_id = backup_entry["backup_id"]
        backup_path = store._backup_dir / backup_entry["path"]
        backup_path.write_text(backup.model_dump_json(indent=2), encoding="utf-8")

        diff = chat_api_module._build_compaction_diff(
            current_conversation=current,
            backup_conversation=backup,
            compaction={"source_message_ids": ["m1"], "backup_id": backup_id},
        )
        assert diff["removed_message_ids"] == ["m1"]
        assert diff["added_message_ids"] == ["m3"]
        assert diff["source_message_previews"][0]["message_id"] == "m1"
        assert diff["added_message_previews"][0]["message_id"] == "m3"

        history_item = chat_api_module._build_compaction_history_item(
            service=service,
            conversation=current,
            compaction={
                "compaction_id": "cmp1",
                "backup_id": backup_id,
                "source_message_ids": ["m1"],
            },
        )
        assert history_item["backup"]["backup_id"] == backup_id
        assert history_item["diff"]["removed_message_ids"] == ["m1"]


class TestAdditionalChatApiRoutes:
    def test_seed_messages(self, app_and_client):
        _, client, _ = app_and_client
        conv_id = _create_conversation_id(client, title="Seeded")
        resp = client.post(
            f"/api/chat/conversations/{conv_id}/_seed-messages",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        assert resp.status_code == 200
        assert resp.json()["seeded"] == 1

        missing = client.post(
            "/api/chat/conversations/missing/_seed-messages",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        assert missing.status_code == 404
        assert missing.json()["detail"]["error_code"] == "resource_not_found"

    def test_context_usage_endpoint(self, app_and_client):
        _, client, store = app_and_client
        conv_id = _create_conversation_id(client, title="Usage")
        none_resp = client.get(f"/api/chat/conversations/{conv_id}/context-usage")
        assert none_resp.status_code == 200
        assert none_resp.json() == {"usage": None}

        conv = store.get(conv_id)
        assert conv is not None
        conv.messages = [Message(message_id="u1", role=MessageRole.USER, content="hello " * 40)]
        store.update(conv)

        usage_resp = client.get(f"/api/chat/conversations/{conv_id}/context-usage")
        assert usage_resp.status_code == 200
        assert usage_resp.json()["usage"] is not None

    def test_list_compactions_and_not_found(self, app_and_client):
        _, client, store = app_and_client
        conv_id = _create_conversation_id(client, title="Compactions")
        conv = store.get(conv_id)
        assert conv is not None
        conv.metadata["compaction_history"] = [
            {"compaction_id": "c1", "backup_id": "", "source_message_ids": ["m1"]},
            {"compaction_id": "c2", "backup_id": "", "source_message_ids": ["m2"]},
            "ignored",
        ]
        store.update(conv)

        resp = client.get(f"/api/chat/conversations/{conv_id}/compactions")
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert [item["compaction"]["compaction_id"] for item in items] == ["c2", "c1"]

        missing = client.get("/api/chat/conversations/missing/compactions")
        assert missing.status_code == 404
        assert missing.json()["detail"]["error_code"] == "conversation_not_found"

    def test_restore_compaction(self, app_and_client):
        _, client, store = app_and_client
        conv_id = _create_conversation_id(client, title="Restore Edge")
        conv = store.get(conv_id)
        assert conv is not None

        missing_conv = client.post("/api/chat/conversations/missing/compactions/c1/restore")
        assert missing_conv.status_code == 404
        assert missing_conv.json()["detail"]["error_code"] == "conversation_not_found"

        conv.metadata["compaction_history"] = [{"compaction_id": "c1"}]
        store.update(conv)

        missing_compaction = client.post(
            f"/api/chat/conversations/{conv_id}/compactions/nope/restore"
        )
        assert missing_compaction.status_code == 404
        assert missing_compaction.json()["detail"]["error_code"] == "compaction_not_found"

        no_backup = client.post(f"/api/chat/conversations/{conv_id}/compactions/c1/restore")
        assert no_backup.status_code == 400
        assert no_backup.json()["detail"]["error_code"] == "compaction_backup_missing"

        conv.metadata["compaction_history"] = [
            {"compaction_id": "c2", "backup_id": "missing-backup"}
        ]
        store.update(conv)
        missing_backup = client.post(f"/api/chat/conversations/{conv_id}/compactions/c2/restore")
        assert missing_backup.status_code == 404
        assert missing_backup.json()["detail"]["error_code"] == "resource_not_found"

        other = Conversation(conversation_id="conv-other", title="Other")
        store.create(other)
        foreign_backup = store.create_backup("conv-other", trigger="manual")
        conv.metadata["compaction_history"] = [
            {"compaction_id": "c3", "backup_id": foreign_backup["backup_id"]}
        ]
        store.update(conv)
        mismatch = client.post(f"/api/chat/conversations/{conv_id}/compactions/c3/restore")
        assert mismatch.status_code == 400
        assert mismatch.json()["detail"]["error_code"] == "backup_conversation_mismatch"

    def test_restore_backup_errors(self, app_and_client):
        _, client, store = app_and_client
        conv_id = _create_conversation_id(client, title="Backup Edge")

        missing_conv = client.post("/api/chat/conversations/missing/backups/any/restore")
        assert missing_conv.status_code == 404
        assert missing_conv.json()["detail"]["error_code"] == "conversation_not_found"

        missing_backup = client.post(f"/api/chat/conversations/{conv_id}/backups/missing/restore")
        assert missing_backup.status_code == 404
        assert missing_backup.json()["detail"]["error_code"] == "backup_not_found"

        other = Conversation(conversation_id="conv-b", title="Other")
        store.create(other)
        foreign_backup = store.create_backup("conv-b", trigger="manual")
        mismatch = client.post(
            f"/api/chat/conversations/{conv_id}/backups/{foreign_backup['backup_id']}/restore"
        )
        assert mismatch.status_code == 400
        assert mismatch.json()["detail"]["error_code"] == "backup_conversation_mismatch"

    def test_delete_message_error_paths(self, app_and_client, monkeypatch):
        _, client, _ = app_and_client
        service = _get_registered_chat_service()

        async def _missing(*args, **kwargs):
            _ = (args, kwargs)
            raise FileNotFoundError("missing")

        async def _bad(*args, **kwargs):
            _ = (args, kwargs)
            raise ValueError("bad delete")

        monkeypatch.setattr(service, "delete_message", _missing)
        missing = client.delete("/api/chat/conversations/c1/messages/m1")
        assert missing.status_code == 404
        assert missing.json()["detail"]["error_code"] == "resource_not_found"

        monkeypatch.setattr(service, "delete_message", _bad)
        bad = client.delete("/api/chat/conversations/c1/messages/m1")
        assert bad.status_code == 400
        assert bad.json()["detail"]["error_code"] == "invalid_request"

    def test_regenerate_error_paths(self, app_and_client, monkeypatch):
        _, client, _ = app_and_client
        service = _get_registered_chat_service()

        async def _missing(*args, **kwargs):
            _ = (args, kwargs)
            raise FileNotFoundError("missing")
            yield

        async def _bad(*args, **kwargs):
            _ = (args, kwargs)
            raise ValueError("bad regenerate")
            yield

        monkeypatch.setattr(service, "regenerate_message", _missing)
        with pytest.raises(RuntimeError, match="response already started"):
            client.post("/api/chat/conversations/c1/messages/m1/regenerate")

        monkeypatch.setattr(service, "regenerate_message", _bad)
        with pytest.raises(RuntimeError, match="response already started"):
            client.post("/api/chat/conversations/c1/messages/m1/regenerate")

    def test_bookmarks_and_search_routes(self, app_and_client):
        _, client, store = app_and_client
        conv = Conversation(
            conversation_id="conv-search",
            title="Searchable Title",
            bookmarked=True,
            messages=[
                Message(message_id="m1", role=MessageRole.USER, content="Hello search world")
            ],
        )
        store.create(conv)

        bookmarks = client.get("/api/chat/bookmarks")
        assert bookmarks.status_code == 200
        assert any(item["type"] == "conversation" for item in bookmarks.json()["bookmarks"])

        search = client.get("/api/chat/search?q=search&limit=5")
        assert search.status_code == 200
        assert search.json()["query"] == "search"
        assert search.json()["results"]

    def test_settings_models(self, tmp_path, monkeypatch):
        settings_store = chat_api_module.SettingsStore(tmp_path / "settings.json")
        service = ChatService(json_store=JsonStore(tmp_path / "conv"), default_model="test-model")
        app = FastAPI()
        register_chat_routes(service, settings_store=settings_store)
        app.include_router(router)
        client = TestClient(app)

        get_settings = client.get("/api/chat/settings")
        assert get_settings.status_code == 200
        assert get_settings.json()["version"] == 1

        put_settings = client.put(
            "/api/chat/settings",
            json={
                "version": 1,
                "providers": [
                    {
                        "id": "p1",
                        "name": "Provider 1",
                        "base_url": "https://example.com",
                        "models": ["m1"],
                        "enabled": True,
                        "api_key": "",
                    }
                ],
                "defaults": {
                    "model": "m1",
                    "system_instructions": "hi",
                    "temperature": 0.7,
                    "max_tokens": 4096,
                    "stream": True,
                },
                "display": {
                    "user_name": "You",
                    "user_avatar": None,
                    "assistant_name": "Assistant",
                    "assistant_avatar": None,
                },
                "updated_at": 0,
            },
        )
        assert put_settings.status_code == 200

        models = client.get("/api/chat/models")
        assert models.status_code == 200
        assert models.json()["models"][0]["model"] == "m1"

        class _Probe:
            async def test_connection(self, base_url, api_key):
                return {"ok": True, "base_url": base_url, "api_key": api_key}

            async def fetch_models(self, base_url, api_key):
                return {"models": [{"id": "m1"}], "base_url": base_url, "api_key": api_key}

        monkeypatch.setattr(chat_api_module, "get_probe", lambda provider_id, base_url: _Probe())
        tested = client.post(
            "/api/chat/providers/test",
            json={"base_url": "https://example.com/", "api_key": "key", "provider_id": "p1"},
        )
        assert tested.status_code == 200
        assert tested.json()["base_url"] == "https://example.com"

        fetched = client.post(
            "/api/chat/providers/fetch-models",
            json={"base_url": "https://example.com/", "api_key": "key", "provider_id": "p1"},
        )
        assert fetched.status_code == 200
        assert fetched.json()["models"][0]["id"] == "m1"
