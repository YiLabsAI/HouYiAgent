"""Integration test: Chat API full endpoint coverage.

Tests all Chat API endpoints via FastAPI TestClient with mocked LLM.
Each endpoint has at least happy path + 1 error path.
"""

from __future__ import annotations

import io
import json
import zipfile
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from houyi_studio.server.chat.chat_api import register_chat_routes, router
from houyi_studio.server.chat.chat_service import ChatService
from houyi_studio.server.chat.json_store import JsonStore

from houyi.llm.models import GPT_4O


def _make_mock_llm():
    """Create a mock LLM adapter that yields predictable content."""
    mock = AsyncMock()

    async def mock_stream_chat(messages, model=None, **kwargs):
        yield ("Hello ", None)
        yield ("from ", None)
        yield ("mock!", None)

    mock.stream_chat = mock_stream_chat
    mock.last_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    return mock


@pytest.fixture
def app_and_client(tmp_path):
    """Create FastAPI app with Chat routes and TestClient."""
    store = JsonStore(data_dir=tmp_path / "conversations")
    service = ChatService(json_store=store, default_model="test-model")
    service._llm_adapter = _make_mock_llm()

    app = FastAPI()
    register_chat_routes(service)
    app.include_router(router)

    client = TestClient(app)
    return app, client, store


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
        create_resp = client.post("/api/chat/conversations", json={"title": "Get Test"})
        conv_id = create_resp.json()["conversation_id"]

        resp = client.get(f"/api/chat/conversations/{conv_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["conversation_id"] == conv_id
        assert data["title"] == "Get Test"
        assert "messages" in data

    def test_get_not_found(self, app_and_client):
        _, client, _ = app_and_client
        resp = client.get("/api/chat/conversations/nonexistent")
        assert resp.status_code == 404


class TestUpdateConversation:
    """PATCH /api/chat/conversations/{id}"""

    def test_update_title(self, app_and_client):
        _, client, _ = app_and_client
        create_resp = client.post("/api/chat/conversations", json={"title": "Old Title"})
        conv_id = create_resp.json()["conversation_id"]

        resp = client.patch(f"/api/chat/conversations/{conv_id}", json={"title": "New Title"})
        assert resp.status_code == 200
        assert resp.json()["title"] == "New Title"

    def test_update_not_found(self, app_and_client):
        _, client, _ = app_and_client
        resp = client.patch("/api/chat/conversations/nonexistent", json={"title": "X"})
        assert resp.status_code == 404

    def test_update_status(self, app_and_client):
        _, client, _ = app_and_client
        create_resp = client.post("/api/chat/conversations", json={})
        conv_id = create_resp.json()["conversation_id"]

        resp = client.patch(f"/api/chat/conversations/{conv_id}", json={"status": "archived"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "archived"

    def test_bookmark_conversation(self, app_and_client):
        _, client, _ = app_and_client
        create_resp = client.post("/api/chat/conversations", json={})
        conv_id = create_resp.json()["conversation_id"]

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


class TestDeleteConversation:
    """DELETE /api/chat/conversations/{id}"""

    def test_delete_success(self, app_and_client):
        _, client, _ = app_and_client
        create_resp = client.post("/api/chat/conversations", json={})
        conv_id = create_resp.json()["conversation_id"]

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


class TestSendMessage:
    """POST /api/chat/conversations/{id}/messages"""

    def test_send_message_sse_stream(self, app_and_client):
        _, client, store = app_and_client
        create_resp = client.post("/api/chat/conversations", json={"title": "SSE Test"})
        conv_id = create_resp.json()["conversation_id"]

        resp = client.post(
            f"/api/chat/conversations/{conv_id}/messages",
            json={"content": "Hello"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        # Parse SSE events
        events = _parse_sse_response(resp.text)
        event_types = [e["event"] for e in events]

        # Must have context.usage, deltas, and finish
        assert "context.usage" in event_types
        assert "message.delta" in event_types
        assert "message.finish" in event_types

        # Verify delta content
        deltas = [e for e in events if e["event"] == "message.delta"]
        full_content = "".join(e["data"].get("content", "") for e in deltas)
        assert "Hello from mock!" in full_content

        # Verify seq increments
        seqs = [e["data"]["seq"] for e in deltas]
        assert seqs == list(range(1, len(seqs) + 1))

    def test_send_message_persists(self, app_and_client):
        _, client, store = app_and_client
        create_resp = client.post("/api/chat/conversations", json={})
        conv_id = create_resp.json()["conversation_id"]

        # Send message (consume the stream)
        client.post(
            f"/api/chat/conversations/{conv_id}/messages",
            json={"content": "Test persist"},
        )

        # Verify persisted
        conv = store.get(conv_id)
        assert conv is not None
        assert len(conv.messages) == 2  # user + assistant
        assert conv.messages[0].role.value == "user"
        assert conv.messages[0].content == "Test persist"
        assert conv.messages[1].role.value == "assistant"
        assert len(conv.messages[1].content) > 0

    def test_send_message_not_found(self, app_and_client):
        _, client, _ = app_and_client
        resp = client.post(
            "/api/chat/conversations/nonexistent/messages",
            json={"content": "Hello"},
        )
        assert resp.status_code == 404


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

    def test_import_non_zip(self, app_and_client):
        _, client, _ = app_and_client
        resp = client.post(
            "/api/chat/import/cherrystudio",
            files={"file": ("backup.txt", io.BytesIO(b"not a zip"), "text/plain")},
        )
        assert resp.status_code == 400

    def test_import_empty_file(self, app_and_client):
        _, client, _ = app_and_client
        resp = client.post(
            "/api/chat/import/cherrystudio",
            files={"file": ("backup.zip", io.BytesIO(b""), "application/zip")},
        )
        assert resp.status_code == 400


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


# --- Helpers ---


def _parse_sse_response(text: str) -> list[dict]:
    """Parse SSE response text into structured events."""
    events = []
    current: dict = {}
    for line in text.split("\n"):
        if line.startswith("id: "):
            current["id"] = line[4:]
        elif line.startswith("event: "):
            current["event"] = line[7:]
        elif line.startswith("data: "):
            try:
                current["data"] = json.loads(line[6:])
            except json.JSONDecodeError:
                current["data"] = line[6:]
        elif line == "" and "event" in current:
            events.append(current)
            current = {}
    if "event" in current:
        events.append(current)
    return events


def _make_backup_zip(data: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data.json", json.dumps(data))
    return buf.getvalue()
