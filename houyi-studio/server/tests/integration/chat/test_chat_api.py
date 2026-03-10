"""Chat API full endpoint coverage.

Tests all Chat API endpoints via FastAPI TestClient with mocked LLM.
Each endpoint has at least happy path + 1 error path.
"""

from __future__ import annotations

import io
import json
import zipfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from houyi_studio.server.chat.chat_api import register_chat_routes, router
from houyi_studio.server.chat.chat_service import ChatService
from houyi_studio.server.chat.json_store import JsonStore

from houyi.adapters.llm.base import StreamChunk
from houyi.adapters.llm.models import GPT_4O


def _make_mock_llm():
    """Create a mock LLM adapter that yields predictable content."""
    mock = AsyncMock()

    async def mock_stream_chat(messages, model=None, **kwargs):
        yield StreamChunk(content_delta="Hello ")
        yield StreamChunk(content_delta="from ")
        yield StreamChunk(content_delta="mock!")

    mock.stream_chat = mock_stream_chat
    mock.last_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    return mock


@pytest.fixture
def app_and_client(tmp_path):
    """Create FastAPI app with Chat routes and TestClient."""
    store = JsonStore(data_dir=tmp_path / "conversations")
    service = ChatService(json_store=store, default_model="test-model")
    service._default_adapter = _make_mock_llm()

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


class TestGetTrace:
    """GET /api/chat/trace/{trace_id}"""

    def test_get_trace_success(self, app_and_client, monkeypatch):
        _, client, _ = app_and_client

        root = SimpleNamespace(
            span_id="root",
            parent_id=None,
            start_time=10.0,
            end_time=10.2,
            name="chat.send_message",
            span_type="llm",
            status="ok",
            attributes={"k": "v"},
            events=[SimpleNamespace(name="started", timestamp=10.0, attributes={"a": 1})],
            tokens=SimpleNamespace(input=100, output=50, total=150),
        )
        child = SimpleNamespace(
            span_id="child1",
            parent_id="root",
            start_time=10.05,
            end_time=10.1,
            name="tool.execute",
            span_type="tool",
            status="ok",
            attributes={},
            events=[],
            tokens=SimpleNamespace(input=20, output=10, total=30),
        )

        fake_trace_view = SimpleNamespace(
            trace_id="trace_123",
            spans=[root, child],
            total_duration_ms=200.0,
        )

        class _FakeObservabilityQuery:
            def get_trace(self, trace_id: str, include_content: bool = False):
                if trace_id == "trace_123" and include_content is False:
                    return fake_trace_view
                return None

        monkeypatch.setattr(
            "houyi_studio.server.chat.chat_api.ObservabilityQuery",
            _FakeObservabilityQuery,
        )

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

    def test_get_trace_not_found(self, app_and_client, monkeypatch):
        _, client, _ = app_and_client

        class _FakeObservabilityQuery:
            def get_trace(self, trace_id: str, include_content: bool = False):
                return None

        monkeypatch.setattr(
            "houyi_studio.server.chat.chat_api.ObservabilityQuery",
            _FakeObservabilityQuery,
        )

        resp = client.get("/api/chat/trace/missing")
        assert resp.status_code == 404
        assert "Trace missing not found" in resp.json()["detail"]


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
        assert "message.complete" in event_types

        # Verify delta content
        deltas = [e for e in events if e["event"] == "message.delta"]
        full_content = "".join(e["data"].get("content", "") for e in deltas)
        assert "Hello from mock!" in full_content

        complete = next(e for e in events if e["event"] == "message.complete")
        metadata = complete["data"]["metadata"]
        assert metadata["usage"]["total_tokens"] == 15
        assert metadata["first_token_latency_ms"] >= 0
        assert metadata["generation_time_ms"] >= 0
        assert metadata["tokens_per_second"] > 0

        conv = store.get(conv_id)
        assert conv is not None
        assert conv.messages[0].metadata["usage"]["input_tokens"] > 0
        assert conv.messages[1].metadata["usage"]["total_tokens"] == 15
        assert conv.messages[1].metadata["first_token_latency_ms"] >= 0
        assert conv.messages[1].metadata["tokens_per_second"] > 0

    def test_streams_final_response_after_tool_loop(self, app_and_client, monkeypatch):
        _, client, _ = app_and_client
        create_resp = client.post("/api/chat/conversations", json={"title": "Tool Loop Replay"})
        conv_id = create_resp.json()["conversation_id"]

        from houyi_studio.server.chat import chat_api

        service = chat_api._chat_service
        assert service is not None

        class _AdapterWithStream:
            last_usage = {"prompt_tokens": 9, "completion_tokens": 6, "total_tokens": 15}
            stream_calls = 0

            async def chat(self, *args, **kwargs):
                _ = (args, kwargs)
                return SimpleNamespace(content="", tool_calls=[], usage={}, metadata={})

            async def stream_chat(self, *args, **kwargs):
                _ = (args, kwargs)
                self.stream_calls += 1
                yield StreamChunk(content_delta="Answer ")
                yield StreamChunk(content_delta="from ")
                yield StreamChunk(content_delta="stream")

        class _FakeToolRunner:
            async def run(self, **kwargs):
                _ = kwargs
                return (
                    SimpleNamespace(
                        content="Answer from tool loop",
                        tool_calls=[],
                        usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                        metadata={},
                    ),
                    [],
                )

        adapter = _AdapterWithStream()
        monkeypatch.setattr(service, "_default_adapter", adapter)
        monkeypatch.setattr(service, "_get_tool_runner", lambda: _FakeToolRunner())
        monkeypatch.setattr(
            "houyi_studio.server.chat.chat_service.ToolBridge.collect_tool_schemas",
            lambda self, skill_filter, include_core: [
                {"type": "function", "function": {"name": "demo"}}
            ],
        )
        monkeypatch.setattr(
            "houyi_studio.server.chat.chat_service.ToolBridge.collect_skills",
            lambda self, skill_filter, include_core: [SimpleNamespace(name="demo")],
        )

        resp = client.post(
            f"/api/chat/conversations/{conv_id}/messages",
            json={"content": "hello", "enable_skills": ["demo"]},
        )
        assert resp.status_code == 200
        events = _parse_sse_response(resp.text)
        deltas = [e for e in events if e["event"] == "message.delta"]
        assert deltas[0]["data"].get("content", "") == ""
        assert "".join(e["data"].get("content", "") for e in deltas) == "Answer from tool loop"
        assert adapter.stream_calls == 0

        complete = next(e for e in events if e["event"] == "message.complete")
        assert complete["data"]["metadata"]["usage"]["total_tokens"] == 18

    def test_emits_tool_lifecycle_events_with_trace_metadata(self, app_and_client, monkeypatch):
        _, client, _ = app_and_client
        create_resp = client.post("/api/chat/conversations", json={"title": "Tool Lifecycle"})
        conv_id = create_resp.json()["conversation_id"]

        from houyi_studio.server.chat import chat_api

        service = chat_api._chat_service
        assert service is not None

        class _AdapterWithStream:
            last_usage = {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}

            async def chat(self, *args, **kwargs):
                _ = (args, kwargs)
                return SimpleNamespace(content="", tool_calls=[], usage={}, metadata={})

            async def stream_chat(self, *args, **kwargs):
                _ = (args, kwargs)
                yield StreamChunk(content_delta="done")

        class _FakeToolRunner:
            async def run(self, **kwargs):
                _ = kwargs
                return (
                    SimpleNamespace(
                        content="Done after tools",
                        tool_calls=[],
                        usage={"prompt_tokens": 13, "completion_tokens": 9, "total_tokens": 22},
                        metadata={},
                    ),
                    [
                        {
                            "tool_call_id": "call_ok",
                            "tool_name": "demo_ok",
                            "parallel_group_id": "round_1",
                            "round_index": 1,
                            "duration_ms": 1280.0,
                            "args": {"q": "alpha"},
                            "result": {"raw": {"success": True, "value": "ok"}},
                        },
                        {
                            "tool_call_id": "call_err",
                            "tool_name": "demo_err",
                            "parallel_group_id": "round_1",
                            "round_index": 1,
                            "duration_ms": 245.0,
                            "args": {"q": "beta"},
                            "result": {"raw": {"error": "boom", "code": "tool_failed"}},
                        },
                    ],
                )

        adapter = _AdapterWithStream()
        monkeypatch.setattr(service, "_default_adapter", adapter)
        monkeypatch.setattr(service, "_get_tool_runner", lambda: _FakeToolRunner())
        monkeypatch.setattr(
            "houyi_studio.server.chat.chat_service.ToolBridge.collect_tool_schemas",
            lambda self, skill_filter, include_core: [
                {"type": "function", "function": {"name": "demo_ok"}},
                {"type": "function", "function": {"name": "demo_err"}},
            ],
        )
        monkeypatch.setattr(
            "houyi_studio.server.chat.chat_service.ToolBridge.collect_skills",
            lambda self, skill_filter, include_core: [
                SimpleNamespace(name="demo_ok"),
                SimpleNamespace(name="demo_err"),
            ],
        )

        resp = client.post(
            f"/api/chat/conversations/{conv_id}/messages",
            json={"content": "run tools", "enable_skills": ["demo_ok", "demo_err"]},
        )
        assert resp.status_code == 200

        events = _parse_sse_response(resp.text)
        event_types = [e["event"] for e in events]
        assert "agent.iteration" in event_types
        assert event_types.count("tool_call.start") == 2
        assert event_types.count("tool_call.result") == 1
        assert event_types.count("tool_call.error") == 1

        iteration = next(e for e in events if e["event"] == "agent.iteration")
        trace_id = iteration["data"]["trace_id"]
        assert trace_id
        assert iteration["data"]["round_index"] == 1

        starts = [e for e in events if e["event"] == "tool_call.start"]
        assert starts[0]["data"]["parallel_group_id"] == "round_1"
        assert starts[0]["data"]["duration_ms"] == 1280.0
        assert starts[0]["data"]["tool_call_id"] == "call_ok"
        assert starts[1]["data"]["tool_call_id"] == "call_err"

        result_evt = next(e for e in events if e["event"] == "tool_call.result")
        error_evt = next(e for e in events if e["event"] == "tool_call.error")
        assert result_evt["data"]["trace_id"] == trace_id
        assert result_evt["data"]["duration_ms"] == 1280.0
        assert result_evt["data"]["result"]["success"] is True
        assert error_evt["data"]["trace_id"] == trace_id
        assert error_evt["data"]["duration_ms"] == 245.0
        assert error_evt["data"]["error"]["error"] == "boom"

        complete = next(e for e in events if e["event"] == "message.complete")
        assert complete["data"]["metadata"]["trace_id"] == trace_id
        assert complete["data"]["metadata"]["usage"]["total_tokens"] == 22

    def test_persists_tool_call_carrier_and_tool_results(self, app_and_client, monkeypatch):
        _, client, store = app_and_client
        create_resp = client.post("/api/chat/conversations", json={"title": "Persist Tool Steps"})
        conv_id = create_resp.json()["conversation_id"]

        from houyi_studio.server.chat import chat_api

        service = chat_api._chat_service
        assert service is not None

        class _AdapterWithStream:
            last_usage = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}

            async def chat(self, *args, **kwargs):
                _ = (args, kwargs)
                return SimpleNamespace(content="", tool_calls=[], usage={}, metadata={})

            async def stream_chat(self, *args, **kwargs):
                _ = (args, kwargs)
                yield StreamChunk(content_delta="final")

        class _FakeToolRunner:
            async def run(self, **kwargs):
                messages = kwargs["messages"]
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "demo",
                                        "arguments": '{"path":"README.md"}',
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "content": '{"matches":["README.md"]}',
                            "tool_call_id": "call_1",
                            "name": "demo",
                        },
                    ]
                )
                return (
                    SimpleNamespace(
                        content="Final answer",
                        tool_calls=[],
                        usage={"prompt_tokens": 7, "completion_tokens": 6, "total_tokens": 13},
                        metadata={},
                    ),
                    [
                        {
                            "tool_call_id": "call_1",
                            "tool_name": "demo",
                            "parallel_group_id": "round_1",
                            "round_index": 1,
                            "duration_ms": 321.0,
                            "args": {"path": "README.md"},
                            "result": {"raw": {"matches": ["README.md"]}},
                        }
                    ],
                )

        adapter = _AdapterWithStream()
        monkeypatch.setattr(service, "_default_adapter", adapter)
        monkeypatch.setattr(service, "_get_tool_runner", lambda: _FakeToolRunner())
        monkeypatch.setattr(
            "houyi_studio.server.chat.chat_service.ToolBridge.collect_tool_schemas",
            lambda self, skill_filter, include_core: [
                {
                    "type": "function",
                    "function": {
                        "name": "demo",
                        "description": "Demo",
                        "parameters": {},
                    },
                }
            ],
        )
        monkeypatch.setattr(
            "houyi_studio.server.chat.chat_service.ToolBridge.collect_skills",
            lambda self, skill_filter, include_core: [SimpleNamespace(name="demo")],
        )

        resp = client.post(
            f"/api/chat/conversations/{conv_id}/messages",
            json={"content": "find file", "enable_skills": ["demo"]},
        )
        assert resp.status_code == 200

        conv = store.get(conv_id)
        assert conv is not None
        assert len(conv.messages) == 4

        user_msg, assistant_carrier, tool_msg, final_assistant = conv.messages
        assert user_msg.role.value == "user"
        assert assistant_carrier.role.value == "assistant"
        assert assistant_carrier.tool_calls is not None
        assert assistant_carrier.tool_calls[0]["id"] == "call_1"
        assert assistant_carrier.tool_calls[0]["function"]["name"] == "demo"

        assert tool_msg.role.value == "tool"
        assert tool_msg.tool_call_id == "call_1"
        assert tool_msg.name == "demo"
        assert "README.md" in tool_msg.content
        assert tool_msg.metadata.get("duration_ms") == 321.0
        assert tool_msg.metadata.get("parallel_group_id") == "round_1"
        assert tool_msg.metadata.get("round_index") == 1

        assert final_assistant.role.value == "assistant"
        assert final_assistant.content == "Final answer"
        assert final_assistant.metadata.get("usage", {}).get("total_tokens") == 13

    def test_send_message_accepts_tooling_fields(self, app_and_client):
        _, client, _ = app_and_client
        create_resp = client.post("/api/chat/conversations", json={})
        conv_id = create_resp.json()["conversation_id"]

        resp = client.post(
            f"/api/chat/conversations/{conv_id}/messages",
            json={
                "content": "Tooling fields",
                "enable_tool_calls": False,
                "tool_call_strategy": "aggressive",
                "enable_skills": ["web_search"],
                "max_tool_iterations": 3,
            },
        )
        assert resp.status_code == 200

    def test_send_message_trace_id_is_queryable(self, app_and_client):
        _, client, _ = app_and_client
        create_resp = client.post("/api/chat/conversations", json={"title": "Trace Query"})
        conv_id = create_resp.json()["conversation_id"]

        resp = client.post(
            f"/api/chat/conversations/{conv_id}/messages",
            json={"content": "trace please"},
        )
        assert resp.status_code == 200

        events = _parse_sse_response(resp.text)
        complete = next(e for e in events if e["event"] == "message.complete")
        trace_id = complete["data"]["metadata"].get("trace_id")
        assert isinstance(trace_id, str) and trace_id

        trace_resp = client.get(f"/api/chat/trace/{trace_id}")
        assert trace_resp.status_code == 200
        trace_data = trace_resp.json()
        assert trace_data["trace_id"] == trace_id
        assert trace_data["root_span"] is not None

        root_span = trace_data["root_span"]
        assert root_span.get("name") == "chat.request"
        assert root_span.get("span_type") == "node"

        stage_names = {
            child.get("name")
            for child in (root_span.get("children") or [])
            if isinstance(child, dict)
        }
        assert "chat.prepare" in stage_names
        assert "chat.tool_loop" in stage_names
        assert "chat.persist" in stage_names
        assert "chat.stream.llm" in stage_names or "chat.stream.replay" in stage_names

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
        assert conv.messages[0].metadata["usage"]["input_tokens"] > 0
        assert conv.messages[1].role.value == "assistant"
        assert len(conv.messages[1].content) > 0
        assert conv.messages[1].metadata["usage"]["total_tokens"] == 15

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


class TestTraceTreeBuilder:
    """Trace tree assembly scenarios for root/child span structures."""

    def test_empty_spans_returns_none(self):
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

    def test_orphan_span_becomes_root(self):
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

    def test_children_sorted_by_start_time(self):
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
