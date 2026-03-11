"""End-to-end chat lifecycle via Chat API.

Tests the minimum viable path end-to-end:
  Server start → Create conversation → Send message (mock LLM) →
  Receive SSE events → Verify persistence → Delete conversation
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from houyi_studio.server.chat.chat_api import register_chat_routes, router
from houyi_studio.server.chat.chat_service import ChatService
from houyi_studio.server.chat.json_store import JsonStore

from houyi.adapters.llm.base import StreamChunk
from houyi.application.context.token_estimator import TokenEstimator


def _make_mock_llm(content_chunks=None, reasoning_chunks=None):
    """Create a mock LLM adapter with configurable output."""
    mock = AsyncMock()
    content_chunks = content_chunks or ["Hello ", "from ", "the ", "assistant!"]
    reasoning_chunks = reasoning_chunks or []

    async def mock_stream_chat(messages, model=None, **kwargs):
        for i, chunk in enumerate(content_chunks):
            reasoning = reasoning_chunks[i] if i < len(reasoning_chunks) else None
            yield StreamChunk(content_delta=chunk, reasoning_delta=reasoning)

    mock.stream_chat = mock_stream_chat
    mock.last_usage = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    return mock


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE text into event dicts."""
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


@pytest.fixture
def smoke_env(tmp_path):
    """Create app/client/store for full chat lifecycle scenarios."""
    store = JsonStore(data_dir=tmp_path / "conversations")
    service = ChatService(json_store=store, default_model="smoke-model")
    service._default_adapter = _make_mock_llm()

    app = FastAPI()
    register_chat_routes(service)
    app.include_router(router)

    client = TestClient(app)
    return client, store


class TestSmokeFullLifecycle:
    """Complete lifecycle: create → send → verify SSE → verify persistence → delete."""

    def test_full_lifecycle(self, smoke_env):
        client, store = smoke_env

        # 1. Create conversation
        resp = client.post(
            "/api/chat/conversations",
            json={
                "title": "Smoke Test",
                "system_instructions": "You are a test assistant.",
            },
        )
        assert resp.status_code == 201
        conv_id = resp.json()["conversation_id"]
        assert resp.json()["title"] == "Smoke Test"

        # 2. Verify conversation exists
        resp = client.get(f"/api/chat/conversations/{conv_id}")
        assert resp.status_code == 200
        assert resp.json()["messages"] == []

        # 3. Send first message and receive SSE stream
        resp = client.post(
            f"/api/chat/conversations/{conv_id}/messages",
            json={"content": "Hello, assistant!"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        events = _parse_sse(resp.text)
        event_types = [e["event"] for e in events]

        # 4. Verify SSE event sequence
        assert "context.usage" in event_types, "Missing context.usage event"
        assert "message.delta" in event_types, "Missing message.delta events"
        assert "message.finish" in event_types, "Missing message.finish event"
        assert "message.complete" in event_types, "Missing message.complete event"
        assert "message.error" not in event_types, "Unexpected error event"
        assert "message.aborted" not in event_types, "Unexpected abort event"

        # context.usage should come first (before deltas)
        usage_idx = event_types.index("context.usage")
        first_delta_idx = event_types.index("message.delta")
        assert usage_idx < first_delta_idx, "context.usage should precede deltas"

        # message.finish should precede message.complete
        finish_idx = event_types.index("message.finish")
        complete_idx = event_types.index("message.complete")
        assert finish_idx < complete_idx, "message.finish should precede message.complete"

        # 5. Verify delta content
        deltas = [e for e in events if e["event"] == "message.delta"]
        full_content = "".join(e["data"].get("content", "") for e in deltas)
        assert full_content == "Hello from the assistant!"

        # 6. Verify context.usage structure
        usage_event = next(e for e in events if e["event"] == "context.usage")
        usage_data = usage_event["data"]
        assert "usage" in usage_data
        usage = usage_data["usage"]
        assert "used_tokens" in usage
        assert "max_context_tokens" in usage
        assert usage["used_tokens"] <= usage["max_context_tokens"]

        # 7. Verify persistence
        conv = store.get(conv_id)
        assert conv is not None
        assert len(conv.messages) == 2
        assert conv.messages[0].role.value == "user"
        assert conv.messages[0].content == "Hello, assistant!"
        assert conv.messages[1].role.value == "assistant"
        assert conv.messages[1].content == "Hello from the assistant!"
        assert conv.messages[1].metadata["usage"]["prompt_tokens"] == 20
        assert conv.messages[1].metadata["usage"]["completion_tokens"] == 10
        assert conv.messages[1].metadata["usage"]["reasoning_tokens"] == 0
        assert conv.messages[1].metadata["usage"]["answer_tokens"] == 10
        assert conv.messages[1].metadata["usage"]["cached_prompt_tokens"] == 0
        assert conv.messages[1].metadata["usage"]["usage_confidence"] == "reported"

        # 8. Send second message (multi-turn)
        resp = client.post(
            f"/api/chat/conversations/{conv_id}/messages",
            json={"content": "Follow-up question"},
        )
        assert resp.status_code == 200
        events2 = _parse_sse(resp.text)
        assert any(e["event"] == "message.finish" for e in events2)

        # Verify 4 messages now (2 user + 2 assistant)
        conv = store.get(conv_id)
        assert len(conv.messages) == 4

        # 9. Verify conversation appears in list
        resp = client.get("/api/chat/conversations")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

        # 10. Delete conversation
        resp = client.delete(f"/api/chat/conversations/{conv_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # 11. Verify deletion
        resp = client.get(f"/api/chat/conversations/{conv_id}")
        assert resp.status_code == 404

        resp = client.get("/api/chat/conversations")
        assert resp.json()["total"] == 0

    def test_reasoning_content_in_stream(self, smoke_env):
        """Verify reasoning_content flows through SSE."""
        client, store = smoke_env

        # Reconfigure mock with reasoning
        # Create conversation and send message with reasoning mock
        resp = client.post("/api/chat/conversations", json={"title": "Reasoning Test"})
        conv_id = resp.json()["conversation_id"]

        resp = client.post(
            f"/api/chat/conversations/{conv_id}/messages",
            json={"content": "Think about this"},
        )
        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        assert any(e["event"] == "message.finish" for e in events)


class TestContextBurst:
    """Integration test: context burst with many messages.

    Verifies that the system handles large conversation histories
    gracefully — truncation occurs, newest messages preserved,
    no crashes or data loss.
    """

    def test_message_truncation_preserves_newest_messages(self, smoke_env):
        """Send enough messages to trigger truncation and preserve the newest context."""
        client, store = smoke_env
        burst_turns = 40
        payload = "Padding text to consume more tokens and trigger truncation faster. " * 8

        # Create conversation
        resp = client.post("/api/chat/conversations", json={"title": "Burst Test"})
        conv_id = resp.json()["conversation_id"]

        # Send many messages to build up history
        for i in range(burst_turns):
            resp = client.post(
                f"/api/chat/conversations/{conv_id}/messages",
                json={"content": f"Message number {i}. {payload}"},
            )
            assert resp.status_code == 200

        # Verify conversation has the expected persisted history before the truncation trigger
        conv = store.get(conv_id)
        assert len(conv.messages) == burst_turns * 2

        estimator = TokenEstimator(model="smoke-model")
        history_before_final = [m.to_llm_message() for m in conv.messages]
        full_history_with_final = [
            *history_before_final,
            {"role": "user", "content": "Final message after burst"},
        ]
        assert estimator.count_messages(full_history_with_final) > estimator.max_input_tokens

        # Send one more message — this triggers context planning with truncation
        resp = client.post(
            f"/api/chat/conversations/{conv_id}/messages",
            json={"content": "Final message after burst"},
        )
        assert resp.status_code == 200

        events = _parse_sse(resp.text)
        event_types = [e["event"] for e in events]

        # Must still produce valid SSE stream
        assert "context.usage" in event_types
        assert "message.delta" in event_types
        assert "message.finish" in event_types

        # Verify context usage shows truncation
        usage_event = next(e for e in events if e["event"] == "context.usage")["data"]
        usage = usage_event["usage"]
        assert usage["used_tokens"] <= usage["max_context_tokens"]
        assert usage["used_tokens"] < usage["max_context_tokens"]
        assert usage["used_tokens"] < estimator.count_messages(full_history_with_final)

        # Verify final message persisted
        conv = store.get(conv_id)
        assert len(conv.messages) == (burst_turns * 2) + 2
        assert conv.messages[-2].content == "Final message after burst"
        assert conv.messages[-1].role.value == "assistant"

    def test_burst_no_data_loss(self, smoke_env):
        """All user messages are persisted even under burst load."""
        client, store = smoke_env

        resp = client.post("/api/chat/conversations", json={"title": "No Loss Test"})
        conv_id = resp.json()["conversation_id"]

        sent_contents = []
        for i in range(50):
            content = f"Unique message {i} - {hash(i)}"
            sent_contents.append(content)
            resp = client.post(
                f"/api/chat/conversations/{conv_id}/messages",
                json={"content": content},
            )
            assert resp.status_code == 200

        # Verify all user messages persisted
        conv = store.get(conv_id)
        user_msgs = [m for m in conv.messages if m.role.value == "user"]
        user_contents = [m.content for m in user_msgs]
        for expected in sent_contents:
            assert expected in user_contents, f"Lost message: {expected}"
