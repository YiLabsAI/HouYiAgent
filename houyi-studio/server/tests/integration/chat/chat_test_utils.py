from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from houyi_studio.server.chat import chat_api as chat_api_module
from houyi_studio.server.chat.chat_api import register_chat_routes, router
from houyi_studio.server.chat.chat_service import ChatService
from houyi_studio.server.chat.json_store import JsonStore

from houyi.adapters.llm.base import StreamChunk
from houyi.infrastructure.observability.storage import reset_storage


def make_mock_llm(
    content_chunks: list[str] | None = None,
    reasoning_chunks: list[str] | None = None,
    usage: dict[str, int] | None = None,
    finish_reason: str | None = "stop",
):
    mock = AsyncMock()
    content_chunks = content_chunks or ["Hello ", "from ", "mock!"]
    reasoning_chunks = reasoning_chunks or []

    async def mock_stream_chat(messages, model=None, **kwargs):
        _ = (messages, model, kwargs)
        for index, chunk in enumerate(content_chunks):
            reasoning = reasoning_chunks[index] if index < len(reasoning_chunks) else None
            yield StreamChunk(content_delta=chunk, reasoning_delta=reasoning)

    mock.stream_chat = mock_stream_chat
    mock.last_usage = usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    mock.last_finish_reason = finish_reason
    return mock


def create_app_client_store(tmp_path, *, default_model: str = "test-model", llm=None):
    store = JsonStore(data_dir=tmp_path / "conversations")
    service = ChatService(json_store=store, default_model=default_model)
    service._default_adapter = llm or make_mock_llm()

    app = FastAPI()
    register_chat_routes(service)
    app.include_router(router)

    client = TestClient(app)
    return app, client, store


def cleanup_app_client(client: TestClient) -> None:
    client.close()
    reset_storage()


def parse_sse_response(text: str) -> list[dict]:
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


def get_sse_events(resp) -> list[dict]:
    return parse_sse_response(resp.text)


def get_event(events: list[dict], event_name: str) -> dict:
    return next(event for event in events if event["event"] == event_name)


def get_events(events: list[dict], event_name: str) -> list[dict]:
    return [event for event in events if event["event"] == event_name]


def get_delta_text(events: list[dict]) -> str:
    return "".join(
        event["data"].get("content", "") for event in get_events(events, "message.delta")
    )


def get_reasoning_text(events: list[dict]) -> str:
    return "".join(
        event["data"].get("reasoning_content", "") for event in get_events(events, "message.delta")
    )


def assert_event_names_present(events: list[dict], expected: Iterable[str]) -> list[str]:
    event_names = [event["event"] for event in events]
    for event_name in expected:
        assert event_name in event_names
    return event_names


def assert_event_order(events: list[dict], earlier_event: str, later_event: str) -> None:
    event_names = [event["event"] for event in events]
    assert event_names.index(earlier_event) < event_names.index(later_event)


def assert_status_code(resp, expected: int = 200):
    assert resp.status_code == expected
    return resp


def assert_event_count(events: list[dict], event_name: str, count: int) -> list[dict]:
    matched = get_events(events, event_name)
    assert len(matched) == count
    return matched


def assert_delta_text(events: list[dict], expected_text: str) -> list[dict]:
    deltas = get_events(events, "message.delta")
    assert get_delta_text(events) == expected_text
    return deltas


def assert_delta_text_contains(events: list[dict], expected_text: str) -> list[dict]:
    deltas = get_events(events, "message.delta")
    assert expected_text in get_delta_text(events)
    return deltas


def assert_reasoning_text(events: list[dict], expected_text: str) -> list[dict]:
    deltas = get_events(events, "message.delta")
    assert get_reasoning_text(events) == expected_text
    return deltas


def get_finish_data(events: list[dict]) -> dict[str, Any]:
    return get_event(events, "message.finish")["data"]


def get_complete_metadata(events: list[dict]) -> dict[str, Any]:
    return get_event(events, "message.complete")["data"]["metadata"]


def get_context_usage_data(events: list[dict]) -> dict[str, Any]:
    return get_event(events, "context.usage")["data"]


def get_compaction_data(events: list[dict]) -> dict[str, Any]:
    return get_event(events, "context.compacted")["data"]["compaction"]


def assert_finish_reason(
    events: list[dict], expected_reason: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    finish = get_finish_data(events)
    metadata = get_complete_metadata(events)
    assert finish["finish_reason"] == expected_reason
    assert metadata["finish_reason"] == expected_reason
    return finish, metadata


def assert_compaction_event(
    events: list[dict],
    *,
    trigger: str,
    reason: str | None = None,
    pressure_level: str | None = None,
    dropped_messages: int | None = None,
    messages_compacted: int | None = None,
    source_message_ids: list[str] | None = None,
) -> dict[str, Any]:
    compaction = get_compaction_data(events)
    assert compaction["trigger"] == trigger
    if reason is not None:
        assert compaction["metadata"]["reason"] == reason
    if pressure_level is not None:
        assert compaction["pressure_level"] == pressure_level
    if dropped_messages is not None:
        assert compaction["metadata"]["dropped_messages"] == dropped_messages
    if messages_compacted is not None:
        assert compaction["metrics"]["messages_compacted"] == messages_compacted
    if source_message_ids is not None:
        assert compaction["source_message_ids"] == source_message_ids
    return compaction


def assert_usage_fields(actual_usage: dict[str, Any], **expected_fields: Any) -> dict[str, Any]:
    for field_name, expected_value in expected_fields.items():
        assert actual_usage[field_name] == expected_value
    return actual_usage


def build_send_message_payload(content: str = "hello", **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"content": content}
    payload.update(overrides)
    return payload


def post_message(client: TestClient, conversation_id: str, **payload_overrides: Any):
    payload = build_send_message_payload(**payload_overrides)
    return client.post(f"/api/chat/conversations/{conversation_id}/messages", json=payload)


def get_conversation_or_fail(store, conversation_id: str):
    conversation = store.get(conversation_id)
    assert conversation is not None
    return conversation


def assert_message_roles(conversation, expected_roles: list[str]) -> None:
    assert [message.role.value for message in conversation.messages] == expected_roles


def assert_last_message(
    conversation,
    *,
    role: str | None = None,
    content: str | None = None,
    reasoning_content: str | None = None,
    finish_reason: str | None = None,
) -> Any:
    message = conversation.messages[-1]
    if role is not None:
        assert message.role.value == role
    if content is not None:
        assert message.content == content
    if reasoning_content is not None:
        assert message.reasoning_content == reasoning_content
    if finish_reason is not None:
        assert message.metadata["finish_reason"] == finish_reason
    return message


def assert_trace_stages(
    trace_data: dict[str, Any], expected_stage_names: Iterable[str]
) -> dict[str, Any]:
    assert trace_data["root_span"] is not None
    root_span = trace_data["root_span"]
    stage_names = {
        child.get("name") for child in (root_span.get("children") or []) if isinstance(child, dict)
    }
    for expected_stage_name in expected_stage_names:
        assert expected_stage_name in stage_names
    return root_span


def create_conversation_id(client: TestClient, **payload: object) -> str:
    resp = client.post("/api/chat/conversations", json=payload)
    assert resp.status_code == 201
    return resp.json()["conversation_id"]


def get_registered_chat_service() -> ChatService:
    service = chat_api_module._chat_service
    assert service is not None
    return service
