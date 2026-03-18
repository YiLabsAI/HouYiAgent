"""End-to-end chat lifecycle via Chat API.

Tests the minimum viable path end-to-end:
  Server start → Create conversation → Send message (mock LLM) →
  Receive SSE events → Verify persistence → Delete conversation
"""

from __future__ import annotations

import pytest

from houyi.application.context.token_estimator import TokenEstimator

from .chat_test_utils import (
    assert_delta_text,
    assert_event_names_present,
    assert_event_order,
    assert_message_roles,
    assert_status_code,
    cleanup_app_client,
    create_app_client_store,
    create_conversation_id,
    get_context_usage_data,
    get_conversation_or_fail,
    get_sse_events,
    make_mock_llm,
    post_message,
)


@pytest.fixture
def smoke_env(tmp_path):
    _, client, store = create_app_client_store(
        tmp_path,
        default_model="smoke-model",
        llm=make_mock_llm(
            content_chunks=["Hello ", "from ", "the ", "assistant!"],
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
                "finish_reason": "stop",
            },
        ),
    )
    try:
        yield client, store
    finally:
        cleanup_app_client(client)


class TestSmokeFullLifecycle:
    """Complete lifecycle: create → send → verify SSE → verify persistence → delete."""

    def test_full_lifecycle(self, smoke_env):
        client, store = smoke_env

        conv_id = create_conversation_id(
            client,
            title="Smoke Test",
            system_instructions="You are a test assistant.",
        )

        resp = client.get(f"/api/chat/conversations/{conv_id}")
        assert_status_code(resp)
        assert resp.json()["messages"] == []
        assert resp.json()["title"] == "Smoke Test"

        resp = assert_status_code(post_message(client, conv_id, content="Hello, assistant!"))
        assert "text/event-stream" in resp.headers["content-type"]

        events = get_sse_events(resp)
        event_types = assert_event_names_present(
            events,
            ["context.usage", "message.delta", "message.finish", "message.complete"],
        )
        assert "message.error" not in event_types
        assert "message.aborted" not in event_types
        assert_event_order(events, "context.usage", "message.delta")
        assert_event_order(events, "message.finish", "message.complete")
        assert_delta_text(events, "Hello from the assistant!")

        usage = get_context_usage_data(events)["usage"]
        assert usage["used_tokens"] <= usage["max_context_tokens"]

        conv = get_conversation_or_fail(store, conv_id)
        assert_message_roles(conv, ["user", "assistant"])
        assert conv.messages[0].content == "Hello, assistant!"
        assert conv.messages[1].content == "Hello from the assistant!"
        assert conv.messages[1].metadata["usage"]["prompt_tokens"] == 20
        assert conv.messages[1].metadata["usage"]["completion_tokens"] == 10
        assert conv.messages[1].metadata["usage"]["reasoning_tokens"] == 0
        assert conv.messages[1].metadata["usage"]["answer_tokens"] == 10
        assert conv.messages[1].metadata["usage"]["cached_prompt_tokens"] == 0
        assert conv.messages[1].metadata["usage"]["usage_confidence"] == "reported"

        resp = assert_status_code(post_message(client, conv_id, content="Follow-up question"))
        events2 = get_sse_events(resp)
        assert_event_names_present(events2, ["message.finish"])

        conv = get_conversation_or_fail(store, conv_id)
        assert len(conv.messages) == 4

        resp = client.get("/api/chat/conversations")
        assert_status_code(resp)
        assert resp.json()["total"] == 1

        resp = client.delete(f"/api/chat/conversations/{conv_id}")
        assert_status_code(resp)
        assert resp.json()["status"] == "deleted"

        resp = client.get(f"/api/chat/conversations/{conv_id}")
        assert resp.status_code == 404

        resp = client.get("/api/chat/conversations")
        assert resp.json()["total"] == 0

    def test_reasoning_content_in_stream(self, smoke_env):
        client, store = smoke_env

        conv_id = create_conversation_id(client, title="Reasoning Test")

        resp = assert_status_code(post_message(client, conv_id, content="Think about this"))
        events = get_sse_events(resp)
        assert_event_names_present(events, ["message.finish"])

    def test_gemini_usage(self, tmp_path):
        llm = make_mock_llm(
            content_chunks=["Gemini ", "provider ", "smoke"],
            usage={
                "prompt_token_count": 14,
                "candidates_token_count": 9,
                "thoughts_token_count": 4,
                "cached_content_token_count": 5,
            },
        )
        _, client, store = create_app_client_store(
            tmp_path,
            default_model="gemini-smoke-model",
            llm=llm,
        )
        try:
            conv_id = create_conversation_id(
                client,
                title="Provider Aware Smoke",
                system_instructions="Please always use chinese to reply",
            )

            resp = assert_status_code(post_message(client, conv_id, content="hello"))
            events = get_sse_events(resp)
            assert_event_names_present(
                events,
                ["context.usage", "message.delta", "message.finish", "message.complete"],
            )
            assert_delta_text(events, "Gemini provider smoke")

            conv = get_conversation_or_fail(store, conv_id)
            assistant = conv.messages[-1]
            usage = assistant.metadata["usage"]
            assert usage["prompt_tokens"] == 14
            assert usage["completion_tokens"] == 9
            assert usage["reasoning_tokens"] == 4
            assert usage["answer_tokens"] == 5
            assert usage["cached_prompt_tokens"] == 5
            assert usage["cache_hit"] is True
            assert usage["usage_confidence"] == "reported"
        finally:
            cleanup_app_client(client)

    def test_openai_compatible_usage(self, tmp_path):
        llm = make_mock_llm(
            content_chunks=["OpenAI ", "compatible ", "smoke"],
            usage={
                "prompt_tokens": 18,
                "completion_tokens": 11,
                "prompt_tokens_details": {"cached_tokens": 6},
                "completion_tokens_details": {"reasoning_tokens": 4},
                "total_tokens": 29,
            },
        )
        _, client, store = create_app_client_store(
            tmp_path,
            default_model="openai-compatible-smoke-model",
            llm=llm,
        )
        try:
            conv_id = create_conversation_id(
                client,
                title="OpenAI Compatible Usage Smoke",
                system_instructions="Please always use chinese to reply",
            )

            resp = assert_status_code(post_message(client, conv_id, content="hello"))
            events = get_sse_events(resp)
            assert_event_names_present(
                events,
                ["context.usage", "message.delta", "message.finish", "message.complete"],
            )
            assert_delta_text(events, "OpenAI compatible smoke")

            conv = get_conversation_or_fail(store, conv_id)
            assistant = conv.messages[-1]
            usage = assistant.metadata["usage"]
            assert usage["prompt_tokens"] == 18
            assert usage["completion_tokens"] == 11
            assert usage["reasoning_tokens"] == 4
            assert usage["answer_tokens"] == 7
            assert usage["cached_prompt_tokens"] == 6
            assert usage["cache_hit"] is True
            assert usage["usage_confidence"] == "reported"
        finally:
            cleanup_app_client(client)

    def test_anthropic_usage(self, tmp_path):
        llm = make_mock_llm(
            content_chunks=["Anthropic ", "provider ", "smoke"],
            usage={
                "input_tokens": 13,
                "output_tokens": 8,
            },
        )
        _, client, store = create_app_client_store(
            tmp_path,
            default_model="anthropic-smoke-model",
            llm=llm,
        )
        try:
            conv_id = create_conversation_id(
                client,
                title="Anthropic Usage Smoke",
                system_instructions="Please always use chinese to reply",
            )

            resp = assert_status_code(post_message(client, conv_id, content="hello"))
            events = get_sse_events(resp)
            assert_event_names_present(
                events,
                ["context.usage", "message.delta", "message.finish", "message.complete"],
            )
            assert_delta_text(events, "Anthropic provider smoke")

            conv = get_conversation_or_fail(store, conv_id)
            assistant = conv.messages[-1]
            usage = assistant.metadata["usage"]
            assert usage["prompt_tokens"] == 13
            assert usage["completion_tokens"] == 8
            assert usage["reasoning_tokens"] == 0
            assert usage["answer_tokens"] == 8
            assert usage["cached_prompt_tokens"] == 0
            assert usage["cache_hit"] is False
            assert usage["usage_confidence"] == "reported"
        finally:
            cleanup_app_client(client)


class TestContextBurst:
    """Integration test: context burst with many messages.

    Verifies that the system handles large conversation histories
    gracefully — truncation occurs, newest messages preserved,
    no crashes or data loss.
    """

    def test_message_truncation(self, smoke_env):
        """Send enough messages to trigger truncation and preserve the newest context."""
        client, store = smoke_env
        burst_turns = 40
        payload = "Padding text to consume more tokens and trigger truncation faster. " * 8

        conv_id = create_conversation_id(client, title="Burst Test")

        for i in range(burst_turns):
            assert_status_code(
                post_message(client, conv_id, content=f"Message number {i}. {payload}")
            )

        conv = get_conversation_or_fail(store, conv_id)
        assert len(conv.messages) == burst_turns * 2

        estimator = TokenEstimator(model="smoke-model")
        history_before_final = [m.to_llm_message() for m in conv.messages]
        full_history_with_final = [
            *history_before_final,
            {"role": "user", "content": "Final message after burst"},
        ]
        assert estimator.count_messages(full_history_with_final) > estimator.max_input_tokens

        resp = assert_status_code(
            post_message(client, conv_id, content="Final message after burst")
        )

        events = get_sse_events(resp)
        assert_event_names_present(events, ["context.usage", "message.delta", "message.finish"])

        usage = get_context_usage_data(events)["usage"]
        assert usage["used_tokens"] <= usage["max_context_tokens"]
        assert usage["used_tokens"] < usage["max_context_tokens"]
        assert usage["used_tokens"] < estimator.count_messages(full_history_with_final)

        conv = get_conversation_or_fail(store, conv_id)
        assert len(conv.messages) == (burst_turns * 2) + 2
        assert conv.messages[-2].content == "Final message after burst"
        assert conv.messages[-1].role.value == "assistant"

    def test_burst_no_data_loss(self, smoke_env):
        """All user messages are persisted even under burst load."""
        client, store = smoke_env

        conv_id = create_conversation_id(client, title="No Loss Test")

        sent_contents = []
        for i in range(50):
            content = f"Unique message {i} - {hash(i)}"
            sent_contents.append(content)
            assert_status_code(post_message(client, conv_id, content=content))

        conv = get_conversation_or_fail(store, conv_id)
        user_msgs = [m for m in conv.messages if m.role.value == "user"]
        user_contents = [m.content for m in user_msgs]
        for expected in sent_contents:
            assert expected in user_contents, f"Lost message: {expected}"
