"""Tool-loop, replay, and send-message edge integration tests for Chat API."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from houyi_studio.server.chat import chat_service as chat_service_module
from pydantic import BaseModel

from houyi.adapters.llm.base import StreamChunk
from houyi.domain.skill.registry import DEFAULT_SKILL_REGISTRY
from houyi.domain.skill.spec import SkillSpec
from houyi.infrastructure.config.env_config import (
    ENV_TOOLCALL_LOOP_MAX_MESSAGE_CHARS,
    ENV_TOOLCALL_LOOP_MAX_TOTAL_CHARS,
    ENV_TOOLCALL_RESULT_SUMMARY_ENABLED,
    ENV_TOOLCALL_RESULT_SUMMARY_MAX_CHARS,
    ENV_TOOLCALL_RESULT_SUMMARY_MAX_ITEMS,
)
from houyi.skills.web_search.skill import build_web_search_skill

from .chat_test_utils import (
    assert_delta_text,
    assert_delta_text_contains,
    assert_event_count,
    assert_event_names_present,
    assert_finish_reason,
    assert_last_message,
    assert_reasoning_text,
    assert_status_code,
    assert_usage_fields,
    cleanup_app_client,
    create_app_client_store,
    create_conversation_id,
    get_complete_metadata,
    get_context_usage_data,
    get_conversation_or_fail,
    get_event,
    get_events,
    get_registered_chat_service,
    get_sse_events,
    post_message,
)


@pytest.fixture
def app_and_client(tmp_path):
    app, client, store = create_app_client_store(tmp_path)
    try:
        yield app, client, store
    finally:
        cleanup_app_client(client)


class TestSendMessageTooling:
    def test_streams_tool_loop(self, app_and_client, monkeypatch):
        _, client, store = app_and_client
        conv_id = create_conversation_id(client, title="Tool Loop Replay")
        service = get_registered_chat_service()

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
                        metadata={"finish_reason": "stop"},
                    ),
                    [],
                )

        adapter = _AdapterWithStream()
        monkeypatch.setattr(service, "_default_adapter", adapter)
        monkeypatch.setattr(service, "_get_tool_runner", lambda: _FakeToolRunner())
        monkeypatch.setattr(
            chat_service_module.ToolBridge,
            "collect_tool_schemas",
            lambda self, skill_filter, include_core: [
                {"type": "function", "function": {"name": "demo"}}
            ],
        )
        monkeypatch.setattr(
            chat_service_module.ToolBridge,
            "collect_skills",
            lambda self, skill_filter, include_core: [SimpleNamespace(name="demo")],
        )

        resp = assert_status_code(
            post_message(client, conv_id, content="hello", enable_skills=["demo"])
        )
        events = get_sse_events(resp)
        deltas = get_events(events, "message.delta")
        assert deltas[0]["data"].get("content", "") == ""
        assert_delta_text(events, "Answer from tool loop")
        assert adapter.stream_calls == 0

        _, metadata = assert_finish_reason(events, "stop")
        assert_usage_fields(
            metadata["usage"],
            total_tokens=18,
            reasoning_tokens=0,
            answer_tokens=7,
            cached_prompt_tokens=0,
            usage_confidence="reported",
        )
        assert metadata["tool_loop_convergence_reason"] == "no_tool_calls_with_replay_payload"
        assert metadata["tool_loop_final_stream_skipped"] is True
        assert metadata["final_stream_messages_reconstructed"] is False
        assert metadata["final_stream_persisted_tool_message_count"] == 0
        assert metadata["final_stream_prepared_message_count"] >= 1

        conv = get_conversation_or_fail(store, conv_id)
        assert_last_message(conv, finish_reason="stop")

    def test_enters_tool_loop(self, app_and_client, monkeypatch):
        _, client, _ = app_and_client
        conv_id = create_conversation_id(client, title="Web Search Alias")
        service = get_registered_chat_service()

        class _AdapterWithNoStream:
            last_usage = {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}

            async def chat(self, *args, **kwargs):
                _ = (args, kwargs)
                return SimpleNamespace(content="", tool_calls=[], usage={}, metadata={})

            async def stream_chat(self, *args, **kwargs):
                _ = (args, kwargs)
                if False:
                    yield None

        class _CapturedToolRunner:
            def __init__(self) -> None:
                self.skill_names: list[str] = []
                self.schema_names: list[str] = []

            async def run(self, **kwargs):
                self.skill_names = [str(getattr(skill, "name", "")) for skill in kwargs["skills"]]
                self.schema_names = [
                    str(tool.get("function", {}).get("name", "")) for tool in kwargs["tools"]
                ]
                return (
                    SimpleNamespace(
                        content="Web search answer",
                        tool_calls=[],
                        usage={"prompt_tokens": 6, "completion_tokens": 5, "total_tokens": 11},
                        metadata={"finish_reason": "stop"},
                    ),
                    [],
                )

        runner = _CapturedToolRunner()
        web_search_skill = build_web_search_skill().model_copy(update={"is_core": True})
        registered_name = DEFAULT_SKILL_REGISTRY.register(web_search_skill, overwrite=True)
        monkeypatch.setattr(service, "_default_adapter", _AdapterWithNoStream())
        monkeypatch.setattr(service, "_get_tool_runner", lambda *args, **kwargs: runner)
        try:
            resp = assert_status_code(
                post_message(
                    client,
                    conv_id,
                    content="look up latest RocketMQ news",
                    enable_web_search=True,
                    enable_skills=["houyi_web_search"],
                )
            )
        finally:
            DEFAULT_SKILL_REGISTRY.unregister(registered_name)

        assert_status_code(resp)
        assert runner.skill_names == ["web_search"]
        assert runner.schema_names == ["web_search"]

    def test_tool_loop_shows_error(self, app_and_client, monkeypatch):
        _, client, store = app_and_client
        conv_id = create_conversation_id(client, title="Tool Loop Error")
        service = get_registered_chat_service()

        class _AdapterWithStream:
            last_usage = None
            last_finish_reason = None

            async def chat(self, *args, **kwargs):
                _ = (args, kwargs)
                return SimpleNamespace(content="", tool_calls=[], usage={}, metadata={})

            async def stream_chat(self, *args, **kwargs):
                _ = (args, kwargs)
                yield StreamChunk(content_delta="should-not-run")

        class _FailingToolRunner:
            async def run(self, **kwargs):
                _ = kwargs
                raise RuntimeError("429 RESOURCE_EXHAUSTED")

        monkeypatch.setattr(service, "_default_adapter", _AdapterWithStream())
        monkeypatch.setattr(service, "_get_tool_runner", lambda: _FailingToolRunner())
        monkeypatch.setattr(
            chat_service_module.ToolBridge,
            "collect_tool_schemas",
            lambda self, skill_filter, include_core: [
                {"type": "function", "function": {"name": "demo"}}
            ],
        )
        monkeypatch.setattr(
            chat_service_module.ToolBridge,
            "collect_skills",
            lambda self, skill_filter, include_core: [SimpleNamespace(name="demo")],
        )

        resp = assert_status_code(
            post_message(client, conv_id, content="hello", enable_skills=["demo"])
        )

        events = get_sse_events(resp)
        assert_delta_text_contains(events, "temporarily rate limited")
        assert any(event["event"] == "message.error" for event in events)
        metadata = get_complete_metadata(events)
        assert metadata["finish_reason"] == "error"

        conv = get_conversation_or_fail(store, conv_id)
        assert "temporarily rate limited" in conv.messages[-1].content
        assert_last_message(conv, finish_reason="error")

    def test_emits_tool_events(self, app_and_client, monkeypatch):
        _, client, _ = app_and_client
        conv_id = create_conversation_id(client, title="Tool Lifecycle")
        service = get_registered_chat_service()

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
                            "requested_tool_name": "demo_ok",
                            "parallel_group_id": "round_1",
                            "round_index": 1,
                            "duration_ms": 1280.0,
                            "args": {"q": "alpha"},
                            "result": {"raw": {"success": True, "value": "ok"}},
                        },
                        {
                            "tool_call_id": "call_err",
                            "tool_name": "demo_err",
                            "requested_tool_name": "demo_err",
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
            chat_service_module.ToolBridge,
            "collect_tool_schemas",
            lambda self, skill_filter, include_core: [
                {"type": "function", "function": {"name": "demo_ok"}},
                {"type": "function", "function": {"name": "demo_err"}},
            ],
        )
        monkeypatch.setattr(
            chat_service_module.ToolBridge,
            "collect_skills",
            lambda self, skill_filter, include_core: [
                SimpleNamespace(name="demo_ok"),
                SimpleNamespace(name="demo_err"),
            ],
        )

        resp = assert_status_code(
            post_message(
                client, conv_id, content="run tools", enable_skills=["demo_ok", "demo_err"]
            )
        )

        events = get_sse_events(resp)
        assert_event_names_present(
            events,
            ["agent.iteration", "tool_call.start", "tool_call.result", "tool_call.error"],
        )
        assert_event_count(events, "tool_call.start", 2)
        assert_event_count(events, "tool_call.result", 1)
        assert_event_count(events, "tool_call.error", 1)

        iteration = get_event(events, "agent.iteration")
        trace_id = iteration["data"]["trace_id"]
        assert trace_id
        assert iteration["data"]["round_index"] == 1

        starts = get_events(events, "tool_call.start")
        assert starts[0]["data"]["parallel_group_id"] == "round_1"
        assert starts[0]["data"]["duration_ms"] == 1280.0
        assert starts[0]["data"]["tool_call_id"] == "call_ok"
        assert starts[0]["data"]["requested_tool_name"] == "demo_ok"
        assert starts[1]["data"]["tool_call_id"] == "call_err"
        assert starts[1]["data"]["requested_tool_name"] == "demo_err"

        result_evt = get_event(events, "tool_call.result")
        error_evt = get_event(events, "tool_call.error")
        assert result_evt["data"]["trace_id"] == trace_id
        assert result_evt["data"]["duration_ms"] == 1280.0
        assert result_evt["data"]["requested_tool_name"] == "demo_ok"
        assert result_evt["data"]["result"]["success"] is True
        assert error_evt["data"]["trace_id"] == trace_id
        assert error_evt["data"]["duration_ms"] == 245.0
        assert error_evt["data"]["requested_tool_name"] == "demo_err"
        assert error_evt["data"]["error"]["error"] == "boom"

        metadata = get_complete_metadata(events)
        assert metadata["trace_id"] == trace_id
        assert metadata["usage"]["total_tokens"] == 22

    def test_persists_tool_results(self, app_and_client, monkeypatch):
        _, client, store = app_and_client
        conv_id = create_conversation_id(client, title="Persist Tool Steps")
        service = get_registered_chat_service()

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
            chat_service_module.ToolBridge,
            "collect_tool_schemas",
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
            chat_service_module.ToolBridge,
            "collect_skills",
            lambda self, skill_filter, include_core: [SimpleNamespace(name="demo")],
        )

        resp = assert_status_code(
            post_message(client, conv_id, content="find file", enable_skills=["demo"])
        )

        conv = get_conversation_or_fail(store, conv_id)
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
        assert final_assistant.metadata.get("usage", {}).get("reasoning_tokens") == 0
        assert final_assistant.metadata.get("usage", {}).get("answer_tokens") == 6
        assert final_assistant.metadata.get("usage", {}).get("cached_prompt_tokens") == 0
        assert final_assistant.metadata.get("usage", {}).get("usage_confidence") == "reported"

    def test_persists_research_trace(self, app_and_client, monkeypatch):
        _, client, store = app_and_client
        conv_id = create_conversation_id(client, title="Research Tool Steps")
        service = get_registered_chat_service()

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
                                        "name": "web_search",
                                        "arguments": '{"query":"skill.md"}',
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "content": json.dumps(
                                {
                                    "data": {
                                        "matches": [
                                            {
                                                "title": f"title-{idx}",
                                                "snippet": f"snippet-{idx}",
                                                "url": f"https://example.com/{idx}",
                                            }
                                            for idx in range(80)
                                        ]
                                    }
                                }
                            ),
                            "tool_call_id": "call_1",
                            "name": "web_search",
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
                            "tool_name": "houyi_web_search",
                            "parallel_group_id": "round_1",
                            "round_index": 1,
                            "duration_ms": 321.0,
                            "args": {"query": "skill.md"},
                            "result": {"raw": {"matches": ["README.md"]}},
                        }
                    ],
                )

        adapter = _AdapterWithStream()
        monkeypatch.setattr(service, "_default_adapter", adapter)
        monkeypatch.setattr(service, "_get_tool_runner", lambda: _FakeToolRunner())
        monkeypatch.setattr(
            chat_service_module.ToolBridge,
            "collect_tool_schemas",
            lambda self, skill_filter, include_core: [
                {
                    "type": "function",
                    "function": {
                        "name": "houyi_web_search",
                        "description": "Search",
                        "parameters": {},
                    },
                }
            ],
        )
        monkeypatch.setattr(
            chat_service_module.ToolBridge,
            "collect_skills",
            lambda self, skill_filter, include_core: [SimpleNamespace(name="houyi_web_search")],
        )

        resp = assert_status_code(
            post_message(
                client,
                conv_id,
                content="research skill.md",
                enable_deep_research=True,
                enable_skills=["houyi_web_search", "deep_research"],
            )
        )

        conv = get_conversation_or_fail(store, conv_id)
        assert len(conv.messages) == 4

        tool_msg = conv.messages[2]
        payload = json.loads(tool_msg.content)
        assert tool_msg.role.value == "tool"
        assert payload["matches"] == ["README.md"]
        assert tool_msg.metadata["round_index"] == 1
        assert tool_msg.metadata["parallel_group_id"] == "round_1"
        assert tool_msg.metadata["duration_ms"] == 321.0
        assert tool_msg.metadata["tool_args"] == {"query": "skill.md"}

    def test_only_toolloop_final_stream(self, app_and_client, monkeypatch):
        _, client, store = app_and_client
        conv_id = create_conversation_id(client, title="Reasoning Replay")
        service = get_registered_chat_service()

        class _AdapterWithNoFinalStream:
            last_usage = {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
            last_finish_reason = "stop"

            async def chat(self, *args, **kwargs):
                _ = (args, kwargs)
                return SimpleNamespace(content="", tool_calls=[], usage={}, metadata={})

            async def stream_chat(self, *args, **kwargs):
                _ = (args, kwargs)
                yield StreamChunk(reasoning_delta="final reasoning only")
                yield StreamChunk(content_delta="final answer after tools")

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
                                    "id": "call_reasoning",
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
                            "content": '{"ok":true}',
                            "tool_call_id": "call_reasoning",
                            "name": "demo",
                        },
                    ]
                )
                return (
                    SimpleNamespace(
                        content="",
                        tool_calls=[],
                        usage={"prompt_tokens": 8, "completion_tokens": 5, "total_tokens": 13},
                        metadata={"reasoning_content": "final reasoning only"},
                    ),
                    [
                        {
                            "tool_call_id": "call_reasoning",
                            "tool_name": "demo",
                            "parallel_group_id": "round_1",
                            "round_index": 1,
                            "duration_ms": 123.0,
                            "args": {"path": "README.md"},
                            "result": {"raw": {"ok": True}},
                        }
                    ],
                )

        adapter = _AdapterWithNoFinalStream()
        monkeypatch.setattr(service, "_default_adapter", adapter)
        monkeypatch.setattr(service, "_get_tool_runner", lambda: _FakeToolRunner())
        monkeypatch.setattr(
            chat_service_module.ToolBridge,
            "collect_tool_schemas",
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
            chat_service_module.ToolBridge,
            "collect_skills",
            lambda self, skill_filter, include_core: [SimpleNamespace(name="demo")],
        )

        resp = assert_status_code(
            post_message(client, conv_id, content="reason please", enable_skills=["demo"])
        )
        events = get_sse_events(resp)
        assert_reasoning_text(events, "final reasoning only")
        assert_delta_text_contains(events, "final answer after tools")
        metadata = get_complete_metadata(events)
        assert metadata["tool_loop_final_stream_skipped"] is False

        conv = get_conversation_or_fail(store, conv_id)
        assert len(conv.messages) == 4

        final_assistant = conv.messages[-1]
        assert final_assistant.role.value == "assistant"
        assert final_assistant.content == "final answer after tools"
        assert final_assistant.reasoning_content == "final reasoning only"
        assert final_assistant.metadata.get("finish_reason") == "stop"

    def test_send_orphan_ignored(self, app_and_client, monkeypatch):
        _, client, store = app_and_client
        conv_id = create_conversation_id(client, title="Historical Orphan Tool")

        conversation = get_conversation_or_fail(store, conv_id)
        conversation.messages.extend(
            [
                chat_service_module.Message(
                    role=chat_service_module.MessageRole.USER, content="older"
                ),
                chat_service_module.Message(
                    role=chat_service_module.MessageRole.TOOL,
                    content='{"legacy":true}',
                    name="legacy_tool",
                ),
            ]
        )
        store.update(conversation)

        service = get_registered_chat_service()

        class _AdapterWithFinalText:
            last_usage = {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10}
            last_finish_reason = "stop"

            async def stream_chat(self, *args, **kwargs):
                _ = (args, kwargs)
                yield StreamChunk(content_delta="Recovered answer")

        monkeypatch.setattr(service, "_default_adapter", _AdapterWithFinalText())

        resp = assert_status_code(post_message(client, conv_id, content="continue"))

        events = get_sse_events(resp)
        assert_delta_text(events, "Recovered answer")

        updated = get_conversation_or_fail(store, conv_id)
        assert_last_message(updated, content="Recovered answer")

    def test_send_stream_error(self, app_and_client, monkeypatch):
        _, client, store = app_and_client
        conv_id = create_conversation_id(client, title="Stream Error Visible")
        service = get_registered_chat_service()

        class _AdapterWithError:
            last_usage = None
            last_finish_reason = None

            async def stream_chat(self, *args, **kwargs):
                _ = (args, kwargs)
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
                yield

        monkeypatch.setattr(service, "_default_adapter", _AdapterWithError())

        resp = assert_status_code(post_message(client, conv_id, content="hello"))

        events = get_sse_events(resp)
        assert_delta_text_contains(events, "temporarily rate limited")
        assert any(e["event"] == "message.error" for e in events)

        conv = get_conversation_or_fail(store, conv_id)
        assert "temporarily rate limited" in conv.messages[-1].content
        assert_last_message(conv, finish_reason="error")

    def test_send_empty_stream(self, app_and_client, monkeypatch):
        _, client, store = app_and_client
        conv_id = create_conversation_id(client, title="Empty Stream Visible")
        service = get_registered_chat_service()

        class _AdapterWithEmptyStream:
            last_usage = {"prompt_tokens": 2, "completion_tokens": 0, "total_tokens": 2}
            last_finish_reason = "stop"

            async def stream_chat(self, *args, **kwargs):
                _ = (args, kwargs)
                if False:
                    yield None

        monkeypatch.setattr(service, "_default_adapter", _AdapterWithEmptyStream())

        resp = assert_status_code(post_message(client, conv_id, content="hello"))

        events = get_sse_events(resp)
        assert_delta_text_contains(events, "empty final response")
        metadata = get_complete_metadata(events)
        assert metadata["final_stream_message_count"] >= 1
        assert metadata["final_stream_input_chars"] >= 1
        assert metadata["final_stream_chunk_count"] >= 1

        conv = get_conversation_or_fail(store, conv_id)
        assert "empty final response" in conv.messages[-1].content
        assert_last_message(conv, finish_reason="stop")

    def test_send_reasoning_persists(self, app_and_client, monkeypatch):
        _, client, store = app_and_client
        conv_id = create_conversation_id(client, title="Reasoning Only Stream")
        service = get_registered_chat_service()

        class _AdapterWithReasoningOnlyStream:
            last_usage = {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
            last_finish_reason = "stop"

            async def stream_chat(self, *args, **kwargs):
                _ = (args, kwargs)
                yield StreamChunk(reasoning_delta="thinking step 1")
                yield StreamChunk(reasoning_delta=" + step 2")

        monkeypatch.setattr(service, "_default_adapter", _AdapterWithReasoningOnlyStream())

        resp = assert_status_code(post_message(client, conv_id, content="hello"))

        events = get_sse_events(resp)
        assert_delta_text(events, "")
        assert_reasoning_text(events, "thinking step 1 + step 2")
        assert "empty final response" not in "".join(
            event["data"].get("content", "") for event in get_events(events, "message.delta")
        )

        conv = get_conversation_or_fail(store, conv_id)
        assert_last_message(
            conv,
            content="",
            reasoning_content="thinking step 1 + step 2",
            finish_reason="stop",
        )

    def test_send_stream_sanitizes_tool_loop(self, app_and_client, monkeypatch):
        _, client, store = app_and_client
        conv_id = create_conversation_id(client, title="Tool Loop Final Stream Sanitize")
        service = get_registered_chat_service()

        captured_messages: list[dict[str, object]] = []
        captured_stream_kwargs: list[dict[str, object]] = []

        class _AdapterWithCapturedStream:
            last_usage = {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
            last_finish_reason = "stop"

            async def chat(self, messages, tools=None, **kwargs):
                _ = (messages, kwargs)
                return SimpleNamespace(content="", tool_calls=[], usage={}, metadata={})

            async def stream_chat(self, *args, **kwargs):
                messages = kwargs.get("messages") or (args[0] if args else [])
                captured_messages.extend(messages)
                captured_stream_kwargs.append(dict(kwargs))
                yield StreamChunk(content_delta="final answer")

        class _PendingToolRunner:
            async def run(self, **kwargs):
                messages = kwargs["messages"]
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "thinking",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "demo", "arguments": '{"q":"x"}'},
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call_1",
                            "name": "demo",
                            "content": '{"results":[1]}',
                        },
                    ]
                )
                return (
                    SimpleNamespace(
                        content="",
                        tool_calls=[
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {"name": "demo", "arguments": '{"q":"y"}'},
                            }
                        ],
                        usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                        metadata={},
                    ),
                    [],
                )

        monkeypatch.setattr(service, "_default_adapter", _AdapterWithCapturedStream())
        monkeypatch.setattr(service, "_get_tool_runner", lambda: _PendingToolRunner())
        monkeypatch.setattr(
            chat_service_module.ToolBridge,
            "collect_tool_schemas",
            lambda self, skill_filter, include_core: [
                {"type": "function", "function": {"name": "demo"}}
            ],
        )
        monkeypatch.setattr(
            chat_service_module.ToolBridge,
            "collect_skills",
            lambda self, skill_filter, include_core: [SimpleNamespace(name="demo")],
        )

        resp = assert_status_code(
            post_message(client, conv_id, content="hello", enable_skills=["demo"])
        )
        events = get_sse_events(resp)
        assert_delta_text(events, "final answer")
        metadata = get_complete_metadata(events)
        assert metadata["tool_loop_convergence_reason"] == "pending_tool_calls_after_tool_loop"
        assert metadata["tool_loop_terminal_tool_call_count"] == 1
        assert metadata["tool_loop_max_rounds_reached"] is True
        assert metadata["final_stream_sanitized_message_count"] >= 1
        assert all(not message.get("tool_calls") for message in captured_messages)
        assert all("reasoning_content" not in message for message in captured_messages)
        assert captured_stream_kwargs[-1]["include_stream_usage"] is False

        conv = get_conversation_or_fail(store, conv_id)
        assert_last_message(conv, content="final answer", finish_reason="stop")

    def test_send_blast(self, app_and_client, monkeypatch):
        _, client, store = app_and_client
        conv_id = create_conversation_id(client, title="Blast Context")

        conv = get_conversation_or_fail(store, conv_id)
        conv.messages = [
            chat_service_module.Message(
                role=(
                    chat_service_module.MessageRole.USER
                    if index % 2 == 0
                    else chat_service_module.MessageRole.ASSISTANT
                ),
                content=f"turn-{index} " + ("x" * 140),
            )
            for index in range(520)
        ]
        store.update(conv)

        service = get_registered_chat_service()

        class _AdapterWithStream:
            last_usage = {"prompt_tokens": 1800, "completion_tokens": 6, "total_tokens": 1806}
            last_finish_reason = "stop"

            async def stream_chat(self, *args, **kwargs):
                _ = (args, kwargs)
                yield StreamChunk(content_delta="blast ok")

        monkeypatch.setattr(service, "_default_adapter", _AdapterWithStream())

        resp = assert_status_code(post_message(client, conv_id, content="continue under load"))

        events = get_sse_events(resp)
        usage = get_context_usage_data(events)["usage"]
        _, metadata = assert_finish_reason(events, "stop")

        assert usage["used_tokens"] > 0
        assert usage["max_context_tokens"] >= usage["used_tokens"]
        assert usage["block_breakdown"]
        assert metadata["finish_reason"] == "stop"

        persisted = get_conversation_or_fail(store, conv_id)
        assert_last_message(persisted, content="blast ok", finish_reason="stop")

    def test_send_message_tooling(self, app_and_client):
        _, client, _ = app_and_client
        conv_id = create_conversation_id(client)

        resp = assert_status_code(
            post_message(
                client,
                conv_id,
                content="Tooling fields",
                enable_tool_calls=False,
                tool_call_strategy="aggressive",
                enable_skills=["houyi_web_search"],
                max_tool_iterations=3,
            )
        )

    def test_send_large_tool_result(self, app_and_client, monkeypatch):
        _, client, _ = app_and_client
        conv_id = create_conversation_id(client, title="Large Tool Result")
        service = get_registered_chat_service()

        monkeypatch.setenv(ENV_TOOLCALL_LOOP_MAX_MESSAGE_CHARS, "20000")
        monkeypatch.setenv(ENV_TOOLCALL_LOOP_MAX_TOTAL_CHARS, "100000")
        monkeypatch.setenv(ENV_TOOLCALL_RESULT_SUMMARY_ENABLED, "1")
        monkeypatch.setenv(ENV_TOOLCALL_RESULT_SUMMARY_MAX_CHARS, "600")
        monkeypatch.setenv(ENV_TOOLCALL_RESULT_SUMMARY_MAX_ITEMS, "5")

        class _In(BaseModel):
            q: str

        class _Out(BaseModel):
            items: list[dict[str, str]]

        async def _execute_large_result(input_data: _In) -> dict[str, object]:
            _ = input_data
            return {
                "items": [{"idx": str(i), "payload": "y" * 500} for i in range(200)],
            }

        skill = SkillSpec(
            name="demo",
            description="Demo",
            input_schema=_In,
            output_schema=_Out,
            executor=_execute_large_result,
        )

        class _RecordingAdapter:
            last_usage = {"prompt_tokens": 18, "completion_tokens": 6, "total_tokens": 24}
            last_finish_reason = "stop"

            def __init__(self) -> None:
                self.chat_payloads: list[list[dict[str, object]]] = []
                self.calls = 0

            async def chat(self, messages, tools=None, **kwargs):
                _ = (tools, kwargs)
                self.calls += 1
                self.chat_payloads.append([dict(message) for message in messages])
                if self.calls == 1:
                    return SimpleNamespace(
                        content="",
                        tool_calls=[
                            {
                                "id": "call_big",
                                "type": "function",
                                "function": {
                                    "name": "demo",
                                    "arguments": json.dumps({"q": "hello"}),
                                },
                            }
                        ],
                        usage={},
                        metadata={},
                    )
                return SimpleNamespace(
                    content="Handled summarized tool result",
                    tool_calls=[],
                    usage={"prompt_tokens": 18, "completion_tokens": 6, "total_tokens": 24},
                    metadata={"finish_reason": "stop"},
                )

            async def stream_chat(self, *args, **kwargs):
                raise AssertionError("tool-loop replay should avoid final stream_chat")

        adapter = _RecordingAdapter()
        monkeypatch.setattr(service, "_default_adapter", adapter)
        monkeypatch.setattr(
            chat_service_module.ToolBridge,
            "collect_tool_schemas",
            lambda self, skill_filter, include_core: [skill.to_tool_schema()],
        )
        monkeypatch.setattr(
            chat_service_module.ToolBridge,
            "collect_skills",
            lambda self, skill_filter, include_core: [skill],
        )

        resp = assert_status_code(
            post_message(
                client,
                conv_id,
                content="Use demo",
                enable_skills=["demo"],
                tool_call_strategy="aggressive",
            )
        )

        events = get_sse_events(resp)
        usage = get_context_usage_data(events)["usage"]
        metadata = get_complete_metadata(events)

        assert len(adapter.chat_payloads) == 2
        second_round_payload = adapter.chat_payloads[-1]
        tool_message = next(msg for msg in second_round_payload if msg.get("role") == "tool")
        assert len(str(tool_message["content"])) < 1000
        assert (
            '"_truncated": true' in str(tool_message["content"]).lower()
            or "[truncated" in str(tool_message["content"]).lower()
        )
        assert usage["used_tokens"] > 0
        assert metadata["finish_reason"] == "stop"
        assert_delta_text(events, "Handled summarized tool result")

    def test_send_round_limit(self, app_and_client, monkeypatch):
        _, client, _ = app_and_client
        conv_id = create_conversation_id(client, title="Tool Iteration Limit")
        service = get_registered_chat_service()

        class _CapturedToolRunner:
            def __init__(self):
                self.max_rounds = None

            async def run(self, **kwargs):
                self.max_rounds = kwargs.get("max_rounds")
                return (
                    SimpleNamespace(
                        content="Done",
                        tool_calls=[],
                        usage={"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
                        metadata={},
                    ),
                    [],
                )

        runner = _CapturedToolRunner()
        monkeypatch.setattr(service, "_get_tool_runner", lambda: runner)
        monkeypatch.setattr(
            chat_service_module.ToolBridge,
            "collect_tool_schemas",
            lambda self, skill_filter, include_core: [
                {
                    "type": "function",
                    "function": {
                        "name": "houyi_web_search",
                        "description": "Web search",
                        "parameters": {},
                    },
                }
            ],
        )
        monkeypatch.setattr(
            chat_service_module.ToolBridge,
            "collect_skills",
            lambda self, skill_filter, include_core: [SimpleNamespace(name="houyi_web_search")],
        )

        resp = assert_status_code(
            post_message(
                client,
                conv_id,
                content="search please",
                enable_skills=["houyi_web_search"],
                max_tool_iterations=7,
            )
        )

        assert runner.max_rounds == 7

    def test_send_tool_replay(self, app_and_client, monkeypatch):
        _, client, store = app_and_client
        conv_id = create_conversation_id(client, title="Tool Replay Timing")
        service = get_registered_chat_service()

        class _ReplayTimingRunner:
            async def run(self, **kwargs):
                _ = kwargs
                return (
                    SimpleNamespace(
                        content="Replay final answer",
                        tool_calls=[],
                        usage={"prompt_tokens": 18, "completion_tokens": 6, "total_tokens": 24},
                        metadata={
                            "finish_reason": "stop",
                            "first_token_ms": 432,
                            "decode_tokens_per_second": 54,
                            "end_to_end_tokens_per_second": 37,
                        },
                    ),
                    [],
                )

        monkeypatch.setattr(service, "_get_tool_runner", lambda: _ReplayTimingRunner())
        monkeypatch.setattr(
            chat_service_module.ToolBridge,
            "collect_tool_schemas",
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
            chat_service_module.ToolBridge,
            "collect_skills",
            lambda self, skill_filter, include_core: [SimpleNamespace(name="demo")],
        )

        resp = assert_status_code(
            post_message(client, conv_id, content="Use demo", enable_skills=["demo"])
        )
        events = get_sse_events(resp)
        metadata = get_complete_metadata(events)

        assert metadata["first_token_ms"] == 432
        assert metadata["decode_tokens_per_second"] == 54
        assert metadata["end_to_end_tokens_per_second"] == 37

        conv = get_conversation_or_fail(store, conv_id)
        final_assistant = conv.messages[-1]
        assert final_assistant.metadata["first_token_ms"] == 432
        assert final_assistant.metadata["decode_tokens_per_second"] == 54
        assert final_assistant.metadata["end_to_end_tokens_per_second"] == 37

    def test_send_message_persists(self, app_and_client):
        _, client, store = app_and_client
        conv_id = create_conversation_id(client)

        assert_status_code(post_message(client, conv_id, content="Test persist"))

        conv = get_conversation_or_fail(store, conv_id)
        assert len(conv.messages) == 2
        assert conv.messages[0].role.value == "user"
        assert conv.messages[0].content == "Test persist"
        assert conv.messages[0].metadata["usage"]["input_tokens"] > 0
        assert conv.messages[1].role.value == "assistant"
        assert len(conv.messages[1].content) > 0
        assert conv.messages[1].metadata["usage"]["total_tokens"] == 15
