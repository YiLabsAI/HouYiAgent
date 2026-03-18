from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from houyi_studio.server.chat.tool_loop_orchestrator import ToolLoopOrchestrator
from houyi_studio.server.chat.types import Message, MessageRole, SendMessageRequest


@contextmanager
def _stage_span(*args, **kwargs):
    yield args[0] if args else None


class TestToolLoopOrchestrator:
    @pytest.mark.asyncio
    async def test_skips_when_no_skills(self):
        orchestrator = ToolLoopOrchestrator(
            default_chat_max_tool_iterations=3,
            get_tool_runner=lambda *_args, **_kwargs: None,
            context_hooks=SimpleNamespace(run_tool_result=lambda messages, trace, span: messages),
            extract_finish_reason=lambda *args: "stop",
            json_safe=lambda value: value,
            normalize_usage_payload=lambda value: value,
            null_hook_span_factory=lambda: None,
            sanitize_tool_loop_messages=lambda messages: messages,
            tool_bridge_factory=lambda: None,
            build_chat_kwargs=lambda **kwargs: kwargs,
            skill_executor_factory=lambda: None,
            stage_span=_stage_span,
        )

        outcome = await orchestrator.run(
            llm_adapter=SimpleNamespace(),
            model="m",
            llm_messages=[{"role": "user", "content": "hi"}],
            llm_kwargs={},
            request=SendMessageRequest(content="hi"),
            runtime_profile=SimpleNamespace(tool_result_max_tokens=128, per_tool_quota=4),
            assistant_message_id="msg-1",
            trace_id="trace-1",
            enabled_chat_skills=[],
        )

        assert outcome.llm_messages == [{"role": "user", "content": "hi"}]
        assert outcome.event_chunks == []

    @pytest.mark.asyncio
    async def test_replay_without_tools(self):
        runner = SimpleNamespace(
            run=AsyncMock(
                return_value=(
                    SimpleNamespace(
                        content="done",
                        tool_calls=[],
                        usage={"prompt_tokens": 4},
                        metadata={"reasoning_content": "think"},
                    ),
                    [],
                )
            )
        )
        deps = SimpleNamespace(
            get_tool_runner=MagicMock(return_value=runner),
            context_hooks=SimpleNamespace(
                run_tool_result=MagicMock(side_effect=lambda messages, trace, span: messages)
            ),
        )
        tool_bridge = SimpleNamespace(
            collect_tool_schemas=MagicMock(return_value=[{"type": "function"}]),
            collect_skills=MagicMock(return_value=[SimpleNamespace(name="demo")]),
        )
        orchestrator = ToolLoopOrchestrator(
            default_chat_max_tool_iterations=3,
            get_tool_runner=deps.get_tool_runner,
            context_hooks=deps.context_hooks,
            extract_finish_reason=lambda *args: "stop",
            json_safe=lambda value: value,
            normalize_usage_payload=lambda value: {"normalized": value["prompt_tokens"]},
            null_hook_span_factory=lambda: None,
            sanitize_tool_loop_messages=lambda messages: messages,
            tool_bridge_factory=lambda: tool_bridge,
            build_chat_kwargs=lambda **kwargs: kwargs,
            skill_executor_factory=lambda: "executor",
            stage_span=_stage_span,
        )

        outcome = await orchestrator.run(
            llm_adapter=SimpleNamespace(chat=MagicMock()),
            model="test-model",
            llm_messages=[{"role": "user", "content": "hi"}],
            llm_kwargs={"max_tokens": 64, "temperature": 0.1},
            request=SendMessageRequest(content="hi", max_tool_iterations=2),
            runtime_profile=SimpleNamespace(tool_result_max_tokens=128, per_tool_quota=4),
            assistant_message_id="msg-1",
            trace_id="trace-1",
            enabled_chat_skills=["demo"],
        )

        assert outcome.replay_response is not None
        assert outcome.usage_payload == {"normalized": 4}
        assert outcome.finish_reason == "stop"
        assert outcome.convergence_reason == "no_tool_calls_with_replay_payload"
        runner.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reasoning_only_payload(self):
        runner = SimpleNamespace(
            run=AsyncMock(
                return_value=(
                    SimpleNamespace(
                        content="",
                        tool_calls=[],
                        usage={"prompt_tokens": 4},
                        metadata={"reasoning_content": "think"},
                    ),
                    [],
                )
            )
        )
        tool_bridge = SimpleNamespace(
            collect_tool_schemas=MagicMock(return_value=[{"type": "function"}]),
            collect_skills=MagicMock(return_value=[SimpleNamespace(name="demo")]),
        )
        orchestrator = ToolLoopOrchestrator(
            default_chat_max_tool_iterations=3,
            get_tool_runner=lambda *_args, **_kwargs: runner,
            context_hooks=SimpleNamespace(run_tool_result=lambda messages, trace, span: messages),
            extract_finish_reason=lambda *args: "stop",
            json_safe=lambda value: value,
            normalize_usage_payload=lambda value: value,
            null_hook_span_factory=lambda: None,
            sanitize_tool_loop_messages=lambda messages: messages,
            tool_bridge_factory=lambda: tool_bridge,
            build_chat_kwargs=lambda **kwargs: kwargs,
            skill_executor_factory=lambda: None,
            stage_span=_stage_span,
        )

        outcome = await orchestrator.run(
            llm_adapter=SimpleNamespace(chat=MagicMock()),
            model="test-model",
            llm_messages=[{"role": "user", "content": "hi"}],
            llm_kwargs={"max_tokens": 64},
            request=SendMessageRequest(content="hi"),
            runtime_profile=SimpleNamespace(tool_result_max_tokens=128, per_tool_quota=4),
            assistant_message_id="msg-1",
            trace_id="trace-1",
            enabled_chat_skills=["demo"],
        )

        assert outcome.replay_response is None
        assert outcome.convergence_reason is None
        assert outcome.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_marks_pending_toolcalls(self):
        runner = SimpleNamespace(
            run=AsyncMock(
                return_value=(
                    SimpleNamespace(
                        content="",
                        tool_calls=[{"id": "call-1", "type": "function"}],
                        usage={"prompt_tokens": 4},
                        metadata={},
                    ),
                    [],
                )
            )
        )
        tool_bridge = SimpleNamespace(
            collect_tool_schemas=MagicMock(return_value=[{"type": "function"}]),
            collect_skills=MagicMock(return_value=[SimpleNamespace(name="demo")]),
        )
        orchestrator = ToolLoopOrchestrator(
            default_chat_max_tool_iterations=3,
            get_tool_runner=lambda *_args, **_kwargs: runner,
            context_hooks=SimpleNamespace(run_tool_result=lambda messages, trace, span: messages),
            extract_finish_reason=lambda *args: "tool_calls",
            json_safe=lambda value: value,
            normalize_usage_payload=lambda value: value,
            null_hook_span_factory=lambda: None,
            sanitize_tool_loop_messages=lambda messages: messages,
            tool_bridge_factory=lambda: tool_bridge,
            build_chat_kwargs=lambda **kwargs: kwargs,
            skill_executor_factory=lambda: None,
            stage_span=_stage_span,
        )

        outcome = await orchestrator.run(
            llm_adapter=SimpleNamespace(chat=MagicMock()),
            model="test-model",
            llm_messages=[{"role": "user", "content": "hi"}],
            llm_kwargs={"max_tokens": 64},
            request=SendMessageRequest(content="hi"),
            runtime_profile=SimpleNamespace(tool_result_max_tokens=128, per_tool_quota=4),
            assistant_message_id="msg-1",
            trace_id="trace-1",
            enabled_chat_skills=["demo"],
        )

        assert outcome.replay_response is None
        assert outcome.terminal_tool_call_count == 1
        assert outcome.convergence_reason == "pending_tool_calls_after_tool_loop"

    def test_emit_tool_result(self):
        stage_calls: list[dict[str, object]] = []

        @contextmanager
        def _recording_stage_span(parent, name, attributes=None):
            stage_calls.append(
                {
                    "parent": parent,
                    "name": name,
                    "attributes": dict(attributes or {}),
                }
            )
            yield parent

        orchestrator = ToolLoopOrchestrator(
            default_chat_max_tool_iterations=3,
            get_tool_runner=lambda *_args, **_kwargs: None,
            context_hooks=SimpleNamespace(run_tool_result=lambda messages, trace, span: messages),
            extract_finish_reason=lambda *args: "stop",
            json_safe=lambda value: value,
            normalize_usage_payload=lambda value: value,
            null_hook_span_factory=lambda: None,
            sanitize_tool_loop_messages=lambda messages: messages,
            tool_bridge_factory=lambda: None,
            build_chat_kwargs=lambda **kwargs: kwargs,
            skill_executor_factory=lambda: None,
            stage_span=_recording_stage_span,
        )

        class _ParentSpan:
            def __init__(self):
                self.attributes: dict[str, object] = {}

            def set_attribute(self, key, value):
                self.attributes[key] = value

        parent_span = _ParentSpan()
        persisted_tool_messages = [
            Message(
                role=MessageRole.TOOL,
                name="houyi_web_search",
                content="{}",
                metadata={
                    "tool_result_profile": {
                        "compressed": True,
                        "tool_category": "search",
                        "compression_strategy": "top_k",
                        "tokens_before": 200,
                        "tokens_after": 40,
                        "tool_result_max_tokens": 128,
                        "per_tool_quota": 5,
                    }
                },
            ),
            Message(
                role=MessageRole.TOOL,
                name="read_file",
                content="{}",
                metadata={
                    "tool_result_profile": {
                        "compressed": True,
                        "tool_category": "read",
                        "compression_strategy": "excerpt",
                        "tokens_before": 120,
                        "tokens_after": 24,
                        "tool_result_max_tokens": 256,
                        "per_tool_quota": 2,
                    }
                },
            ),
            Message(
                role=MessageRole.TOOL,
                name="code_exec",
                content="{}",
                metadata={"tool_result_profile": {"compressed": False}},
            ),
        ]

        orchestrator.emit_tool_result_profile_spans(
            parent_span=parent_span,
            persisted_tool_messages=persisted_tool_messages,
        )

        assert len(stage_calls) == 2
        assert stage_calls[0]["name"] == "tool_result.compress"
        assert stage_calls[0]["attributes"] == {
            "tool.name": "houyi_web_search",
            "tool.category": "search",
            "tool.compression_strategy": "top_k",
            "tool.tokens_before": 200,
            "tool.tokens_after": 40,
            "tool.token_budget": 128,
            "tool.item_quota": 5,
        }
        assert stage_calls[1]["attributes"] == {
            "tool.name": "read_file",
            "tool.category": "read",
            "tool.compression_strategy": "excerpt",
            "tool.tokens_before": 120,
            "tool.tokens_after": 24,
            "tool.token_budget": 256,
            "tool.item_quota": 2,
        }
        assert parent_span.attributes["chat.tool_result.compress.count"] == 2
        assert parent_span.attributes["chat.tool_result.compress.tokens_before"] == 320
        assert parent_span.attributes["chat.tool_result.compress.tokens_after"] == 64
