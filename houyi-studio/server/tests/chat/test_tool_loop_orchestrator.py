from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from houyi_studio.server.chat.tool_loop_orchestrator import ToolLoopOrchestrator
from houyi_studio.server.chat.types import Message, MessageRole, SendMessageRequest

from houyi.adapters.llm.base import LLMResponse
from houyi.adapters.llm.models import DEEPSEEK_V3_2
from houyi.adapters.llm.siliconflow_adapter import SiliconFlowAdapter
from houyi.application.tool_calling.orchestrator import (
    ToolLoopOrchestrator as SdkToolLoopOrchestrator,
)


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

    @pytest.mark.asyncio
    async def test_relaxes_siliconflow_deepseek_tool_loop_under_wrapper_adapter(self):
        runner = SimpleNamespace(
            run=AsyncMock(
                return_value=(
                    SimpleNamespace(
                        content="done",
                        tool_calls=[],
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

        wrapped_adapter = SimpleNamespace(
            _inner=SiliconFlowAdapter(
                api_key="test-key",
                base_url="https://example.invalid/v1",
                default_model=DEEPSEEK_V3_2,
            ),
            chat=MagicMock(),
        )

        await orchestrator.run(
            llm_adapter=wrapped_adapter,
            model=DEEPSEEK_V3_2,
            llm_messages=[{"role": "user", "content": "hi"}],
            llm_kwargs={"max_tokens": 64},
            request=SendMessageRequest(content="hi", max_tool_iterations=2),
            runtime_profile=SimpleNamespace(tool_result_max_tokens=128, per_tool_quota=4),
            assistant_message_id="msg-1",
            trace_id="trace-1",
            enabled_chat_skills=["demo"],
        )

        runner.run.assert_awaited_once()
        call_kwargs = runner.run.await_args.kwargs
        assert "transport" not in call_kwargs["chat_kwargs"]
        assert "parallel_tool_calls" not in call_kwargs["chat_kwargs"]
        assert "tool_choice" not in call_kwargs["chat_kwargs"]

    @pytest.mark.asyncio
    async def test_relaxes_deepseek_for_wrapped_openai_compat_adapter_with_siliconflow_base_url(
        self,
    ):
        runner = SimpleNamespace(
            run=AsyncMock(
                return_value=(
                    SimpleNamespace(
                        content="done",
                        tool_calls=[],
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

        class _CompatInner:
            def __init__(self) -> None:
                self.base_url = "https://api.siliconflow.cn/v1"
                self.model = DEEPSEEK_V3_2

        wrapped_adapter = SimpleNamespace(
            _inner=_CompatInner(),
            chat=MagicMock(),
        )

        await orchestrator.run(
            llm_adapter=wrapped_adapter,
            model=DEEPSEEK_V3_2,
            llm_messages=[{"role": "user", "content": "hi"}],
            llm_kwargs={"max_tokens": 64},
            request=SendMessageRequest(content="hi", max_tool_iterations=2),
            runtime_profile=SimpleNamespace(tool_result_max_tokens=128, per_tool_quota=4),
            assistant_message_id="msg-1",
            trace_id="trace-1",
            enabled_chat_skills=["demo"],
        )

        runner.run.assert_awaited_once()
        call_kwargs = runner.run.await_args.kwargs
        assert "transport" not in call_kwargs["chat_kwargs"]
        assert "parallel_tool_calls" not in call_kwargs["chat_kwargs"]
        assert "tool_choice" not in call_kwargs["chat_kwargs"]

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

        assert outcome.replay_response is not None
        assert outcome.convergence_reason == "no_tool_calls_with_replay_payload"
        assert outcome.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_payload_with_persisted_tools(self):
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
        orchestrator.collect_persisted_tool_messages = MagicMock(
            return_value=[
                Message(
                    role=MessageRole.TOOL,
                    content='{"ok":true}',
                    tool_call_id="call-1",
                    name="demo",
                    metadata={"tool_name": "demo"},
                )
            ]
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

        assert outcome.persisted_tool_messages
        assert outcome.replay_response is None
        assert outcome.convergence_reason is None
        assert outcome.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_tool_marker_only_payload(self):
        runner = SimpleNamespace(
            run=AsyncMock(
                return_value=(
                    SimpleNamespace(
                        content="[tool_call]",
                        tool_calls=[],
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
    async def test_payload_without_tools_replays(self):
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

        assert outcome.replay_response is not None
        assert outcome.convergence_reason == "no_tool_calls_with_replay_payload"
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

    @pytest.mark.asyncio
    async def test_toolloop_defaults(self):
        runner = SimpleNamespace(
            run=AsyncMock(
                return_value=(
                    SimpleNamespace(
                        content="done",
                        tool_calls=[],
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

        await orchestrator.run(
            llm_adapter=SimpleNamespace(chat=MagicMock()),
            model="test-model",
            llm_messages=[{"role": "user", "content": "hi"}],
            llm_kwargs={"max_tokens": 64, "temperature": 0.1},
            request=SendMessageRequest(content="hi"),
            runtime_profile=SimpleNamespace(tool_result_max_tokens=128, per_tool_quota=4),
            assistant_message_id="msg-1",
            trace_id="trace-1",
            enabled_chat_skills=["demo"],
        )

        runner.run.assert_awaited_once()
        assert runner.run.await_args.kwargs["chat_kwargs"]["tool_choice"] == "required"

    @pytest.mark.asyncio
    async def test_toolloop_explicit_toolchoice(self):
        runner = SimpleNamespace(
            run=AsyncMock(
                return_value=(
                    SimpleNamespace(
                        content="done",
                        tool_calls=[],
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

        explicit_tool_choice = {"type": "function", "function": {"name": "houyi_web_search"}}
        await orchestrator.run(
            llm_adapter=SimpleNamespace(chat=MagicMock()),
            model="test-model",
            llm_messages=[{"role": "user", "content": "hi"}],
            llm_kwargs={"max_tokens": 64, "tool_choice": explicit_tool_choice},
            request=SendMessageRequest(content="hi"),
            runtime_profile=SimpleNamespace(tool_result_max_tokens=128, per_tool_quota=4),
            assistant_message_id="msg-1",
            trace_id="trace-1",
            enabled_chat_skills=["demo"],
        )

        runner.run.assert_awaited_once()
        assert runner.run.await_args.kwargs["chat_kwargs"]["tool_choice"] == explicit_tool_choice

    @pytest.mark.asyncio
    async def test_toolloop_relaxes_deepseek(self):
        class SiliconFlowAdapter:
            def chat(self):
                return None

        runner = SimpleNamespace(
            run=AsyncMock(
                return_value=(
                    SimpleNamespace(
                        content="done",
                        tool_calls=[],
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

        await orchestrator.run(
            llm_adapter=SiliconFlowAdapter(),
            model="deepseek-ai/DeepSeek-R1",
            llm_messages=[{"role": "user", "content": "hi"}],
            llm_kwargs={"max_tokens": 64, "temperature": 0.1},
            request=SendMessageRequest(content="hi"),
            runtime_profile=SimpleNamespace(tool_result_max_tokens=128, per_tool_quota=4),
            assistant_message_id="msg-1",
            trace_id="trace-1",
            enabled_chat_skills=["demo"],
        )

        assert "tool_choice" not in runner.run.await_args.kwargs["chat_kwargs"]
        assert "parallel_tool_calls" not in runner.run.await_args.kwargs["chat_kwargs"]
        assert "max_parallel_calls" not in runner.run.await_args.kwargs["chat_kwargs"]
        assert "transport" not in runner.run.await_args.kwargs["chat_kwargs"]

    @pytest.mark.asyncio
    async def test_toolloop_required_for_nodeepseek(self):
        runner = SimpleNamespace(
            run=AsyncMock(
                return_value=(
                    SimpleNamespace(
                        content="done",
                        tool_calls=[],
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

        await orchestrator.run(
            llm_adapter=SimpleNamespace(chat=MagicMock()),
            model="deepseek-ai/DeepSeek-V3",
            llm_messages=[{"role": "user", "content": "hi"}],
            llm_kwargs={"max_tokens": 64, "temperature": 0.1},
            request=SendMessageRequest(content="hi"),
            runtime_profile=SimpleNamespace(tool_result_max_tokens=128, per_tool_quota=4),
            assistant_message_id="msg-1",
            trace_id="trace-1",
            enabled_chat_skills=["demo"],
        )

        runner.run.assert_awaited_once()
        assert runner.run.await_args.kwargs["chat_kwargs"]["tool_choice"] == "required"
        assert runner.run.await_args.kwargs["chat_kwargs"]["parallel_tool_calls"] is True
        assert "transport" not in runner.run.await_args.kwargs["chat_kwargs"]

    @pytest.mark.asyncio
    async def test_toolloop_relaxes_deepseek_v32(self):
        class SiliconFlowAdapter:
            def chat(self):
                return None

        runner = SimpleNamespace(
            run=AsyncMock(
                return_value=(
                    SimpleNamespace(
                        content="done",
                        tool_calls=[],
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

        await orchestrator.run(
            llm_adapter=SiliconFlowAdapter(),
            model=DEEPSEEK_V3_2,
            llm_messages=[{"role": "user", "content": "hi"}],
            llm_kwargs={"max_tokens": 64, "temperature": 0.1},
            request=SendMessageRequest(content="hi"),
            runtime_profile=SimpleNamespace(tool_result_max_tokens=128, per_tool_quota=4),
            assistant_message_id="msg-1",
            trace_id="trace-1",
            enabled_chat_skills=["demo"],
        )

        runner.run.assert_awaited_once()
        assert "tool_choice" not in runner.run.await_args.kwargs["chat_kwargs"]
        assert "parallel_tool_calls" not in runner.run.await_args.kwargs["chat_kwargs"]
        assert "max_parallel_calls" not in runner.run.await_args.kwargs["chat_kwargs"]
        assert "transport" not in runner.run.await_args.kwargs["chat_kwargs"]

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


class TestSdkToolLoopOrchestratorLogging:
    @pytest.mark.asyncio
    async def test_logs_round_response(self, caplog: pytest.LogCaptureFixture):
        response = LLMResponse(
            content="done",
            tool_calls=[],
            metadata={"reasoning_content": "think"},
            finish_reason="stop",
            model="test-model",
        )
        runner = SimpleNamespace(
            _call_llm_with_cache=AsyncMock(return_value=(response, None)),
        )
        ctx = SimpleNamespace(
            runner=runner,
            config=SimpleNamespace(
                tool_loop_max_rounds=1,
                tool_loop_enable_timing=False,
                tool_loop_max_message_chars=1000,
                tool_loop_max_total_chars=2000,
            ),
            state=SimpleNamespace(
                tool_loop_messages=[{"role": "user", "content": "hi"}],
                tool_loop_started_at_monotonic=None,
            ),
            services=SimpleNamespace(
                model_adapter=SimpleNamespace(),
                available_tool_schemas=[],
                model_request_options={},
                llm_response_cache=None,
            ),
        )

        with caplog.at_level("DEBUG"):
            result = await SdkToolLoopOrchestrator.execute_rounds(ctx)

        assert result is response
        assert any(
            "response_shape" in record.message
            and "tool_calls=0" in record.message
            and "content_len=4" in record.message
            and "reasoning_len=5" in record.message
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_warns_on_textualmarker(self, caplog: pytest.LogCaptureFixture):
        response = LLMResponse(
            content="[tool:houyi_web_search]",
            tool_calls=[],
            metadata={},
            finish_reason="stop",
            model="test-model",
        )
        runner = SimpleNamespace(
            _call_llm_with_cache=AsyncMock(return_value=(response, None)),
        )
        ctx = SimpleNamespace(
            runner=runner,
            config=SimpleNamespace(
                tool_loop_max_rounds=1,
                tool_loop_enable_timing=False,
                tool_loop_max_message_chars=1000,
                tool_loop_max_total_chars=2000,
            ),
            state=SimpleNamespace(
                tool_loop_messages=[{"role": "user", "content": "hi"}],
                tool_loop_started_at_monotonic=None,
            ),
            services=SimpleNamespace(
                model_adapter=SimpleNamespace(),
                available_tool_schemas=[],
                model_request_options={},
                llm_response_cache=None,
            ),
        )

        with caplog.at_level("WARNING"):
            result = await SdkToolLoopOrchestrator.execute_rounds(ctx)

        assert result is response
        assert any(
            "textual tool markers without structured tool_calls" in record.message
            for record in caplog.records
        )
