from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from houyi.adapters.llm import LLMAdapter
from houyi.infrastructure.observability import Span

from .tool_loop_runtime import build_tool_trace_events, collect_persisted_tool_messages
from .types import Message, SendMessageRequest


@dataclass
class ToolLoopOutcome:
    llm_messages: list[dict[str, Any]]
    event_chunks: list[str] = field(default_factory=list)
    persisted_tool_messages: list[Message] = field(default_factory=list)
    replay_response: Any | None = None
    usage_payload: dict[str, Any] | None = None
    response_metadata: dict[str, Any] | None = None
    finish_reason: str | None = None
    convergence_reason: str | None = None
    terminal_tool_call_count: int = 0


class ToolLoopOrchestrator:
    """Executes the skill/tool loop and projects its persisted artifacts."""

    def __init__(
        self,
        *,
        default_chat_max_tool_iterations: int,
        get_tool_runner: Callable[..., Any],
        context_hooks: Any,
        extract_finish_reason: Any,
        json_safe: Any,
        normalize_usage_payload: Any,
        null_hook_span_factory: Any,
        sanitize_tool_loop_messages: Any,
        tool_bridge_factory: Any,
        build_chat_kwargs: Any,
        skill_executor_factory: Any,
        stage_span: Any,
    ) -> None:
        self._default_chat_max_tool_iterations = max(1, int(default_chat_max_tool_iterations))
        self._get_tool_runner = get_tool_runner
        self._context_hooks = context_hooks
        self._extract_finish_reason = extract_finish_reason
        self._json_safe = json_safe
        self._normalize_usage_payload = normalize_usage_payload
        self._null_hook_span_factory = null_hook_span_factory
        self._sanitize_tool_loop_messages = sanitize_tool_loop_messages
        self._tool_bridge_factory = tool_bridge_factory
        self._build_chat_kwargs = build_chat_kwargs
        self._skill_executor_factory = skill_executor_factory
        self._stage_span = stage_span

    def collect_persisted_tool_messages(
        self,
        intermediate_messages: list[dict[str, Any]],
        *,
        model: str,
        runtime_profile: Any,
        tool_trace: list[dict[str, Any]] | None = None,
    ) -> list[Message]:
        return collect_persisted_tool_messages(
            intermediate_messages=intermediate_messages,
            model=model,
            tool_result_max_tokens=runtime_profile.tool_result_max_tokens,
            per_tool_quota=runtime_profile.per_tool_quota,
            tool_trace=tool_trace,
        )

    def emit_tool_result_profile_spans(
        self,
        *,
        parent_span: Span | None,
        persisted_tool_messages: list[Message],
    ) -> None:
        if parent_span is None:
            return
        compressed_count = 0
        tokens_before = 0
        tokens_after = 0
        for message in persisted_tool_messages:
            metadata = message.metadata if isinstance(message.metadata, dict) else {}
            profile = metadata.get("tool_result_profile")
            if not isinstance(profile, dict) or profile.get("compressed") is not True:
                continue
            compressed_count += 1
            before = profile.get("tokens_before")
            after = profile.get("tokens_after")
            if isinstance(before, int):
                tokens_before += before
            if isinstance(after, int):
                tokens_after += after
            with self._stage_span(
                parent_span,
                "tool_result.compress",
                attributes={
                    "tool.name": message.name or "tool",
                    "tool.category": profile.get("tool_category"),
                    "tool.compression_strategy": profile.get("compression_strategy"),
                    "tool.tokens_before": before or 0,
                    "tool.tokens_after": after or 0,
                    "tool.token_budget": profile.get("tool_result_max_tokens") or 0,
                    "tool.item_quota": profile.get("per_tool_quota") or 0,
                },
            ):
                pass
        parent_span.set_attribute("chat.tool_result.compress.count", compressed_count)
        parent_span.set_attribute("chat.tool_result.compress.tokens_before", tokens_before)
        parent_span.set_attribute("chat.tool_result.compress.tokens_after", tokens_after)

    async def run(
        self,
        *,
        llm_adapter: LLMAdapter,
        model: str,
        llm_messages: list[dict[str, Any]],
        llm_kwargs: dict[str, Any],
        request: SendMessageRequest,
        runtime_profile: Any,
        assistant_message_id: str,
        trace_id: str,
        enabled_chat_skills: list[str],
        parent_span: Span | None = None,
    ) -> ToolLoopOutcome:
        if not enabled_chat_skills:
            return ToolLoopOutcome(llm_messages=llm_messages)

        tool_bridge = self._tool_bridge_factory()
        tool_schemas = tool_bridge.collect_tool_schemas(
            skill_filter=enabled_chat_skills,
            include_core=True,
        )
        tool_specs = tool_bridge.collect_skills(
            skill_filter=enabled_chat_skills,
            include_core=True,
        )
        if not tool_schemas or not tool_specs or not hasattr(llm_adapter, "chat"):
            return ToolLoopOutcome(llm_messages=llm_messages)

        try:
            tool_runner = self._get_tool_runner(parent_span)
        except TypeError:
            tool_runner = self._get_tool_runner()
        tool_loop_messages = self._sanitize_tool_loop_messages(list(llm_messages))
        max_tool_iterations = request.max_tool_iterations or self._default_chat_max_tool_iterations
        tool_chat_kwargs = self._build_chat_kwargs(
            max_tokens=llm_kwargs.get("max_tokens"),
            temperature=llm_kwargs.get("temperature"),
            parallel_tool_calls=True,
            max_parallel_calls=None,
            prompt_cache_key=None,
        )
        tool_chat_kwargs["model"] = model
        tool_executor = self._skill_executor_factory()
        tool_loop_response, tool_trace = await tool_runner.run(
            adapter=llm_adapter,
            messages=tool_loop_messages,
            tools=tool_schemas,
            skills=tool_specs,
            executor=tool_executor,
            max_rounds=max_tool_iterations,
            chat_kwargs=tool_chat_kwargs,
            allow_tool_replace=False,
        )

        event_chunks = build_tool_trace_events(
            tool_trace=tool_trace,
            assistant_message_id=assistant_message_id,
            trace_id=trace_id,
        )

        intermediate_messages = [
            msg for msg in tool_loop_messages[len(llm_messages) :] if isinstance(msg, dict)
        ]
        persisted_tool_messages = self.collect_persisted_tool_messages(
            intermediate_messages,
            model=model,
            runtime_profile=runtime_profile,
            tool_trace=tool_trace,
        )
        persisted_tool_messages = self._context_hooks.run_tool_result(
            persisted_tool_messages,
            tool_trace,
            span=parent_span or self._null_hook_span_factory(),
        )
        self.emit_tool_result_profile_spans(
            parent_span=parent_span,
            persisted_tool_messages=persisted_tool_messages,
        )

        usage_payload: dict[str, Any] | None = None
        if isinstance(getattr(tool_loop_response, "usage", None), dict):
            usage_payload = self._normalize_usage_payload(self._json_safe(tool_loop_response.usage))
        response_metadata: dict[str, Any] | None = None
        if isinstance(getattr(tool_loop_response, "metadata", None), dict):
            response_metadata = dict(tool_loop_response.metadata)

        replay_response: Any | None = None
        convergence_reason: str | None = None
        terminal_tool_call_count = len(list(getattr(tool_loop_response, "tool_calls", []) or []))
        replay_content = str(getattr(tool_loop_response, "content", "") or "")
        if tool_loop_response and terminal_tool_call_count == 0 and replay_content:
            replay_response = tool_loop_response
            convergence_reason = "no_tool_calls_with_replay_payload"
        elif terminal_tool_call_count > 0:
            convergence_reason = "pending_tool_calls_after_tool_loop"

        return ToolLoopOutcome(
            llm_messages=tool_loop_messages,
            event_chunks=event_chunks,
            persisted_tool_messages=persisted_tool_messages,
            replay_response=replay_response,
            usage_payload=usage_payload,
            response_metadata=response_metadata,
            finish_reason=self._extract_finish_reason(tool_loop_response, replay_response),
            convergence_reason=convergence_reason,
            terminal_tool_call_count=terminal_tool_call_count,
        )
