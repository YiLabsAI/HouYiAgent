"""Chat Service: orchestrates SDK components for chat interactions.

Bridges Studio Server with SDK Context Engine, Memory Engine, and LLM Adapter.
Owns the business logic for sending messages, managing context, and streaming.

"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

from houyi.config.env_config import (
    ENV_CHAT_TOOL_LOOP_MAX_MESSAGE_CHARS,
    ENV_CHAT_TOOL_LOOP_MAX_TOTAL_CHARS,
)
from houyi.context import ContextPlanner, ContextRenderer, TokenEstimator
from houyi.core.skill.tool_bridge import ToolBridge
from houyi.core.skill_registry import DEFAULT_SKILL_REGISTRY
from houyi.execution.skill_executor import SkillExecutor
from houyi.execution.tool_call_orchestrator import build_chat_kwargs
from houyi.execution.tool_call_runner import ToolCallRunner
from houyi.llm.base import LLMAdapter, LLMMessage
from houyi.llm.factory import LLMAdapterFactory, _create_vertex_adapter
from houyi.llm.models import DEFAULT_MODEL
from houyi.llm.siliconflow_adapter import SiliconFlowAdapter
from houyi.memory import MemoryStore
from houyi.observability.context import TraceContext
from houyi.observability.storage import get_storage
from houyi.observability.trace_manager import Span
from houyi.observability.types import SpanSchema, SpanStatus, SpanType

from ..skill.service import get_skill_service
from .json_store import JsonStore
from .sse_adapter import SSEEvent, stream_chat_sse
from .types import EditMessageRequest, Message, MessageRole, SendMessageRequest

logger = logging.getLogger(__name__)

# Vision-capable model patterns.
# Models matching these patterns support image_url in content arrays.
_VISION_PATTERNS = [
    r"gemini-[12345]\.\d",
    r"gemini-(?:flash|pro|flash-lite)",
    r"claude-3",
    r"claude-(?:haiku|sonnet|opus)-4",
    r"gpt-4o",
    r"gpt-4-turbo",
    r"gpt-4\.1",
    r"gpt-4\.5",
    r"gpt-5",
    r"o[134]-",
    r"qwen-vl",
    r"qwen2-vl",
    r"qwen2\.5-vl",
    r"qwen-omni",
    r"deepseek-vl",
    r"glm-4v",
    r"internvl",
    r"pixtral",
    r"llava",
    r"minicpm",
    r"grok-vision",
    r"vision",
]
_VISION_RE = re.compile("|".join(_VISION_PATTERNS), re.IGNORECASE)

_DEFAULT_CHAT_MAX_TOOL_ITERATIONS = 10


def _read_positive_int_env(env_name: str, default: int) -> int:
    raw = os.getenv(env_name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, fallback to %d", env_name, raw, default)
        return default
    if parsed <= 0:
        logger.warning("Invalid %s=%r (must be > 0), fallback to %d", env_name, raw, default)
        return default
    return parsed


_TOOL_LOOP_MAX_MESSAGE_CHARS = _read_positive_int_env(
    ENV_CHAT_TOOL_LOOP_MAX_MESSAGE_CHARS,
    12_000,
)
_TOOL_LOOP_MAX_TOTAL_CHARS = _read_positive_int_env(
    ENV_CHAT_TOOL_LOOP_MAX_TOTAL_CHARS,
    160_000,
)
_CHAT_BUILTIN_TOOL_NAMES = frozenset(
    {
        "houyi_read_file",
        "houyi_write_file",
        "houyi_find_files",
        "houyi_list_dir",
        "houyi_grep",
        "houyi_shell_exec",
    }
)
_WEB_SEARCH_SKILL_NAME = "web_search"
_TOOL_CALL_STRATEGY_CONSERVATIVE = "conservative"
_TOOL_CALL_STRATEGY_BALANCED = "balanced"
_TOOL_CALL_STRATEGY_AGGRESSIVE = "aggressive"
_TOOL_INTENT_KEYWORDS = (
    "grep",
    "find",
    "search",
    "read file",
    "write file",
    "list dir",
    "terminal",
    "shell",
    "command",
    "codebase",
    "path",
    "folder",
    "目录",
    "文件",
    "查找",
    "搜索",
    "读取",
    "写入",
    "命令",
    "执行",
    "代码库",
)
_TOOL_INTENT_REGEXES = (
    re.compile(r"`[^`]+`"),
    re.compile(r"(?:^|\s)/(?:Users|home|var|tmp|opt|etc)/"),
    re.compile(r"(?:^|\s)\./[\w./-]+"),
    re.compile(r"(?:^|\s)\.{2}/[\w./-]+"),
    re.compile(r"\b[a-zA-Z0-9_./-]+\.(?:py|ts|tsx|js|jsx|md|json|yaml|yml|toml|sql|sh|txt)\b"),
)


@dataclass
class _PreparedSendContext:
    conv_lock: Any
    model: str
    llm_messages: list[dict[str, Any]]
    context_usage: dict[str, Any]
    llm_kwargs: dict[str, Any]


@dataclass
class _ToolLoopOutcome:
    llm_messages: list[dict[str, Any]]
    event_chunks: list[str] = field(default_factory=list)
    persisted_tool_messages: list[Message] = field(default_factory=list)
    replay_response: Any | None = None
    usage_payload: dict[str, Any] | None = None
    convergence_reason: str | None = None


@dataclass
class _ToolLoopGateDecision:
    enabled_skills: list[str]
    mode: str
    reason: str


def is_vision_model(model_id: str) -> bool:
    """Return True if *model_id* is known to accept image_url content parts."""
    return bool(_VISION_RE.search(model_id))


def _json_safe(value: Any) -> Any:
    """Best-effort conversion into JSON-serializable payloads."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    with contextlib.suppress(Exception):
        return json.loads(json.dumps(value, default=str))
    return str(value)


def _coerce_text_content(value: Any) -> str:
    """Coerce chat content into plain text for tool-loop chat adapters."""
    return LLMAdapter._coerce_message_content_to_text(value)


def _looks_like_tool_intent(user_content: str) -> bool:
    lowered = user_content.lower()
    if any(keyword in lowered for keyword in _TOOL_INTENT_KEYWORDS):
        return True
    if "```" in user_content:
        return True
    return any(regex.search(user_content) for regex in _TOOL_INTENT_REGEXES)


def _truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return f"{text[:head]}\n...[truncated]...\n{text[-tail:]}"


def _message_budget_chars(message: dict[str, Any]) -> int:
    content_len = len(str(message.get("content") or ""))
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return content_len

    args_len = 0
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function")
        if not isinstance(fn, dict):
            continue
        args = fn.get("arguments")
        args_len += len(args) if isinstance(args, str) else len(str(args or ""))
    return content_len + args_len


def _truncate_tool_call_arguments(message: dict[str, Any], max_chars: int) -> dict[str, Any]:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return message

    fixed_calls: list[dict[str, Any]] = []
    for call in tool_calls:
        fixed = LLMAdapter._sanitize_tool_call(call)
        if fixed is None:
            continue
        fn = fixed.get("function")
        if isinstance(fn, dict):
            args = fn.get("arguments")
            if isinstance(args, str):
                fn["arguments"] = _truncate_middle(args, max_chars)
        fixed_calls.append(fixed)

    message["tool_calls"] = fixed_calls
    return message


def _cap_total_payload_chars(
    items: list[dict[str, Any]], max_total_chars: int
) -> list[dict[str, Any]]:
    total_chars = sum(_message_budget_chars(msg) for msg in items)
    if total_chars <= max_total_chars:
        return items

    system_messages = [msg for msg in items if msg.get("role") == MessageRole.SYSTEM.value]
    non_system = [msg for msg in items if msg.get("role") != MessageRole.SYSTEM.value]
    system_chars = sum(_message_budget_chars(msg) for msg in system_messages)
    budget_for_non_system = max(0, max_total_chars - system_chars)

    kept_non_system: list[dict[str, Any]] = []
    used = 0
    for msg in reversed(non_system):
        payload_chars = _message_budget_chars(msg)
        if kept_non_system and used + payload_chars > budget_for_non_system:
            continue
        kept_non_system.append(msg)
        used += payload_chars

    trimmed = system_messages + list(reversed(kept_non_system))
    logger.warning(
        "Tool-loop message budget applied: total_payload=%d -> %d messages=%d -> %d",
        total_chars,
        sum(_message_budget_chars(msg) for msg in trimmed),
        len(items),
        len(trimmed),
    )
    return trimmed


def _sanitize_tool_loop_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy with string ``content`` fields for tool-loop chat calls."""
    base_sanitized = LLMAdapter._sanitize_messages(messages)
    sanitized: list[dict[str, Any]] = []
    for msg in base_sanitized:
        normalized = dict(msg)
        normalized["content"] = _truncate_middle(
            _coerce_text_content(normalized.get("content")),
            _TOOL_LOOP_MAX_MESSAGE_CHARS,
        )
        normalized = _truncate_tool_call_arguments(normalized, _TOOL_LOOP_MAX_MESSAGE_CHARS)
        sanitized.append(normalized)
    return _cap_total_payload_chars(sanitized, _TOOL_LOOP_MAX_TOTAL_CHARS)


def _flatten_span_tree(root: Span) -> list[Span]:
    stack = [root]
    ordered: list[Span] = []
    while stack:
        current = stack.pop()
        ordered.append(current)
        if current.children:
            stack.extend(reversed(current.children))
    return ordered


def _persist_trace_tree(root: Span) -> None:
    try:
        storage = get_storage()
        payload = [
            SpanSchema(
                name=span.name,
                trace_id=span.trace_id,
                span_id=span.span_id,
                parent_id=span.parent_id,
                start_time=span.start_time,
                end_time=span.end_time,
                status=cast(SpanStatus, span.status),
                status_description=span.status_description,
                span_type=span.span_type,
                node_id=span.node_id,
                model=span.model,
                provider=span.provider,
                tokens=span.tokens,
                cost=span.cost,
                cache_hit=span.cache_hit,
                tool_name=span.tool_name,
                kb_name=span.kb_name,
                docs_count=span.docs_count,
                top_k=span.top_k,
                group_id=span.group_id,
                lane_id=span.lane_id,
                seq=span.seq,
                parent_trace_id=span.parent_trace_id,
                restore_checkpoint_id=span.restore_checkpoint_id,
                replay_mode=span.replay_mode,
                attributes=span.attributes,
                events=span.events,
            )
            for span in _flatten_span_tree(root)
        ]
        storage.save_batch(payload)
    except Exception as exc:
        logger.warning("Persist trace tree failed: trace_id=%s error=%s", root.trace_id, exc)


@contextlib.contextmanager
def _stage_span(
    parent: Span, name: str, attributes: dict[str, Any] | None = None
) -> Iterator[Span]:
    span = Span(
        name=name,
        parent=parent,
        span_type=SpanType.INTERNAL,
        attributes=attributes or {},
    )
    try:
        yield span
    except Exception as exc:
        span.set_status("error", str(exc))
        raise
    else:
        span.set_status("ok")
    finally:
        span.end()


class ChatService:
    """Orchestrates chat interactions between UI, Server, and SDK layers.

    Responsibilities:
    - Receive user message → persist → build context → call LLM → stream response
    - Manage conversation lifecycle (create, update, delete)
    - Track context usage for UI display

    Concurrency: Uses per-conversation locking via JsonStore.lock() to
    serialize read-modify-write cycles. Safe for concurrent async callers.
    """

    def __init__(
        self,
        json_store: JsonStore,
        memory_store: MemoryStore | None = None,
        default_model: str = "",
        default_system_instructions: str = "",
        settings_store: Any | None = None,
    ):
        """Initialize chat service.

        Args:
            json_store: Conversation persistence store.
            memory_store: Optional memory store for context injection.
            default_model: Default LLM model name.
            default_system_instructions: Default system prompt.
            settings_store: Optional SettingsStore for provider-based model routing.
        """
        self.json_store = json_store
        self.memory_store = memory_store
        self.default_model = default_model or DEFAULT_MODEL
        self.default_system_instructions = default_system_instructions
        self._settings_store = settings_store
        self._default_adapter = LLMAdapterFactory.create()
        self._adapter_cache: dict[str, LLMAdapter] = {}

    def invalidate_adapter_cache(self) -> None:
        """Clear cached adapters. Call when provider settings change."""
        self._adapter_cache.clear()
        logger.info("LLM adapter cache invalidated")

    def _get_tool_runner(self, parent_span: Span | None = None) -> ToolCallRunner:
        """Build a ToolCallRunner with governance components from SkillService."""
        policy_enforcer = None
        consent_manager = None
        metrics_store = None
        with contextlib.suppress(Exception):
            skill_service = get_skill_service()
            policy_enforcer = skill_service.policy_enforcer
            consent_manager = skill_service.consent_manager
            metrics_store = skill_service.metrics_store
        trace_manager = None
        if parent_span is not None:
            trace_manager = SimpleNamespace(current_span=parent_span, root_spans=[])
        return ToolCallRunner(
            trace_manager=trace_manager,
            policy_enforcer=policy_enforcer,
            consent_manager=consent_manager,
            metrics_store=metrics_store,
        )

    def _resolve_enabled_chat_skills(self, request: SendMessageRequest) -> list[str]:
        """Resolve chat-allowed skills from request toggles and defaults."""
        if request.enable_tool_calls is False:
            return []
        resolved: set[str] = set(_CHAT_BUILTIN_TOOL_NAMES)
        if request.enable_web_search:
            resolved.add(_WEB_SEARCH_SKILL_NAME)
        if request.enable_skills:
            for skill_name in request.enable_skills:
                if isinstance(skill_name, str) and skill_name.strip():
                    resolved.add(skill_name.strip())
        return sorted(resolved)

    def _gate_tool_loop(
        self,
        *,
        request: SendMessageRequest,
        resolved_skills: list[str],
    ) -> _ToolLoopGateDecision:
        """Lightweight tool-loop gating for latency-sensitive enterprise chat."""
        if request.enable_tool_calls is False:
            return _ToolLoopGateDecision([], "disabled_by_request", "request_disable")

        if not resolved_skills:
            return _ToolLoopGateDecision([], "disabled_no_skills", "no_resolved_skills")

        explicit_skills = [
            name.strip()
            for name in (request.enable_skills or [])
            if isinstance(name, str) and name.strip()
        ]
        strategy = (request.tool_call_strategy or _TOOL_CALL_STRATEGY_BALANCED).strip().lower()
        if strategy not in {
            _TOOL_CALL_STRATEGY_CONSERVATIVE,
            _TOOL_CALL_STRATEGY_BALANCED,
            _TOOL_CALL_STRATEGY_AGGRESSIVE,
        }:
            strategy = _TOOL_CALL_STRATEGY_BALANCED

        if request.enable_web_search or explicit_skills:
            return _ToolLoopGateDecision(resolved_skills, "enabled", "explicit_skill_request")

        if strategy == _TOOL_CALL_STRATEGY_AGGRESSIVE:
            return _ToolLoopGateDecision(
                resolved_skills, "enabled", "strategy_aggressive_default_on"
            )

        if strategy == _TOOL_CALL_STRATEGY_CONSERVATIVE:
            return _ToolLoopGateDecision(
                [], "disabled_by_gating", "strategy_conservative_requires_explicit"
            )

        user_content = str(request.content or "").strip()
        if _looks_like_tool_intent(user_content):
            return _ToolLoopGateDecision(resolved_skills, "enabled", "heuristic_tool_intent")

        return _ToolLoopGateDecision([], "disabled_by_gating", "heuristic_no_tool_intent")

    def _get_adapter_for_model(self, model: str) -> LLMAdapter:
        """Get the LLM adapter for a given model by looking up its provider.

        Routes the model to the correct provider's adapter based on settings.
        Falls back to the default adapter if no provider match is found.
        """
        if not self._settings_store:
            return self._default_adapter

        settings = self._settings_store.get()
        for provider in settings.providers:
            if not provider.enabled:
                continue
            if model in provider.models:
                cache_key = provider.id
                if cache_key in self._adapter_cache:
                    return self._adapter_cache[cache_key]

                # Detect if this is a Vertex AI / Gemini provider
                provider_url = (provider.base_url or "").rstrip("/")
                is_vertex = (
                    "aiplatform.googleapis.com" in provider_url
                    or "vertex" in provider.name.lower()
                    or provider.id.startswith("vertex")
                )
                if is_vertex:
                    adapter = _create_vertex_adapter()
                    logger.info(
                        "Model '%s' routed to Gemini adapter (provider='%s')", model, provider.name
                    )
                elif provider_url and "/v1" in provider_url:
                    adapter = SiliconFlowAdapter(
                        api_key=provider.api_key or None,
                        base_url=provider_url,
                        default_model=model,
                    )
                else:
                    # No usable base_url, fall back to default
                    logger.info(
                        "Model '%s' provider '%s' has no OpenAI-compatible base_url, using default adapter",
                        model,
                        provider.name,
                    )
                    return self._default_adapter
                self._adapter_cache[cache_key] = adapter
                logger.info(
                    "Model '%s' routed to provider '%s' (%s)",
                    model,
                    provider.name,
                    provider.base_url or "default",
                )
                return adapter

        return self._default_adapter

    def _resolve_llm_kwargs(
        self,
        request: SendMessageRequest,
        conversation: Any,
    ) -> dict[str, Any]:
        llm_kwargs: dict[str, Any] = {}
        if request.temperature is not None:
            llm_kwargs["temperature"] = request.temperature
        elif conversation.temperature is not None:
            llm_kwargs["temperature"] = conversation.temperature
        if request.max_tokens is not None:
            llm_kwargs["max_tokens"] = request.max_tokens
        elif conversation.max_tokens is not None:
            llm_kwargs["max_tokens"] = conversation.max_tokens
        if conversation.top_p is not None:
            llm_kwargs["top_p"] = conversation.top_p
        if request.enable_reasoning:
            llm_kwargs["enable_reasoning"] = True
        return llm_kwargs

    def _build_context_messages(
        self,
        conversation: Any,
        model: str,
        sys_instructions: str,
        chat_span: Span,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        estimator = TokenEstimator(model=model)
        planner = ContextPlanner(
            token_estimator=estimator,
            system_instructions=sys_instructions,
        )
        renderer = ContextRenderer()

        memory_text = ""
        if self.memory_store:
            from houyi.memory.types import MemoryScope

            memory_text = self.memory_store.as_context_text(MemoryScope.SESSION)

        _vision = is_vision_model(model)
        history_messages = [
            m.to_llm_message(vision=_vision)
            for m in conversation.messages
            if m.role != MessageRole.SYSTEM
        ]

        plan = planner.plan(
            messages=history_messages,
            system_instructions=sys_instructions,
            memory_context=memory_text if memory_text else None,
        )

        llm_messages = renderer.render(plan)
        context_usage = plan.usage.model_dump(mode="json")
        chat_span.set_attribute("chat.context_tokens_used", plan.usage.used_tokens)
        chat_span.set_attribute("chat.context_tokens_max", plan.usage.max_context_tokens)
        chat_span.set_attribute("chat.llm_messages_count", len(llm_messages))

        logger.info(
            "Chat context: %d messages, %d tokens used / %d max (%s)",
            len(llm_messages),
            plan.usage.used_tokens,
            plan.usage.max_context_tokens,
            model,
        )
        return llm_messages, context_usage

    async def _prepare_send_context(
        self,
        conversation_id: str,
        request: SendMessageRequest,
        chat_span: Span,
    ) -> _PreparedSendContext:
        conv_lock = await self.json_store.lock(conversation_id)
        conversation_snapshot: Any | None = None
        async with conv_lock:
            conversation = self.json_store.get(conversation_id)
            if conversation is None:
                raise FileNotFoundError(f"Conversation {conversation_id} not found")

            model = (
                (request.model if request.model else None)
                or (conversation.model if conversation.model else None)
                or self.default_model
            )
            sys_instructions = conversation.system_instructions or self.default_system_instructions
            chat_span.set_attribute("chat.model", model)

            user_msg = Message(
                role=MessageRole.USER,
                content=request.content,
                attachments=request.attachments,
            )
            conversation.messages.append(user_msg)
            conversation.updated_at = time.time()
            self.json_store.update(conversation)
            llm_kwargs = self._resolve_llm_kwargs(request=request, conversation=conversation)
            conversation_snapshot = conversation.model_copy(deep=True)

        if conversation_snapshot is None:
            raise RuntimeError(f"Conversation snapshot unavailable: {conversation_id}")

        llm_messages, context_usage = self._build_context_messages(
            conversation=conversation_snapshot,
            model=model,
            sys_instructions=sys_instructions,
            chat_span=chat_span,
        )

        return _PreparedSendContext(
            conv_lock=conv_lock,
            model=model,
            llm_messages=llm_messages,
            context_usage=context_usage,
            llm_kwargs=llm_kwargs,
        )

    def _collect_persisted_tool_messages(
        self,
        intermediate_messages: list[dict[str, Any]],
    ) -> list[Message]:
        persisted_tool_messages: list[Message] = []
        for intermediate in intermediate_messages:
            role = intermediate.get("role")
            if role == MessageRole.ASSISTANT.value and intermediate.get("tool_calls"):
                persisted_tool_messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=str(intermediate.get("content") or ""),
                        reasoning_content=(
                            str(intermediate.get("reasoning_content"))
                            if isinstance(intermediate.get("reasoning_content"), str)
                            else None
                        ),
                        tool_calls=intermediate.get("tool_calls"),
                    )
                )
                continue
            if role == MessageRole.TOOL.value:
                persisted_tool_messages.append(
                    Message(
                        role=MessageRole.TOOL,
                        content=str(intermediate.get("content") or ""),
                        tool_call_id=(
                            str(intermediate.get("tool_call_id"))
                            if intermediate.get("tool_call_id")
                            else None
                        ),
                        name=(str(intermediate.get("name")) if intermediate.get("name") else None),
                    )
                )
        return persisted_tool_messages

    async def _run_tool_loop(
        self,
        *,
        llm_adapter: LLMAdapter,
        model: str,
        llm_messages: list[dict[str, Any]],
        llm_kwargs: dict[str, Any],
        request: SendMessageRequest,
        assistant_message_id: str,
        trace_id: str,
        enabled_chat_skills: list[str],
        parent_span: Span | None = None,
    ) -> _ToolLoopOutcome:
        if not enabled_chat_skills:
            return _ToolLoopOutcome(llm_messages=llm_messages)

        tool_bridge = ToolBridge(DEFAULT_SKILL_REGISTRY)
        tool_schemas = tool_bridge.collect_tool_schemas(
            skill_filter=enabled_chat_skills,
            include_core=True,
        )
        tool_specs = tool_bridge.collect_skills(
            skill_filter=enabled_chat_skills,
            include_core=True,
        )
        if not tool_schemas or not tool_specs or not hasattr(llm_adapter, "chat"):
            return _ToolLoopOutcome(llm_messages=llm_messages)

        try:
            tool_runner = self._get_tool_runner(parent_span)
        except TypeError:
            # Backward compatibility for tests/overrides that monkeypatch
            # _get_tool_runner as a zero-arg callable.
            tool_runner = self._get_tool_runner()
        tool_loop_messages = _sanitize_tool_loop_messages(list(llm_messages))
        max_tool_iterations = request.max_tool_iterations or _DEFAULT_CHAT_MAX_TOOL_ITERATIONS
        tool_chat_kwargs = build_chat_kwargs(
            max_tokens=llm_kwargs.get("max_tokens"),
            temperature=llm_kwargs.get("temperature"),
            parallel_tool_calls=True,
            max_parallel_calls=None,
            prompt_cache_key=None,
        )
        # Keep tool-loop model consistent with the request-selected model.
        tool_chat_kwargs["model"] = model
        tool_executor = SkillExecutor(max_retries=3, timeout=30.0)
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

        event_chunks: list[str] = []
        round_indexes = sorted(
            {
                round_index
                for round_index in (
                    entry.get("round_index") if isinstance(entry, dict) else None
                    for entry in tool_trace
                )
                if isinstance(round_index, int)
            }
        )
        for round_index in round_indexes:
            event_chunks.append(
                SSEEvent(
                    event="agent.iteration",
                    data={
                        "message_id": assistant_message_id,
                        "trace_id": trace_id,
                        "round_index": round_index,
                    },
                ).encode()
            )

        for entry in tool_trace:
            if not isinstance(entry, dict):
                continue
            tool_call_id = entry.get("tool_call_id")
            tool_name = entry.get("tool_name")
            parallel_group_id = entry.get("parallel_group_id")
            round_value = entry.get("round_index")
            args = entry.get("args")
            result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
            raw_result = result.get("raw") if isinstance(result, dict) else None

            event_chunks.append(
                SSEEvent(
                    event="tool_call.start",
                    data={
                        "message_id": assistant_message_id,
                        "trace_id": trace_id,
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "parallel_group_id": parallel_group_id,
                        "round_index": round_value,
                        "arguments": args,
                    },
                ).encode()
            )

            if isinstance(raw_result, dict) and raw_result.get("error"):
                event_chunks.append(
                    SSEEvent(
                        event="tool_call.error",
                        data={
                            "message_id": assistant_message_id,
                            "trace_id": trace_id,
                            "tool_call_id": tool_call_id,
                            "tool_name": tool_name,
                            "parallel_group_id": parallel_group_id,
                            "round_index": round_value,
                            "error": raw_result,
                        },
                    ).encode()
                )
            else:
                event_chunks.append(
                    SSEEvent(
                        event="tool_call.result",
                        data={
                            "message_id": assistant_message_id,
                            "trace_id": trace_id,
                            "tool_call_id": tool_call_id,
                            "tool_name": tool_name,
                            "parallel_group_id": parallel_group_id,
                            "round_index": round_value,
                            "result": raw_result,
                        },
                    ).encode()
                )

        intermediate_messages = [
            msg for msg in tool_loop_messages[len(llm_messages) :] if isinstance(msg, dict)
        ]
        persisted_tool_messages = self._collect_persisted_tool_messages(intermediate_messages)

        usage_payload: dict[str, Any] | None = None
        if isinstance(getattr(tool_loop_response, "usage", None), dict):
            usage_payload = _json_safe(tool_loop_response.usage)

        replay_response: Any | None = None
        convergence_reason: str | None = None
        if (
            tool_loop_response
            and not list(getattr(tool_loop_response, "tool_calls", []) or [])
            and str(getattr(tool_loop_response, "content", "") or "")
        ):
            replay_response = tool_loop_response
            convergence_reason = "no_tool_calls_with_content"

        return _ToolLoopOutcome(
            llm_messages=tool_loop_messages,
            event_chunks=event_chunks,
            persisted_tool_messages=persisted_tool_messages,
            replay_response=replay_response,
            usage_payload=usage_payload,
            convergence_reason=convergence_reason,
        )

    def get_context_usage(self, conversation_id: str) -> dict[str, Any] | None:
        """Calculate context usage for a conversation without sending a message.

        Returns the same usage dict that the SSE context.usage event provides,
        or None if the conversation has no messages.
        """
        conversation = self.json_store.get(conversation_id)
        if conversation is None:
            return None

        model = conversation.model or self.default_model
        sys_instructions = conversation.system_instructions or self.default_system_instructions

        history_messages = [
            m.to_llm_message() for m in conversation.messages if m.role != MessageRole.SYSTEM
        ]
        if not history_messages:
            return None

        estimator = TokenEstimator(model=model)
        planner = ContextPlanner(
            token_estimator=estimator,
            system_instructions=sys_instructions,
        )

        memory_text = ""
        if self.memory_store:
            from houyi.memory.types import MemoryScope

            memory_text = self.memory_store.as_context_text(MemoryScope.SESSION)

        plan = planner.plan(
            messages=history_messages,
            system_instructions=sys_instructions,
            memory_context=memory_text if memory_text else None,
        )

        return plan.usage.model_dump(mode="json")

    async def _stream_replay_chunks(
        self,
        *,
        replay_response: Any,
        assistant_message_id: str,
        model: str,
        context_usage: dict[str, Any],
    ) -> tuple[list[str], list[str], list[str]]:
        replay_content = str(getattr(replay_response, "content", "") or "")
        replay_reasoning: str | None = None
        metadata = getattr(replay_response, "metadata", None)
        if isinstance(metadata, dict) and metadata.get("reasoning_content") is not None:
            replay_reasoning = str(metadata.get("reasoning_content") or "")

        content_parts = [replay_content] if replay_content else []
        reasoning_parts = [replay_reasoning] if replay_reasoning else []

        async def replay_stream() -> AsyncIterator[tuple[str, str | None]]:
            yield replay_content, replay_reasoning

        sse_chunks = [
            chunk
            async for chunk in stream_chat_sse(
                llm_stream=replay_stream(),
                message_id=assistant_message_id,
                model=model,
                context_usage=context_usage,
            )
        ]
        return sse_chunks, content_parts, reasoning_parts

    async def _stream_adapter_chunks(
        self,
        *,
        llm_adapter: LLMAdapter,
        llm_messages: list[dict[str, Any]],
        llm_kwargs: dict[str, Any],
        assistant_message_id: str,
        model: str,
        context_usage: dict[str, Any],
        chat_span: Span,
    ) -> tuple[list[str], list[str], list[str], dict[str, Any] | None]:
        llm_span = Span(
            name="llm.call",
            parent=chat_span,
            span_type=SpanType.LLM,
            model=model,
            attributes={
                "llm.model": model,
                "llm.message_count": len(llm_messages),
            },
        )

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        stream_started_at = time.perf_counter()
        first_token_ms: float | None = None
        llm_stream = llm_adapter.stream_chat(
            messages=cast(list[LLMMessage | dict[str, Any]], llm_messages),
            model=model,
            **llm_kwargs,
        )

        async def accumulating_stream() -> AsyncIterator[tuple[str, str | None]]:
            nonlocal first_token_ms
            async for chunk in llm_stream:
                content_delta = chunk.content_delta
                reasoning_delta = chunk.reasoning_delta
                if first_token_ms is None and (content_delta or reasoning_delta):
                    first_token_ms = (time.perf_counter() - stream_started_at) * 1000
                    llm_span.set_attribute("chat.first_token_ms", round(first_token_ms, 2))
                if content_delta:
                    content_parts.append(content_delta)
                if reasoning_delta:
                    reasoning_parts.append(reasoning_delta)
                yield content_delta, reasoning_delta

        sse_chunks = [
            chunk
            async for chunk in stream_chat_sse(
                llm_stream=accumulating_stream(),
                message_id=assistant_message_id,
                model=model,
                context_usage=context_usage,
            )
        ]

        usage_payload = _json_safe(getattr(llm_adapter, "last_usage", None))
        llm_span.set_attribute(
            "chat.stream_total_ms",
            round((time.perf_counter() - stream_started_at) * 1000, 2),
        )
        llm_span.set_attribute("chat.stream_chunk_count", len(sse_chunks))
        if first_token_ms is None:
            llm_span.set_attribute("chat.first_token_ms", None)
        if isinstance(usage_payload, dict) and usage_payload:
            llm_span.set_tokens(
                input_tokens=int(usage_payload.get("prompt_tokens", 0) or 0),
                output_tokens=int(usage_payload.get("completion_tokens", 0) or 0),
            )
        llm_span.set_status("ok")
        llm_span.end()
        return sse_chunks, content_parts, reasoning_parts, usage_payload

    async def _persist_assistant_message(
        self,
        *,
        conversation_id: str,
        conv_lock: Any,
        assistant_msg: Message,
        content_parts: list[str],
        reasoning_parts: list[str],
        persisted_tool_messages: list[Message],
        usage_payload: dict[str, Any] | None,
        chat_span: Span,
    ) -> bool:
        assistant_msg.content = "".join(content_parts)
        if reasoning_parts:
            assistant_msg.reasoning_content = "".join(reasoning_parts)

        if not (assistant_msg.content or assistant_msg.reasoning_content):
            chat_span.set_status("error", "LLM returned no content")
            logger.warning(
                "Chat response empty (LLM error): conversation=%s, message=%s — not persisted",
                conversation_id,
                assistant_msg.message_id,
            )
            return False

        async with conv_lock:
            if isinstance(usage_payload, dict) and usage_payload:
                assistant_msg.metadata["usage"] = usage_payload
            assistant_msg.metadata["trace_id"] = chat_span.trace_id

            conversation = self.json_store.get(conversation_id)
            if conversation is not None:
                if persisted_tool_messages:
                    conversation.messages.extend(persisted_tool_messages)
                conversation.messages.append(assistant_msg)
                conversation.updated_at = time.time()
                self.json_store.update(conversation)

        chat_span.set_attribute("chat.response_content_len", len(assistant_msg.content))
        chat_span.set_status("ok")
        logger.info(
            "Chat response complete: conversation=%s, message=%s, content_len=%d",
            conversation_id,
            assistant_msg.message_id,
            len(assistant_msg.content),
        )
        return True

    async def send_message(
        self,
        conversation_id: str,
        request: SendMessageRequest,
    ) -> AsyncIterator[str]:
        """Send a user message and stream the assistant response as SSE.

        Flow:
        1. Load conversation from store
        2. Append user message
        3. Build context plan (TokenEstimator → ContextPlanner → ContextRenderer)
        4. Call LLM with rendered messages
        5. Stream response as SSE events
        6. On completion, persist assistant message

        Args:
            conversation_id: Target conversation.
            request: User message and optional overrides.

        Yields:
            SSE-encoded event strings.

        Raises:
            FileNotFoundError: If conversation does not exist.
        """
        # --- Observability: create chat.request root span ---
        chat_span = Span(
            name="chat.request",
            parent=TraceContext.current(),
            span_type=SpanType.NODE,
            attributes={
                "chat.conversation_id": conversation_id,
                "chat.user_content_len": len(request.content),
            },
        )
        chat_token = TraceContext.push(chat_span)

        try:
            with _stage_span(chat_span, "chat.prepare"):
                prepared = await self._prepare_send_context(
                    conversation_id=conversation_id,
                    request=request,
                    chat_span=chat_span,
                )

            assistant_msg = Message(role=MessageRole.ASSISTANT, content="")
            llm_messages = prepared.llm_messages
            context_usage = prepared.context_usage
            llm_adapter = self._get_adapter_for_model(prepared.model)

            resolved_chat_skills = self._resolve_enabled_chat_skills(request)
            tool_gate = self._gate_tool_loop(
                request=request,
                resolved_skills=resolved_chat_skills,
            )
            enabled_chat_skills = tool_gate.enabled_skills

            # Keep an early assistant anchor only when tool loop is enabled,
            # so tool_call.* events can consistently reference one message id.
            if enabled_chat_skills:
                yield SSEEvent(
                    event="message.delta",
                    data={
                        "message_id": assistant_msg.message_id,
                        "seq": 0,
                        "content": "",
                    },
                ).encode()

            with _stage_span(
                chat_span,
                "chat.tool_loop",
                attributes={
                    "chat.enabled_skill_count": len(enabled_chat_skills),
                    "chat.tool_loop.mode": tool_gate.mode,
                    "chat.tool_loop.gating_reason": tool_gate.reason,
                    "chat.tool_loop.strategy": request.tool_call_strategy
                    or _TOOL_CALL_STRATEGY_BALANCED,
                },
            ) as tool_loop_span:
                tool_outcome = await self._run_tool_loop(
                    llm_adapter=llm_adapter,
                    model=prepared.model,
                    llm_messages=llm_messages,
                    llm_kwargs=prepared.llm_kwargs,
                    request=request,
                    assistant_message_id=assistant_msg.message_id,
                    trace_id=chat_span.trace_id,
                    enabled_chat_skills=enabled_chat_skills,
                    parent_span=tool_loop_span,
                )
                final_stream_skipped = tool_outcome.replay_response is not None
                tool_loop_span.set_attribute(
                    "chat.tool_loop.final_stream_skipped", final_stream_skipped
                )
                tool_loop_span.set_attribute(
                    "chat.tool_loop.convergence_reason",
                    tool_outcome.convergence_reason
                    if final_stream_skipped
                    else "needs_final_stream",
                )
            for event_chunk in tool_outcome.event_chunks:
                yield event_chunk
            llm_messages = tool_outcome.llm_messages
            persisted_tool_messages = tool_outcome.persisted_tool_messages

            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            usage_payload: dict[str, Any] | None = None
            if tool_outcome.replay_response is not None:
                with _stage_span(chat_span, "chat.stream.replay"):
                    (
                        replay_chunks,
                        replay_content_parts,
                        replay_reasoning_parts,
                    ) = await self._stream_replay_chunks(
                        replay_response=tool_outcome.replay_response,
                        assistant_message_id=assistant_msg.message_id,
                        model=prepared.model,
                        context_usage=context_usage,
                    )
                    for sse_chunk in replay_chunks:
                        yield sse_chunk
                    content_parts = replay_content_parts
                    reasoning_parts = replay_reasoning_parts
                usage_payload = tool_outcome.usage_payload
            else:
                with _stage_span(chat_span, "chat.stream.llm"):
                    llm_span = Span(
                        name="llm.call",
                        parent=chat_span,
                        span_type=SpanType.LLM,
                        model=prepared.model,
                        attributes={
                            "llm.model": prepared.model,
                            "llm.message_count": len(llm_messages),
                        },
                    )
                    stream_started_at = time.perf_counter()
                    first_token_ms: float | None = None
                    llm_chunk_count = 0
                    llm_stream = llm_adapter.stream_chat(
                        messages=cast(list[LLMMessage | dict[str, Any]], llm_messages),
                        model=prepared.model,
                        **prepared.llm_kwargs,
                    )

                    async def accumulating_stream() -> AsyncIterator[tuple[str, str | None]]:
                        nonlocal first_token_ms
                        async for chunk in llm_stream:
                            content_delta = chunk.content_delta
                            reasoning_delta = chunk.reasoning_delta
                            if first_token_ms is None and (content_delta or reasoning_delta):
                                first_token_ms = (time.perf_counter() - stream_started_at) * 1000
                                llm_span.set_attribute(
                                    "chat.first_token_ms", round(first_token_ms, 2)
                                )
                            if content_delta:
                                content_parts.append(content_delta)
                            if reasoning_delta:
                                reasoning_parts.append(reasoning_delta)
                            yield content_delta, reasoning_delta

                    async for sse_chunk in stream_chat_sse(
                        llm_stream=accumulating_stream(),
                        message_id=assistant_msg.message_id,
                        model=prepared.model,
                        context_usage=context_usage,
                    ):
                        llm_chunk_count += 1
                        yield sse_chunk

                    usage_payload = _json_safe(getattr(llm_adapter, "last_usage", None))
                    llm_span.set_attribute(
                        "chat.stream_total_ms",
                        round((time.perf_counter() - stream_started_at) * 1000, 2),
                    )
                    llm_span.set_attribute("chat.stream_chunk_count", llm_chunk_count)
                    if first_token_ms is None:
                        llm_span.set_attribute("chat.first_token_ms", None)
                    if isinstance(usage_payload, dict) and usage_payload:
                        llm_span.set_tokens(
                            input_tokens=int(usage_payload.get("prompt_tokens", 0) or 0),
                            output_tokens=int(usage_payload.get("completion_tokens", 0) or 0),
                        )
                    llm_span.set_status("ok")
                    llm_span.end()

            completion_metadata: dict[str, Any] = {"trace_id": chat_span.trace_id}
            if isinstance(usage_payload, dict) and usage_payload:
                completion_metadata["usage"] = usage_payload
            yield SSEEvent(
                event="message.complete",
                data={
                    "message_id": assistant_msg.message_id,
                    "metadata": completion_metadata,
                },
            ).encode()

            with _stage_span(chat_span, "chat.persist"):
                await self._persist_assistant_message(
                    conversation_id=conversation_id,
                    conv_lock=prepared.conv_lock,
                    assistant_msg=assistant_msg,
                    content_parts=content_parts,
                    reasoning_parts=reasoning_parts,
                    persisted_tool_messages=persisted_tool_messages,
                    usage_payload=usage_payload,
                    chat_span=chat_span,
                )

        except Exception as e:
            chat_span.set_status("error", str(e))
            raise
        finally:
            chat_span.end()
            _persist_trace_tree(chat_span)
            # GeneratorExit during async streaming may cause cleanup
            # in a different asyncio Context, making ContextVar.reset()
            # fail. This is safe to ignore — the span is already ended.
            with contextlib.suppress(ValueError):
                TraceContext.pop(chat_token)

    async def edit_message(
        self,
        conversation_id: str,
        message_id: str,
        request: EditMessageRequest,
    ) -> Message:
        """Edit a message's content.

        Only user messages can be edited. Updates the message in place
        and persists the conversation.

        Args:
            conversation_id: Target conversation.
            message_id: Message to edit.
            request: New content.

        Returns:
            The updated Message.

        Raises:
            FileNotFoundError: If conversation not found.
            ValueError: If message not found or not a user message.
        """
        conv_lock = await self.json_store.lock(conversation_id)
        async with conv_lock:
            conversation = self.json_store.get(conversation_id)
            if conversation is None:
                raise FileNotFoundError(f"Conversation {conversation_id} not found")

            msg = next((m for m in conversation.messages if m.message_id == message_id), None)
            if msg is None:
                raise ValueError(f"Message {message_id} not found")
            if msg.role != MessageRole.USER:
                raise ValueError("Only user messages can be edited")

            msg.content = request.content
            msg.metadata["edited"] = True
            msg.metadata["edited_at"] = time.time()
            conversation.updated_at = time.time()
            self.json_store.update(conversation)
            return msg

    async def delete_message(
        self,
        conversation_id: str,
        message_id: str,
    ) -> None:
        """Delete a single message from a conversation.

        Args:
            conversation_id: Target conversation.
            message_id: Message to delete.

        Raises:
            FileNotFoundError: If conversation not found.
            ValueError: If message not found.
        """
        conv_lock = await self.json_store.lock(conversation_id)
        async with conv_lock:
            conversation = self.json_store.get(conversation_id)
            if conversation is None:
                raise FileNotFoundError(f"Conversation {conversation_id} not found")

            message_index = next(
                (
                    idx
                    for idx, item in enumerate(conversation.messages)
                    if item.message_id == message_id
                ),
                None,
            )
            if message_index is None:
                raise ValueError(f"Message {message_id} not found")

            target_message = conversation.messages[message_index]
            to_remove_ids = {message_id}
            if target_message.role == MessageRole.ASSISTANT:
                removed_tool_call_ids: set[str] = set()
                cursor = message_index - 1
                while cursor >= 0 and conversation.messages[cursor].role == MessageRole.TOOL:
                    tool_msg = conversation.messages[cursor]
                    to_remove_ids.add(tool_msg.message_id)
                    if tool_msg.tool_call_id:
                        removed_tool_call_ids.add(str(tool_msg.tool_call_id))
                    cursor -= 1

                if cursor >= 0:
                    carrier = conversation.messages[cursor]
                    if carrier.role == MessageRole.ASSISTANT and carrier.tool_calls:
                        carrier_call_ids = {
                            str(call.get("id"))
                            for call in carrier.tool_calls
                            if isinstance(call, dict) and call.get("id")
                        }
                        if not removed_tool_call_ids or bool(
                            carrier_call_ids & removed_tool_call_ids
                        ):
                            to_remove_ids.add(carrier.message_id)

            conversation.messages = [
                item for item in conversation.messages if item.message_id not in to_remove_ids
            ]

            conversation.updated_at = time.time()
            self.json_store.update(conversation)

    async def regenerate_message(
        self,
        conversation_id: str,
        message_id: str,
    ) -> AsyncIterator[str]:
        """Regenerate an assistant message.

        Removes the target assistant message and all subsequent messages,
        then re-sends the last user message to get a fresh response.

        Args:
            conversation_id: Target conversation.
            message_id: Assistant message to regenerate.

        Yields:
            SSE-encoded event strings (same as send_message).

        Raises:
            FileNotFoundError: If conversation not found.
            ValueError: If message not found or not an assistant message.
        """
        conv_lock = await self.json_store.lock(conversation_id)
        async with conv_lock:
            conversation = self.json_store.get(conversation_id)
            if conversation is None:
                raise FileNotFoundError(f"Conversation {conversation_id} not found")

            # Find the target message index
            msg_idx = next(
                (i for i, m in enumerate(conversation.messages) if m.message_id == message_id),
                None,
            )
            if msg_idx is None:
                raise ValueError(f"Message {message_id} not found")
            if conversation.messages[msg_idx].role != MessageRole.ASSISTANT:
                raise ValueError("Only assistant messages can be regenerated")

            # Find the last user message before this assistant message
            last_user_content = None
            for i in range(msg_idx - 1, -1, -1):
                if conversation.messages[i].role == MessageRole.USER:
                    last_user_content = conversation.messages[i].content
                    break

            if last_user_content is None:
                raise ValueError("No user message found before the assistant message")

            # Remove the assistant message and everything after it
            conversation.messages = conversation.messages[:msg_idx]
            conversation.updated_at = time.time()
            self.json_store.update(conversation)

        # Re-send the last user message (send_message will append it again)
        # We need to remove the last user message too since send_message will re-add it
        conv_lock2 = await self.json_store.lock(conversation_id)
        async with conv_lock2:
            conversation = self.json_store.get(conversation_id)
            if (
                conversation
                and conversation.messages
                and conversation.messages[-1].role == MessageRole.USER
            ):
                conversation.messages.pop()
                conversation.updated_at = time.time()
                self.json_store.update(conversation)

        request = SendMessageRequest(content=last_user_content)
        async for chunk in self.send_message(conversation_id, request):
            yield chunk
