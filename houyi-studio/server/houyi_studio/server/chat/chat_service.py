"""Chat Service: orchestrates SDK components for chat interactions.

Bridges Studio Server with Context Engine, Memory Engine, and LLM Adapter.
Owns the business logic for sending messages, managing context, and streaming.

"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

from houyi.adapters.llm import (
    DEFAULT_MODEL,
    LLMAdapter,
    LLMAdapterFactory,
    SiliconFlowAdapter,
    create_vertex_adapter,
)
from houyi.adapters.llm.openai_compat_adapter import OpenAICompatibleAdapter
from houyi.adapters.memory import MemoryStore
from houyi.application.context.context_lifecycle import (
    ContextLifecycleHookService as ChatContextHookService,
)
from houyi.application.context.stream_metrics import (
    build_generation_metadata,
    normalize_usage_payload,
)
from houyi.application.context.token_budget_policy import TokenBudgetPolicy
from houyi.application.context.token_estimator import TokenEstimator
from houyi.application.tool_calling.runner import ToolCallRunner
from houyi.application.tool_calling.runtime_options import build_chat_kwargs
from houyi.application.tool_calling.tool_bridge import ToolBridge
from houyi.application.workflow.skill_executor import SkillExecutor
from houyi.domain.skill.registry import DEFAULT_SKILL_REGISTRY
from houyi.infrastructure.config import (
    ENV_CHAT_TOOL_LOOP_MAX_MESSAGE_CHARS,
    ENV_CHAT_TOOL_LOOP_MAX_TOTAL_CHARS,
)
from houyi.infrastructure.observability import (
    Span,
    SpanSchema,
    SpanStatus,
    SpanType,
    TraceContext,
    get_storage,
)

from ..skill.service import get_skill_service
from .assistant_response_streamer import (
    AssistantResponseStreamer,
    FinalStreamCapture,
    ReplayStreamCapture,
)
from .assistant_turn_persistence import AssistantTurnPersistence
from .chat_context_adapter import ChatContextAdapter
from .chat_errors import (
    build_public_stream_error_message,
    build_stream_error_content,
    normalize_chat_error,
)
from .chat_request_preparation import ChatRequestPreparation, PreparedSendContext
from .chat_tool_loop_policy import ChatToolLoopPolicy
from .context_compressor import SummaryBuildResult
from .conversation_compaction_coordinator import ConversationCompactionCoordinator
from .conversation_context_adapter import ConversationContextAdapter
from .conversation_context_state_updater import ConversationContextStateUpdater
from .conversation_message_manager import ConversationMessageManager
from .conversation_streaming_state_manager import ConversationStreamingStateManager
from .json_store import JsonStore
from .llm_request_options_resolver import LLMRequestOptionsResolver
from .model_adapter_resolver import ModelAdapterResolver
from .pinned_context_store import PinnedContextStore
from .provider_service import _is_vertex_provider
from .sse_adapter import SSEEvent
from .tool_loop_orchestrator import ToolLoopOrchestrator
from .types import (
    Conversation,
    CreateConversationRequest,
    EditMessageRequest,
    Message,
    MessageRole,
    SendMessageRequest,
    UpdateConversationRequest,
)

logger = logging.getLogger(__name__)
_TOKEN_BUDGET_POLICY = TokenBudgetPolicy(default_answer_reserve=512)

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
_ROLLING_CONTEXT_CAPACITY = max(
    1,
    int(os.getenv("HOUYI_CHAT_ROLLING_CONTEXT_CAPACITY", "272000") or "272000"),
)
_SUMMARY_MODEL_ENV = "HOUYI_CHAT_SUMMARY_MODEL"


def _finalize_stream_result(
    *,
    llm_adapter: Any,
    llm_span: Span,
    first_token_ms: float | None,
    generation_time_ms: float,
    chunk_count: int,
    finish_reason_sources: tuple[Any, ...] = (),
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any]]:
    usage_payload = normalize_usage_payload(
        _json_safe(getattr(llm_adapter, "last_usage", None)),
        first_token_ms=first_token_ms,
        generation_time_ms=generation_time_ms,
    )
    finish_reason = _extract_finish_reason(
        getattr(llm_adapter, "last_finish_reason", None),
        *finish_reason_sources,
    )
    llm_span.set_attribute("chat.stream_total_ms", round(generation_time_ms, 2))
    llm_span.set_attribute("chat.stream_chunk_count", chunk_count)
    if first_token_ms is None:
        llm_span.set_attribute("chat.first_token_ms", None)
    if isinstance(usage_payload, dict) and usage_payload:
        llm_span.set_tokens(
            input_tokens=int(usage_payload.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage_payload.get("completion_tokens", 0) or 0),
        )
    llm_span.set_status("ok")
    llm_span.end()
    generation_metadata = build_generation_metadata(
        usage_payload=usage_payload,
        first_token_ms=first_token_ms,
        generation_time_ms=generation_time_ms,
    )
    return usage_payload, finish_reason, generation_metadata


def _apply_reasoning_budget_guardrail(llm_kwargs: dict[str, Any]) -> dict[str, Any] | None:
    if not llm_kwargs.get("enable_reasoning"):
        return None

    explicit_max_tokens = llm_kwargs.get("max_tokens")
    if explicit_max_tokens is None:
        return {
            "reasoning_enabled": True,
            "max_tokens_guardrail_applied": False,
            "answer_reserve": _REASONING_MIN_ANSWER_RESERVE,
            "max_tokens_effective": None,
            "max_tokens_source": "provider_default",
        }

    effective_max_tokens = int(explicit_max_tokens or 0)
    guardrail_applied = False
    if effective_max_tokens < _REASONING_MIN_ANSWER_RESERVE:
        effective_max_tokens = _REASONING_MIN_ANSWER_RESERVE
        llm_kwargs["max_tokens"] = effective_max_tokens
        guardrail_applied = True

    return {
        "reasoning_enabled": True,
        "max_tokens_guardrail_applied": guardrail_applied,
        "answer_reserve": _REASONING_MIN_ANSWER_RESERVE,
        "max_tokens_effective": effective_max_tokens,
        "max_tokens_source": "request_or_conversation",
    }


def _apply_budget_policy(
    *,
    model: str,
    request: SendMessageRequest,
    conversation: Any,
    llm_kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    estimator = TokenEstimator(model=model)
    requested_max_tokens = llm_kwargs.get("max_tokens")
    decision = _TOKEN_BUDGET_POLICY.decide(
        context_window=estimator.context_window,
        enable_reasoning=bool(llm_kwargs.get("enable_reasoning")),
        requested_max_tokens=(
            int(requested_max_tokens) if requested_max_tokens is not None else None
        ),
    )

    max_tokens_guardrail_applied = False
    max_tokens_effective = decision.max_tokens_to_send
    max_tokens_source = decision.max_tokens_source
    if llm_kwargs.get("enable_reasoning"):
        if max_tokens_effective is None:
            max_tokens_source = "provider_default"
        elif max_tokens_effective < _REASONING_MIN_ANSWER_RESERVE:
            max_tokens_effective = _REASONING_MIN_ANSWER_RESERVE
            max_tokens_guardrail_applied = True
            max_tokens_source = "request_or_conversation"
            decision.output_budget = max_tokens_effective
            decision.input_budget = max(
                0,
                decision.context_window - decision.output_budget - decision.tool_reserve,
            )
            decision.answer_reserve = _REASONING_MIN_ANSWER_RESERVE
            decision.reasoning_budget = 0
            decision.max_tokens_to_send = max_tokens_effective

    if decision.should_set_max_tokens and max_tokens_effective is not None:
        llm_kwargs["max_tokens"] = max_tokens_effective
    else:
        llm_kwargs.pop("max_tokens", None)

    budget_metadata = decision.model_dump(mode="json")
    if llm_kwargs.get("enable_reasoning"):
        budget_metadata.update(
            {
                "reasoning_enabled": True,
                "max_tokens_guardrail_applied": max_tokens_guardrail_applied,
                "answer_reserve": _REASONING_MIN_ANSWER_RESERVE,
                "max_tokens_effective": max_tokens_effective,
                "max_tokens_source": max_tokens_source,
            }
        )
    return budget_metadata


def is_vision_model(model: str | None) -> bool:
    if not model:
        return False
    return bool(_VISION_RE.search(model))


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
_REASONING_MIN_ANSWER_RESERVE = 512
_CHAT_BUILTIN_TOOL_NAMES = frozenset(
    {
        "houyi_read_file",
        "houyi_write_file",
        "houyi_find_files",
        "houyi_list_dir",
        "houyi_grep",
    }
)
_CHAT_EXPLICIT_TOOL_NAMES = frozenset(
    {
        "houyi_shell_exec",
    }
)
_WEB_SEARCH_SKILL_NAME = "houyi_web_search"
_TOOL_CALL_STRATEGY_CONSERVATIVE = "conservative"
_TOOL_CALL_STRATEGY_BALANCED = "balanced"
_TOOL_CALL_STRATEGY_AGGRESSIVE = "aggressive"
_TOOL_INTENT_KEYWORDS = (
    "grep",
    "find",
    "read file",
    "write file",
    "list dir",
    "terminal",
    "shell",
    "codebase",
    "run command",
    "execute command",
    "open file",
    "save file",
)
_TOOL_INTENT_REGEXES = (
    re.compile(r"`[^`]+`"),
    re.compile(r"(?:^|\s)/(?:Users|home|var|tmp|opt|etc)/"),
    re.compile(r"(?:^|\s)\./[\w./-]+"),
    re.compile(r"(?:^|\s)\.{2}/[\w./-]+"),
    re.compile(r"\b[a-zA-Z0-9_./-]+\.(?:py|ts|tsx|js|jsx|md|json|yaml|yml|toml|sql|sh|txt)\b"),
)
_REPO_INTENT_KEYWORDS = (
    "github",
    "repo",
    "repository",
    "readme",
    "read me",
)
_WEB_INTENT_KEYWORDS = (
    "search web",
    "search the web",
    "web search",
    "online search",
    "browse web",
    "browse the web",
    "search online",
    "look up",
    "lookup",
    "google",
    "duckduckgo",
    "news",
    "latest",
    "recent",
    "\u4e0a\u7f51\u641c\u7d22",
    "\u4e0a\u7f51\u641c\u7d20",
    "\u8054\u7f51\u641c\u7d22",
    "\u8054\u7f51\u641c\u7d20",
    "\u7f51\u4e0a\u641c\u7d22",
    "\u7f51\u4e0a\u641c\u7d20",
    "\u5728\u7ebf\u641c\u7d22",
    "\u5728\u7ebf\u641c\u7d20",
    "\u7f51\u7edc\u641c\u7d22",
    "\u7f51\u7edc\u641c\u7d20",
    "\u4e0a\u7f51\u67e5",
    "\u8054\u7f51\u67e5",
    "\u7f51\u4e0a\u67e5",
    "\u5728\u7ebf\u67e5",
    "\u641c\u4e00\u4e0b",
    "\u641c\u7d20\u4e00\u4e0b",
    "\u67e5\u4e00\u4e0b",
    "\u6700\u65b0",
    "\u8fd1\u671f",
)
_REPO_INTENT_REGEXES = (
    re.compile(r"https?://(?:www\.)?github\.com/[^\s]+", re.IGNORECASE),
    re.compile(r"\bgithub\.com/[^\s]+\b", re.IGNORECASE),
)
_REPO_TOOL_SKILLS = frozenset(
    {
        "houyi_list_dir",
        "houyi_find_files",
        "houyi_grep",
    }
)
_WEB_TOOL_SKILLS = frozenset({_WEB_SEARCH_SKILL_NAME})


@dataclass
class _ToolLoopGateDecision:
    enabled_skills: list[str]
    mode: str
    reason: str


@dataclass(frozen=True)
class _ChatRuntimeProfile:
    name: str
    keep_n: int | None
    low_watermark: float
    compression_threshold: float
    overflow_threshold: float
    cooldown_messages: int
    cooldown_seconds: float
    tool_result_max_tokens: int | None
    per_tool_quota: dict[str, int] | None


_CHAT_DEFAULT_PROFILE = _ChatRuntimeProfile(
    name="chat.default",
    keep_n=None,
    low_watermark=0.6,
    compression_threshold=0.7,
    overflow_threshold=0.9,
    cooldown_messages=4,
    cooldown_seconds=30.0,
    tool_result_max_tokens=None,
    per_tool_quota=None,
)
_DEEP_RESEARCH_PROFILE = _ChatRuntimeProfile(
    name="agent.deep_research",
    keep_n=5,
    low_watermark=0.55,
    compression_threshold=0.6,
    overflow_threshold=0.85,
    cooldown_messages=2,
    cooldown_seconds=15.0,
    tool_result_max_tokens=4096,
    per_tool_quota={"search": 50, "read": 100, "exec": 20},
)


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


def _extract_finish_reason(*sources: Any) -> str | None:
    for source in sources:
        if source is None:
            continue
        if isinstance(source, str) and source:
            return source
        if isinstance(source, dict):
            reason = source.get("finish_reason")
            if isinstance(reason, str) and reason:
                return reason
            metadata = source.get("metadata")
            if isinstance(metadata, dict):
                nested = metadata.get("finish_reason")
                if isinstance(nested, str) and nested:
                    return nested
            continue
        reason = getattr(source, "finish_reason", None)
        if isinstance(reason, str) and reason:
            return reason
        metadata = getattr(source, "metadata", None)
        if isinstance(metadata, dict):
            nested = metadata.get("finish_reason")
            if isinstance(nested, str) and nested:
                return nested
    return None


def _looks_like_tool_intent(user_content: str) -> bool:
    lowered = user_content.lower()
    if any(keyword in lowered for keyword in _TOOL_INTENT_KEYWORDS):
        return True
    if "```" in user_content:
        return True
    return any(regex.search(user_content) for regex in _TOOL_INTENT_REGEXES)


def _looks_like_repo_intent(user_content: str) -> bool:
    lowered = user_content.lower()
    if any(regex.search(user_content) for regex in _REPO_INTENT_REGEXES):
        return True
    return any(keyword in lowered for keyword in _REPO_INTENT_KEYWORDS)


def _looks_like_web_intent(user_content: str) -> bool:
    lowered = user_content.lower()
    if _looks_like_repo_intent(user_content):
        return True
    if "http://" in lowered or "https://" in lowered:
        return True
    if any(
        phrase in lowered
        for phrase in (
            "summarize this url",
            "summarize this link",
            "总结这个链接",
            "总结这个网页",
            "读取这个链接",
            "打开这个链接",
            "访问这个链接",
        )
    ):
        return True
    return any(keyword in lowered for keyword in _WEB_INTENT_KEYWORDS)


def _is_deep_research_enabled(request: SendMessageRequest) -> bool:
    if request.enable_deep_research is True:
        return True
    return any(
        isinstance(skill_name, str) and skill_name.strip() == "deep_research"
        for skill_name in (request.enable_skills or [])
    )


def _truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text.strip()
    if max_chars <= len("\n...[truncated]...\n"):
        return text[:max_chars].strip()
    half = max(1, (max_chars - len("\n...[truncated]...\n")) // 2)
    return f"{text[:half].rstrip()}\n...[truncated]...\n{text[-half:].lstrip()}".strip()


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
    messages: list[dict[str, Any]],
    max_total_chars: int,
) -> list[dict[str, Any]]:
    total_chars = sum(_message_budget_chars(message) for message in messages)
    if total_chars <= max_total_chars:
        return messages

    capped = list(messages)
    while total_chars > max_total_chars:
        drop_index = next(
            (index for index, message in enumerate(capped) if message.get("role") != "system"),
            None,
        )
        if drop_index is None:
            break
        dropped = capped.pop(drop_index)
        total_chars -= _message_budget_chars(dropped)
    return capped


def _sanitize_tool_loop_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy with string ``content`` fields for tool-loop chat calls."""
    base_sanitized = LLMAdapter._sanitize_messages(messages)
    sanitized: list[dict[str, Any]] = []
    for original, msg in zip(messages, base_sanitized, strict=False):
        normalized = dict(msg)
        has_original_content = isinstance(original, dict) and "content" in original
        if has_original_content:
            normalized["content"] = _truncate_middle(
                _coerce_text_content(normalized.get("content")),
                _TOOL_LOOP_MAX_MESSAGE_CHARS,
            )
        else:
            normalized.pop("content", None)
        normalized = _truncate_tool_call_arguments(normalized, _TOOL_LOOP_MAX_MESSAGE_CHARS)
        sanitized.append(normalized)
    capped = _cap_total_payload_chars(sanitized, _TOOL_LOOP_MAX_TOTAL_CHARS)
    return capped


def _sanitize_tool_loop_structure(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized = _sanitize_tool_loop_messages(messages)
    normalized: list[dict[str, Any]] = []
    pending_carrier: dict[str, Any] | None = None
    pending_steps: list[dict[str, Any]] = []
    pending_ids: set[str] = set()
    resolved_ids: set[str] = set()

    def flush_pending() -> None:
        nonlocal pending_carrier, pending_steps, pending_ids, resolved_ids
        if pending_carrier is not None and (not pending_ids or pending_ids.issubset(resolved_ids)):
            normalized.append(pending_carrier)
            normalized.extend(pending_steps)
        pending_carrier = None
        pending_steps = []
        pending_ids = set()
        resolved_ids = set()

    for message in sanitized:
        role = message.get("role")
        tool_calls = message.get("tool_calls")
        if role == MessageRole.ASSISTANT.value and isinstance(tool_calls, list) and tool_calls:
            flush_pending()
            pending_carrier = message
            pending_steps = []
            pending_ids = {
                str(call.get("id"))
                for call in tool_calls
                if isinstance(call, dict) and isinstance(call.get("id"), str) and call.get("id")
            }
            resolved_ids = set()
            continue

        if role == MessageRole.TOOL.value and pending_carrier is not None:
            tool_call_id = message.get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id:
                resolved_ids.add(tool_call_id)
            pending_steps.append(message)
            continue

        flush_pending()
        normalized.append(message)

    flush_pending()
    return normalized


def _sanitize_context_history_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized = _sanitize_tool_loop_structure(messages)
    normalized: list[dict[str, Any]] = []
    for message in sanitized:
        role = str(message.get("role") or "")
        if role == MessageRole.ASSISTANT.value and "reasoning_content" in message:
            cleaned = dict(message)
            cleaned.pop("reasoning_content", None)
            has_content = bool(str(cleaned.get("content") or "").strip())
            has_tool_calls = bool(cleaned.get("tool_calls"))
            if not has_content and not has_tool_calls:
                continue
            normalized.append(cleaned)
            continue
        normalized.append(dict(message))
    return normalized


def _sanitize_final_stream_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    sanitized = _sanitize_tool_loop_structure(messages)
    normalized: list[dict[str, Any]] = []
    stats = {
        "assistant_tool_call_carrier_count": 0,
        "assistant_reasoning_removed_count": 0,
        "assistant_reasoning_only_removed_count": 0,
        "tool_result_projection_count": 0,
    }
    for message in sanitized:
        role = str(message.get("role") or "")
        if role == MessageRole.ASSISTANT.value and message.get("tool_calls"):
            cleaned = dict(message)
            stats["assistant_tool_call_carrier_count"] += 1
            cleaned.pop("tool_calls", None)
            cleaned.pop("tool_call_id", None)
            if "reasoning_content" in cleaned:
                stats["assistant_reasoning_removed_count"] += 1
                if not str(cleaned.get("content") or "").strip():
                    stats["assistant_reasoning_only_removed_count"] += 1
            cleaned.pop("reasoning_content", None)
            if str(cleaned.get("content") or "").strip():
                normalized.append(cleaned)
            continue
        if role == MessageRole.TOOL.value:
            tool_name = str(message.get("name") or "tool")
            tool_content = str(message.get("content") or "")
            stats["tool_result_projection_count"] += 1
            normalized.append(
                {
                    "role": MessageRole.USER.value,
                    "content": f"[tool:{tool_name}] {tool_content}",
                }
            )
            continue
        if role == MessageRole.ASSISTANT.value and "reasoning_content" in message:
            cleaned = dict(message)
            stats["assistant_reasoning_removed_count"] += 1
            if not str(cleaned.get("content") or "").strip():
                stats["assistant_reasoning_only_removed_count"] += 1
            cleaned.pop("reasoning_content", None)
            normalized.append(cleaned)
            continue
        normalized.append(dict(message))
    return normalized, stats


def _summarize_message_shapes(messages: list[dict[str, Any]]) -> dict[str, int]:
    stats = {
        "message_count": len(messages),
        "assistant_message_count": 0,
        "assistant_reasoning_message_count": 0,
        "assistant_reasoning_only_message_count": 0,
        "assistant_tool_call_message_count": 0,
        "tool_message_count": 0,
        "user_message_count": 0,
    }
    for message in messages:
        role = str(message.get("role") or "")
        if role == MessageRole.ASSISTANT.value:
            stats["assistant_message_count"] += 1
            if (
                isinstance(message.get("reasoning_content"), str)
                and str(message.get("reasoning_content") or "").strip()
            ):
                stats["assistant_reasoning_message_count"] += 1
                if not str(message.get("content") or "").strip():
                    stats["assistant_reasoning_only_message_count"] += 1
            if isinstance(message.get("tool_calls"), list) and message.get("tool_calls"):
                stats["assistant_tool_call_message_count"] += 1
        elif role == MessageRole.TOOL.value:
            stats["tool_message_count"] += 1
        elif role == MessageRole.USER.value:
            stats["user_message_count"] += 1
    return stats


def _flatten_span_tree(root: Span) -> list[Span]:
    stack = [root]
    ordered: list[Span] = []
    while stack:
        current = stack.pop()
        ordered.append(current)
        if current.children:
            stack.extend(reversed(current.children))
    return ordered


def _build_empty_stream_content() -> str:
    return "The model returned an empty final response. Please retry."


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


class _NullHookSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        return None


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
        self._model_adapter_resolver = ModelAdapterResolver(
            get_settings_store=lambda: self._settings_store,
            get_default_adapter=lambda: self._default_adapter,
            is_vertex_provider=lambda provider_id, provider_url: _is_vertex_provider(
                provider_id, provider_url
            ),
            create_vertex_adapter=lambda: create_vertex_adapter(),
            openai_compat_adapter_cls=OpenAICompatibleAdapter,
            siliconflow_adapter_cls=SiliconFlowAdapter,
        )
        self._adapter_cache = self._model_adapter_resolver.adapter_cache
        self.pinned_context_store = PinnedContextStore(json_store=json_store)
        self._conversation_context = ConversationContextAdapter(
            json_store=json_store,
            default_model=self.default_model,
            rolling_capacity=_ROLLING_CONTEXT_CAPACITY,
            is_vision_model=is_vision_model,
        )
        self._context_state_updater = ConversationContextStateUpdater(
            conversation_context=self._conversation_context,
        )
        self._context_hooks = ChatContextHookService()
        self._context_runtime = ChatContextAdapter(
            memory_store=memory_store,
            is_vision_model=is_vision_model,
            sanitize_tool_loop_structure=_sanitize_context_history_messages,
            hook_service=self._context_hooks,
        )
        self._llm_request_options_resolver = LLMRequestOptionsResolver(
            apply_budget_policy=_apply_budget_policy,
        )
        self._tool_loop_policy = ChatToolLoopPolicy(
            builtin_tool_names=_CHAT_BUILTIN_TOOL_NAMES,
            explicit_tool_names=_CHAT_EXPLICIT_TOOL_NAMES,
            web_search_skill_name=_WEB_SEARCH_SKILL_NAME,
            repo_tool_skills=_REPO_TOOL_SKILLS,
            web_tool_skills=_WEB_TOOL_SKILLS,
            conservative_strategy=_TOOL_CALL_STRATEGY_CONSERVATIVE,
            balanced_strategy=_TOOL_CALL_STRATEGY_BALANCED,
            aggressive_strategy=_TOOL_CALL_STRATEGY_AGGRESSIVE,
            looks_like_repo_intent=_looks_like_repo_intent,
            looks_like_web_intent=_looks_like_web_intent,
            looks_like_tool_intent=_looks_like_tool_intent,
        )
        self._streaming_state_manager = ConversationStreamingStateManager(json_store=json_store)
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._compaction_coordinator = ConversationCompactionCoordinator(
            json_store=json_store,
            default_model=self.default_model,
            is_vision_model=is_vision_model,
            context_state_updater=self._context_state_updater,
            repo_intent_detector=_looks_like_repo_intent,
            hook_service=self._context_hooks,
            get_adapter_for_model=lambda model: self._get_adapter_for_model(model),
            resolve_summary_model=lambda model: self._resolve_summary_model(model),
            background_tasks=self._background_tasks,
        )
        self._context_compressor = self._compaction_coordinator.context_compressor
        self._request_preparation = self._build_request_preparation(json_store)
        self._tool_loop_orchestrator = self._build_tool_loop_orchestrator()
        self._response_streamer = self._build_response_streamer()
        self._turn_persistence = self._build_turn_persistence(json_store)
        self._message_manager = ConversationMessageManager(
            json_store=json_store,
            send_message=lambda conversation_id, request: self.send_message(
                conversation_id,
                request,
            ),
            context_state_updater=self._context_state_updater,
            default_model=self.default_model,
        )

    def build_initial_conversation_context_state(
        self,
        conversation_id: str,
        *,
        now: float | None = None,
    ):
        return self._conversation_context.build_initial_state(conversation_id, now=now)

    def describe_context_hook_contract(self) -> dict[str, dict[str, str]]:
        return self._context_hooks.describe_contract()

    def create_conversation(self, request: CreateConversationRequest) -> dict[str, Any]:
        conversation = Conversation(
            title=request.title,
            model=request.model,
            system_instructions=request.system_instructions,
            metadata=request.metadata,
        )
        conversation.conversation_context_state = self.build_initial_conversation_context_state(
            conversation.conversation_id,
            now=conversation.created_at,
        )
        created = self.json_store.create(conversation)
        return created.to_summary()

    def seed_messages(
        self,
        conversation_id: str,
        *,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        conversation = self.json_store.get(conversation_id)
        if conversation is None:
            raise FileNotFoundError(f"Conversation {conversation_id} not found")
        for message_data in messages:
            conversation.messages.append(
                Message(
                    role=message_data["role"],
                    content=message_data["content"],
                )
            )
        conversation.updated_at = max(conversation.updated_at, conversation.created_at)
        self.ensure_conversation_context_state(conversation, persist=False)
        self.json_store.update(conversation)
        return {"seeded": len(messages)}

    def list_conversations(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        conversations = self.json_store.list_conversations(
            status=status,
            limit=limit,
            offset=offset,
        )
        total = self.json_store.count(status=status)
        return {
            "conversations": conversations,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        conversation = self.json_store.get(conversation_id)
        if conversation is None:
            raise FileNotFoundError(f"Conversation {conversation_id} not found")
        self.ensure_conversation_context_state(
            conversation,
            model=conversation.model or self.default_model,
            persist=True,
        )
        return conversation.model_dump(mode="json")

    def update_conversation(
        self,
        conversation_id: str,
        request: UpdateConversationRequest,
    ) -> dict[str, Any]:
        conversation = self.json_store.get(conversation_id)
        if conversation is None:
            raise FileNotFoundError(f"Conversation {conversation_id} not found")
        if request.title is not None:
            conversation.title = request.title
        if request.status is not None:
            conversation.status = request.status
        if request.system_instructions is not None:
            conversation.system_instructions = request.system_instructions
        if request.model is not None:
            conversation.model = request.model
        raw_body = request.model_dump(exclude_unset=True)
        if "temperature" in raw_body:
            conversation.temperature = request.temperature
        if "max_tokens" in raw_body:
            conversation.max_tokens = request.max_tokens
        if "top_p" in raw_body:
            conversation.top_p = request.top_p
        if "stream" in raw_body:
            conversation.stream = request.stream
        if request.bookmarked is not None:
            conversation.bookmarked = request.bookmarked
        updated = self.json_store.update(conversation)
        return updated.to_summary()

    def delete_conversation(self, conversation_id: str) -> dict[str, str]:
        deleted = self.json_store.delete(conversation_id)
        if not deleted:
            raise FileNotFoundError(f"Conversation {conversation_id} not found")
        return {"status": "deleted", "conversation_id": conversation_id}

    def toggle_message_bookmark(
        self,
        *,
        conversation_id: str,
        message_id: str,
        bookmarked: bool,
    ) -> dict[str, Any]:
        conversation = self.json_store.get(conversation_id)
        if conversation is None:
            raise FileNotFoundError(f"Conversation {conversation_id} not found")
        message = next(
            (item for item in conversation.messages if item.message_id == message_id), None
        )
        if message is None:
            raise ValueError(f"Message {message_id} not found")
        message.bookmarked = bookmarked
        self.json_store.update(conversation)
        return message.model_dump(mode="json")

    def ensure_conversation_context_state(
        self,
        conversation: Conversation,
        *,
        model: str | None = None,
        persist: bool = False,
    ):
        return self._conversation_context.ensure_state(
            conversation,
            model=model,
            persist=persist,
        )

    def _resolve_runtime_profile(self, request: SendMessageRequest) -> _ChatRuntimeProfile:
        return (
            _DEEP_RESEARCH_PROFILE if _is_deep_research_enabled(request) else _CHAT_DEFAULT_PROFILE
        )

    async def _set_active_streaming_state(
        self,
        *,
        conversation_id: str,
        conv_lock: Any,
        message_id: str,
        request_id: str,
        status: str = "streaming",
        started_at: float | None = None,
    ) -> None:
        await self._streaming_state_manager.set_active_streaming_state(
            conversation_id=conversation_id,
            conv_lock=conv_lock,
            message_id=message_id,
            request_id=request_id,
            status=status,
            started_at=started_at,
        )

    async def _clear_active_streaming_state(
        self,
        *,
        conversation_id: str,
        conv_lock: Any,
        message_id: str,
    ) -> None:
        await self._streaming_state_manager.clear_active_streaming_state(
            conversation_id=conversation_id,
            conv_lock=conv_lock,
            message_id=message_id,
        )

    def invalidate_adapter_cache(self) -> None:
        """Clear cached adapters. Call when provider settings change."""
        self._model_adapter_resolver.invalidate_adapter_cache()

    def _build_request_preparation(self, json_store: JsonStore) -> ChatRequestPreparation:
        return ChatRequestPreparation(
            json_store=json_store,
            default_model=self.default_model,
            default_system_instructions=self.default_system_instructions,
            conversation_context=self._conversation_context,
            context_state_updater=self._context_state_updater,
            resolve_llm_kwargs=self._resolve_llm_kwargs,
            resolve_runtime_profile=self._resolve_runtime_profile,
            context_compressor=self._context_compressor,
            build_context_messages=self._build_context_messages,
        )

    def _build_tool_loop_orchestrator(self) -> ToolLoopOrchestrator:
        return ToolLoopOrchestrator(
            default_chat_max_tool_iterations=_DEFAULT_CHAT_MAX_TOOL_ITERATIONS,
            get_tool_runner=lambda *args, **kwargs: self._get_tool_runner(*args, **kwargs),
            context_hooks=self._context_hooks,
            extract_finish_reason=_extract_finish_reason,
            json_safe=_json_safe,
            normalize_usage_payload=normalize_usage_payload,
            null_hook_span_factory=_NullHookSpan,
            sanitize_tool_loop_messages=_sanitize_tool_loop_messages,
            tool_bridge_factory=lambda: ToolBridge(DEFAULT_SKILL_REGISTRY),
            build_chat_kwargs=build_chat_kwargs,
            skill_executor_factory=lambda: SkillExecutor(max_retries=2, timeout=30.0),
            stage_span=_stage_span,
        )

    def _build_response_streamer(self) -> AssistantResponseStreamer:
        return AssistantResponseStreamer(
            build_stream_error_content=build_stream_error_content,
            build_public_stream_error_message=build_public_stream_error_message,
            build_empty_stream_content=_build_empty_stream_content,
            extract_finish_reason=_extract_finish_reason,
            finalize_stream_result=_finalize_stream_result,
            json_safe=_json_safe,
            normalize_chat_error=normalize_chat_error,
            normalize_usage_payload=normalize_usage_payload,
            stage_span=_stage_span,
        )

    def _build_turn_persistence(self, json_store: JsonStore) -> AssistantTurnPersistence:
        return AssistantTurnPersistence(
            json_store=json_store,
            context_state_updater=self._context_state_updater,
        )

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
        return self._tool_loop_policy.resolve_enabled_chat_skills(request)

    def _gate_tool_loop(
        self,
        *,
        request: SendMessageRequest,
        resolved_skills: list[str],
        context_usage: dict[str, Any] | None = None,
        runtime_profile: Any | None = None,
    ) -> _ToolLoopGateDecision:
        """Lightweight tool-loop gating for latency-sensitive enterprise chat."""
        decision = self._tool_loop_policy.gate_tool_loop(
            request=request,
            resolved_skills=resolved_skills,
        )
        used_tokens = None
        max_context_tokens = None
        if isinstance(context_usage, dict):
            used_tokens = context_usage.get("used_tokens")
            max_context_tokens = context_usage.get("max_context_tokens")
        try:
            utilization = (
                float(used_tokens) / float(max_context_tokens)
                if used_tokens is not None and max_context_tokens not in (None, 0)
                else None
            )
        except (TypeError, ValueError, ZeroDivisionError):
            utilization = None
        compression_threshold = getattr(runtime_profile, "compression_threshold", None)
        try:
            threshold = float(compression_threshold) if compression_threshold is not None else None
        except (TypeError, ValueError):
            threshold = None
        if (
            utilization is not None
            and threshold is not None
            and utilization >= threshold
            and decision.reason.startswith("heuristic_")
        ):
            if decision.reason == "heuristic_web_intent" and decision.enabled_skills:
                logger.info(
                    "Chat tool-loop gate: model=%s mode=%s reason=%s enabled_skills=%s strategy=%s context_utilization=%.4f compression_threshold=%.4f",
                    getattr(request, "model", None) or "(conversation-default)",
                    "enabled_under_pressure",
                    "context_pressure_preserve_web_search",
                    decision.enabled_skills,
                    request.tool_call_strategy or _TOOL_CALL_STRATEGY_BALANCED,
                    utilization,
                    threshold,
                )
                return _ToolLoopGateDecision(
                    decision.enabled_skills,
                    "enabled_under_pressure",
                    "context_pressure_preserve_web_search",
                )
            logger.info(
                "Chat tool-loop gate: model=%s mode=%s reason=%s enabled_skills=%s strategy=%s context_utilization=%.4f compression_threshold=%.4f",
                getattr(request, "model", None) or "(conversation-default)",
                "disabled_by_pressure",
                "context_pressure_near_compaction",
                [],
                request.tool_call_strategy or _TOOL_CALL_STRATEGY_BALANCED,
                utilization,
                threshold,
            )
            return _ToolLoopGateDecision(
                [],
                "disabled_by_pressure",
                "context_pressure_near_compaction",
            )
        final_decision = _ToolLoopGateDecision(
            decision.enabled_skills,
            decision.mode,
            decision.reason,
        )
        logger.info(
            "Chat tool-loop gate: model=%s mode=%s reason=%s enabled_skills=%s strategy=%s enable_tool_calls=%s enable_web_search=%s explicit_skills=%s content_len=%s used_tokens=%s max_context_tokens=%s",
            getattr(request, "model", None) or "(conversation-default)",
            final_decision.mode,
            final_decision.reason,
            final_decision.enabled_skills,
            request.tool_call_strategy or _TOOL_CALL_STRATEGY_BALANCED,
            request.enable_tool_calls,
            request.enable_web_search,
            [
                skill
                for skill in (request.enable_skills or [])
                if isinstance(skill, str) and skill.strip()
            ],
            len(str(request.content or "")),
            used_tokens,
            max_context_tokens,
        )
        return final_decision

    def _get_adapter_for_model(self, model: str) -> LLMAdapter:
        """Get the LLM adapter for a given model by looking up its provider.

        Routes the model to the correct provider's adapter based on settings.
        Falls back to the default adapter if no provider match is found.
        """
        return self._model_adapter_resolver.get_adapter_for_model(model)

    def _resolve_summary_model(self, model: str) -> str | None:
        return self._model_adapter_resolver.resolve_summary_model(model)

    async def _build_compaction_summary(
        self,
        messages: list[Message],
        *,
        model: str,
        chat_span: Any | None = None,
    ) -> SummaryBuildResult:
        return await self._compaction_coordinator.build_compaction_summary(
            messages,
            model=model,
            chat_span=chat_span,
        )

    def _resolve_llm_kwargs(
        self,
        model: str,
        request: SendMessageRequest,
        conversation: Any,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        return self._llm_request_options_resolver.resolve_llm_kwargs(
            model=model,
            request=request,
            conversation=conversation,
        )

    def _build_context_messages(
        self,
        conversation: Any,
        model: str,
        sys_instructions: str,
        span: Span,
        input_budget: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return self._context_runtime.build_context_messages(
            conversation=conversation,
            model=model,
            sys_instructions=sys_instructions,
            span=span,
            input_budget=input_budget,
            truncation_log_label="chat_send",
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
        return self._context_runtime.get_context_usage(
            conversation=conversation,
            model=model,
            sys_instructions=sys_instructions,
            truncation_log_label=None,
        )

    async def _run_post_turn_compaction(
        self,
        *,
        conversation_id: str,
        model: str,
    ) -> None:
        await self._compaction_coordinator.run_post_turn_compaction(
            conversation_id=conversation_id,
            model=model,
        )

    def _schedule_post_turn_compaction(
        self,
        *,
        conversation_id: str,
        model: str,
    ) -> None:
        self._compaction_coordinator.schedule_post_turn_compaction(
            conversation_id=conversation_id,
            model=model,
        )

    async def compact_conversation(
        self,
        conversation_id: str,
    ) -> dict[str, Any]:
        return await self._compaction_coordinator.compact_conversation(conversation_id)

    async def send_message(
        self,
        conversation_id: str,
        request: SendMessageRequest,
    ) -> AsyncIterator[str]:
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
        prepared: PreparedSendContext | None = None
        assistant_message_id = ""
        active_streaming_registered = False
        assistant_persisted = False

        try:
            with _stage_span(chat_span, "chat.prepare"):
                prepared = await self._request_preparation.prepare(
                    conversation_id=conversation_id,
                    request=request,
                    chat_span=chat_span,
                )

            assistant_msg = Message(role=MessageRole.ASSISTANT, content="")
            assistant_message_id = assistant_msg.message_id
            chat_span.set_attribute("chat.request_id", assistant_msg.message_id)
            await self._set_active_streaming_state(
                conversation_id=conversation_id,
                conv_lock=prepared.conv_lock,
                message_id=assistant_msg.message_id,
                request_id=assistant_msg.message_id,
                status="streaming",
            )
            active_streaming_registered = True
            llm_messages = prepared.llm_messages
            context_usage = prepared.context_usage
            llm_adapter = self._get_adapter_for_model(prepared.model)
            generation_metadata: dict[str, Any] = {}
            generation_metadata["request_adapter_class"] = llm_adapter.__class__.__name__
            generation_metadata["request_adapter_strict_message_string_contract"] = bool(
                getattr(llm_adapter, "strict_message_string_contract", False)
            )
            request_message_shape = _summarize_message_shapes(llm_messages)
            generation_metadata["request_message_count"] = request_message_shape["message_count"]
            generation_metadata["request_user_message_count"] = request_message_shape[
                "user_message_count"
            ]
            generation_metadata["request_assistant_message_count"] = request_message_shape[
                "assistant_message_count"
            ]
            generation_metadata["request_assistant_reasoning_message_count"] = (
                request_message_shape["assistant_reasoning_message_count"]
            )
            generation_metadata["request_assistant_reasoning_only_message_count"] = (
                request_message_shape["assistant_reasoning_only_message_count"]
            )
            generation_metadata["request_assistant_tool_call_message_count"] = (
                request_message_shape["assistant_tool_call_message_count"]
            )
            generation_metadata["request_tool_message_count"] = request_message_shape[
                "tool_message_count"
            ]

            if isinstance(prepared.context_state_event, dict) and prepared.context_state_event:
                yield SSEEvent(
                    event="context.state.updated",
                    data={
                        "message_id": assistant_msg.message_id,
                        **prepared.context_state_event,
                    },
                ).encode()

            if isinstance(prepared.compaction_event, dict) and prepared.compaction_event:
                yield SSEEvent(
                    event="context.compacted",
                    data={
                        "message_id": assistant_msg.message_id,
                        **prepared.compaction_event,
                    },
                ).encode()

            if (
                isinstance(prepared.compaction_state_event, dict)
                and prepared.compaction_state_event
            ):
                yield SSEEvent(
                    event="context.state.updated",
                    data={
                        "message_id": assistant_msg.message_id,
                        **prepared.compaction_state_event,
                    },
                ).encode()

            resolved_chat_skills = self._resolve_enabled_chat_skills(request)
            tool_gate = self._gate_tool_loop(
                request=request,
                resolved_skills=resolved_chat_skills,
                context_usage=context_usage,
                runtime_profile=prepared.runtime_profile,
            )
            enabled_chat_skills = tool_gate.enabled_skills

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
                try:
                    tool_outcome = await self._tool_loop_orchestrator.run(
                        llm_adapter=llm_adapter,
                        model=prepared.model,
                        llm_messages=llm_messages,
                        llm_kwargs=prepared.llm_kwargs,
                        request=request,
                        runtime_profile=prepared.runtime_profile,
                        assistant_message_id=assistant_msg.message_id,
                        trace_id=chat_span.trace_id,
                        enabled_chat_skills=enabled_chat_skills,
                        parent_span=tool_loop_span,
                    )
                except Exception as exc:
                    visible_error = build_stream_error_content(exc)
                    normalized_error = normalize_chat_error(exc)
                    public_error = normalized_error.public_message
                    logger.warning(
                        "Chat tool-loop failed: conversation=%s message=%s model=%s error_code=%s status_code=%s provider_code=%s internal=%s",
                        conversation_id,
                        assistant_msg.message_id,
                        prepared.model,
                        normalized_error.error_code,
                        normalized_error.status_code,
                        normalized_error.provider_code,
                        normalized_error.internal_message,
                    )
                    if visible_error:
                        yield SSEEvent(
                            event="message.delta",
                            data={
                                "message_id": assistant_msg.message_id,
                                "seq": 1,
                                "content": visible_error,
                            },
                        ).encode()
                    yield SSEEvent(
                        event="message.error",
                        data={
                            "message_id": assistant_msg.message_id,
                            "error": public_error,
                            "error_code": normalized_error.error_code,
                            "public_message": normalized_error.public_message,
                            "retryable": normalized_error.retryable,
                            "status_code": normalized_error.status_code,
                            "provider_code": normalized_error.provider_code,
                            "error_type": type(exc).__name__,
                            "chunks_sent": 1 if visible_error else 0,
                            "timestamp": time.time(),
                        },
                    ).encode()
                    yield SSEEvent(
                        event="message.complete",
                        data={
                            "message_id": assistant_msg.message_id,
                            "metadata": {
                                "trace_id": chat_span.trace_id,
                                "finish_reason": "error",
                            },
                        },
                    ).encode()
                    completion_emitted_at = time.perf_counter()
                    persist_result = await self._turn_persistence.persist(
                        conversation_id=conversation_id,
                        conv_lock=prepared.conv_lock,
                        assistant_msg=assistant_msg,
                        content_parts=[visible_error] if visible_error else [],
                        reasoning_parts=[],
                        persisted_tool_messages=[],
                        usage_payload=None,
                        finish_reason="error",
                        budget_metadata=prepared.budget_metadata,
                        generation_metadata={},
                        completion_emitted_at=completion_emitted_at,
                        chat_span=chat_span,
                        model=prepared.model,
                    )
                    if (
                        isinstance(persist_result.context_state_event, dict)
                        and persist_result.context_state_event
                    ):
                        yield SSEEvent(
                            event="context.state.updated",
                            data={
                                "message_id": assistant_msg.message_id,
                                **persist_result.context_state_event,
                            },
                        ).encode()
                    return
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
            persisted_tool_messages = tool_outcome.persisted_tool_messages
            llm_messages = tool_outcome.llm_messages
            final_stream_messages_reconstructed = False
            if tool_outcome.event_chunks or persisted_tool_messages:
                yield SSEEvent(
                    event="agent.finalizing",
                    data={
                        "message_id": assistant_msg.message_id,
                        "trace_id": chat_span.trace_id,
                    },
                ).encode()
            if not llm_messages:
                reconstructed_messages = list(prepared.llm_messages)
                reconstructed_messages.extend(
                    message.to_llm_message() for message in persisted_tool_messages
                )
                llm_messages = reconstructed_messages
                final_stream_messages_reconstructed = True
                logger.warning(
                    "Tool loop returned empty final-stream messages; reconstructed from prepared context: conversation=%s, message=%s, persisted_tool_messages=%d",
                    conversation_id,
                    assistant_msg.message_id,
                    len(persisted_tool_messages),
                )

            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            usage_payload: dict[str, Any] | None = None
            finish_reason: str | None = tool_outcome.finish_reason
            generation_metadata["tool_loop_convergence_reason"] = (
                tool_outcome.convergence_reason
                if tool_outcome.convergence_reason
                in {
                    "no_tool_calls_with_replay_payload",
                    "pending_tool_calls_after_tool_loop",
                }
                else "needs_final_stream"
            )
            generation_metadata["tool_loop_final_stream_skipped"] = (
                tool_outcome.replay_response is not None
            )
            generation_metadata["tool_loop_terminal_tool_call_count"] = (
                tool_outcome.terminal_tool_call_count
            )
            generation_metadata["tool_loop_max_rounds_reached"] = (
                tool_outcome.convergence_reason == "pending_tool_calls_after_tool_loop"
            )
            generation_metadata["final_stream_messages_reconstructed"] = (
                final_stream_messages_reconstructed
            )
            generation_metadata["final_stream_persisted_tool_message_count"] = len(
                persisted_tool_messages
            )
            generation_metadata["final_stream_prepared_message_count"] = len(llm_messages)
            if isinstance(tool_outcome.response_metadata, dict) and tool_outcome.response_metadata:
                generation_metadata.update(
                    {
                        key: value
                        for key, value in tool_outcome.response_metadata.items()
                        if key not in {"reasoning_content", "usage", "finish_reason"}
                    }
                )
            if tool_outcome.replay_response is not None:
                logger.info(
                    "Chat stream phase: conversation=%s message=%s phase=replay convergence_reason=%s persisted_tool_messages=%s final_stream_skipped=%s",
                    conversation_id,
                    assistant_msg.message_id,
                    tool_outcome.convergence_reason,
                    len(persisted_tool_messages),
                    True,
                )
                with _stage_span(chat_span, "chat.stream.replay"):
                    replay_capture = ReplayStreamCapture()
                    replay_metadata = getattr(tool_outcome.replay_response, "metadata", None)
                    replay_has_reasoning = bool(
                        isinstance(replay_metadata, dict)
                        and str(replay_metadata.get("reasoning_content") or "").strip()
                    )
                    async for sse_chunk in self._response_streamer.iter_replay_chunks(
                        replay_response=tool_outcome.replay_response,
                        assistant_message_id=assistant_msg.message_id,
                        model=prepared.model,
                        context_usage=context_usage,
                        finish_reason=finish_reason,
                        capture=replay_capture,
                        stream_reasoning=(not bool(persisted_tool_messages))
                        or replay_has_reasoning,
                    ):
                        yield sse_chunk
                    content_parts = replay_capture.content_parts
                    reasoning_parts = replay_capture.reasoning_parts
                usage_payload = tool_outcome.usage_payload
            else:
                final_stream_messages, final_stream_sanitize_stats = (
                    _sanitize_final_stream_messages(llm_messages)
                )
                logger.info(
                    "Chat stream phase: conversation=%s message=%s phase=final_stream convergence_reason=%s prepared_messages=%s sanitized_messages=%s reconstructed=%s persisted_tool_messages=%s",
                    conversation_id,
                    assistant_msg.message_id,
                    tool_outcome.convergence_reason or "needs_final_stream",
                    len(llm_messages),
                    len(final_stream_messages),
                    final_stream_messages_reconstructed,
                    len(persisted_tool_messages),
                )
                generation_metadata["final_stream_sanitized_message_count"] = len(
                    final_stream_messages
                )
                generation_metadata["final_stream_assistant_tool_call_carrier_count"] = (
                    final_stream_sanitize_stats["assistant_tool_call_carrier_count"]
                )
                generation_metadata["final_stream_assistant_reasoning_removed_count"] = (
                    final_stream_sanitize_stats["assistant_reasoning_removed_count"]
                )
                generation_metadata["final_stream_assistant_reasoning_only_removed_count"] = (
                    final_stream_sanitize_stats["assistant_reasoning_only_removed_count"]
                )
                generation_metadata["final_stream_tool_result_projection_count"] = (
                    final_stream_sanitize_stats["tool_result_projection_count"]
                )
                prior_generation_metadata = dict(generation_metadata)
                final_stream_capture = FinalStreamCapture()
                async for sse_chunk in self._response_streamer.iter_final_response(
                    llm_adapter=llm_adapter,
                    llm_messages=final_stream_messages,
                    llm_kwargs=prepared.llm_kwargs,
                    model=prepared.model,
                    conversation_id=conversation_id,
                    assistant_message_id=assistant_msg.message_id,
                    context_usage=context_usage,
                    chat_span=chat_span,
                    capture=final_stream_capture,
                    require_visible_content=bool(persisted_tool_messages),
                ):
                    yield sse_chunk
                content_parts = final_stream_capture.content_parts
                reasoning_parts = final_stream_capture.reasoning_parts
                usage_payload = final_stream_capture.usage_payload
                finish_reason = final_stream_capture.finish_reason
                generation_metadata = {
                    **prior_generation_metadata,
                    **final_stream_capture.generation_metadata,
                }

            completion_metadata: dict[str, Any] = {"trace_id": chat_span.trace_id}
            await self._set_active_streaming_state(
                conversation_id=conversation_id,
                conv_lock=prepared.conv_lock,
                message_id=assistant_msg.message_id,
                request_id=assistant_msg.message_id,
                status="finishing",
            )
            if isinstance(usage_payload, dict) and usage_payload:
                completion_metadata["usage"] = usage_payload
            if isinstance(finish_reason, str) and finish_reason:
                completion_metadata["finish_reason"] = finish_reason
            if isinstance(prepared.budget_metadata, dict) and prepared.budget_metadata:
                completion_metadata["budget"] = prepared.budget_metadata
            completion_metadata.update(generation_metadata)
            completion_emitted_at = time.perf_counter()
            yield SSEEvent(
                event="message.complete",
                data={
                    "message_id": assistant_msg.message_id,
                    "metadata": completion_metadata,
                },
            ).encode()

            with _stage_span(chat_span, "chat.persist"):
                persist_result = await self._turn_persistence.persist(
                    conversation_id=conversation_id,
                    conv_lock=prepared.conv_lock,
                    assistant_msg=assistant_msg,
                    content_parts=content_parts,
                    reasoning_parts=reasoning_parts,
                    persisted_tool_messages=persisted_tool_messages,
                    usage_payload=usage_payload,
                    finish_reason=finish_reason,
                    budget_metadata=prepared.budget_metadata,
                    generation_metadata=generation_metadata,
                    completion_emitted_at=completion_emitted_at,
                    chat_span=chat_span,
                    model=prepared.model,
                )
                assistant_persisted = persist_result.persisted
                if (
                    isinstance(persist_result.context_state_event, dict)
                    and persist_result.context_state_event
                ):
                    yield SSEEvent(
                        event="context.state.updated",
                        data={
                            "message_id": assistant_msg.message_id,
                            **persist_result.context_state_event,
                        },
                    ).encode()

        except Exception as e:
            chat_span.set_status("error", str(e))
            raise
        finally:
            if active_streaming_registered and prepared is not None and assistant_message_id:
                with contextlib.suppress(Exception):
                    await self._clear_active_streaming_state(
                        conversation_id=conversation_id,
                        conv_lock=prepared.conv_lock,
                        message_id=assistant_message_id,
                    )
            if assistant_persisted and prepared is not None:
                self._schedule_post_turn_compaction(
                    conversation_id=conversation_id,
                    model=prepared.model,
                )
            chat_span.end()
            _persist_trace_tree(chat_span)
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
        return await self._message_manager.edit_message(
            conversation_id,
            message_id,
            request,
        )

    async def delete_message(
        self,
        conversation_id: str,
        message_id: str,
    ) -> Any:
        """Delete a single message from a conversation.

        Args:
            conversation_id: Target conversation.
            message_id: Message to delete.

        Returns:
            The result of the delete operation.

        Raises:
            FileNotFoundError: If conversation not found.
            ValueError: If message not found.
        """
        return await self._message_manager.delete_message(conversation_id, message_id)

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
        preparation = await self._message_manager.prepare_regeneration(
            conversation_id=conversation_id,
            message_id=message_id,
        )
        if isinstance(preparation.context_state_event, dict) and preparation.context_state_event:
            yield SSEEvent(
                event="context.state.updated",
                data={
                    "message_id": message_id,
                    **preparation.context_state_event,
                },
            ).encode()
        request = SendMessageRequest(content=preparation.last_user_content)
        async for chunk in self.send_message(conversation_id, request):
            yield chunk
