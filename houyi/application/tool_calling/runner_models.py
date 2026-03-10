"""Shared models and parsing helpers for tool-calling runner collaborators."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, TypedDict

from houyi.domain.skill.spec import SkillSpec

logger = logging.getLogger(__name__)

_DEFAULT_TOOL_RESULT_SUMMARY_ENABLED = True
_DEFAULT_TOOL_RESULT_SUMMARY_MAX_CHARS = 4_000
_DEFAULT_TOOL_RESULT_SUMMARY_MAX_ITEMS = 50


def _read_positive_int_env_or_none(env_name: str) -> int | None:
    raw = os.getenv(env_name)
    if raw is None:
        return None
    try:
        parsed = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, ignore and use auto/default budget", env_name, raw)
        return None
    if parsed <= 0:
        logger.warning(
            "Invalid %s=%r (must be > 0), ignore and use auto/default budget", env_name, raw
        )
        return None
    return parsed


def _read_bool_env(env_name: str, default: bool) -> bool:
    raw = os.getenv(env_name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid %s=%r, fallback to %s", env_name, raw, default)
    return default


def _parse_max_parallel_calls(chat_kwargs: dict[str, Any]) -> int:
    default_max = 5
    if "max_parallel_calls" not in chat_kwargs:
        return default_max
    raw_value = chat_kwargs.get("max_parallel_calls")
    try:
        if raw_value is None:
            raise ValueError("max_parallel_calls is None")
        parsed = int(raw_value)
        if parsed > 0:
            return parsed
        logger.warning(
            "Invalid max_parallel_calls=%s (must be > 0), using default=%s",
            raw_value,
            default_max,
        )
        return default_max
    except (TypeError, ValueError):
        logger.warning(
            "Invalid max_parallel_calls=%s (must be int), using default=%s",
            raw_value,
            default_max,
        )
        return default_max


def _parse_tool_latency_seconds() -> float | None:
    from houyi.infrastructure.config import ENV_TOOLCALL_TOOL_LATENCY_MS

    tool_latency_env = os.getenv(ENV_TOOLCALL_TOOL_LATENCY_MS)
    if not tool_latency_env:
        return None
    try:
        tool_latency_ms = float(tool_latency_env)
        if tool_latency_ms > 0:
            return tool_latency_ms / 1000.0
    except ValueError:
        logger.warning(
            "Invalid HOUYI_TOOLCALL_TOOL_LATENCY_MS=%s",
            tool_latency_env,
        )
    return None


class _HookCtx(TypedDict):
    tool_name: str | None
    args: dict[str, Any]
    skill: SkillSpec | None
    tool_call_id: str | None


@dataclass(frozen=True)
class _PreparedToolCall:
    requested_tool_name: str | None
    tool_name: str | None
    tool_call_id: str | None
    args: dict[str, Any]
    skill: SkillSpec | None
    hook_context: _HookCtx
    attempted_tool_name: str | None
    cache_key: str | None


@dataclass(frozen=True)
class _ExecutedToolCall:
    result: dict[str, Any]
    cache_hit_for_reporting: bool
    tool_elapsed: float
    latency_ms: Any


@dataclass(frozen=True)
class _ToolCallPreparationRequest:
    tool_call: Any
    parsed_args: dict[str, Any] | None
    resolved_outputs: dict[str, Any] | None
    skills_by_name: dict[str, SkillSpec]
    tool_hooks: list[Any]
    allow_tool_replace: bool
    index: int
    round_index_value: int | None
    parallel_group_id: str | None


@dataclass(frozen=True)
class _ToolCallPresentationRequest:
    tool_name: str | None
    requested_tool_name: str | None
    tool_call_id: str | None
    round_index_value: int | None
    parallel_group_id: str | None
    duration_ms: float | None
    args: dict[str, Any]
    result: dict[str, Any]
    attempted_tool_name: str | None
    allow_tool_replace: bool
    tool_result_summary_enabled: bool
    tool_result_summary_max_chars: int
    tool_result_summary_max_items: int


@dataclass(frozen=True)
class _BlockedToolCallPresentationRequest:
    tool_name: str
    args: dict[str, Any]
    tool_call_id: str | None
    error_code: str
    message: str
    block_reason: str
