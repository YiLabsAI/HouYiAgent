from __future__ import annotations

import asyncio
import difflib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from houyi.application.tool_calling.tool_results import ToolResultBuilder
from houyi.domain.skill.exceptions import SkillExecutionError
from houyi.domain.skill.spec import SkillSpec


@dataclass(frozen=True)
class ToolSkillExecutionRequest:
    """Input payload for one concrete skill execution."""

    tool_name: str | None
    args: dict[str, Any]
    tool_call_id: str | None


@dataclass(frozen=True)
class ToolSkillExecutionServices:
    """Runtime collaborators required to execute one skill call."""

    skill_specs_by_name: dict[str, SkillSpec]
    tool_executor: Any
    record_metrics: Callable[[str, float, bool, bool], None]


def _available_tool_names(services: ToolSkillExecutionServices) -> list[str]:
    return sorted(name for name in services.skill_specs_by_name if name)


def _build_missing_tool_recovery(
    *,
    requested_tool_name: str | None,
    services: ToolSkillExecutionServices,
) -> dict[str, Any]:
    available_tools = _available_tool_names(services)
    similar_tools = (
        difflib.get_close_matches(requested_tool_name, available_tools, n=3, cutoff=0.5)
        if requested_tool_name
        else []
    )
    next_steps = [
        "Choose one of the registered tool names before retrying.",
    ]
    if similar_tools:
        next_steps.append("Retry with one of the suggested tool names.")
    elif available_tools:
        next_steps.append("Review the available tool list and select the closest match.")
    return {
        "code": "missing_tool",
        "title": "Select a registered tool name",
        "message": "The requested tool name is missing or does not match a registered tool.",
        "similar_tools": similar_tools,
        "available_tools": available_tools[:10],
        "next_steps": next_steps,
    }


def _build_argument_validation_recovery(
    *,
    tool_name: str,
    message: str,
) -> dict[str, Any]:
    return {
        "code": "argument_validation",
        "title": "Review required arguments",
        "message": message,
        "required_fields": [],
        "next_steps": [
            f"Review the expected arguments for '{tool_name}'.",
            "Retry with a smaller, fully populated argument set.",
        ],
    }


def _build_execution_failure_recovery(
    *,
    tool_name: str,
    error_type: str,
    message: str,
    retry_count: Any,
    timeout: Any,
) -> dict[str, Any]:
    normalized_error_type = str(error_type or "execution_error").lower()
    if normalized_error_type == "timeout":
        return {
            "code": "execution_timeout",
            "title": "Reduce the scope and retry",
            "message": message,
            "next_steps": [
                f"Retry '{tool_name}' with a narrower scope or fewer returned items.",
                f"Current timeout setting: {timeout}."
                if timeout is not None
                else "Retry after reducing the tool workload.",
            ],
        }
    if "validation" in normalized_error_type or "required" in message.lower():
        return _build_argument_validation_recovery(tool_name=tool_name, message=message)
    return {
        "code": "execution_failure",
        "title": "Inspect tool inputs before retrying",
        "message": message,
        "next_steps": [
            f"Verify that '{tool_name}' is the correct tool for this task.",
            "Retry with simpler inputs or inspect the previous tool output before retrying.",
            f"Configured retries: {retry_count}."
            if retry_count is not None
            else "Review tool-specific retry policy if repeated failures continue.",
        ],
    }


async def execute_tool_skill(
    request: ToolSkillExecutionRequest,
    services: ToolSkillExecutionServices,
) -> dict[str, Any]:
    """Execute one skill call and normalize result payload/error shape."""
    tool_name = request.tool_name
    if not tool_name:
        return ToolResultBuilder.build(
            {
                "error": "tool_name_missing",
                "recovery_guidance": _build_missing_tool_recovery(
                    requested_tool_name=tool_name,
                    services=services,
                ),
            },
            call_id=request.tool_call_id,
            metadata={"tool_name": tool_name},
        )

    skill = services.skill_specs_by_name.get(tool_name)
    if not skill:
        return ToolResultBuilder.build(
            {
                "error": f"tool_not_found: {tool_name}",
                "recovery_guidance": _build_missing_tool_recovery(
                    requested_tool_name=tool_name,
                    services=services,
                ),
            },
            call_id=request.tool_call_id,
            metadata={"tool_name": tool_name},
        )

    start_time = time.time()
    try:
        raw_result = await services.tool_executor.execute(skill, request.args)
        latency_ms = (time.time() - start_time) * 1000
        services.record_metrics(tool_name, latency_ms, True, False)
        return ToolResultBuilder.build(
            raw_result,
            call_id=request.tool_call_id,
            metadata={"tool_name": tool_name, "latency_ms": latency_ms},
        )
    except SkillExecutionError as exc:
        latency_ms = (time.time() - start_time) * 1000
        is_timeout = isinstance(exc.original_error, asyncio.TimeoutError)
        services.record_metrics(tool_name, latency_ms, False, is_timeout)
        error_type = "timeout" if is_timeout else "execution_error"
        message = exc.message or str(exc.original_error or "Tool execution failed")
        return ToolResultBuilder.build(
            {
                "error": "tool_execution_failed",
                "error_type": error_type,
                "message": message,
                "skill_name": exc.skill_name,
                "cause": str(exc.original_error) if exc.original_error else None,
                "retry_count": getattr(services.tool_executor, "max_retries", None),
                "timeout": getattr(services.tool_executor, "timeout", None),
                "recovery_guidance": _build_execution_failure_recovery(
                    tool_name=tool_name,
                    error_type=error_type,
                    message=message,
                    retry_count=getattr(services.tool_executor, "max_retries", None),
                    timeout=getattr(services.tool_executor, "timeout", None),
                ),
            },
            call_id=request.tool_call_id,
            metadata={"tool_name": tool_name, "latency_ms": latency_ms},
        )
    except Exception as exc:
        latency_ms = (time.time() - start_time) * 1000
        services.record_metrics(tool_name, latency_ms, False, False)
        message = str(exc)
        return ToolResultBuilder.build(
            {
                "error": "tool_execution_failed",
                "error_type": exc.__class__.__name__,
                "message": message,
                "retry_count": getattr(services.tool_executor, "max_retries", None),
                "timeout": getattr(services.tool_executor, "timeout", None),
                "recovery_guidance": _build_execution_failure_recovery(
                    tool_name=tool_name,
                    error_type=exc.__class__.__name__,
                    message=message,
                    retry_count=getattr(services.tool_executor, "max_retries", None),
                    timeout=getattr(services.tool_executor, "timeout", None),
                ),
            },
            call_id=request.tool_call_id,
            metadata={"tool_name": tool_name, "latency_ms": latency_ms},
        )


__all__ = [
    "ToolSkillExecutionRequest",
    "ToolSkillExecutionServices",
    "execute_tool_skill",
]
