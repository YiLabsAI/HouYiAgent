from __future__ import annotations

import asyncio
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


async def execute_tool_skill(
    request: ToolSkillExecutionRequest,
    services: ToolSkillExecutionServices,
) -> dict[str, Any]:
    """Execute one skill call and normalize result payload/error shape."""
    tool_name = request.tool_name
    if not tool_name:
        return ToolResultBuilder.build(
            {"error": "tool_name_missing"},
            call_id=request.tool_call_id,
            metadata={"tool_name": tool_name},
        )

    skill = services.skill_specs_by_name.get(tool_name)
    if not skill:
        return ToolResultBuilder.build(
            {"error": f"tool_not_found: {tool_name}"},
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
        return ToolResultBuilder.build(
            {
                "error": "tool_execution_failed",
                "error_type": error_type,
                "message": exc.message,
                "skill_name": exc.skill_name,
                "cause": str(exc.original_error) if exc.original_error else None,
                "retry_count": getattr(services.tool_executor, "max_retries", None),
                "timeout": getattr(services.tool_executor, "timeout", None),
            },
            call_id=request.tool_call_id,
            metadata={"tool_name": tool_name, "latency_ms": latency_ms},
        )
    except Exception as exc:
        latency_ms = (time.time() - start_time) * 1000
        services.record_metrics(tool_name, latency_ms, False, False)
        return ToolResultBuilder.build(
            {
                "error": "tool_execution_failed",
                "error_type": exc.__class__.__name__,
                "message": str(exc),
                "retry_count": getattr(services.tool_executor, "max_retries", None),
                "timeout": getattr(services.tool_executor, "timeout", None),
            },
            call_id=request.tool_call_id,
            metadata={"tool_name": tool_name, "latency_ms": latency_ms},
        )


__all__ = [
    "ToolSkillExecutionRequest",
    "ToolSkillExecutionServices",
    "execute_tool_skill",
]
