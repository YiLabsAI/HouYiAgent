"""Execution wrapper for prepared tool calls."""

from __future__ import annotations

import logging
import time
from typing import Any

from houyi.application.tool_calling.runner_models import _ExecutedToolCall, _PreparedToolCall

logger = logging.getLogger(__name__)


class _PreparedToolCallExecutor:
    """Execute a prepared tool call and normalize runtime reporting metadata."""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    async def execute(
        self,
        *,
        prepared: _PreparedToolCall,
        config: Any,
        state: Any,
        services: Any,
        tool_start: float,
    ) -> _ExecutedToolCall:
        """Execute one prepared tool call and normalize timing/cache metadata."""
        result, cache_hit = await self._runner._execution_service._execute_tool_with_cache(
            tool_name=prepared.tool_name,
            requested_tool_name=prepared.requested_tool_name,
            tool_call_id=prepared.tool_call_id,
            parallel_group_id=state.parallel_group_id,
            args=prepared.args,
            skill=prepared.skill,
            cache_key=prepared.cache_key,
            tool_cache=services.tool_result_cache,
            skills_by_name=services.skill_specs_by_name,
            executor=services.tool_executor,
        )

        raw_result = result.get("raw")
        raw_metadata = raw_result.get("metadata") if isinstance(raw_result, dict) else None
        result_metadata = result.get("metadata")
        tool_reported_cache_hit = (
            bool(result_metadata.get("cache_hit")) if isinstance(result_metadata, dict) else False
        ) or (bool(raw_metadata.get("cache_hit")) if isinstance(raw_metadata, dict) else False)

        result = self._runner._enrich_result_with_cache_metadata(
            result,
            cache_hit,
            prepared.cache_key,
            tool_reported_cache_hit,
        )
        cache_hit_for_reporting = cache_hit or tool_reported_cache_hit
        tool_elapsed = time.perf_counter() - tool_start if config.tool_loop_enable_timing else 0.0
        if config.tool_loop_enable_timing:
            logger.info(
                "[ToolCallRunner] tool=%s call_id=%s elapsed=%.3fs",
                prepared.tool_name,
                prepared.tool_call_id,
                tool_elapsed,
            )

        result_meta = result.get("metadata", {})
        latency_ms = result_meta.get("latency_ms") if isinstance(result_meta, dict) else None
        return _ExecutedToolCall(
            result=result,
            cache_hit_for_reporting=cache_hit_for_reporting,
            tool_elapsed=tool_elapsed,
            latency_ms=latency_ms,
        )


__all__ = ["_PreparedToolCallExecutor"]
