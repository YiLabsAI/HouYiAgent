"""Run settings configuration helpers.

This module is intentionally free of any server-specific dependencies.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ConfigService:
    """Resolve and normalize run settings for execution."""

    def normalize_run_settings(self, run_settings: dict[str, Any] | None) -> dict[str, Any]:
        if not run_settings:
            return {}
        resolved = self.resolve_tool_settings(run_settings)
        retry_policy = run_settings.get("retry_policy") if isinstance(run_settings, dict) else None
        if retry_policy is not None:
            resolved["retry_policy"] = retry_policy
        return resolved

    def resolve_tool_settings(
        self,
        settings: dict[str, Any] | None,
        default_max_tool_calls: int = 6,
    ) -> dict[str, Any]:
        source = settings or {}
        tool_names = self._coerce_tool_names(source.get("tool_names"))
        tool_choice = self._normalize_tool_choice(source.get("tool_choice"))
        max_tool_calls = self._coerce_max_tool_calls(
            source.get("max_tool_calls"),
            default=default_max_tool_calls,
        )
        temperature = self._coerce_temperature(source.get("temperature"))
        parallel_tool_calls = self._coerce_parallel_tool_calls(source.get("parallel_tool_calls"))
        enable_tool_calls_value = (
            source.get("enable_tool_calls") if "enable_tool_calls" in source else True
        )
        enable_tool_calls = bool(enable_tool_calls_value or tool_names)
        prompt_cache_key = source.get("prompt_cache_key")
        web_search_provider = source.get("web_search_provider")
        return {
            "enable_tool_calls": enable_tool_calls,
            "tool_names": tool_names,
            "tool_choice": tool_choice,
            "max_tool_calls": max_tool_calls,
            "temperature": temperature,
            "parallel_tool_calls": parallel_tool_calls,
            "prompt_cache_key": prompt_cache_key,
            "web_search_provider": web_search_provider,
        }

    @staticmethod
    def _normalize_tool_choice(tool_choice: Any | None) -> Any | None:
        """Normalize tool_choice input and guard against invalid types."""
        normalized = ConfigService._coerce_tool_choice(tool_choice)
        if isinstance(normalized, bool):
            logger.warning("Invalid tool_choice boolean=%s; resetting to None", normalized)
            return None
        return normalized

    @staticmethod
    def _coerce_tool_names(tool_names: Any | None) -> list[str]:
        if isinstance(tool_names, str):
            try:
                parsed = json.loads(tool_names)
                if isinstance(parsed, list):
                    return [str(name) for name in parsed if str(name).strip()]
                if parsed:
                    return [str(parsed)]
            except json.JSONDecodeError:
                return [name.strip() for name in tool_names.split(",") if name.strip()]
        if tool_names is None:
            return []
        if isinstance(tool_names, list):
            return [str(name) for name in tool_names if str(name).strip()]
        return [str(tool_names)]

    @staticmethod
    def _coerce_tool_choice(tool_choice: Any | None) -> Any | None:
        if isinstance(tool_choice, str):
            try:
                return json.loads(tool_choice)
            except json.JSONDecodeError:
                return tool_choice
        return tool_choice

    @staticmethod
    def _coerce_max_tool_calls(max_tool_calls: Any | None, default: int = 6) -> int:
        if isinstance(max_tool_calls, str):
            try:
                return int(max_tool_calls)
            except ValueError:
                return default
        if isinstance(max_tool_calls, int):
            return max_tool_calls
        if isinstance(max_tool_calls, float):
            return int(max_tool_calls)
        return default

    @staticmethod
    def _coerce_temperature(value: Any | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    @staticmethod
    def _coerce_parallel_tool_calls(value: Any | None) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y"}:
                return True
            if normalized in {"false", "0", "no", "n"}:
                return False
        return None
