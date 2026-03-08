"""Configurable argument coercion for tool calls.

Instead of hardcoding tool-name-specific logic in the runner, coercion
functions are registered in a registry keyed by tool name. The runner
calls ``coerce_args(tool_name, args, resolved_outputs)`` and dispatches
to the matching handler (or returns ``args`` unchanged).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

CoercionFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]

_REGISTRY: dict[str, CoercionFn] = {}


def register_arg_coercion(tool_name: str, fn: CoercionFn) -> None:
    """Register a coercion function for ``tool_name``."""
    _REGISTRY[tool_name] = fn


def coerce_args(
    tool_name: str,
    args: dict[str, Any],
    resolved_outputs: dict[str, Any],
) -> dict[str, Any]:
    """Apply registered coercion (if any) for ``tool_name``."""
    fn = _REGISTRY.get(tool_name)
    if fn is None:
        return args
    try:
        return fn(args, resolved_outputs)
    except Exception:
        logger.debug("Coercion for %s failed, returning original args", tool_name, exc_info=True)
        return args


# Built-in coercion: weather tools


def _coerce_weather_args(args: dict[str, Any], resolved_outputs: dict[str, Any]) -> dict[str, Any]:
    """Enrich weather tool arguments with date/location from prior tool outputs."""
    updated = dict(args)
    date = resolved_outputs.get("get_date")
    if isinstance(date, str):
        updated["date"] = date
    location = resolved_outputs.get("get_location")
    if isinstance(location, dict):
        if "lat" in location:
            updated["lat"] = location["lat"]
        if "lon" in location:
            updated["lon"] = location["lon"]
    return updated


register_arg_coercion("get_weather", _coerce_weather_args)
