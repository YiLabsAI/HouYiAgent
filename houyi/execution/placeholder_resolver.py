"""Placeholder resolution for tool-call argument chaining.

Supports ``$tool.<name>.<path>`` and ``$call.<id>.<path>`` placeholders that
reference outputs from previously executed tool calls within the same session.
"""

from __future__ import annotations

from typing import Any


class PlaceholderResolver:
    """Resolve ``$tool.*`` / ``$call.*`` placeholders in tool arguments."""

    @staticmethod
    def contains(value: Any) -> bool:
        """Return True if *value* contains at least one placeholder reference."""
        if isinstance(value, str):
            return value.startswith("$tool.") or value.startswith("$call.")
        if isinstance(value, dict):
            return any(PlaceholderResolver.contains(item) for item in value.values())
        if isinstance(value, list):
            return any(PlaceholderResolver.contains(item) for item in value)
        return False

    @staticmethod
    def resolve(value: Any, resolved_outputs: dict[str, Any]) -> Any:
        """Recursively resolve placeholder references in *value*."""
        if isinstance(value, str):
            parsed = PlaceholderResolver.extract(value)
            if parsed is None:
                return value
            root_key, path = parsed
            payload = resolved_outputs.get(root_key)
            if payload is None:
                return value
            return PlaceholderResolver.resolve_path(payload, path)
        if isinstance(value, dict):
            return {
                key: PlaceholderResolver.resolve(item, resolved_outputs)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [PlaceholderResolver.resolve(item, resolved_outputs) for item in value]
        return value

    @staticmethod
    def extract(value: str) -> tuple[str, list[str]] | None:
        """Extract ``(root_key, path_segments)`` from a placeholder string."""
        if value.startswith("$tool."):
            path = value[len("$tool.") :]
        elif value.startswith("$call."):
            path = value[len("$call.") :]
        else:
            return None
        parts = [segment for segment in path.split(".") if segment]
        if not parts:
            return None
        return parts[0], parts[1:]

    @staticmethod
    def resolve_path(payload: Any, path: list[str]) -> Any:
        """Walk *path* segments into *payload* (dict keys or list indices)."""
        current = payload
        for segment in path:
            if isinstance(current, dict) and segment in current:
                current = current[segment]
                continue
            if isinstance(current, list):
                try:
                    index = int(segment)
                except ValueError:
                    return payload
                if 0 <= index < len(current):
                    current = current[index]
                    continue
            return payload
        return current
