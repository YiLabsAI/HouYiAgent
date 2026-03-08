"""Helpers for resolving node inputs and extracting output payloads."""

from __future__ import annotations

from typing import Any


def resolve_value(value: Any, context_values: dict[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        return context_values.get(value[1:], value)
    if isinstance(value, dict):
        return {key: resolve_value(val, context_values) for key, val in value.items()}
    if isinstance(value, list):
        return [resolve_value(item, context_values) for item in value]
    return value


def resolve_inputs(inputs: dict[str, Any], context_values: dict[str, Any]) -> dict[str, Any]:
    return {key: resolve_value(value, context_values) for key, value in inputs.items()}


def extract_output_payload(outputs: dict[str, Any]) -> dict[str, Any]:
    if (
        isinstance(outputs, dict)
        and "output" in outputs
        and isinstance(outputs.get("output"), dict)
    ):
        payload = outputs.get("output") or {}
        if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
            return payload.get("result") or {}
        return payload
    if isinstance(outputs, dict) and isinstance(outputs.get("result"), dict):
        return outputs.get("result") or {}
    return outputs or {}
