"""Helpers for TOOL node execution."""

from __future__ import annotations

import json
from typing import Any


def normalize_tool_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def extract_schema_fields(input_schema: type | None) -> set[str]:
    if not input_schema:
        return set()
    fields = getattr(input_schema, "model_fields", None)
    if isinstance(fields, dict):
        return set(fields.keys())
    fields = getattr(input_schema, "__fields__", None)
    if isinstance(fields, dict):
        return set(fields.keys())
    return set()


def build_inputs_from_context_values(
    *,
    schema_fields: set[str],
    context_values: dict[str, Any] | None,
) -> dict[str, Any]:
    if not schema_fields:
        return {}
    context_values = context_values or {}
    fallback_values: dict[str, Any] = {}
    result_payload = context_values.get("result")
    if isinstance(result_payload, dict):
        fallback_values.update(result_payload)
    elif isinstance(result_payload, str):
        try:
            parsed_payload = json.loads(result_payload)
        except json.JSONDecodeError:
            parsed_payload = None
        if isinstance(parsed_payload, dict):
            fallback_values.update(parsed_payload)
    nested_result = fallback_values.get("result")
    if isinstance(nested_result, dict):
        fallback_values.update(nested_result)

    return {
        key: context_values.get(key, fallback_values.get(key))
        for key in schema_fields
        if key in context_values or key in fallback_values
    }
