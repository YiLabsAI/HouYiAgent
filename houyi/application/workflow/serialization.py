"""Serialization helpers for workflow outputs and traces."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any


def to_wire_data(value: Any, *, by_alias: bool = True) -> Any:
    """Convert model-like objects into plain JSON-friendly Python structures."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return to_wire_data(model_dump(by_alias=by_alias), by_alias=by_alias)

    if is_dataclass(value):
        return {
            field.name: to_wire_data(getattr(value, field.name), by_alias=by_alias)
            for field in fields(value)
        }

    if isinstance(value, dict):
        return {key: to_wire_data(item, by_alias=by_alias) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_wire_data(item, by_alias=by_alias) for item in value]

    return value
