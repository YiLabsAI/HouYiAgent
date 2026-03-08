"""Decorators for HouYi framework."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, cast, get_type_hints

from pydantic import BaseModel, create_model

from houyi.domain.skill.spec import SkillSpec


class _EmptyToolInput(BaseModel):
    pass


def _schema_model_prefix(name: str) -> str:
    parts = [part for part in name.replace("-", "_").split("_") if part]
    if not parts:
        return "Tool"
    return "".join(part[:1].upper() + part[1:] for part in parts)


def _build_input_schema(func: Callable[..., Any], hints: dict[str, Any]) -> type[BaseModel]:
    sig = inspect.signature(func)
    input_fields: dict[str, tuple[Any, Any]] = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        param_type = hints.get(param_name, str)
        default = ... if param.default == inspect.Parameter.empty else param.default
        input_fields[param_name] = (param_type, default)

    if not input_fields:
        return _EmptyToolInput

    model_name = f"{_schema_model_prefix(func.__name__)}Input"
    return create_model(model_name, **cast(dict[str, Any], input_fields))


def _build_output_schema(func: Callable[..., Any], hints: dict[str, Any]) -> type[BaseModel]:
    model_name = f"{_schema_model_prefix(func.__name__)}Output"
    return_type = hints.get("return", str)
    return create_model(model_name, result=(return_type, ...))


def tool(func: Callable[..., Any]) -> SkillSpec:
    """Decorator to convert a function into a SkillSpec.

    Automatically infers input/output schemas from type hints.

    Usage:
        @tool
        def search(query: str) -> list[str]:
            '''Search the web for information.'''
            return ["result1", "result2"]

    Args:
        func: Function to convert to a skill

    Returns:
        SkillSpec instance
    """
    name = func.__name__
    description = func.__doc__ or f"Execute {name}"
    description = description.strip()

    hints = get_type_hints(func)
    input_schema = _build_input_schema(func, hints)
    output_schema = _build_output_schema(func, hints)

    skill = SkillSpec(
        name=name,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        executor=func,
    )

    skill._original_func = func  # type: ignore[attr-defined]

    return skill
