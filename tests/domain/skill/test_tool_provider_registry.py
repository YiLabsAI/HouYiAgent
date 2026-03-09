"""Unit tests for ToolProviderRegistry selection policy."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from houyi.domain.skill.spec import ExecutionMode, SkillSpec
from houyi.domain.skill.tool_provider_registry import ToolProviderRegistry


@dataclass
class _DummyRegistry:
    skills: list[SkillSpec]

    def list(self) -> list[SkillSpec]:
        return self.skills


class _DummyInput(BaseModel):
    value: int | None = None


class _DummyOutput(BaseModel):
    result: int | None = None


def _build_skill(name: str, mode: ExecutionMode) -> SkillSpec:
    return SkillSpec(
        name=name,
        description="",
        input_schema=_DummyInput,
        output_schema=_DummyOutput,
        executor=lambda *_args, **_kwargs: None,
        execution_mode=mode,
    )


def test_tool_provider_registry_internal_first() -> None:
    """Registry should prioritize internal/remote tools."""

    skills = [
        _build_skill("a", ExecutionMode.CLIENT),
        _build_skill("b", ExecutionMode.PLUGIN),
    ]
    registry = ToolProviderRegistry(skill_registry=_DummyRegistry(skills))
    selected = registry.select_skills()
    assert [skill.name for skill in selected] == ["b"]


def test_tool_provider_registry_fallback_all() -> None:
    """Registry should return all skills when no internal match exists."""

    skills = [_build_skill("b", ExecutionMode.PLUGIN)]
    registry = ToolProviderRegistry(skill_registry=_DummyRegistry(skills))
    selected = registry.select_skills()
    assert [skill.name for skill in selected] == ["b"]


def test_tool_provider_registry_select_tool_names() -> None:
    """Registry should filter by tool names."""

    skills = [
        _build_skill("a", ExecutionMode.CLIENT),
        _build_skill("b", ExecutionMode.PLUGIN),
    ]
    registry = ToolProviderRegistry(skill_registry=_DummyRegistry(skills))
    selected = registry.select_skills(tool_names=["b"])
    assert [skill.name for skill in selected] == ["b"]
