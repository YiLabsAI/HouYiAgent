from __future__ import annotations

from typing import Any

from houyi.core.skill import SkillSpec


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillSpec] = {}

    def register(self, skill: SkillSpec, *, overwrite: bool = False) -> None:
        name = str(getattr(skill, "name", "") or "").strip()
        if not name:
            raise ValueError("Skill name is required")
        if not overwrite and name in self._skills:
            raise ValueError(f"Skill already registered: {name}")
        self._skills[name] = skill

    def get(self, name: str) -> SkillSpec | None:
        return self._skills.get(name)

    def list(self) -> list[SkillSpec]:
        return list(self._skills.values())

    def as_tool_schemas(self) -> list[dict[str, Any]]:
        return [skill.to_tool_schema() for skill in self._skills.values()]


DEFAULT_SKILL_REGISTRY = SkillRegistry()

__all__ = [
    "DEFAULT_SKILL_REGISTRY",
    "SkillRegistry",
]
