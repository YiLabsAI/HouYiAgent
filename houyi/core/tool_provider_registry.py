"""Tool provider registry and selection policy."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from houyi.core.skill import ExecutionMode, SkillSpec
from houyi.core.skill_registry import SkillRegistry

_INTERNAL_MODES = {ExecutionMode.PLUGIN, ExecutionMode.MCP}


@dataclass(slots=True)
class ToolProviderRegistry:
    """Select tools using internal-first policy and execution_mode filtering."""

    skill_registry: SkillRegistry

    def list_skills(self) -> list[SkillSpec]:
        return self.skill_registry.list()

    def select_skills(
        self,
        tool_names: Iterable[str] | None = None,
        *,
        execution_modes: set[ExecutionMode] | None = None,
    ) -> list[SkillSpec]:
        skills = self.skill_registry.list()
        tool_names = [name for name in tool_names or [] if name]
        if tool_names:
            return [skill for skill in skills if skill.name in tool_names]
        modes = execution_modes or _INTERNAL_MODES
        internal = [skill for skill in skills if skill.execution_mode in modes]
        return internal or skills
