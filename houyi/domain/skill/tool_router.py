"""Tool Router — runtime allowed-tools enforcement and selection middleware."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from houyi.domain.skill.policy import InvocationDecision, PolicyEnforcer
    from houyi.domain.skill.spec import SkillSpec

logger = logging.getLogger(__name__)


@dataclass
class ToolRouteResult:
    """Result of a tool routing decision."""

    tool_name: str
    allowed: bool
    reason: str = ""
    requires_consent: bool = False
    matched_skill: str | None = None


class ToolRouter:
    """Enforce allowed-tools white-lists and InvocationPolicy at runtime."""

    def __init__(
        self,
        skills: list[SkillSpec] | None = None,
        policy_enforcer: PolicyEnforcer | None = None,
    ) -> None:
        self._policy_enforcer = policy_enforcer
        self._skill_allowed_tools: dict[str, set[str]] = {}
        self._global_allowed: set[str] = set()
        self._unrestricted_skills: set[str] = set()
        self._tool_to_skill: dict[str, str] = {}

        for skill in skills or []:
            allowed = set(skill.allowed_tools) if skill.allowed_tools else set()
            if allowed:
                self._skill_allowed_tools[skill.name] = allowed
                self._global_allowed |= allowed
                for tool_name in allowed:
                    self._tool_to_skill[tool_name] = skill.name
            else:
                self._unrestricted_skills.add(skill.name)

    @property
    def has_restrictions(self) -> bool:
        return len(self._skill_allowed_tools) > 0

    def filter_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.has_restrictions:
            return tools

        filtered: list[dict[str, Any]] = []
        for tool in tools:
            name = _extract_tool_name(tool)
            if not name:
                filtered.append(tool)
                continue
            if name in self._global_allowed or self._unrestricted_skills:
                filtered.append(tool)
            else:
                logger.debug(
                    "ToolRouter: filtered out tool '%s' (not in allowed-tools)",
                    name,
                )
        return filtered

    def check(
        self,
        tool_name: str,
        is_model_initiated: bool = True,
        user_consent_given: bool = False,
    ) -> ToolRouteResult:
        if (
            self.has_restrictions
            and tool_name not in self._global_allowed
            and not self._unrestricted_skills
        ):
            return ToolRouteResult(
                tool_name=tool_name,
                allowed=False,
                reason=f"Tool '{tool_name}' not in any skill's allowed-tools whitelist",
            )

        matched_skill = self._tool_to_skill.get(tool_name)
        if self._policy_enforcer and matched_skill:
            decision: InvocationDecision = self._policy_enforcer.check_invocation(
                skill_name=matched_skill,
                is_model_initiated=is_model_initiated,
                user_consent_given=user_consent_given,
            )
            if not decision.allowed:
                return ToolRouteResult(
                    tool_name=tool_name,
                    allowed=False,
                    reason=decision.reason or "Denied by InvocationPolicy",
                    requires_consent=decision.requires_consent,
                    matched_skill=matched_skill,
                )

        return ToolRouteResult(
            tool_name=tool_name,
            allowed=True,
            matched_skill=matched_skill,
        )

    def check_batch(
        self,
        tool_names: list[str],
        is_model_initiated: bool = True,
        user_consent_given: bool = False,
    ) -> dict[str, ToolRouteResult]:
        return {
            name: self.check(name, is_model_initiated, user_consent_given) for name in tool_names
        }


def _extract_tool_name(tool: dict[str, Any]) -> str | None:
    if "function" in tool and isinstance(tool["function"], dict):
        return tool["function"].get("name")
    return tool.get("name")


__all__ = ["ToolRouteResult", "ToolRouter"]
