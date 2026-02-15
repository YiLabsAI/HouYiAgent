"""Tool Router — runtime allowed-tools enforcement and selection middleware.

The Tool Router sits between the LLM's tool selection and actual execution,
enforcing:
  1. ``allowed-tools`` white-lists from skill manifests
  2. ``InvocationPolicy`` checks before execution
  3. Centralized logging for audit and observability

Design reference: §3.4 (Tool Router), §4.6 of simpleskill-design.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from houyi.core.skill.policy import InvocationDecision, PolicyEnforcer
    from houyi.core.skill.spec import SkillSpec

logger = logging.getLogger(__name__)


@dataclass
class ToolRouteResult:
    """Result of a tool routing decision."""

    tool_name: str
    """Name of the tool being routed."""

    allowed: bool
    """Whether the tool is allowed to execute."""

    reason: str = ""
    """Human-readable reason for deny/consent decisions."""

    requires_consent: bool = False
    """Whether user consent is needed before execution."""

    matched_skill: str | None = None
    """Name of the skill that declares this tool (if any)."""


class ToolRouter:
    """Enforce allowed-tools white-lists and InvocationPolicy at runtime.

    The router is initialized with registered skills and optionally a
    ``PolicyEnforcer``.  For each tool call, it checks:

    1. Is the tool in the global tools list?
    2. Is the tool allowed by any registered skill's ``allowed_tools``?
    3. Does the ``InvocationPolicy`` permit execution?

    Usage::

        router = ToolRouter(skills, policy_enforcer)
        filtered_tools = router.filter_tools(tools)
        decision = router.check(tool_name, is_model_initiated=True)
    """

    def __init__(
        self,
        skills: list[SkillSpec] | None = None,
        policy_enforcer: PolicyEnforcer | None = None,
    ) -> None:
        self._policy_enforcer = policy_enforcer
        # Build the allowed-tools index: skill_name -> set of allowed tools
        self._skill_allowed_tools: dict[str, set[str]] = {}
        # Global set of all explicitly allowed tools (union across all skills)
        self._global_allowed: set[str] = set()
        # Skills that have an empty allowed_tools (no restriction)
        self._unrestricted_skills: set[str] = set()
        # Map tool name -> owning skill name (for policy lookups)
        self._tool_to_skill: dict[str, str] = {}

        for skill in skills or []:
            allowed = set(skill.allowed_tools) if skill.allowed_tools else set()
            if allowed:
                self._skill_allowed_tools[skill.name] = allowed
                self._global_allowed |= allowed
                for tool_name in allowed:
                    self._tool_to_skill[tool_name] = skill.name
            else:
                # Skill has no allowed_tools restriction — all tools permitted
                self._unrestricted_skills.add(skill.name)

    @property
    def has_restrictions(self) -> bool:
        """Whether any skill declares an allowed-tools whitelist."""
        return len(self._skill_allowed_tools) > 0

    def filter_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter the tools list to only those allowed by registered skills.

        If no skill declares ``allowed_tools``, all tools pass through.
        If any skill declares ``allowed_tools``, only tools in the union
        of all whitelists are kept.

        Args:
            tools: Raw tool definitions (each must have a ``function.name`` key).

        Returns:
            Filtered list of tools (new list, originals not mutated).
        """
        if not self.has_restrictions:
            return tools

        filtered: list[dict[str, Any]] = []
        for tool in tools:
            name = _extract_tool_name(tool)
            if not name:
                filtered.append(tool)  # Keep malformed tools for debugging
                continue
            if name in self._global_allowed:
                filtered.append(tool)
            elif self._unrestricted_skills:
                # If some skills have no restriction, allow their tools
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
        """Check whether a tool call should be allowed.

        Performs two-level checks:
        1. Allowed-tools whitelist (if any skill declares one)
        2. InvocationPolicy via PolicyEnforcer (if configured)

        Args:
            tool_name: Name of the tool being called.
            is_model_initiated: Whether the LLM initiated the call.
            user_consent_given: Whether the user has granted consent.

        Returns:
            A ToolRouteResult with the routing decision.
        """
        # Level 1: Allowed-tools whitelist
        if self.has_restrictions:
            if tool_name not in self._global_allowed and not self._unrestricted_skills:
                return ToolRouteResult(
                    tool_name=tool_name,
                    allowed=False,
                    reason=f"Tool '{tool_name}' not in any skill's allowed-tools whitelist",
                )

        # Level 2: InvocationPolicy
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
        """Check multiple tool calls at once.

        Args:
            tool_names: Names of tools being called.
            is_model_initiated: Whether the LLM initiated the calls.
            user_consent_given: Whether the user has granted consent.

        Returns:
            Mapping of tool_name -> ToolRouteResult.
        """
        return {
            name: self.check(name, is_model_initiated, user_consent_given) for name in tool_names
        }


def _extract_tool_name(tool: dict[str, Any]) -> str | None:
    """Extract the tool name from a tool definition dict.

    Supports both OpenAI-style (``function.name``) and flat (``name``) formats.
    """
    if "function" in tool and isinstance(tool["function"], dict):
        return tool["function"].get("name")
    return tool.get("name")
