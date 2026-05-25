"""CoreGuard: built-in global PreToolUse hook for core tool hijacking prevention.

This module implements the HouYi Core Protection scheme.
It evaluates calls to ext__ tools with dangerous side effects and either
allows, denies, or requires user consent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from houyi.domain.skill.registry import SkillRegistry

logger = logging.getLogger(__name__)

EXT_PREFIX = "ext__"


class CoreGuardDecision(str, Enum):
    """Decision returned by CoreGuard for a tool call attempt."""

    ALLOW = "allow"
    DENY = "deny"
    ALLOW_WITH_CONSENT = "allow_with_consent"


@dataclass(frozen=True)
class CoreGuardResult:
    """Result of a CoreGuard evaluation."""

    decision: CoreGuardDecision
    tool_name: str
    reason: str = ""

    @property
    def is_allowed(self) -> bool:
        return self.decision == CoreGuardDecision.ALLOW

    @property
    def is_denied(self) -> bool:
        return self.decision == CoreGuardDecision.DENY

    @property
    def needs_consent(self) -> bool:
        return self.decision == CoreGuardDecision.ALLOW_WITH_CONSENT

    @classmethod
    def allow(cls, tool_name: str) -> CoreGuardResult:
        return cls(decision=CoreGuardDecision.ALLOW, tool_name=tool_name)

    @classmethod
    def deny(cls, tool_name: str, reason: str) -> CoreGuardResult:
        return cls(decision=CoreGuardDecision.DENY, tool_name=tool_name, reason=reason)

    @classmethod
    def consent(cls, tool_name: str, reason: str) -> CoreGuardResult:
        return cls(
            decision=CoreGuardDecision.ALLOW_WITH_CONSENT,
            tool_name=tool_name,
            reason=reason,
        )


def _resolve_side_effect(skill: Any) -> str | None:
    policy = getattr(skill, "invocation_policy", None)
    if policy is None:
        return None
    if hasattr(policy, "side_effect"):
        side_effect = policy.side_effect
        return str(side_effect.value) if hasattr(side_effect, "value") else str(side_effect)
    if isinstance(policy, dict):
        return str(policy.get("side_effect", ""))
    return None


def _has_write_permission(skill: Any) -> bool:
    perms = getattr(skill, "permissions", None)
    if perms is None:
        return False
    if hasattr(perms, "filesystem"):
        filesystem = perms.filesystem
        if filesystem is None:
            return False
        return bool(getattr(filesystem, "write", False) or getattr(filesystem, "delete", False))
    if isinstance(perms, dict):
        filesystem = perms.get("filesystem", {})
        if isinstance(filesystem, dict):
            return bool(filesystem.get("write", False) or filesystem.get("delete", False))
    return False


def evaluate(tool_name: str, registry: SkillRegistry | None = None) -> CoreGuardResult:
    """Evaluate whether a tool call should be allowed, denied, or consented."""
    if not tool_name.startswith(EXT_PREFIX):
        return CoreGuardResult.allow(tool_name)

    if registry is None:
        logger.debug(
            "CoreGuard: no registry provided for tool '%s'; defaulting to ALLOW",
            tool_name,
        )
        return CoreGuardResult.allow(tool_name)

    skill = registry.get(tool_name)
    if skill is None:
        logger.debug(
            "CoreGuard: tool '%s' not found in registry; defaulting to ALLOW",
            tool_name,
        )
        return CoreGuardResult.allow(tool_name)

    side_effect = _resolve_side_effect(skill)

    if side_effect == "exec":
        reason = (
            f"CoreGuard: ext__ tool '{tool_name}' with SideEffect.EXEC is blocked. "
            "Use the equivalent [CORE OFFICIAL TOOL] instead."
        )
        logger.warning("CoreGuard DENY: %s", reason)
        return CoreGuardResult.deny(tool_name, reason)

    if side_effect == "filesystem":
        if _has_write_permission(skill):
            reason = (
                f"CoreGuard: ext__ tool '{tool_name}' with filesystem write/delete "
                "access is blocked. Use the equivalent [CORE OFFICIAL TOOL] instead."
            )
            logger.warning("CoreGuard DENY: %s", reason)
            return CoreGuardResult.deny(tool_name, reason)
        reason = (
            f"CoreGuard: ext__ tool '{tool_name}' requests filesystem read access. "
            "Explicit user consent is required."
        )
        logger.info("CoreGuard CONSENT required: %s", reason)
        return CoreGuardResult.consent(tool_name, reason)

    return CoreGuardResult.allow(tool_name)


__all__ = [
    "EXT_PREFIX",
    "CoreGuardDecision",
    "CoreGuardResult",
    "evaluate",
]
