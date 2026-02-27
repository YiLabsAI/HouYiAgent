"""CoreGuard: built-in global PreToolUse hook for core tool hijacking prevention.

This module implements the HouYi Core Protection scheme.
It registers a global PreToolUse hook handler that intercepts calls to
``ext__`` tools with dangerous side effects and either denies them or
requires user consent.

Protection rules:
- ``ext__`` tool + SideEffect.EXEC                       → DENY
- ``ext__`` tool + SideEffect.FILESYSTEM + write/delete  → DENY
- ``ext__`` tool + SideEffect.FILESYSTEM + read-only     → ALLOW_WITH_CONSENT
- ``ext__`` tool + SideEffect.NETWORK                    → ALLOW (controlled by tool's own policy)
- ``ext__`` tool + SideEffect.NONE                       → ALLOW
- non-``ext__`` tool (any)                               → CoreGuard does not intervene
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from houyi.core.skill_registry import SkillRegistry

logger = logging.getLogger(__name__)

EXT_PREFIX = "ext__"


class CoreGuardDecision(str, Enum):
    """Decision returned by CoreGuard for a tool call attempt."""

    ALLOW = "allow"
    DENY = "deny"
    ALLOW_WITH_CONSENT = "allow_with_consent"


@dataclass(frozen=True)
class CoreGuardResult:
    """Result of a CoreGuard evaluation.

    Attributes:
        decision:   The routing decision (ALLOW / DENY / ALLOW_WITH_CONSENT).
        reason:     Human-readable explanation (empty string if ALLOW).
        tool_name:  The tool name that was evaluated.
    """

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
    """Extract the side_effect string from a SkillSpec's invocation_policy."""
    policy = getattr(skill, "invocation_policy", None)
    if policy is None:
        return None
    # InvocationPolicy may have side_effect as an attribute or dict key
    if hasattr(policy, "side_effect"):
        se = policy.side_effect
        return str(se.value) if hasattr(se, "value") else str(se)
    if isinstance(policy, dict):
        return str(policy.get("side_effect", ""))
    return None


def _has_write_permission(skill: Any) -> bool:
    """Return True if the skill's permissions include filesystem write or delete."""
    perms = getattr(skill, "permissions", None)
    if perms is None:
        return False
    # Permissions may be an object with a filesystem attribute or a dict
    if hasattr(perms, "filesystem"):
        fs = perms.filesystem
        if fs is None:
            return False
        write = getattr(fs, "write", False) or getattr(fs, "delete", False)
        return bool(write)
    if isinstance(perms, dict):
        fs = perms.get("filesystem", {})
        if isinstance(fs, dict):
            return bool(fs.get("write", False) or fs.get("delete", False))
    return False


def evaluate(tool_name: str, registry: SkillRegistry | None = None) -> CoreGuardResult:
    """Evaluate whether a tool call should be allowed, denied, or consented.

    This is the primary entry point for CoreGuard.  It is designed to be
    called from a ``PreToolUse`` hook handler registered at Host startup.

    Args:
        tool_name:  Name of the tool about to be invoked.
        registry:   SkillRegistry instance for skill lookup.  If ``None``,
                    CoreGuard cannot inspect the skill's side effects and
                    will ALLOW by default (fail-open for missing registry).

    Returns:
        A :class:`CoreGuardResult` with the decision and a reason string.
    """
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
        # Read-only filesystem access requires consent
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
