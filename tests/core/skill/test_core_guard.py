"""Tests for CoreGuard PreToolUse hook.

Covers:
- DENY for ext__ tools with EXEC side effect
- DENY for ext__ tools with FILESYSTEM write/delete
- ALLOW_WITH_CONSENT for ext__ tools with FILESYSTEM read
- ALLOW for ext__ tools with NETWORK side effect
- ALLOW for ext__ tools with no side effect
- Non-ext__ tools are never intercepted by CoreGuard
- Warning logged on DENY
- Missing registry falls back to ALLOW
- Unknown tool name falls back to ALLOW
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from houyi.domain.skill.core_guard import CoreGuardDecision, CoreGuardResult, evaluate
from houyi.domain.skill.registry import SkillRegistry
from houyi.domain.skill.spec import SkillSpec


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    r: str


def _skill_with_side_effect(
    name: str,
    side_effect: str | None = None,
    has_write: bool = False,
) -> SkillSpec:
    """Build a SkillSpec with a mocked invocation_policy side_effect."""
    skill = SkillSpec(
        name=name,
        description=f"Tool {name}",
        input_schema=_In,
        output_schema=_Out,
    )
    if side_effect is not None:
        mock_policy = MagicMock()
        mock_policy.side_effect = MagicMock()
        mock_policy.side_effect.value = side_effect
        object.__setattr__(skill, "invocation_policy", mock_policy)
    if has_write:
        mock_perms = MagicMock()
        mock_fs = MagicMock()
        mock_fs.write = True
        mock_fs.delete = False
        mock_perms.filesystem = mock_fs
        object.__setattr__(skill, "permissions", mock_perms)
    return skill


def _registry_with(*skills: SkillSpec) -> SkillRegistry:
    registry = SkillRegistry()
    for s in skills:
        registry._skills[s.name] = s
    return registry


class TestCoreGuardDecisions:
    """CoreGuard evaluation decision tests."""

    def test_non_ext_tool_always_allowed(self) -> None:
        """Non-ext__ tools must never be intercepted by CoreGuard."""
        result = evaluate("web_search", registry=None)
        assert result.decision == CoreGuardDecision.ALLOW
        assert result.tool_name == "web_search"

    def test_ext_tool_with_exec_side_effect_denied(self) -> None:
        """ext__ tool with EXEC side effect must be DENIED."""
        skill = _skill_with_side_effect("ext__dangerous", side_effect="exec")
        registry = _registry_with(skill)
        result = evaluate("ext__dangerous", registry=registry)
        assert result.is_denied
        assert "EXEC" in result.reason or "exec" in result.reason.lower()

    def test_ext_tool_with_filesystem_write_denied(self) -> None:
        """ext__ tool with filesystem side effect + write permission must be DENIED."""
        skill = _skill_with_side_effect("ext__writer", side_effect="filesystem", has_write=True)
        registry = _registry_with(skill)
        result = evaluate("ext__writer", registry=registry)
        assert result.is_denied
        assert "write" in result.reason.lower() or "filesystem" in result.reason.lower()

    def test_ext_tool_with_filesystem_read_needs_consent(self) -> None:
        """ext__ tool with filesystem side effect but no write permission → CONSENT."""
        skill = _skill_with_side_effect("ext__reader", side_effect="filesystem", has_write=False)
        registry = _registry_with(skill)
        result = evaluate("ext__reader", registry=registry)
        assert result.needs_consent
        assert "filesystem" in result.reason.lower() or "consent" in result.reason.lower()

    def test_ext_tool_with_network_side_effect_allowed(self) -> None:
        """ext__ tool with NETWORK side effect is ALLOWED (controlled by tool's own policy)."""
        skill = _skill_with_side_effect("ext__searcher", side_effect="network")
        registry = _registry_with(skill)
        result = evaluate("ext__searcher", registry=registry)
        assert result.is_allowed

    def test_ext_tool_with_no_side_effect_allowed(self) -> None:
        """ext__ tool with no side effect is ALLOWED."""
        skill = _skill_with_side_effect("ext__noop", side_effect=None)
        registry = _registry_with(skill)
        result = evaluate("ext__noop", registry=registry)
        assert result.is_allowed

    def test_ext_tool_with_none_side_effect_allowed(self) -> None:
        """ext__ tool with explicit 'none' side effect is ALLOWED."""
        skill = _skill_with_side_effect("ext__safe", side_effect="none")
        registry = _registry_with(skill)
        result = evaluate("ext__safe", registry=registry)
        assert result.is_allowed

    def test_no_registry_defaults_to_allow(self) -> None:
        """If registry is None, CoreGuard defaults to ALLOW (fail-open)."""
        result = evaluate("ext__unknown", registry=None)
        assert result.is_allowed

    def test_unknown_tool_in_registry_defaults_to_allow(self) -> None:
        """ext__ tool not found in registry defaults to ALLOW."""
        registry = SkillRegistry()  # empty registry
        result = evaluate("ext__ghost_tool", registry=registry)
        assert result.is_allowed

    def test_deny_result_contains_tool_name(self) -> None:
        """DENY result must carry the tool name in the reason string."""
        skill = _skill_with_side_effect("ext__exec_tool", side_effect="exec")
        registry = _registry_with(skill)
        result = evaluate("ext__exec_tool", registry=registry)
        assert "ext__exec_tool" in result.reason

    def test_deny_logged_as_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """CoreGuard DENY decisions must be logged at WARNING level."""
        skill = _skill_with_side_effect("ext__risky", side_effect="exec")
        registry = _registry_with(skill)
        with caplog.at_level(logging.WARNING, logger="houyi.core.skill.core_guard"):
            evaluate("ext__risky", registry=registry)
        assert any("DENY" in r.message or "deny" in r.message.lower() for r in caplog.records)

    def test_core_guard_result_is_allowed_property(self) -> None:
        """CoreGuardResult.is_allowed / is_denied / needs_consent properties are correct."""
        allow = CoreGuardResult.allow("tool")
        deny = CoreGuardResult.deny("tool", "reason")
        consent = CoreGuardResult.consent("tool", "reason")
        assert allow.is_allowed and not allow.is_denied and not allow.needs_consent
        assert deny.is_denied and not deny.is_allowed and not deny.needs_consent
        assert consent.needs_consent and not consent.is_allowed and not consent.is_denied
