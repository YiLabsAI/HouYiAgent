from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from houyi.application.tool_calling.preparation_policy_service import (
    _ToolCallPreparationPolicyService,
)


def _mock_runner(
    *,
    policy_enforcer=None,
    consent_manager=None,
    consent_cache: dict | None = None,
) -> MagicMock:
    runner = MagicMock()
    runner.policy_enforcer = policy_enforcer
    runner.consent_manager = consent_manager
    runner._consent_cache = consent_cache or {}
    runner._emit_tool_usage_blocked = MagicMock()
    runner._build_blocked_tool_trace_and_message = MagicMock(
        return_value=({"trace": True}, {"msg": True})
    )
    return runner


class TestCheckConsent:
    @pytest.mark.asyncio
    async def test_no_enforcer_grants(self) -> None:
        svc = _ToolCallPreparationPolicyService(_mock_runner(policy_enforcer=None))
        granted, trace, msg = await svc.check_consent_if_required("tool", {}, "c1")
        assert granted is True
        assert trace is None and msg is None

    @pytest.mark.asyncio
    async def test_policy_allowed(self) -> None:
        enforcer = MagicMock()
        enforcer.check_invocation.return_value = SimpleNamespace(
            requires_consent=False, allowed=True, reason=None
        )
        svc = _ToolCallPreparationPolicyService(_mock_runner(policy_enforcer=enforcer))
        granted, _, _ = await svc.check_consent_if_required("tool", {}, "c1")
        assert granted is True

    @pytest.mark.asyncio
    async def test_policy_denied(self) -> None:
        enforcer = MagicMock()
        enforcer.check_invocation.return_value = SimpleNamespace(
            requires_consent=False, allowed=False, reason="blocked"
        )
        runner = _mock_runner(policy_enforcer=enforcer)
        svc = _ToolCallPreparationPolicyService(runner)
        granted, trace, msg = await svc.check_consent_if_required("tool", {}, "c1")
        assert granted is False
        assert trace is not None
        runner._emit_tool_usage_blocked.assert_called_once()

    @pytest.mark.asyncio
    async def test_consent_granted(self) -> None:
        enforcer = MagicMock()
        enforcer.check_invocation.return_value = SimpleNamespace(
            requires_consent=True, allowed=True, reason=None
        )
        enforcer.get_policy.return_value = MagicMock()

        consent_mgr = AsyncMock()
        consent_mgr.request_consent.return_value = MagicMock(is_granted=lambda: True)

        runner = _mock_runner(policy_enforcer=enforcer, consent_manager=consent_mgr)
        svc = _ToolCallPreparationPolicyService(runner)
        granted, _, _ = await svc.check_consent_if_required("tool", {}, "c1")
        assert granted is True
        assert runner._consent_cache["tool"] is True

    @pytest.mark.asyncio
    async def test_consent_denied(self) -> None:
        enforcer = MagicMock()
        enforcer.check_invocation.return_value = SimpleNamespace(
            requires_consent=True, allowed=True, reason=None
        )
        enforcer.get_policy.return_value = MagicMock()

        consent_mgr = AsyncMock()
        consent_mgr.request_consent.return_value = MagicMock(is_granted=lambda: False)

        runner = _mock_runner(policy_enforcer=enforcer, consent_manager=consent_mgr)
        svc = _ToolCallPreparationPolicyService(runner)
        granted, trace, msg = await svc.check_consent_if_required("tool", {}, "c1")
        assert granted is False
        runner._emit_tool_usage_blocked.assert_called_once()


class TestHandleConsentRejection:
    @pytest.mark.asyncio
    async def test_none_tool_name(self) -> None:
        svc = _ToolCallPreparationPolicyService(_mock_runner())
        result = await svc.handle_consent_rejection(
            tool_name=None,
            args={},
            tool_call_id="c1",
            index=0,
            round_index_value=0,
            parallel_group_id=None,
            requested_tool_name=None,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_consent_granted(self) -> None:
        enforcer = MagicMock()
        enforcer.check_invocation.return_value = SimpleNamespace(
            requires_consent=False, allowed=True, reason=None
        )
        svc = _ToolCallPreparationPolicyService(_mock_runner(policy_enforcer=enforcer))
        result = await svc.handle_consent_rejection(
            tool_name="t",
            args={},
            tool_call_id="c1",
            index=0,
            round_index_value=1,
            parallel_group_id="pg",
            requested_tool_name="t",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_rejection_returns_tuple(self) -> None:
        enforcer = MagicMock()
        enforcer.check_invocation.return_value = SimpleNamespace(
            requires_consent=False, allowed=False, reason="denied"
        )
        runner = _mock_runner(policy_enforcer=enforcer)
        svc = _ToolCallPreparationPolicyService(runner)
        result = await svc.handle_consent_rejection(
            tool_name="t",
            args={},
            tool_call_id="c1",
            index=3,
            round_index_value=2,
            parallel_group_id="pg1",
            requested_tool_name="t",
        )
        assert result is not None
        idx, trace, msg, duration = result
        assert idx == 3
