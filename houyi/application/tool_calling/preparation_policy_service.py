"""Consent and policy gating helpers for tool-call preparation."""

from __future__ import annotations

from typing import Any


class _ToolCallPreparationPolicyService:
    """Resolve consent/policy decisions before tool execution is prepared."""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    async def handle_consent_rejection(
        self,
        *,
        tool_name: str | None,
        args: dict[str, Any],
        tool_call_id: str | None,
        index: int,
        round_index_value: int | None,
        parallel_group_id: str | None,
        requested_tool_name: str | None,
    ) -> tuple[int, dict[str, Any], dict[str, Any], float] | None:
        if not tool_name:
            return None
        consent_granted, trace_entry, tool_message = await self.check_consent_if_required(
            tool_name,
            args,
            tool_call_id,
        )
        if consent_granted:
            return None
        if trace_entry and tool_message:
            trace_entry["round_index"] = round_index_value
            trace_entry["parallel_group_id"] = parallel_group_id
            trace_entry["requested_tool_name"] = requested_tool_name
        return index, trace_entry or {}, tool_message or {}, 0.0

    async def check_consent_if_required(
        self,
        tool_name: str,
        args: dict[str, Any],
        tool_call_id: str | None,
    ) -> tuple[bool, dict[str, Any] | None, dict[str, Any] | None]:
        if not self._runner.policy_enforcer:
            return True, None, None

        consent_given = self._runner._consent_cache.get(tool_name, False)
        decision = self._runner.policy_enforcer.check_invocation(
            skill_name=tool_name,
            is_model_initiated=True,
            user_consent_given=consent_given,
        )

        if decision.requires_consent and self._runner.consent_manager:
            from houyi.domain.skill.consent import ConsentRequest, ConsentType

            policy = self._runner.policy_enforcer.get_policy(tool_name)
            consent_request = ConsentRequest(
                consent_type=ConsentType.INVOKE_CONFIRM,
                skill_name=tool_name,
                operation=f"invoke tool '{tool_name}'",
                policy=policy,
                context={"args": args, "tool_call_id": tool_call_id},
            )
            consent_response = await self._runner.consent_manager.request_consent(consent_request)
            if consent_response.is_granted():
                self._runner._consent_cache[tool_name] = True
                return True, None, None

            self._runner._emit_tool_usage_blocked(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                reason="consent_denied",
            )
            trace_entry, tool_message = self._runner._build_blocked_tool_trace_and_message(
                tool_name=tool_name,
                args=args,
                tool_call_id=tool_call_id,
                error_code="consent_denied",
                message=f"User denied consent for tool '{tool_name}'",
                block_reason="consent_denied",
            )
            return False, trace_entry, tool_message

        if not decision.allowed:
            policy_reason = decision.reason or "policy_denied"
            self._runner._emit_tool_usage_blocked(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                reason=policy_reason,
            )
            trace_entry, tool_message = self._runner._build_blocked_tool_trace_and_message(
                tool_name=tool_name,
                args=args,
                tool_call_id=tool_call_id,
                error_code="policy_denied",
                message=decision.reason or f"Policy denied invocation of tool '{tool_name}'",
                block_reason=policy_reason,
            )
            return False, trace_entry, tool_message

        return True, None, None


__all__ = ["_ToolCallPreparationPolicyService"]
