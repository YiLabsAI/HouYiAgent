"""Constraint checking implementation."""

from typing import Any

from houyi.verification.verifier import VerificationResult, VerificationRule, Verifier


class ConstraintChecker(Verifier):
    """Verifies constraints like type, range, and business rules."""

    def __init__(self, use_cache: bool = True):
        """Initialize constraint checker."""
        super().__init__(use_cache=use_cache)

    async def _verify_impl(
        self,
        output: Any,
        rule: VerificationRule,
        context: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """Verify constraints."""
        rule_spec = rule.rule_spec

        # Type checking
        expected_type = rule_spec.get("expected_type")
        if expected_type and not isinstance(output, expected_type):
            return VerificationResult(
                rule_id=rule.rule_id,
                passed=False,
                error_message=f"Type mismatch: expected {expected_type}, got {type(output)}",
                error_type="type_mismatch",
                auto_fixable=True,
                severity=rule.severity,
            )

        # Range checking
        min_val = rule_spec.get("min_value")
        max_val = rule_spec.get("max_value")
        if min_val is not None and output < min_val:
            return VerificationResult(
                rule_id=rule.rule_id,
                passed=False,
                error_message=f"Value {output} below minimum {min_val}",
                error_type="range_violation",
                severity=rule.severity,
            )
        if max_val is not None and output > max_val:
            return VerificationResult(
                rule_id=rule.rule_id,
                passed=False,
                error_message=f"Value {output} above maximum {max_val}",
                error_type="range_violation",
                severity=rule.severity,
            )

        return VerificationResult(rule_id=rule.rule_id, passed=True)
