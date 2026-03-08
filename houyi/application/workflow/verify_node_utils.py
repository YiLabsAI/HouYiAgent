"""Helpers for VERIFY node execution."""

from __future__ import annotations

from typing import Any

from houyi.assurance.verification import VerificationRule


def build_verification_rules(config: dict[str, Any]) -> list[VerificationRule]:
    rules: list[VerificationRule] = []

    rule_dicts = config.get("verification_rules")
    if isinstance(rule_dicts, list):
        for raw in rule_dicts:
            if isinstance(raw, dict):
                rules.append(VerificationRule.model_validate(raw))

    # Backward-compatible lightweight config -> unified VerificationRule format
    if rules:
        return rules

    require_keys = config.get("require_keys") or []
    if isinstance(require_keys, list) and require_keys:
        rules.append(
            VerificationRule(
                rule_id="require_keys",
                verifier_type="constraint",
                rule_spec={"require_keys": [str(k) for k in require_keys]},
                severity="error",
            )
        )

    min_results = config.get("min_results")
    results_path = config.get("results_path", "results")
    if min_results is not None:
        try:
            min_results_int = int(min_results)
        except Exception:
            min_results_int = None
        if min_results_int is not None:
            rules.append(
                VerificationRule(
                    rule_id="min_results",
                    verifier_type="constraint",
                    rule_spec={
                        "min_items_path": str(results_path),
                        "min_items": min_results_int,
                    },
                    severity="error",
                )
            )

    return rules
