"""Helpers for ROUTE node execution."""

from __future__ import annotations

from typing import Any


def evaluate_route(
    *,
    config: dict[str, Any],
    inputs: dict[str, Any] | None,
) -> tuple[str, bool, list[str], dict[str, Any] | None, list[dict[str, Any]]]:
    """Evaluate route configuration.

    Returns:
        (route_mode, condition, target_ids, matched_rule, trace)
    """

    inputs = inputs or {}

    route_rules = config.get("route_rules")
    rule_trace: list[dict[str, Any]] = []
    matched_rule: dict[str, Any] | None = None
    route_mode = "legacy"
    target_ids: list[str] = []
    condition: bool | None = None

    if isinstance(route_rules, list) and route_rules:
        route_mode = "rules"
        for idx, rule in enumerate(route_rules):
            if not isinstance(rule, dict):
                continue
            when = rule.get("when", rule.get("condition"))
            label = rule.get("label")
            matched = False
            reason = None

            if isinstance(when, bool):
                matched = when
                reason = "bool"
            elif isinstance(when, str):
                matched = bool(inputs.get(when))
                reason = "key"
            elif when is None:
                matched = False
                reason = "missing"

            rule_trace.append(
                {"index": idx, "label": label, "when": when, "matched": matched, "reason": reason}
            )

            if not matched:
                continue

            then_block = rule.get("then") if isinstance(rule.get("then"), dict) else {}
            disable_nodes = rule.get("disable_nodes")
            if disable_nodes is None:
                disable_nodes = then_block.get("disable_nodes")  # type: ignore[union-attr]

            if isinstance(disable_nodes, list):
                target_ids = [str(item) for item in disable_nodes]
            else:
                target_ids = []

            matched_rule = {
                "index": idx,
                "label": label,
            }
            condition = True
            break

        if condition is None:
            condition = False

    # Legacy config fallback
    if route_mode == "legacy" or (route_mode == "rules" and not matched_rule):
        condition_value = inputs.get("condition")
        if condition_value is None:
            condition_value = inputs.get("verified")
        condition = bool(condition_value)

        disable_on_true = config.get("disable_nodes_on_true") or []
        disable_on_false = config.get("disable_nodes_on_false") or []

        if condition:
            if isinstance(disable_on_true, list):
                target_ids = [str(item) for item in disable_on_true]
        else:
            if isinstance(disable_on_false, list):
                target_ids = [str(item) for item in disable_on_false]

    return route_mode, bool(condition), target_ids, matched_rule, rule_trace
