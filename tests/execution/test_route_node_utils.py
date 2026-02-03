"""Unit tests for execution/route_node_utils.py."""

from houyi.execution.route_node_utils import evaluate_route


class TestEvaluateRouteRules:
    def test_rules_first_match_by_bool_when(self):
        config = {
            "route_rules": [
                {"label": "a", "when": False, "then": {"disable_nodes": ["n1"]}},
                {"label": "b", "when": True, "then": {"disable_nodes": ["n2", 3]}},
            ]
        }
        route_mode, condition, target_ids, matched_rule, trace = evaluate_route(
            config=config, inputs={}
        )

        assert route_mode == "rules"
        assert condition is True
        assert target_ids == ["n2", "3"]
        assert matched_rule == {"index": 1, "label": "b"}
        assert trace[0]["matched"] is False
        assert trace[1]["matched"] is True

    def test_rules_match_by_key_lookup(self):
        config = {
            "route_rules": [
                {"label": "k", "when": "flag", "disable_nodes": ["a"]},
            ]
        }
        route_mode, condition, target_ids, matched_rule, trace = evaluate_route(
            config=config, inputs={"flag": 1}
        )

        assert route_mode == "rules"
        assert condition is True
        assert target_ids == ["a"]
        assert matched_rule == {"index": 0, "label": "k"}
        assert trace[0]["reason"] == "key"

    def test_rules_no_match_falls_back_to_legacy(self):
        config = {
            "route_rules": [
                {"label": "k", "when": "flag", "disable_nodes": ["a"]},
            ],
            "disable_nodes_on_true": ["t"],
            "disable_nodes_on_false": ["f"],
        }
        route_mode, condition, target_ids, matched_rule, trace = evaluate_route(
            config=config, inputs={"condition": False}
        )

        assert route_mode == "rules"
        assert condition is False
        assert target_ids == ["f"]
        assert matched_rule is None
        assert trace


class TestEvaluateRouteLegacy:
    def test_legacy_condition_prefers_condition_over_verified(self):
        config = {"disable_nodes_on_true": ["a"], "disable_nodes_on_false": ["b"]}
        route_mode, condition, target_ids, matched_rule, trace = evaluate_route(
            config=config, inputs={"condition": 1, "verified": False}
        )

        assert route_mode == "legacy"
        assert condition is True
        assert target_ids == ["a"]
        assert matched_rule is None
        assert trace == []

    def test_legacy_condition_uses_verified_when_condition_missing(self):
        config = {"disable_nodes_on_true": ["a"], "disable_nodes_on_false": ["b"]}
        route_mode, condition, target_ids, matched_rule, trace = evaluate_route(
            config=config, inputs={"verified": "non-empty"}
        )

        assert route_mode == "legacy"
        assert condition is True
        assert target_ids == ["a"]
        assert matched_rule is None
        assert trace == []
