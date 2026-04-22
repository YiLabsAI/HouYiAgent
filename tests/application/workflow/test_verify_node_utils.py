"""Verification rule normalization scenarios for workflow VERIFY nodes."""

from houyi.application.workflow.verify_node_utils import build_verification_rules


class TestBuildVerificationRules:
    def test_build_rules_from_dicts(self):
        config = {
            "verification_rules": [
                {
                    "rule_id": "r1",
                    "verifier_type": "constraint",
                    "rule_spec": {"require_keys": ["a"]},
                    "severity": "error",
                }
            ]
        }
        rules = build_verification_rules(config)
        assert len(rules) == 1
        assert rules[0].rule_id == "r1"

    def test_rules_backward_require_keys(self):
        rules = build_verification_rules({"require_keys": ["a", 2]})
        assert len(rules) == 1
        assert rules[0].rule_id == "require_keys"
        assert rules[0].rule_spec["require_keys"] == ["a", "2"]

    def test_rules_backward_min_results(self):
        rules = build_verification_rules({"min_results": "2", "results_path": "items"})
        assert len(rules) == 1
        assert rules[0].rule_id == "min_results"
        assert rules[0].rule_spec["min_items"] == 2
        assert rules[0].rule_spec["min_items_path"] == "items"

    def test_build_rules_ignores_invalid(self):
        rules = build_verification_rules({"min_results": "bad"})
        assert rules == []
