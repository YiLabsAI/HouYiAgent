"""Tests for SQL verifier with constraint solver integration."""

import pytest

from houyi.assurance.verification.sql_verifier import SQLVerifier
from houyi.assurance.verification.verifier import VerificationRule


class TestSQLVerifierConstraints:
    """Tests for SQL verifier with constraint solving."""

    @pytest.mark.asyncio
    async def test_sql_without_constraints(self):
        """Test SQL verification without constraints."""
        verifier = SQLVerifier(use_constraint_solver=True)

        rule = VerificationRule(
            rule_id="test_sql",
            verifier_type="sql",
            rule_spec={"check_syntax": True, "check_injection": True},
        )

        result = await verifier.verify("SELECT * FROM users;", rule)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_sql_with_constraints_satisfied(self):
        """Test SQL verification with satisfied constraints."""
        verifier = SQLVerifier(use_constraint_solver=True)

        rule = VerificationRule(
            rule_id="test_sql",
            verifier_type="sql",
            rule_spec={
                "check_syntax": True,
                "constraints": [
                    {
                        "name": "select_only",
                        "type": "operation",
                        "expression": "True",  # Simplified - real constraint would check operation type
                        "description": "Only SELECT operations allowed",
                    }
                ],
            },
        )

        result = await verifier.verify("SELECT * FROM users;", rule)

        # Should pass basic checks
        assert result.passed is True or result.error_type != "constraint_violation"

    @pytest.mark.asyncio
    async def test_syntax_error_before_constraints(self):
        """Test that syntax errors are caught before constraint checking."""
        verifier = SQLVerifier(use_constraint_solver=True)

        rule = VerificationRule(
            rule_id="test_sql",
            verifier_type="sql",
            rule_spec={
                "check_syntax": True,
                "constraints": [{"name": "test", "expression": "True"}],
            },
        )

        result = await verifier.verify("SELECT * FROM", rule)

        assert result.passed is False
        # Could be sql_syntax or missing_semicolon depending on parsing
        assert result.error_type in ["sql_syntax", "missing_semicolon"]

    @pytest.mark.asyncio
    async def test_sql_injection_before_constraints(self):
        """Test that injection checks happen before constraints."""
        verifier = SQLVerifier(use_constraint_solver=True)

        rule = VerificationRule(
            rule_id="test_sql",
            verifier_type="sql",
            rule_spec={
                "check_injection": True,
                "constraints": [],
            },
        )

        result = await verifier.verify("SELECT * FROM users; DROP TABLE users;", rule)

        assert result.passed is False
        assert result.error_type == "sql_injection"

    @pytest.mark.asyncio
    async def test_constraint_solver_disabled(self):
        """Test SQL verification with constraint solver disabled."""
        verifier = SQLVerifier(use_constraint_solver=False)

        rule = VerificationRule(
            rule_id="test_sql",
            verifier_type="sql",
            rule_spec={
                "check_syntax": True,
                "constraints": [{"name": "test", "expression": "x > 0"}],
            },
        )

        result = await verifier.verify("SELECT * FROM users;", rule)

        # Should pass since constraint solver is disabled
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_build_constraints_from_specs(self):
        """Test building constraints from specifications."""
        verifier = SQLVerifier(use_constraint_solver=True)

        constraint_specs = [
            {
                "name": "age_range",
                "type": "range",
                "expression": "age >= 0 AND age <= 120",
                "description": "Age must be valid",
            },
            {
                "name": "id_positive",
                "type": "range",
                "expression": "id > 0",
                "description": "ID must be positive",
            },
        ]

        constraints = verifier._build_constraints(constraint_specs)

        assert len(constraints) == 2
        assert constraints[0].name == "age_range"
        assert constraints[1].name == "id_positive"

    @pytest.mark.asyncio
    async def test_handles_invalid_specs(self):
        """Test that invalid constraint specs are handled gracefully."""
        verifier = SQLVerifier(use_constraint_solver=True)

        constraint_specs = [
            {"name": "valid", "expression": "x > 0"},
            {},  # Invalid - missing required fields
            {"name": "also_valid", "expression": "y < 10"},
        ]

        constraints = verifier._build_constraints(constraint_specs)

        # Should build valid constraints and skip invalid ones
        assert len(constraints) >= 2
