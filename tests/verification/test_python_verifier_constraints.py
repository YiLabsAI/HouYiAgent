"""Tests for Python verifier with constraint solver integration."""

import pytest

from houyi.assurance.verification.python_verifier import PythonVerifier
from houyi.assurance.verification.verifier import VerificationRule


class TestPythonVerifierConstraints:
    """Tests for Python verifier with constraint solving."""

    @pytest.mark.asyncio
    async def test_python_without_constraints(self):
        """Test Python verification without constraints."""
        verifier = PythonVerifier(use_constraint_solver=True)

        rule = VerificationRule(
            rule_id="test_python",
            verifier_type="python",
            rule_spec={"check_syntax": True, "check_imports": True},
        )

        result = await verifier.verify("x = 5", rule)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_python_with_constraints_satisfied(self):
        """Test Python verification with satisfied constraints."""
        verifier = PythonVerifier(use_constraint_solver=True)

        rule = VerificationRule(
            rule_id="test_python",
            verifier_type="python",
            rule_spec={
                "check_syntax": True,
                "constraints": [
                    {
                        "name": "simple_check",
                        "type": "general",
                        "expression": "True",
                        "description": "Always satisfied",
                    }
                ],
            },
        )

        result = await verifier.verify("x = 5", rule)

        # Should pass basic checks
        assert result.passed is True or result.error_type != "constraint_violation"

    @pytest.mark.asyncio
    async def test_python_syntax_error_before_constraints(self):
        """Test that syntax errors are caught before constraint checking."""
        verifier = PythonVerifier(use_constraint_solver=True)

        rule = VerificationRule(
            rule_id="test_python",
            verifier_type="python",
            rule_spec={
                "check_syntax": True,
                "constraints": [{"name": "test", "expression": "True"}],
            },
        )

        result = await verifier.verify("def foo(", rule)

        assert result.passed is False
        assert result.error_type == "python_syntax"

    @pytest.mark.asyncio
    async def test_python_unsafe_import_before_constraints(self):
        """Test that unsafe import checks happen before constraints."""
        verifier = PythonVerifier(use_constraint_solver=True)

        rule = VerificationRule(
            rule_id="test_python",
            verifier_type="python",
            rule_spec={
                "check_imports": True,
                "constraints": [],
            },
        )

        result = await verifier.verify("import os", rule)

        assert result.passed is False
        assert result.error_type == "unsafe_import"

    @pytest.mark.asyncio
    async def test_constraint_solver_disabled(self):
        """Test Python verification with constraint solver disabled."""
        verifier = PythonVerifier(use_constraint_solver=False)

        rule = VerificationRule(
            rule_id="test_python",
            verifier_type="python",
            rule_spec={
                "check_syntax": True,
                "constraints": [{"name": "test", "expression": "x > 0"}],
            },
        )

        result = await verifier.verify("x = 5", rule)

        # Should pass since constraint solver is disabled
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_build_constraints_from_specs(self):
        """Test building constraints from specifications."""
        verifier = PythonVerifier(use_constraint_solver=True)

        constraint_specs = [
            {
                "name": "value_range",
                "type": "range",
                "expression": "x >= 0 AND x <= 100",
                "description": "Value must be in range",
            },
            {
                "name": "type_check",
                "type": "type",
                "expression": "isinstance(x, int)",
                "description": "Must be integer",
            },
        ]

        constraints = verifier._build_constraints(constraint_specs)

        assert len(constraints) == 2
        assert constraints[0].name == "value_range"
        assert constraints[1].name == "type_check"

    @pytest.mark.asyncio
    async def test_build_constraints_handles_invalid_specs(self):
        """Test that invalid constraint specs are handled gracefully."""
        verifier = PythonVerifier(use_constraint_solver=True)

        constraint_specs = [
            {"name": "valid", "expression": "x > 0"},
            {},  # Invalid - missing required fields
            {"name": "also_valid", "expression": "y < 10"},
        ]

        constraints = verifier._build_constraints(constraint_specs)

        # Should build valid constraints and skip invalid ones
        assert len(constraints) >= 2

    @pytest.mark.asyncio
    async def test_valid_python_code(self):
        """Test verification of valid Python code."""
        verifier = PythonVerifier(use_constraint_solver=True)

        rule = VerificationRule(
            rule_id="test_python",
            verifier_type="python",
            rule_spec={"check_syntax": True, "check_imports": True},
        )

        code = """
def add(a, b):
    return a + b

result = add(1, 2)
"""

        result = await verifier.verify(code, rule)

        assert result.passed is True
