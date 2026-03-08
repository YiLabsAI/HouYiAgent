"""Tests for constraint checker."""

import pytest

from houyi.assurance.verification.constraint_checker import ConstraintChecker
from houyi.assurance.verification.verifier import VerificationRule


@pytest.fixture
def constraint_checker():
    """Create constraint checker instance."""
    return ConstraintChecker()


@pytest.mark.asyncio
async def test_type_check_pass(constraint_checker):
    """Test type check passes for correct type."""
    rule = VerificationRule(
        rule_id="test_type",
        verifier_type="constraint",
        rule_spec={"expected_type": int},
    )
    result = await constraint_checker.verify(42, rule)
    assert result.passed is True


@pytest.mark.asyncio
async def test_type_check_fail(constraint_checker):
    """Test type check fails for wrong type."""
    rule = VerificationRule(
        rule_id="test_type",
        verifier_type="constraint",
        rule_spec={"expected_type": int},
    )
    result = await constraint_checker.verify("42", rule)
    assert result.passed is False
    assert result.error_type == "type_mismatch"


@pytest.mark.asyncio
async def test_range_check_pass(constraint_checker):
    """Test range check passes for value in range."""
    rule = VerificationRule(
        rule_id="test_range",
        verifier_type="constraint",
        rule_spec={"min_value": 0, "max_value": 100},
    )
    result = await constraint_checker.verify(50, rule)
    assert result.passed is True


@pytest.mark.asyncio
async def test_range_check_below_min(constraint_checker):
    """Test range check fails for value below minimum."""
    rule = VerificationRule(
        rule_id="test_range",
        verifier_type="constraint",
        rule_spec={"min_value": 0, "max_value": 100},
    )
    result = await constraint_checker.verify(-10, rule)
    assert result.passed is False
    assert result.error_type == "range_violation"


@pytest.mark.asyncio
async def test_range_check_above_max(constraint_checker):
    """Test range check fails for value above maximum."""
    rule = VerificationRule(
        rule_id="test_range",
        verifier_type="constraint",
        rule_spec={"min_value": 0, "max_value": 100},
    )
    result = await constraint_checker.verify(150, rule)
    assert result.passed is False
    assert result.error_type == "range_violation"
