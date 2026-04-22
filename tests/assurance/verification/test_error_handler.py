"""Tests for error handler."""

import pytest

from houyi.assurance.verification.error_handler import AutoFixer, ErrorHandler
from houyi.assurance.verification.sql_verifier import SQLVerifier
from houyi.assurance.verification.verifier import VerificationResult


@pytest.fixture
def error_handler():
    """Create error handler instance."""
    return ErrorHandler()


@pytest.fixture
def auto_fixer():
    """Create auto fixer instance."""
    return AutoFixer()


@pytest.mark.asyncio
async def test_auto_fixer_success(auto_fixer):
    """Test auto-fixer successfully fixes error."""
    verifier = SQLVerifier()
    error = VerificationResult(
        rule_id="test",
        passed=False,
        error_type="missing_semicolon",
        auto_fixable=True,
    )

    sql = "SELECT * FROM users"
    fixed, success = await auto_fixer.fix(sql, error, verifier)

    assert success is True
    assert fixed == "SELECT * FROM users;"


@pytest.mark.asyncio
async def test_auto_fixer_not_fixable(auto_fixer):
    """Test auto-fixer returns False for non-fixable errors."""
    verifier = SQLVerifier()
    error = VerificationResult(
        rule_id="test",
        passed=False,
        error_type="sql_injection",
        auto_fixable=False,
    )

    sql = "SELECT * FROM users"
    fixed, success = await auto_fixer.fix(sql, error, verifier)

    assert success is False
    assert fixed == sql


def test_classify_error_security(error_handler):
    """Test security error classification."""
    error = VerificationResult(
        rule_id="test",
        passed=False,
        error_type="sql_injection",
    )

    classification = error_handler.classify_error(error)
    assert classification == "security"


def test_classify_error_auto_fixable(error_handler):
    """Test auto-fixable error classification."""
    error = VerificationResult(
        rule_id="test",
        passed=False,
        error_type="sql_format",
        auto_fixable=True,
    )

    classification = error_handler.classify_error(error)
    assert classification == "auto_fixable"


def test_should_escalate_security(error_handler):
    """Test security errors always escalate."""
    error = VerificationResult(
        rule_id="test",
        passed=False,
        error_type="sql_injection",
    )

    should_escalate = error_handler.should_escalate(error, 0, 3)
    assert should_escalate is True


def test_should_escalate_max_retries(error_handler):
    """Test escalation on max retries."""
    error = VerificationResult(
        rule_id="test",
        passed=False,
        error_type="sql_format",
        auto_fixable=True,
    )

    should_escalate = error_handler.should_escalate(error, 3, 3)
    assert should_escalate is True


@pytest.mark.asyncio
async def test_handle_error_security(error_handler):
    """Test handling security error."""
    verifier = SQLVerifier()
    error = VerificationResult(
        rule_id="test",
        passed=False,
        error_type="sql_injection",
    )

    sql = "SELECT * FROM users"
    fixed, action = await error_handler.handle_error(sql, error, verifier, 0, 3)

    assert action == "escalate"


@pytest.mark.asyncio
async def test_handle_error_auto_fix(error_handler):
    """Test handling auto-fixable error."""
    verifier = SQLVerifier()
    error = VerificationResult(
        rule_id="test",
        passed=False,
        error_type="missing_semicolon",
        auto_fixable=True,
    )

    sql = "SELECT * FROM users"
    fixed, action = await error_handler.handle_error(sql, error, verifier, 0, 3)

    assert action == "retry"
    assert fixed == "SELECT * FROM users;"


@pytest.mark.asyncio
async def test_auto_fix_failed(error_handler):
    """Test handling auto-fixable error when fix fails."""
    verifier = SQLVerifier()
    error = VerificationResult(
        rule_id="test",
        passed=False,
        error_type="sql_format",
        auto_fixable=True,
    )

    # Invalid SQL that can't be fixed
    sql = None
    fixed, action = await error_handler.handle_error(sql, error, verifier, 0, 3)

    # Should retry even if fix failed
    assert action == "retry"


def test_classify_error_escalate(error_handler):
    """Test escalate error classification."""
    error = VerificationResult(
        rule_id="test",
        passed=False,
        error_type="unknown_error",
        severity="error",
    )

    classification = error_handler.classify_error(error)
    assert classification == "escalate"


def test_classify_error_fail(error_handler):
    """Test fail error classification."""
    error = VerificationResult(
        rule_id="test",
        passed=False,
        error_type="unknown_error",
        severity="warning",
    )

    classification = error_handler.classify_error(error)
    assert classification == "fail"


def test_escalate_non_auto_fixable(error_handler):
    """Test escalation for non-auto-fixable errors."""
    error = VerificationResult(
        rule_id="test",
        passed=False,
        error_type="unknown_error",
        severity="error",
    )

    should_escalate = error_handler.should_escalate(error, 0, 3)
    assert should_escalate is True


def test_no_escalate_auto_fixable(error_handler):
    """Test no escalation for auto-fixable errors under retry limit."""
    error = VerificationResult(
        rule_id="test",
        passed=False,
        error_type="sql_format",
        auto_fixable=True,
    )

    should_escalate = error_handler.should_escalate(error, 1, 3)
    assert should_escalate is False


@pytest.mark.asyncio
async def test_escalate_non_fixable(error_handler):
    """Test handling non-fixable error that should escalate."""
    verifier = SQLVerifier()
    error = VerificationResult(
        rule_id="test",
        passed=False,
        error_type="unknown_error",
        severity="error",
    )

    sql = "SELECT * FROM users"
    fixed, action = await error_handler.handle_error(sql, error, verifier, 0, 3)

    assert action == "escalate"
    assert fixed == sql


@pytest.mark.asyncio
async def test_auto_fixer_exception_handling(auto_fixer):
    """Test auto-fixer handles exceptions gracefully."""
    verifier = SQLVerifier()
    error = VerificationResult(
        rule_id="test",
        passed=False,
        error_type="missing_semicolon",
        auto_fixable=True,
    )

    # Pass invalid type to trigger exception
    fixed, success = await auto_fixer.fix(12345, error, verifier)

    assert success is False
    assert fixed == 12345
