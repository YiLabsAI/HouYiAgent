"""Tests for SQL verifier."""

import pytest

from houyi.assurance.verification.sql_verifier import SQLVerifier
from houyi.assurance.verification.verifier import VerificationRule


@pytest.fixture
def sql_verifier():
    """Create SQL verifier instance."""
    return SQLVerifier()


@pytest.fixture
def basic_rule():
    """Create basic verification rule."""
    return VerificationRule(
        rule_id="test_sql",
        verifier_type="sql",
        rule_spec={"check_syntax": True, "check_injection": True},
    )


@pytest.mark.asyncio
async def test_valid_sql(sql_verifier, basic_rule):
    """Test valid SQL query passes verification."""
    sql = "SELECT * FROM users WHERE id = 1;"
    result = await sql_verifier.verify(sql, basic_rule)
    assert result.passed is True


@pytest.mark.asyncio
async def test_sql_injection_detected(sql_verifier, basic_rule):
    """Test SQL injection is detected."""
    sql = "SELECT * FROM users WHERE id = 1 OR 1=1;"
    result = await sql_verifier.verify(sql, basic_rule)
    assert result.passed is False
    assert result.error_type == "sql_injection"


@pytest.mark.asyncio
async def test_missing_semicolon(sql_verifier, basic_rule):
    """Test missing semicolon is detected."""
    sql = "SELECT * FROM users"
    result = await sql_verifier.verify(sql, basic_rule)
    assert result.passed is False
    assert result.error_type == "missing_semicolon"
    assert result.auto_fixable is True


@pytest.mark.asyncio
async def test_auto_fix_semicolon(sql_verifier):
    """Test auto-fix adds semicolon."""
    sql = "SELECT * FROM users"
    error_result = VerificationRule(
        rule_id="test",
        verifier_type="sql",
        error_type="missing_semicolon",
    )

    from houyi.assurance.verification.verifier import VerificationResult

    error = VerificationResult(
        rule_id="test",
        passed=False,
        error_type="missing_semicolon",
    )

    fixed, success = await sql_verifier.auto_fix(sql, error)
    assert success is True
    assert fixed == "SELECT * FROM users;"


@pytest.mark.asyncio
async def test_forbidden_operation(sql_verifier):
    """Test forbidden SQL operation is detected."""
    rule = VerificationRule(
        rule_id="test_sql",
        verifier_type="sql",
        rule_spec={"allowed_operations": ["SELECT"]},
    )
    sql = "DELETE FROM users WHERE id = 1;"
    result = await sql_verifier.verify(sql, rule)
    assert result.passed is False
    assert result.error_type == "forbidden_operation"


@pytest.mark.asyncio
async def test_forbidden_table(sql_verifier):
    """Test forbidden table access is detected."""
    rule = VerificationRule(
        rule_id="test_sql",
        verifier_type="sql",
        rule_spec={"forbidden_tables": ["admin_users"]},
    )
    sql = "SELECT * FROM admin_users;"
    result = await sql_verifier.verify(sql, rule)
    assert result.passed is False
    assert result.error_type == "forbidden_table"


@pytest.mark.asyncio
async def test_invalid_type(sql_verifier, basic_rule):
    """Test non-string input is rejected."""
    result = await sql_verifier.verify(123, basic_rule)
    assert result.passed is False
    assert result.error_type == "invalid_type"


@pytest.mark.asyncio
async def test_empty_query(sql_verifier, basic_rule):
    """Test empty SQL query is rejected."""
    result = await sql_verifier.verify("", basic_rule)
    assert result.passed is False
    assert result.error_type == "empty_query"


@pytest.mark.asyncio
async def test_sql_format_auto_fix(sql_verifier):
    """Test SQL formatting auto-fix."""
    from houyi.assurance.verification.verifier import VerificationResult

    sql = "select*from users where id=1"
    error = VerificationResult(
        rule_id="test",
        passed=False,
        error_type="sql_format",
    )

    fixed, success = await sql_verifier.auto_fix(sql, error)
    assert success is True
    assert "SELECT" in fixed
    assert "FROM" in fixed


@pytest.mark.asyncio
async def test_auto_fix_non_string(sql_verifier):
    """Test auto-fix handles non-string input gracefully."""
    from houyi.assurance.verification.verifier import VerificationResult

    error = VerificationResult(
        rule_id="test",
        passed=False,
        error_type="sql_format",
    )

    fixed, success = await sql_verifier.auto_fix(123, error)
    assert success is False
    assert fixed == 123


@pytest.mark.asyncio
async def test_auto_fix_unsupported_error(sql_verifier):
    """Test auto-fix returns False for unsupported errors."""
    from houyi.assurance.verification.verifier import VerificationResult

    sql = "SELECT * FROM users"
    error = VerificationResult(
        rule_id="test",
        passed=False,
        error_type="sql_injection",
    )

    fixed, success = await sql_verifier.auto_fix(sql, error)
    assert success is False


@pytest.mark.asyncio
async def test_supports_auto_fix(sql_verifier):
    """Test supports_auto_fix method."""
    assert sql_verifier.supports_auto_fix("sql_format") is True
    assert sql_verifier.supports_auto_fix("missing_semicolon") is True
    assert sql_verifier.supports_auto_fix("sql_injection") is False


@pytest.mark.asyncio
async def test_multiple_injection_patterns(sql_verifier, basic_rule):
    """Test various SQL injection patterns are detected."""
    injection_queries = [
        "SELECT * FROM users; DROP TABLE users;",
        "SELECT * FROM users WHERE name = 'admin' OR '1'='1';",
        "SELECT * FROM users UNION SELECT * FROM passwords;",
        "SELECT * FROM users -- comment",
    ]

    for sql in injection_queries:
        result = await sql_verifier.verify(sql, basic_rule)
        assert result.passed is False
        assert result.error_type == "sql_injection"


@pytest.mark.asyncio
async def test_syntax_check_disabled(sql_verifier):
    """Test syntax check can be disabled."""
    rule = VerificationRule(
        rule_id="test_sql",
        verifier_type="sql",
        rule_spec={"check_syntax": False, "check_injection": False},
    )
    # Valid SQL with checks disabled
    sql = "SELECT * FROM users;"
    result = await sql_verifier.verify(sql, rule)
    # Should pass because all checks are disabled
    assert result.passed is True


@pytest.mark.asyncio
async def test_injection_check_disabled(sql_verifier):
    """Test injection check can be disabled."""
    rule = VerificationRule(
        rule_id="test_sql",
        verifier_type="sql",
        rule_spec={"check_syntax": True, "check_injection": False},
    )
    # Injection pattern but check disabled
    sql = "SELECT * FROM users WHERE id = 1 OR 1=1;"
    result = await sql_verifier.verify(sql, rule)
    assert result.passed is True
