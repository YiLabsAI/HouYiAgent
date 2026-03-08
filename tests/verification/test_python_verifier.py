"""Tests for Python verifier."""

import pytest

from houyi.assurance.verification.python_verifier import PythonVerifier
from houyi.assurance.verification.verifier import VerificationRule


@pytest.fixture
def python_verifier():
    """Create Python verifier instance."""
    return PythonVerifier()


@pytest.fixture
def basic_rule():
    """Create basic verification rule."""
    return VerificationRule(
        rule_id="test_python",
        verifier_type="python",
        rule_spec={"check_syntax": True, "check_imports": True},
    )


@pytest.mark.asyncio
async def test_valid_python(python_verifier, basic_rule):
    """Test valid Python code passes verification."""
    code = "x = 1 + 2\nprint(x)"
    result = await python_verifier.verify(code, basic_rule)
    assert result.passed is True


@pytest.mark.asyncio
async def test_syntax_error(python_verifier, basic_rule):
    """Test syntax error is detected."""
    code = "x = 1 +\nprint(x)"
    result = await python_verifier.verify(code, basic_rule)
    assert result.passed is False
    assert result.error_type == "python_syntax"
    assert result.auto_fixable is True


@pytest.mark.asyncio
async def test_unsafe_import(python_verifier, basic_rule):
    """Test unsafe import is detected."""
    code = "import os\nos.system('rm -rf /')"
    result = await python_verifier.verify(code, basic_rule)
    assert result.passed is False
    assert result.error_type == "unsafe_import"
    assert "os" in result.error_message


@pytest.mark.asyncio
async def test_safe_import(python_verifier, basic_rule):
    """Test safe import passes."""
    code = "import json\ndata = json.loads('{}')"
    result = await python_verifier.verify(code, basic_rule)
    assert result.passed is True


@pytest.mark.asyncio
async def test_invalid_type(python_verifier, basic_rule):
    """Test non-string input is rejected."""
    result = await python_verifier.verify(123, basic_rule)
    assert result.passed is False
    assert result.error_type == "invalid_type"


@pytest.mark.asyncio
async def test_empty_code(python_verifier, basic_rule):
    """Test empty code is rejected."""
    result = await python_verifier.verify("", basic_rule)
    assert result.passed is False


@pytest.mark.asyncio
async def test_import_check_disabled(python_verifier):
    """Test import check can be disabled."""
    rule = VerificationRule(
        rule_id="test_python",
        verifier_type="python",
        rule_spec={"check_syntax": True, "check_imports": False},
    )
    # Unsafe import but check disabled
    code = "import os\nos.system('ls')"
    result = await python_verifier.verify(code, rule)
    assert result.passed is True


@pytest.mark.asyncio
async def test_supports_auto_fix(python_verifier):
    """Test supports_auto_fix method."""
    assert python_verifier.supports_auto_fix("python_syntax") is True
    assert python_verifier.supports_auto_fix("python_indent") is True
    assert python_verifier.supports_auto_fix("unsafe_import") is False


@pytest.mark.asyncio
async def test_auto_fix_unsupported(python_verifier):
    """Test auto-fix returns False for unsupported errors."""
    from houyi.assurance.verification.verifier import VerificationResult

    code = "import os"
    error = VerificationResult(
        rule_id="test",
        passed=False,
        error_type="unsafe_import",
    )

    fixed, success = await python_verifier.auto_fix(code, error)
    assert success is False
