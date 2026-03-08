"""Error handling and auto-fixing for verification."""

import logging
from typing import Any

from houyi.assurance.verification.verifier import VerificationResult, Verifier

logger = logging.getLogger(__name__)


class AutoFixer:
    """Automatically fixes common verification errors."""

    async def fix(
        self,
        output: Any,
        error: VerificationResult,
        verifier: Verifier,
    ) -> tuple[Any, bool]:
        """Attempt to fix the error.

        Args:
            output: Original output with error
            error: Verification error result
            verifier: Verifier that detected the error

        Returns:
            (fixed_output, success)
        """
        if not error.auto_fixable:
            return output, False

        error_type = error.error_type
        if not isinstance(error_type, str):
            return output, False

        if not verifier.supports_auto_fix(error_type):
            return output, False

        try:
            fixed, success = await verifier.auto_fix(output, error)
            if success:
                logger.info("Auto-fixed error: %s", error.error_type)
            return fixed, success
        except Exception as e:
            logger.error("Auto-fix failed: %s", e)
            return output, False


class ErrorHandler:
    """Handles verification errors with classification and escalation."""

    SECURITY_ERRORS = {"sql_injection", "unsafe_import", "security_risk"}
    AUTO_FIXABLE_ERRORS = {
        "sql_format",
        "missing_semicolon",
        "python_syntax",
        "python_indent",
        "type_mismatch",
    }

    def __init__(self):
        """Initialize error handler."""
        self.auto_fixer = AutoFixer()

    def classify_error(self, error: VerificationResult) -> str:
        """Classify error for handling decision.

        Args:
            error: Verification error result

        Returns:
            Classification: 'security', 'auto_fixable', 'escalate', 'fail'
        """
        if error.error_type in self.SECURITY_ERRORS:
            return "security"

        if error.auto_fixable and error.error_type in self.AUTO_FIXABLE_ERRORS:
            return "auto_fixable"

        if error.severity == "error":
            return "escalate"

        return "fail"

    def should_escalate(
        self, error: VerificationResult, retry_count: int, max_retries: int
    ) -> bool:
        """Determine if error should be escalated to human review.

        Args:
            error: Verification error result
            retry_count: Current retry count
            max_retries: Maximum allowed retries

        Returns:
            True if should escalate
        """
        classification = self.classify_error(error)

        # Always escalate security issues
        if classification == "security":
            return True

        # Escalate if max retries exceeded
        if retry_count >= max_retries:
            return True

        return classification == "escalate"

    async def handle_error(
        self,
        output: Any,
        error: VerificationResult,
        verifier: Verifier,
        retry_count: int,
        max_retries: int,
    ) -> tuple[Any, str]:
        """Handle verification error.

        Args:
            output: Original output with error
            error: Verification error result
            verifier: Verifier that detected the error
            retry_count: Current retry count
            max_retries: Maximum allowed retries

        Returns:
            (fixed_output, action) where action is 'retry', 'escalate', or 'fail'
        """
        classification = self.classify_error(error)

        # Security errors always escalate
        if classification == "security":
            logger.warning("Security error detected: %s", error.error_type)
            return output, "escalate"

        # Try auto-fix
        if classification == "auto_fixable":
            fixed, success = await self.auto_fixer.fix(output, error, verifier)
            if success:
                return fixed, "retry"

        # Check if should escalate
        if self.should_escalate(error, retry_count, max_retries):
            logger.info("Escalating error after %d retries: %s", retry_count, error.error_type)
            return output, "escalate"

        # Retry
        return output, "retry"
