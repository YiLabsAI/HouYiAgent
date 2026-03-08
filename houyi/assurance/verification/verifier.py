"""Base verifier interface and data models."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Lazy import cache to avoid circular dependencies
_verification_cache = None


def _get_verification_cache():
    """Lazy load verification cache."""
    global _verification_cache
    if _verification_cache is None:
        from houyi.assurance.verification.cache import get_verification_cache

        _verification_cache = get_verification_cache()
    return _verification_cache


class VerificationResult(BaseModel):
    """Result of a verification check."""

    rule_id: str = Field(..., description="ID of the verification rule")
    passed: bool = Field(..., description="Whether verification passed")
    error_message: str | None = Field(None, description="Error message if failed")
    error_type: str | None = Field(
        None, description="Type of error (e.g., 'sql_syntax', 'sql_injection')"
    )
    fix_suggestion: str | None = Field(None, description="Suggestion for fixing the error")
    auto_fixable: bool = Field(False, description="Whether this error can be auto-fixed")
    severity: str = Field("error", description="Severity level: 'error', 'warning', 'info'")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class VerificationError(Exception):
    """Exception raised when verification fails."""

    def __init__(self, results: list[VerificationResult]):
        self.results = results
        failed = [r for r in results if not r.passed]
        messages = [f"{r.rule_id}: {r.error_message}" for r in failed]
        super().__init__(f"Verification failed: {'; '.join(messages)}")


class VerificationRule(BaseModel):
    """A single verification rule."""

    rule_id: str = Field(..., description="Unique rule identifier")
    verifier_type: str = Field(
        ..., description="Type of verifier: 'sql', 'python', 'constraint', 'schema'"
    )
    rule_spec: dict[str, Any] = Field(
        default_factory=dict, description="Verifier-specific configuration"
    )
    severity: str = Field("error", description="Severity level: 'error', 'warning', 'info'")
    auto_fixable: bool = Field(False, description="Whether errors can be auto-fixed")
    fix_strategy: str | None = Field(None, description="Strategy for auto-fixing")


class Verifier(ABC):
    """Base class for all verifiers.

    Supports optional caching of verification results for improved performance.
    """

    def __init__(self, use_cache: bool = True):
        """Initialize verifier.

        Args:
            use_cache: Enable caching of verification results (default: True)
        """
        self.use_cache = use_cache
        self._cache = _get_verification_cache() if use_cache else None

    async def verify(
        self,
        output: Any,
        rule: VerificationRule,
        context: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """Verify output against rule with optional caching.

        Args:
            output: The output to verify
            rule: The verification rule to apply
            context: Additional context for verification

        Returns:
            VerificationResult with pass/fail status and details
        """
        # Check cache first if enabled
        if self.use_cache and self._cache:
            # CRITICAL FIX: Include type information in cache key
            # to prevent collision between different types with same string representation
            # Example: int(42) vs str("42") both have str() = "42"
            # but they should have different verification results
            cache_key = f"{type(output).__name__}:{output!r}"

            cached_result = self._cache.get_result(cache_key, rule.rule_id, rule.rule_spec)
            if cached_result is not None:
                logger.debug("Cache hit for rule %s", rule.rule_id)
                return cached_result

        # Perform actual verification
        result = await self._verify_impl(output, rule, context)

        # Cache result if enabled
        if self.use_cache and self._cache:
            # Use same cache key format: type:repr
            cache_key = f"{type(output).__name__}:{output!r}"

            self._cache.put_result(cache_key, rule.rule_id, rule.rule_spec, result)
            logger.debug("Cached result for rule %s", rule.rule_id)

        return result

    @abstractmethod
    async def _verify_impl(
        self,
        output: Any,
        rule: VerificationRule,
        context: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """Internal verification implementation (to be overridden by subclasses).

        Args:
            output: The output to verify
            rule: The verification rule to apply
            context: Additional context for verification

        Returns:
            VerificationResult with pass/fail status and details
        """
        pass

    def invalidate_cache(self, rule_id: str | None = None) -> None:
        """Invalidate cached results.

        Args:
            rule_id: Optional rule ID to invalidate (None = clear all)
        """
        if self._cache:
            if rule_id:
                self._cache.invalidate_for_rule(rule_id)
            else:
                self._cache.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics or empty dict if caching disabled
        """
        if self._cache:
            return self._cache.get_stats()
        return {}

    def supports_auto_fix(self, error_type: str) -> bool:
        """Check if this verifier can auto-fix the error type.

        Args:
            error_type: Type of error (e.g., 'sql_format', 'python_indent')

        Returns:
            True if auto-fix is supported for this error type
        """
        return False

    async def auto_fix(
        self,
        output: Any,
        error: VerificationResult,
    ) -> tuple[Any, bool]:
        """Attempt to auto-fix the error.

        Args:
            output: The original output with error
            error: The verification error result

        Returns:
            Tuple of (fixed_output, success)
        """
        return output, False
