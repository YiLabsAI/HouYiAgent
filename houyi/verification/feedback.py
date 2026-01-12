"""Feedback protocol and builder for LLM regeneration."""

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FeedbackProtocol(BaseModel):
    """Standard protocol for feedback data to LLM.

    This structured feedback enables LLM to understand verification failures
    and regenerate improved outputs.
    """

    error_type: str = Field(
        ..., description="Error classification (e.g., 'sql_injection', 'syntax_error')"
    )
    error_message: str = Field(..., description="Human-readable error description")
    violated_constraint: str = Field(
        default="", description="Violated constraint in formal representation"
    )
    fix_suggestion: str = Field(default="", description="Specific suggestion for fixing the error")
    severity: Literal["error", "warning", "info"] = Field(
        default="error", description="Error severity level"
    )

    # Context information
    input_context: dict[str, Any] = Field(
        default_factory=dict, description="Input context that led to the error"
    )
    output_context: dict[str, Any] = Field(
        default_factory=dict, description="Output context (partial output, error location)"
    )
    previous_attempts: list[str] = Field(
        default_factory=list, description="Historical failed attempts to avoid repetition"
    )

    def to_llm_prompt(self) -> str:
        """Convert feedback to LLM-understandable prompt.

        Returns:
            Structured prompt string for LLM consumption.
        """
        prompt_parts = [
            f"## Verification Failed: {self.error_type}",
            f"\n**Error**: {self.error_message}",
        ]

        if self.violated_constraint:
            prompt_parts.append(f"\n**Violated Constraint**: {self.violated_constraint}")

        if self.fix_suggestion:
            prompt_parts.append(f"\n**Suggestion**: {self.fix_suggestion}")

        if self.previous_attempts:
            prompt_parts.append("\n**Previous Failed Attempts**:")
            for i, attempt in enumerate(self.previous_attempts[-3:], 1):  # Last 3 attempts
                prompt_parts.append(f"{i}. {attempt[:100]}...")  # Truncate to 100 chars

        prompt_parts.append("\n**Please regenerate the output addressing the above issues.**")

        return "\n".join(prompt_parts)

    class Config:
        """Pydantic config."""

        frozen = False  # Allow mutation for context accumulation


class FeedbackBuilder:
    """Builder for constructing structured feedback from verification results.

    Responsibilities:
    1. Extract key error information from VerificationResult
    2. Generate actionable fix suggestions
    3. Accumulate historical context
    4. Limit feedback length to avoid context overflow
    """

    MAX_FEEDBACK_LENGTH = 500  # Max chars for error message
    MAX_ATTEMPTS_HISTORY = 5  # Max historical attempts to track

    def __init__(self):
        """Initialize feedback builder."""
        self._attempt_history: list[str] = []

    def build_feedback(
        self,
        error_type: str,
        error_message: str,
        output: Any,
        violated_constraint: str = "",
        input_context: dict[str, Any] | None = None,
    ) -> FeedbackProtocol:
        """Build structured feedback from error information.

        Args:
            error_type: Error classification
            error_message: Error description
            output: The output that failed verification
            violated_constraint: Violated constraint (optional)
            input_context: Input context (optional)

        Returns:
            Structured feedback protocol
        """
        # Truncate error message if too long
        truncated_message = self._truncate_message(error_message)

        # Generate fix suggestion based on error type
        fix_suggestion = self._generate_fix_suggestion(error_type, error_message)

        # Determine severity
        severity = self._determine_severity(error_type)

        # Build output context
        output_context = self._build_output_context(output, error_message)

        # Add current output to attempt history
        self._add_to_history(output)

        feedback = FeedbackProtocol(
            error_type=error_type,
            error_message=truncated_message,
            violated_constraint=violated_constraint,
            fix_suggestion=fix_suggestion,
            severity=severity,
            input_context=input_context or {},
            output_context=output_context,
            previous_attempts=self._attempt_history.copy(),
        )

        logger.info(f"Built feedback for error: {error_type}")
        return feedback

    def _truncate_message(self, message: str) -> str:
        """Truncate message to avoid context overflow."""
        if len(message) <= self.MAX_FEEDBACK_LENGTH:
            return message
        return message[: self.MAX_FEEDBACK_LENGTH] + "..."

    def _generate_fix_suggestion(self, error_type: str, error_message: str) -> str:
        """Generate actionable fix suggestion based on error type.

        Args:
            error_type: Error classification
            error_message: Error description

        Returns:
            Fix suggestion string
        """
        suggestions = {
            "sql_injection": "Use parameterized queries instead of string concatenation. Avoid user input in SQL strings.",
            "sql_syntax": "Check SQL syntax. Ensure proper use of keywords, parentheses, and semicolons.",
            "python_syntax": "Check Python syntax. Ensure proper indentation and valid Python keywords.",
            "python_indent": "Fix indentation. Python requires consistent use of spaces or tabs.",
            "unsafe_import": "Avoid importing dangerous modules. Use only approved standard library modules.",
            "type_mismatch": "Ensure output matches expected type schema.",
            "missing_semicolon": "Add semicolon at the end of SQL statement.",
        }

        suggestion = suggestions.get(error_type, "Review the error message and fix the issue.")

        # Add specific context from error message if available
        if "expected" in error_message.lower():
            suggestion += " Check expected vs actual values."

        return suggestion

    def _determine_severity(self, error_type: str) -> Literal["error", "warning", "info"]:
        """Determine severity level based on error type."""
        security_errors = {"sql_injection", "unsafe_import", "security_risk"}

        if error_type in security_errors:
            return "error"

        warning_errors = {"type_mismatch", "missing_semicolon"}
        if error_type in warning_errors:
            return "warning"

        return "error"  # Default to error

    def _build_output_context(self, output: Any, error_message: str) -> dict[str, Any]:
        """Build output context with error location if available."""
        context = {
            "output_type": type(output).__name__,
            "output_preview": str(output)[:200] if output else "",
        }

        # Extract line/column info if available in error message
        if "line" in error_message.lower():
            context["has_line_info"] = True

        return context

    def _add_to_history(self, output: Any) -> None:
        """Add output to attempt history, maintaining max size."""
        output_str = str(output)[:100] if output else ""  # Truncate to 100 chars

        self._attempt_history.append(output_str)

        # Keep only last N attempts
        if len(self._attempt_history) > self.MAX_ATTEMPTS_HISTORY:
            self._attempt_history = self._attempt_history[-self.MAX_ATTEMPTS_HISTORY :]

    def reset_history(self) -> None:
        """Reset attempt history (useful for new tasks)."""
        self._attempt_history.clear()
        logger.debug("Reset feedback history")

    def get_history_summary(self) -> dict[str, Any]:
        """Get summary of feedback history."""
        return {
            "total_attempts": len(self._attempt_history),
            "attempts": self._attempt_history.copy(),
        }
