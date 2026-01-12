"""Neuro-symbolic engine for generate-verify-execute pattern."""

import logging
from collections.abc import Callable
from typing import Any

from houyi.verification.config import VerificationConfig, VerificationMode
from houyi.verification.error_handler import ErrorHandler
from houyi.verification.feedback import FeedbackBuilder, FeedbackProtocol
from houyi.verification.review_queue import ReviewQueue, ReviewRequest
from houyi.verification.verifier import VerificationResult, VerificationRule, Verifier

logger = logging.getLogger(__name__)


class VerificationMetrics:
    """Metrics for verification operations."""

    def __init__(self):
        """Initialize metrics."""
        self.total_verifications = 0
        self.passed = 0
        self.failed = 0
        self.auto_fixed = 0
        self.escalated = 0
        self.retries = 0

    def record_verification(self, passed: bool) -> None:
        """Record a verification attempt."""
        self.total_verifications += 1
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def record_auto_fix(self) -> None:
        """Record an auto-fix."""
        self.auto_fixed += 1

    def record_escalation(self) -> None:
        """Record an escalation."""
        self.escalated += 1

    def record_retry(self) -> None:
        """Record a retry."""
        self.retries += 1

    def get_stats(self) -> dict[str, Any]:
        """Get verification statistics."""
        return {
            "total": self.total_verifications,
            "passed": self.passed,
            "failed": self.failed,
            "auto_fixed": self.auto_fixed,
            "escalated": self.escalated,
            "retries": self.retries,
            "success_rate": self.passed / self.total_verifications
            if self.total_verifications > 0
            else 0,
        }


class NeuroSymbolicEngine:
    """Engine for neuro-symbolic reasoning with verification."""

    def __init__(
        self,
        config: VerificationConfig | None = None,
        review_queue: ReviewQueue | None = None,
        feedback_builder: FeedbackBuilder | None = None,
    ):
        """Initialize neuro-symbolic engine.

        Args:
            config: Verification configuration
            review_queue: Queue for human review
            feedback_builder: Builder for constructing LLM feedback
        """
        self.config = config or VerificationConfig.lenient()
        self.error_handler = ErrorHandler()
        self.review_queue = review_queue or ReviewQueue()
        self.feedback_builder = feedback_builder or FeedbackBuilder()
        self.metrics = VerificationMetrics()
        self._feedback_context: list[FeedbackProtocol] = []

    async def generate_and_verify(
        self,
        generator: Callable[[list[FeedbackProtocol]], Any],
        verifier: Verifier,
        rule: VerificationRule,
        task_id: str | None = None,
        input_context: dict[str, Any] | None = None,
    ) -> tuple[Any, bool]:
        """Generate output and verify it with retry logic and feedback loop.

        Args:
            generator: Function that generates output (takes feedback_context)
            verifier: Verifier to use
            rule: Verification rule
            task_id: Task identifier for tracking
            input_context: Input context for feedback building

        Returns:
            (output, success) tuple
        """
        if not self.config.enabled:
            output = await generator([])
            return output, True

        retry_count = 0
        last_error = None

        # Reset feedback context for new task
        self._feedback_context.clear()
        self.feedback_builder.reset_history()

        while retry_count <= self.config.max_retries:
            # Generate output with feedback context
            try:
                output = await generator(self._feedback_context)
            except Exception as e:
                logger.error("Generation failed: %s", e)
                return None, False

            # Verify output
            result = await verifier.verify(output, rule)
            self.metrics.record_verification(result.passed)

            if result.passed:
                logger.info("Verification passed on attempt %d", retry_count + 1)
                return output, True

            last_error = result

            # Audit mode: log but don't block
            if self.config.mode == VerificationMode.AUDIT:
                logger.warning("Verification failed (audit mode): %s", result.error_message)
                return output, True

            # Strict mode: fail immediately
            if self.config.mode == VerificationMode.STRICT:
                logger.error("Verification failed (strict mode): %s", result.error_message)
                return output, False

            # Build structured feedback for LLM
            feedback = self.feedback_builder.build_feedback(
                error_type=result.error_type,
                error_message=result.error_message,
                output=output,
                violated_constraint=getattr(result, "violated_constraint", ""),
                input_context=input_context,
            )
            self._feedback_context.append(feedback)

            logger.info("Built feedback for retry %d: %s", retry_count + 1, result.error_type)

            # Handle error with auto-fix if enabled
            if self.config.auto_fix:
                fixed_output, action = await self.error_handler.handle_error(
                    output, result, verifier, retry_count, self.config.max_retries
                )

                if action == "escalate":
                    # Escalate to human review
                    decision = await self._escalate_to_human(output, result, task_id or "unknown")
                    if decision == "approved":
                        return output, True
                    else:
                        return output, False

                elif action == "retry":
                    if fixed_output != output:
                        self.metrics.record_auto_fix()
                        # Re-verify the fixed output
                        verify_result = await verifier.verify(fixed_output, rule)
                        if verify_result.passed:
                            return fixed_output, True
                    retry_count += 1
                    self.metrics.record_retry()
                    continue
            else:
                # No auto-fix: just retry with feedback
                retry_count += 1
                self.metrics.record_retry()

        # Max retries exceeded
        logger.error("Verification failed after %d retries", retry_count)

        # Final escalation
        if self.config.on_failure == "escalate":
            decision = await self._escalate_to_human(output, last_error, task_id or "unknown")
            return output, decision == "approved"

        return output, False

    def get_feedback_context(self) -> list[FeedbackProtocol]:
        """Get current feedback context.

        Returns:
            List of feedback protocols accumulated during generation.
        """
        return self._feedback_context.copy()

    def get_feedback_prompt(self) -> str:
        """Get formatted feedback prompt for LLM.

        Returns:
            Formatted prompt string with all accumulated feedback.
        """
        if not self._feedback_context:
            return ""

        prompt_parts = ["## Previous Verification Failures\n"]
        for i, feedback in enumerate(self._feedback_context, 1):
            prompt_parts.append(f"\n### Attempt {i}")
            prompt_parts.append(feedback.to_llm_prompt())

        return "\n".join(prompt_parts)

    def clear_feedback_context(self) -> None:
        """Clear feedback context (useful for new tasks)."""
        self._feedback_context.clear()
        self.feedback_builder.reset_history()
        logger.debug("Cleared feedback context")

    async def _escalate_to_human(
        self,
        output: Any,
        error: VerificationResult,
        task_id: str,
    ) -> str:
        """Escalate verification failure to human review.

        Args:
            output: Output that failed verification
            error: Verification error
            task_id: Task identifier

        Returns:
            Decision: 'approved', 'rejected', or 'timeout'
        """
        self.metrics.record_escalation()

        request = ReviewRequest(
            request_id=f"{task_id}_{error.rule_id}",
            task_id=task_id,
            error_type=error.error_type or "unknown",
            error_message=error.error_message or "Verification failed",
            original_output=output,
            suggested_fix=error.fix_suggestion,
            timeout_seconds=self.config.escalation_policy.timeout_seconds
            if self.config.escalation_policy
            else 300,
        )

        decision = await self.review_queue.submit(request)
        logger.info("Human review decision: %s", decision)

        return decision

    def get_metrics(self) -> dict[str, Any]:
        """Get verification metrics.

        Returns:
            Metrics dictionary
        """
        return self.metrics.get_stats()
