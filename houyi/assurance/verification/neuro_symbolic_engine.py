"""Neuro-symbolic engine for generate-verify-execute pattern."""

import logging
from collections.abc import Callable
from typing import Any

from houyi.assurance.verification.config import VerificationConfig, VerificationMode
from houyi.assurance.verification.error_handler import ErrorHandler
from houyi.assurance.verification.feedback import FeedbackBuilder, FeedbackProtocol
from houyi.assurance.verification.review_queue import ReviewQueue, ReviewRequest
from houyi.assurance.verification.verifier import VerificationResult, VerificationRule, Verifier

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
        self._reset_feedback_state()

        while retry_count <= self.config.max_retries:
            output = await self._generate_output(generator)
            if output is None:
                return None, False

            result = await verifier.verify(output, rule)
            self.metrics.record_verification(result.passed)

            if result.passed:
                logger.info("Verification passed on attempt %d", retry_count + 1)
                return output, True

            last_error = result

            mode_outcome = self._handle_mode_failure(output, result)
            if mode_outcome is not None:
                return mode_outcome

            self._append_feedback(result=result, output=output, input_context=input_context)

            if not self.config.auto_fix:
                retry_count = self._record_retry(retry_count)
                continue

            auto_fix_outcome = await self._handle_auto_fix(
                output=output,
                result=result,
                verifier=verifier,
                rule=rule,
                retry_count=retry_count,
                task_id=task_id,
            )
            if auto_fix_outcome is not None:
                return auto_fix_outcome

            retry_count = self._record_retry(retry_count)

        logger.error("Verification failed after %d retries", retry_count)
        return await self._finalize_verification_failure(
            output=output,
            last_error=last_error,
            task_id=task_id,
        )

    def _reset_feedback_state(self) -> None:
        self._feedback_context.clear()
        self.feedback_builder.reset_history()

    async def _generate_output(
        self,
        generator: Callable[[list[FeedbackProtocol]], Any],
    ) -> Any | None:
        try:
            return await generator(self._feedback_context)
        except Exception as exc:
            logger.error("Generation failed: %s", exc)
            return None

    def _handle_mode_failure(
        self,
        output: Any,
        result: VerificationResult,
    ) -> tuple[Any, bool] | None:
        if self.config.mode == VerificationMode.AUDIT:
            logger.warning("Verification failed (audit mode): %s", result.error_message)
            return output, True
        if self.config.mode == VerificationMode.STRICT:
            logger.error("Verification failed (strict mode): %s", result.error_message)
            return output, False
        return None

    def _append_feedback(
        self,
        *,
        result: VerificationResult,
        output: Any,
        input_context: dict[str, Any] | None,
    ) -> None:
        error_type = result.error_type or "unknown"
        error_message = result.error_message or "Verification failed"
        feedback = self.feedback_builder.build_feedback(
            error_type=error_type,
            error_message=error_message,
            output=output,
            violated_constraint=getattr(result, "violated_constraint", ""),
            input_context=input_context,
        )
        self._feedback_context.append(feedback)
        logger.info("Built feedback for retry %d: %s", len(self._feedback_context), error_type)

    async def _handle_auto_fix(
        self,
        *,
        output: Any,
        result: VerificationResult,
        verifier: Verifier,
        rule: VerificationRule,
        retry_count: int,
        task_id: str | None,
    ) -> tuple[Any, bool] | None:
        fixed_output, action = await self.error_handler.handle_error(
            output,
            result,
            verifier,
            retry_count,
            self.config.max_retries,
        )
        if action == "escalate":
            return await self._resolve_escalation(output, result, task_id)
        if action != "retry" or fixed_output == output:
            return None
        self.metrics.record_auto_fix()
        verify_result = await verifier.verify(fixed_output, rule)
        if verify_result.passed:
            return fixed_output, True
        return None

    async def _resolve_escalation(
        self,
        output: Any,
        result: VerificationResult,
        task_id: str | None,
    ) -> tuple[Any, bool]:
        decision = await self._escalate_to_human(output, result, task_id or "unknown")
        return output, decision == "approved"

    def _record_retry(self, retry_count: int) -> int:
        self.metrics.record_retry()
        return retry_count + 1

    async def _finalize_verification_failure(
        self,
        *,
        output: Any,
        last_error: VerificationResult | None,
        task_id: str | None,
    ) -> tuple[Any, bool]:
        if self.config.on_failure != "escalate" or last_error is None:
            return output, False
        decision = await self._escalate_to_human(output, last_error, task_id or "unknown")
        return output, decision == "approved"

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
