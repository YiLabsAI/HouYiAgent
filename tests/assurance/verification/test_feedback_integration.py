"""Integration tests for feedback loop in NeuroSymbolicEngine."""

import pytest

from houyi.assurance.verification.config import VerificationConfig, VerificationMode
from houyi.assurance.verification.feedback import FeedbackProtocol
from houyi.assurance.verification.neuro_symbolic_engine import NeuroSymbolicEngine
from houyi.assurance.verification.sql_verifier import SQLVerifier
from houyi.assurance.verification.verifier import VerificationRule


class TestFeedbackIntegration:
    """Tests for feedback loop integration."""

    @pytest.mark.asyncio
    async def test_feedback_accumulation_on_failure(self):
        """Test that feedback accumulates on verification failures."""
        engine = NeuroSymbolicEngine(
            config=VerificationConfig(
                enabled=True,
                mode=VerificationMode.LENIENT,
                max_retries=2,
                auto_fix=False,
            )
        )

        verifier = SQLVerifier()
        rule = VerificationRule(
            rule_id="sql_syntax", verifier_type="sql", rule_spec={"check_syntax": True}
        )

        attempt_count = 0

        async def generator(feedback_context):
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count == 1:
                return "SELECT * FROM users"
            elif attempt_count == 2:
                return "SELECT id FROM users"
            else:
                return "SELECT * FROM users;"

        output, success = await engine.generate_and_verify(
            generator=generator,
            verifier=verifier,
            rule=rule,
        )

        assert success
        assert output == "SELECT * FROM users;"
        assert attempt_count == 3

        feedback_context = engine.get_feedback_context()
        assert len(feedback_context) == 2
        assert all(isinstance(f, FeedbackProtocol) for f in feedback_context)

    @pytest.mark.asyncio
    async def test_feedback_prompt_generation(self):
        """Test feedback prompt generation."""
        engine = NeuroSymbolicEngine(
            config=VerificationConfig(
                enabled=True,
                mode=VerificationMode.LENIENT,
                max_retries=1,
                auto_fix=False,
            )
        )

        verifier = SQLVerifier()
        rule = VerificationRule(
            rule_id="sql_syntax", verifier_type="sql", rule_spec={"check_syntax": True}
        )

        async def generator(feedback_context):
            if not feedback_context:
                return "SELECT * FROM users"
            else:
                return "SELECT * FROM users;"

        output, success = await engine.generate_and_verify(
            generator=generator,
            verifier=verifier,
            rule=rule,
        )

        assert success

        prompt = engine.get_feedback_prompt()
        assert "Previous Verification Failures" in prompt
        assert "Attempt 1" in prompt

    @pytest.mark.asyncio
    async def test_context_cleared_between_tasks(self):
        """Test that feedback context is cleared for new tasks."""
        engine = NeuroSymbolicEngine(
            config=VerificationConfig(
                enabled=True,
                mode=VerificationMode.LENIENT,
                max_retries=1,
                auto_fix=False,
            )
        )

        verifier = SQLVerifier()
        rule = VerificationRule(
            rule_id="sql_syntax", verifier_type="sql", rule_spec={"check_syntax": True}
        )

        async def generator1(feedback_context):
            return "SELECT * FROM users"

        await engine.generate_and_verify(generator1, verifier, rule)
        # With max_retries=1, we get initial attempt + 1 retry = 2 feedback entries
        assert len(engine.get_feedback_context()) == 2

        async def generator2(feedback_context):
            return "SELECT * FROM products;"

        await engine.generate_and_verify(generator2, verifier, rule)
        assert len(engine.get_feedback_context()) == 0

    @pytest.mark.asyncio
    async def test_feedback_includes_fix_suggestions(self):
        """Test that feedback includes actionable fix suggestions."""
        engine = NeuroSymbolicEngine(
            config=VerificationConfig(
                enabled=True,
                mode=VerificationMode.LENIENT,
                max_retries=1,
                auto_fix=False,
            )
        )

        verifier = SQLVerifier()
        rule = VerificationRule(
            rule_id="sql_syntax", verifier_type="sql", rule_spec={"check_syntax": True}
        )

        async def generator(feedback_context):
            if not feedback_context:
                return "SELECT * FROM users"
            else:
                feedback = feedback_context[0]
                assert feedback.fix_suggestion != ""
                assert "semicolon" in feedback.fix_suggestion.lower()
                return "SELECT * FROM users;"

        output, success = await engine.generate_and_verify(
            generator=generator,
            verifier=verifier,
            rule=rule,
        )

        assert success

    @pytest.mark.asyncio
    async def test_feedback_tracks_previous_attempts(self):
        """Test that feedback tracks previous failed attempts."""
        engine = NeuroSymbolicEngine(
            config=VerificationConfig(
                enabled=True,
                mode=VerificationMode.LENIENT,
                max_retries=3,
                auto_fix=False,
            )
        )

        verifier = SQLVerifier()
        rule = VerificationRule(
            rule_id="sql_syntax", verifier_type="sql", rule_spec={"check_syntax": True}
        )

        attempt_count = 0

        async def generator(feedback_context):
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count <= 3:
                return f"SELECT {attempt_count} FROM users"
            else:
                return "SELECT * FROM users;"

        output, success = await engine.generate_and_verify(
            generator=generator,
            verifier=verifier,
            rule=rule,
        )

        assert success

        feedback_context = engine.get_feedback_context()
        assert len(feedback_context) == 3

        for i, feedback in enumerate(feedback_context):
            assert len(feedback.previous_attempts) == i + 1
