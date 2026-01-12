"""End-to-end integration tests for neuro-symbolic verification."""

import pytest

from houyi.verification.config import VerificationConfig, VerificationMode
from houyi.verification.feedback import FeedbackProtocol
from houyi.verification.neuro_symbolic_engine import NeuroSymbolicEngine
from houyi.verification.python_verifier import PythonVerifier
from houyi.verification.sql_verifier import SQLVerifier
from houyi.verification.verifier import VerificationRule


class TestE2EVerification:
    """End-to-end tests for complete verification workflow."""

    @pytest.mark.asyncio
    async def test_e2e_sql_verification_with_feedback(self):
        """Test complete SQL verification with feedback loop."""
        engine = NeuroSymbolicEngine(
            config=VerificationConfig(
                enabled=True,
                mode=VerificationMode.LENIENT,
                max_retries=2,
                auto_fix=False,
            )
        )

        verifier = SQLVerifier(use_constraint_solver=True)
        rule = VerificationRule(
            rule_id="sql_check",
            verifier_type="sql",
            rule_spec={"check_syntax": True, "check_injection": True},
        )

        # Generator that improves based on feedback
        attempt_count = 0

        async def sql_generator(feedback_context: list[FeedbackProtocol]):
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count == 1:
                # First attempt: missing semicolon
                return "SELECT * FROM users"
            elif attempt_count == 2:
                # Second attempt: still missing semicolon
                return "SELECT id, name FROM users"
            else:
                # Third attempt: correct
                return "SELECT * FROM users;"

        output, success = await engine.generate_and_verify(
            generator=sql_generator,
            verifier=verifier,
            rule=rule,
        )

        assert success is True
        assert output == "SELECT * FROM users;"
        assert attempt_count == 3

        # Verify feedback was accumulated
        feedback_context = engine.get_feedback_context()
        assert len(feedback_context) == 2  # Two failures before success

        # Verify feedback contains useful information
        for feedback in feedback_context:
            assert feedback.error_type == "missing_semicolon"
            assert "semicolon" in feedback.fix_suggestion.lower()

    @pytest.mark.asyncio
    async def test_e2e_python_verification_with_feedback(self):
        """Test complete Python verification with feedback loop."""
        engine = NeuroSymbolicEngine(
            config=VerificationConfig(
                enabled=True,
                mode=VerificationMode.LENIENT,
                max_retries=2,
                auto_fix=False,
            )
        )

        verifier = PythonVerifier(use_constraint_solver=True)
        rule = VerificationRule(
            rule_id="python_check",
            verifier_type="python",
            rule_spec={"check_syntax": True, "check_imports": True},
        )

        # Generator that improves based on feedback
        attempt_count = 0

        async def python_generator(feedback_context: list[FeedbackProtocol]):
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count == 1:
                # First attempt: unsafe import
                return "import os\nx = 5"
            elif attempt_count == 2:
                # Second attempt: syntax error
                return "def foo("
            else:
                # Third attempt: correct
                return "x = 5\ny = 10"

        output, success = await engine.generate_and_verify(
            generator=python_generator,
            verifier=verifier,
            rule=rule,
        )

        assert success is True
        assert "x = 5" in output
        assert attempt_count == 3

        # Verify feedback was accumulated
        feedback_context = engine.get_feedback_context()
        assert len(feedback_context) == 2

        # Verify different error types
        assert feedback_context[0].error_type == "unsafe_import"
        assert feedback_context[1].error_type == "python_syntax"

    @pytest.mark.asyncio
    async def test_e2e_verification_max_retries_exceeded(self):
        """Test that verification fails after max retries."""
        engine = NeuroSymbolicEngine(
            config=VerificationConfig(
                enabled=True,
                mode=VerificationMode.LENIENT,
                max_retries=2,
                auto_fix=False,
            )
        )

        verifier = SQLVerifier(use_constraint_solver=True)
        rule = VerificationRule(
            rule_id="sql_check",
            verifier_type="sql",
            rule_spec={"check_syntax": True},
        )

        # Generator that always fails
        attempt_count = 0

        async def failing_generator(feedback_context: list[FeedbackProtocol]):
            nonlocal attempt_count
            attempt_count += 1
            return "SELECT * FROM users"  # Always missing semicolon

        output, success = await engine.generate_and_verify(
            generator=failing_generator,
            verifier=verifier,
            rule=rule,
        )

        assert success is False
        assert attempt_count == 3  # Initial + 2 retries

        # Verify feedback accumulated for all attempts
        feedback_context = engine.get_feedback_context()
        assert len(feedback_context) == 3

    @pytest.mark.asyncio
    async def test_e2e_verification_with_constraints(self):
        """Test verification with constraint solving."""
        engine = NeuroSymbolicEngine(
            config=VerificationConfig(
                enabled=True,
                mode=VerificationMode.LENIENT,
                max_retries=1,
                auto_fix=False,
            )
        )

        verifier = SQLVerifier(use_constraint_solver=True)
        rule = VerificationRule(
            rule_id="sql_check",
            verifier_type="sql",
            rule_spec={
                "check_syntax": True,
                "constraints": [
                    {
                        "name": "simple_constraint",
                        "type": "general",
                        "expression": "True",  # Always satisfied
                        "description": "Test constraint",
                    }
                ],
            },
        )

        async def generator(feedback_context: list[FeedbackProtocol]):
            return "SELECT * FROM users;"

        output, success = await engine.generate_and_verify(
            generator=generator,
            verifier=verifier,
            rule=rule,
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_e2e_feedback_prompt_generation(self):
        """Test that feedback can be formatted for LLM consumption."""
        engine = NeuroSymbolicEngine(
            config=VerificationConfig(
                enabled=True,
                mode=VerificationMode.LENIENT,
                max_retries=1,
                auto_fix=False,
            )
        )

        verifier = SQLVerifier(use_constraint_solver=True)
        rule = VerificationRule(
            rule_id="sql_check",
            verifier_type="sql",
            rule_spec={"check_syntax": True},
        )

        async def generator(feedback_context: list[FeedbackProtocol]):
            if not feedback_context:
                return "SELECT * FROM users"
            else:
                # In real scenario, LLM would use this feedback
                prompt = engine.get_feedback_prompt()
                assert "Previous Verification Failures" in prompt
                assert "missing_semicolon" in prompt.lower() or "semicolon" in prompt.lower()
                return "SELECT * FROM users;"

        output, success = await engine.generate_and_verify(
            generator=generator,
            verifier=verifier,
            rule=rule,
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_e2e_verification_disabled(self):
        """Test that verification can be disabled."""
        engine = NeuroSymbolicEngine(
            config=VerificationConfig(
                enabled=False,  # Disabled
            )
        )

        verifier = SQLVerifier(use_constraint_solver=True)
        rule = VerificationRule(
            rule_id="sql_check",
            verifier_type="sql",
            rule_spec={"check_syntax": True},
        )

        async def generator(feedback_context: list[FeedbackProtocol]):
            return "INVALID SQL"  # Would fail if verification was enabled

        output, success = await engine.generate_and_verify(
            generator=generator,
            verifier=verifier,
            rule=rule,
        )

        assert success is True  # Passes because verification is disabled
        assert output == "INVALID SQL"

    @pytest.mark.asyncio
    async def test_e2e_strict_mode_fails_immediately(self):
        """Test that strict mode fails immediately without retries."""
        engine = NeuroSymbolicEngine(
            config=VerificationConfig(
                enabled=True,
                mode=VerificationMode.STRICT,
                max_retries=5,  # Should not retry in strict mode
                auto_fix=False,
            )
        )

        verifier = SQLVerifier(use_constraint_solver=True)
        rule = VerificationRule(
            rule_id="sql_check",
            verifier_type="sql",
            rule_spec={"check_syntax": True},
        )

        attempt_count = 0

        async def generator(feedback_context: list[FeedbackProtocol]):
            nonlocal attempt_count
            attempt_count += 1
            return "SELECT * FROM users"  # Missing semicolon

        output, success = await engine.generate_and_verify(
            generator=generator,
            verifier=verifier,
            rule=rule,
        )

        assert success is False
        assert attempt_count == 1  # No retries in strict mode

        # In strict mode, feedback context is cleared at start but not built
        # since we fail immediately without building feedback
        feedback_context = engine.get_feedback_context()
        assert len(feedback_context) == 0  # No feedback in strict mode (fails immediately)
