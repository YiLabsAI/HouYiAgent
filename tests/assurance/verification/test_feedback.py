"""Tests for feedback protocol and builder."""

from houyi.assurance.verification.feedback import FeedbackBuilder, FeedbackProtocol


class TestFeedbackProtocol:
    """Tests for FeedbackProtocol."""

    def test_create_basic_feedback(self):
        """Test creating basic feedback."""
        feedback = FeedbackProtocol(
            error_type="sql_syntax",
            error_message="Missing semicolon at end of statement",
        )

        assert feedback.error_type == "sql_syntax"
        assert feedback.error_message == "Missing semicolon at end of statement"
        assert feedback.severity == "error"
        assert feedback.violated_constraint == ""
        assert feedback.fix_suggestion == ""

    def test_create_full_feedback(self):
        """Test creating feedback with all fields."""
        feedback = FeedbackProtocol(
            error_type="sql_injection",
            error_message="Potential SQL injection detected",
            violated_constraint="No user input in SQL strings",
            fix_suggestion="Use parameterized queries",
            severity="error",
            input_context={"user_input": "'; DROP TABLE users--"},
            output_context={"line": 5},
            previous_attempts=["SELECT * FROM users WHERE id = 1"],
        )

        assert feedback.error_type == "sql_injection"
        assert feedback.severity == "error"
        assert feedback.violated_constraint == "No user input in SQL strings"
        assert len(feedback.previous_attempts) == 1

    def test_to_llm_prompt_basic(self):
        """Test converting feedback to LLM prompt."""
        feedback = FeedbackProtocol(
            error_type="sql_syntax",
            error_message="Missing semicolon",
        )

        prompt = feedback.to_llm_prompt()

        assert "Verification Failed: sql_syntax" in prompt
        assert "Missing semicolon" in prompt
        assert "Please regenerate" in prompt

    def test_prompt_with_suggestion(self):
        """Test prompt with fix suggestion."""
        feedback = FeedbackProtocol(
            error_type="sql_injection",
            error_message="SQL injection risk",
            fix_suggestion="Use parameterized queries",
        )

        prompt = feedback.to_llm_prompt()

        assert "Suggestion" in prompt
        assert "parameterized queries" in prompt

    def test_prompt_with_constraint(self):
        """Test prompt with violated constraint."""
        feedback = FeedbackProtocol(
            error_type="type_mismatch",
            error_message="Expected int, got str",
            violated_constraint="output must be integer",
        )

        prompt = feedback.to_llm_prompt()

        assert "Violated Constraint" in prompt
        assert "output must be integer" in prompt

    def test_prompt_with_history(self):
        """Test prompt includes previous attempts."""
        feedback = FeedbackProtocol(
            error_type="sql_syntax",
            error_message="Syntax error",
            previous_attempts=[
                "SELECT * FROM users",
                "SELECT id FROM users WHERE",
                "SELECT id FROM users WHERE name =",
            ],
        )

        prompt = feedback.to_llm_prompt()

        assert "Previous Failed Attempts" in prompt
        # Should show last 3 attempts
        assert "1." in prompt
        assert "2." in prompt
        assert "3." in prompt

    def test_prompt_truncates_long(self):
        """Test that long attempts are truncated."""
        long_attempt = "SELECT " + "x, " * 100 + "FROM table"
        feedback = FeedbackProtocol(
            error_type="sql_syntax",
            error_message="Too many columns",
            previous_attempts=[long_attempt],
        )

        prompt = feedback.to_llm_prompt()

        # Should be truncated to ~100 chars
        assert "..." in prompt


class TestFeedbackBuilder:
    """Tests for FeedbackBuilder."""

    def test_build_basic_feedback(self):
        """Test building basic feedback."""
        builder = FeedbackBuilder()

        feedback = builder.build_feedback(
            error_type="sql_syntax",
            error_message="Missing semicolon at end",
            output="SELECT * FROM users",
        )

        assert feedback.error_type == "sql_syntax"
        assert feedback.error_message == "Missing semicolon at end"
        assert feedback.fix_suggestion != ""  # Should have suggestion
        assert len(feedback.previous_attempts) == 1

    def test_build_feedback_with_context(self):
        """Test building feedback with input context."""
        builder = FeedbackBuilder()

        feedback = builder.build_feedback(
            error_type="sql_injection",
            error_message="SQL injection detected",
            output="SELECT * FROM users WHERE id = '1'",
            violated_constraint="No string concatenation",
            input_context={"user_id": "1"},
        )

        assert feedback.input_context == {"user_id": "1"}
        assert feedback.violated_constraint == "No string concatenation"
        assert "output_type" in feedback.output_context

    def test_truncate_long_message(self):
        """Test that long error messages are truncated."""
        builder = FeedbackBuilder()

        long_message = "Error: " + "x" * 600
        feedback = builder.build_feedback(
            error_type="sql_syntax",
            error_message=long_message,
            output="SELECT * FROM users",
        )

        assert len(feedback.error_message) <= builder.MAX_FEEDBACK_LENGTH + 3  # +3 for "..."
        assert feedback.error_message.endswith("...")

    def test_generate_fix_suggestions(self):
        """Test fix suggestion generation for different error types."""
        builder = FeedbackBuilder()

        # SQL injection
        feedback = builder.build_feedback(
            error_type="sql_injection",
            error_message="SQL injection risk",
            output="SELECT * FROM users WHERE id = '1'",
        )
        assert "parameterized queries" in feedback.fix_suggestion.lower()

        # Python syntax
        feedback = builder.build_feedback(
            error_type="python_syntax",
            error_message="Invalid syntax",
            output="def foo(",
        )
        assert "syntax" in feedback.fix_suggestion.lower()

        # Unsafe import
        feedback = builder.build_feedback(
            error_type="unsafe_import",
            error_message="Dangerous import",
            output="import os",
        )
        assert "import" in feedback.fix_suggestion.lower()

    def test_determine_severity(self):
        """Test severity determination."""
        builder = FeedbackBuilder()

        # Security error
        feedback = builder.build_feedback(
            error_type="sql_injection",
            error_message="SQL injection",
            output="SELECT * FROM users",
        )
        assert feedback.severity == "error"

        # Warning
        feedback = builder.build_feedback(
            error_type="type_mismatch",
            error_message="Type mismatch",
            output="123",
        )
        assert feedback.severity == "warning"

    def test_attempt_history_accumulation(self):
        """Test that attempt history accumulates."""
        builder = FeedbackBuilder()

        # Build multiple feedbacks
        for i in range(3):
            feedback = builder.build_feedback(
                error_type="sql_syntax",
                error_message=f"Error {i}",
                output=f"SELECT {i} FROM users",
            )

        # Last feedback should have 3 attempts in history
        assert len(feedback.previous_attempts) == 3

    def test_attempt_history_max_size(self):
        """Test that history is limited to max size."""
        builder = FeedbackBuilder()

        # Build more than MAX_ATTEMPTS_HISTORY feedbacks
        for i in range(10):
            feedback = builder.build_feedback(
                error_type="sql_syntax",
                error_message=f"Error {i}",
                output=f"SELECT {i} FROM users",
            )

        # Should only keep last MAX_ATTEMPTS_HISTORY
        assert len(feedback.previous_attempts) == builder.MAX_ATTEMPTS_HISTORY
        assert len(builder._attempt_history) == builder.MAX_ATTEMPTS_HISTORY

    def test_reset_history(self):
        """Test resetting attempt history."""
        builder = FeedbackBuilder()

        # Build some feedbacks
        for i in range(3):
            builder.build_feedback(
                error_type="sql_syntax",
                error_message=f"Error {i}",
                output=f"SELECT {i} FROM users",
            )

        assert len(builder._attempt_history) == 3

        # Reset
        builder.reset_history()

        assert len(builder._attempt_history) == 0

    def test_get_history_summary(self):
        """Test getting history summary."""
        builder = FeedbackBuilder()

        # Build some feedbacks
        for i in range(3):
            builder.build_feedback(
                error_type="sql_syntax",
                error_message=f"Error {i}",
                output=f"SELECT {i} FROM users",
            )

        summary = builder.get_history_summary()

        assert summary["total_attempts"] == 3
        assert len(summary["attempts"]) == 3

    def test_output_context_includes_type(self):
        """Test that output context includes type information."""
        builder = FeedbackBuilder()

        feedback = builder.build_feedback(
            error_type="type_mismatch",
            error_message="Expected int",
            output="not an int",
        )

        assert "output_type" in feedback.output_context
        assert feedback.output_context["output_type"] == "str"

    def test_output_context_preview_truncated(self):
        """Test that output preview is truncated."""
        builder = FeedbackBuilder()

        long_output = "SELECT " + "x, " * 200 + "FROM table"
        feedback = builder.build_feedback(
            error_type="sql_syntax",
            error_message="Too long",
            output=long_output,
        )

        assert len(feedback.output_context["output_preview"]) <= 200
