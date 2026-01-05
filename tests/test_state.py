"""Tests for orchestration/state.py."""

from houyi.orchestration.state import SessionState, TaskState, TaskStatus, VerificationResult


class TestSessionState:
    """Test SessionState class."""

    def test_session_state_creation(self):
        """Test SessionState creation."""
        state = SessionState(session_id="test_session", agent_id="test_agent")

        assert state.session_id == "test_session"
        assert state.agent_id == "test_agent"
        assert state.current_plan_id is None
        assert state.memory_stack == []

    def test_session_state_with_plan(self):
        """Test SessionState with plan."""
        state = SessionState(
            session_id="test_session", agent_id="test_agent", current_plan_id="plan_123"
        )

        assert state.current_plan_id == "plan_123"

    def test_session_state_with_memory(self):
        """Test SessionState with memory stack."""
        memory = [{"key": "value"}, {"count": 5}]
        state = SessionState(session_id="test_session", agent_id="test_agent", memory_stack=memory)

        assert len(state.memory_stack) == 2
        assert state.memory_stack[0]["key"] == "value"

    def test_session_state_with_metadata(self):
        """Test SessionState with metadata."""
        metadata = {"version": "1.0", "env": "test"}
        state = SessionState(session_id="test_session", agent_id="test_agent", metadata=metadata)

        assert state.metadata["version"] == "1.0"
        assert state.metadata["env"] == "test"


class TestTaskState:
    """Test TaskState class."""

    def test_task_state_creation(self):
        """Test TaskState creation."""
        state = TaskState(
            task_id="task_123", status=TaskStatus.PENDING, input_data={"query": "test"}
        )

        assert state.task_id == "task_123"
        assert state.status == TaskStatus.PENDING
        assert state.input_data["query"] == "test"

    def test_task_state_with_output(self):
        """Test TaskState with output data."""
        state = TaskState(
            task_id="task_123",
            status=TaskStatus.SUCCEEDED,
            input_data={"query": "test"},
            output_data={"result": "success"},
        )

        assert state.output_data["result"] == "success"


class TestTaskStatus:
    """Test TaskStatus enum."""

    def test_task_status_values(self):
        """Test TaskStatus enum values."""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.RUNNING == "running"
        assert TaskStatus.SUCCEEDED == "succeeded"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.CANCELLED == "cancelled"


class TestVerificationResult:
    """Test VerificationResult class."""

    def test_verification_result_passed(self):
        """Test VerificationResult for passed assertion."""
        result = VerificationResult(
            assertion_name="test_assertion", passed=True, message="Assertion passed"
        )

        assert result.assertion_name == "test_assertion"
        assert result.passed is True
        assert result.message == "Assertion passed"

    def test_verification_result_failed(self):
        """Test VerificationResult for failed assertion."""
        result = VerificationResult(
            assertion_name="test_assertion", passed=False, message="Assertion failed"
        )

        assert result.passed is False
        assert result.message == "Assertion failed"

    def test_verification_result_with_context(self):
        """Test VerificationResult with context."""
        context = {"expected": 10, "actual": 5}
        result = VerificationResult(assertion_name="test_assertion", passed=False, context=context)

        assert result.context["expected"] == 10
        assert result.context["actual"] == 5
