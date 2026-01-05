"""Tests for core/task.py and runtime/task.py."""


from houyi.core.task import TaskSpec
from houyi.orchestration.state import TaskStatus
from houyi.runtime.task import Task


class TestTaskSpec:
    """Test TaskSpec class."""

    def test_task_spec_basic(self):
        """Test basic TaskSpec creation."""
        task = TaskSpec(
            description="Test task",
            expected_output="Expected result"
        )

        assert task.description == "Test task"
        assert task.expected_output == "Expected result"
        assert task.agent is None
        assert task.context is None

    def test_task_spec_with_agent(self):
        """Test TaskSpec with agent."""
        task = TaskSpec(
            description="Research task",
            expected_output="Research report",
            agent="researcher"
        )

        assert task.agent == "researcher"

    def test_task_spec_with_context(self):
        """Test TaskSpec with context."""
        context = [1, 2, 3]

        task = TaskSpec(
            description="Research task",
            expected_output="Report",
            context=context
        )

        assert task.context == [1, 2, 3]
        assert len(task.context) == 3


class TestTask:
    """Test Task runtime class."""

    def test_task_creation(self):
        """Test Task creation."""
        task = Task(
            description="Test task",
            expected_output="Result"
        )

        assert task.description == "Test task"
        assert task.expected_output == "Result"
        assert task.state.status == TaskStatus.PENDING

    def test_task_has_spec(self):
        """Test Task has TaskSpec."""
        task = Task(
            description="Test task",
            expected_output="Result"
        )

        assert isinstance(task.spec, TaskSpec)
        assert task.spec.description == "Test task"

    def test_task_with_agent(self):
        """Test Task with assigned agent."""
        from houyi.core.agent import AgentSpec
        agent = AgentSpec(role="test_agent")

        task = Task(
            description="Test task",
            expected_output="Result",
            agent=agent
        )

        assert task.agent == agent

    def test_task_repr(self):
        """Test Task string representation."""
        task = Task(
            description="Test task",
            expected_output="Result"
        )

        repr_str = repr(task)
        assert "Test task" in repr_str
        assert "Task" in repr_str
