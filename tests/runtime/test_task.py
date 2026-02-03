"""Tests for runtime/task.py."""

from houyi.core.task import TaskSpec
from houyi.orchestration.state import TaskStatus
from houyi.runtime.task import Task


class TestTask:
    """Test Task runtime class."""

    def test_task_creation(self):
        """Test Task creation."""
        task = Task(description="Test task", expected_output="Result")

        assert task.description == "Test task"
        assert task.expected_output == "Result"
        assert task.state.status == TaskStatus.PENDING

    def test_task_has_spec(self):
        """Test Task has TaskSpec."""
        task = Task(description="Test task", expected_output="Result")

        assert isinstance(task.spec, TaskSpec)
        assert task.spec.description == "Test task"

    def test_task_with_agent(self):
        """Test Task with assigned agent."""
        from houyi.core.agent import AgentSpec

        agent = AgentSpec(role="test_agent")

        task = Task(description="Test task", expected_output="Result", agent=agent)

        assert task.agent == agent

    def test_task_repr(self):
        """Test Task string representation."""
        task = Task(description="Test task", expected_output="Result")

        repr_str = repr(task)
        assert "Test task" in repr_str
        assert "Task" in repr_str
