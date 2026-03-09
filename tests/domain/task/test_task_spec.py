"""Tests for core/task.py."""

from houyi.domain.task import TaskSpec


class TestTaskSpec:
    """Test TaskSpec class."""

    def test_task_spec_basic(self):
        """Test basic TaskSpec creation."""
        task = TaskSpec(description="Test task", expected_output="Expected result")

        assert task.description == "Test task"
        assert task.expected_output == "Expected result"
        assert task.agent is None
        assert task.context is None

    def test_task_spec_with_agent(self):
        """Test TaskSpec with agent."""
        task = TaskSpec(
            description="Research task", expected_output="Research report", agent="researcher"
        )

        assert task.agent == "researcher"

    def test_task_spec_with_context(self):
        """Test TaskSpec with context."""
        context = [1, 2, 3]

        task = TaskSpec(description="Research task", expected_output="Report", context=context)

        assert task.context == [1, 2, 3]
        assert len(task.context) == 3
