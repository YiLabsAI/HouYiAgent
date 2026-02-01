"""Tests for execution/executor.py"""

import pytest

from houyi.core.agent import AgentSpec
from houyi.execution.executor import SimplifiedExecutor


def test_simplified_executor_basic():
    """Test basic execution."""
    executor = SimplifiedExecutor()

    # Create a simple agent
    agent = AgentSpec(name="TestAgent", role="Test", goal="Test goal", skills=[])

    with pytest.warns(DeprecationWarning):
        result = executor.execute(task_description="Test task", agent_spec=agent)

    assert result is not None
    assert "task" in result
    assert result["task"] == "Test task"


def test_simplified_executor_with_skills():
    """Test execution with skills."""
    executor = SimplifiedExecutor()

    # Create skill using decorator
    from houyi import tool

    @tool
    def test_skill(x: int) -> int:
        """Test skill."""
        return x * 2

    # Create agent with skill
    agent = AgentSpec(name="TestAgent", role="Test", goal="Test goal", skills=[test_skill])

    with pytest.warns(DeprecationWarning):
        result = executor.execute(task_description="Double the number 5", agent_spec=agent)

    assert result is not None
    assert "available_tools" in result
    assert "test_skill" in result["available_tools"]


def test_simplified_executor_with_expected_output():
    """Test execution with expected output."""
    executor = SimplifiedExecutor()

    agent = AgentSpec(name="TestAgent", role="Test", goal="Test goal", skills=[])

    with pytest.warns(DeprecationWarning):
        result = executor.execute(
            task_description="Test task", agent_spec=agent, expected_output="A number"
        )

    assert result is not None
    assert result["expected_output"] == "A number"


def test_simplified_executor_system_prompt():
    """Test that system prompt is generated."""
    executor = SimplifiedExecutor()

    agent = AgentSpec(name="TestAgent", role="Tester", goal="Test everything", skills=[])

    with pytest.warns(DeprecationWarning):
        result = executor.execute(task_description="Test task", agent_spec=agent)

    assert "system_prompt" in result
    assert "Tester" in result["system_prompt"]
    # Note: goal is not included in system prompt by default
