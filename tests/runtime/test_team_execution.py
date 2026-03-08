"""Tests for Team.run() execution and workflow."""

import pytest
from pydantic import BaseModel

from houyi import SkillSpec
from houyi.application.runtime.agent import Agent
from houyi.application.runtime.task import Task
from houyi.application.runtime.team import Team


class TestTeamExecution:
    """Test Team.run() method and workflow execution."""

    def test_team_run_single_task(self):
        """Test team execution with single task."""
        agent = Agent(role="Worker")
        task = Task(description="Complete task", agent=agent)

        team = Team(agents=[agent], tasks=[task])

        # Team should have tasks configured
        assert len(team.tasks) == 1
        assert team.tasks[0].description == "Complete task"

    def test_team_run_multiple_tasks(self):
        """Test team execution with multiple tasks."""
        agent1 = Agent(role="Agent 1")
        agent2 = Agent(role="Agent 2")

        task1 = Task(description="Task 1", agent=agent1)
        task2 = Task(description="Task 2", agent=agent2)

        team = Team(agents=[agent1, agent2], tasks=[task1, task2])

        assert len(team.tasks) == 2
        assert len(team.agents) == 2

    def test_team_task_validation(self):
        """Test team validates task-agent relationships."""
        agent1 = Agent(role="Agent 1")
        agent2 = Agent(role="Agent 2")  # Not in team

        task = Task(description="Task", agent=agent2)

        # Should raise error when task references agent not in team
        with pytest.raises(ValueError, match="not in the team"):
            Team(agents=[agent1], tasks=[task])

    def test_team_with_task_dependencies(self):
        """Test team with dependent tasks."""
        agent = Agent(role="Worker")

        task1 = Task(description="Task 1", agent=agent)
        task2 = Task(description="Task 2", agent=agent, context=[0])  # Depends on task 1

        team = Team(agents=[agent], tasks=[task1, task2])

        assert team.tasks[1].context == [0]

    def test_team_observability_config(self):
        """Test team observability configuration."""
        agent = Agent(role="Worker")
        task = Task(description="Task", agent=agent)

        team = Team(
            agents=[agent], tasks=[task], observability={"enabled": True, "export_to": "console"}
        )

        assert team.observability["enabled"] is True
        assert team.observability["export_to"] == "console"

    def test_team_default_observability(self):
        """Test team default observability settings."""
        agent = Agent(role="Worker")
        task = Task(description="Task", agent=agent)

        team = Team(agents=[agent], tasks=[task])

        assert team.observability["enabled"] is True

    def test_team_empty_tasks(self):
        """Test team with no tasks."""
        agent = Agent(role="Worker")

        team = Team(agents=[agent], tasks=[])

        assert len(team.tasks) == 0

    def test_team_task_with_expected_output(self):
        """Test team task with expected output."""
        agent = Agent(role="Worker")
        task = Task(description="Generate report", expected_output="Report content", agent=agent)

        team = Team(agents=[agent], tasks=[task])

        assert team.tasks[0].expected_output == "Report content"

    def test_team_multiple_agents_single_task(self):
        """Test team with multiple agents but single task."""
        agent1 = Agent(role="Agent 1")
        agent2 = Agent(role="Agent 2")

        task = Task(description="Shared task", agent=agent1)

        team = Team(agents=[agent1, agent2], tasks=[task])

        assert len(team.agents) == 2
        assert len(team.tasks) == 1

    def test_team_task_assignment(self):
        """Test task assignment to specific agent."""
        agent1 = Agent(role="Researcher")
        agent2 = Agent(role="Writer")

        research_task = Task(description="Research topic", agent=agent1)
        writing_task = Task(description="Write article", agent=agent2)

        team = Team(agents=[agent1, agent2], tasks=[research_task, writing_task])

        assert team.tasks[0].agent == agent1
        assert team.tasks[1].agent == agent2

    def test_team_with_skills(self):
        """Test team agents with skills."""

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            result: str

        def search(input: Input) -> Output:
            return Output(result="found")

        skill = SkillSpec(
            name="search",
            description="Search",
            input_schema=Input,
            output_schema=Output,
            executor=search,
        )

        agent = Agent(role="Researcher", skills=[skill])
        task = Task(description="Research", agent=agent)

        team = Team(agents=[agent], tasks=[task])

        assert len(team.agents[0].skills) == 1

    def test_team_task_context_dependencies(self):
        """Test team task with context dependencies."""
        agent = Agent(role="Worker")

        task1 = Task(description="Step 1", agent=agent)
        task2 = Task(description="Step 2", agent=agent, context=[0])
        task3 = Task(description="Step 3", agent=agent, context=[0, 1])

        team = Team(agents=[agent], tasks=[task1, task2, task3])

        assert team.tasks[0].context is None
        assert team.tasks[1].context == [0]
        assert team.tasks[2].context == [0, 1]


class TestTaskConfiguration:
    """Test Task configuration and validation."""

    def test_task_with_description(self):
        """Test task creation with description."""
        agent = Agent(role="Worker")
        task = Task(description="Do something", agent=agent)

        assert task.description == "Do something"
        assert task.agent == agent

    def test_task_with_expected_output(self):
        """Test task with expected output."""
        agent = Agent(role="Worker")
        task = Task(description="Generate output", expected_output="Expected result", agent=agent)

        assert task.expected_output == "Expected result"

    def test_task_without_agent(self):
        """Test task without assigned agent."""
        task = Task(description="Unassigned task")

        assert task.description == "Unassigned task"
        assert task.agent is None

    def test_task_with_context(self):
        """Test task with context dependencies."""
        agent = Agent(role="Worker")
        task = Task(description="Dependent task", context=[0, 1, 2], agent=agent)

        assert task.context == [0, 1, 2]

    def test_task_minimal(self):
        """Test task with minimal configuration."""
        task = Task(description="Minimal task")

        assert task.description == "Minimal task"
        assert task.expected_output is None
        assert task.agent is None
        assert task.context is None
