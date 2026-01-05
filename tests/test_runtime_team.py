"""Tests for runtime Team class."""

import pytest
from pydantic import BaseModel

from houyi.runtime.team import Team
from houyi.runtime.agent import Agent
from houyi.core.skill import SkillSpec


class TestTeam:
    """Test Team runtime class."""

    def test_team_initialization(self):
        """Test basic team initialization."""
        from houyi.runtime.task import Task
        
        agent1 = Agent(role="Agent 1")
        agent2 = Agent(role="Agent 2")
        
        task1 = Task(description="Task 1", agent=agent1)
        task2 = Task(description="Task 2", agent=agent2)
        
        team = Team(
            agents=[agent1, agent2],
            tasks=[task1, task2]
        )
        
        assert len(team.agents) == 2
        assert len(team.tasks) == 2
        assert team.agents[0].role == "Agent 1"
        assert team.agents[1].role == "Agent 2"

    def test_team_single_agent(self):
        """Test team with single agent."""
        from houyi.runtime.task import Task
        
        agent = Agent(role="Solo Agent")
        task = Task(description="Solo Task", agent=agent)
        
        team = Team(
            agents=[agent],
            tasks=[task]
        )
        
        assert len(team.agents) == 1
        assert team.agents[0].role == "Solo Agent"

    def test_team_with_skills(self):
        """Test team agents with skills."""
        from houyi.runtime.task import Task
        
        class Input(BaseModel):
            x: int
        
        class Output(BaseModel):
            result: int
        
        def adder(input: Input) -> Output:
            return Output(result=input.x + 10)
        
        def multiplier(input: Input) -> Output:
            return Output(result=input.x * 2)
        
        skill1 = SkillSpec(
            name="adder",
            description="Add 10",
            input_schema=Input,
            output_schema=Output,
            executor=adder,
        )
        
        skill2 = SkillSpec(
            name="multiplier",
            description="Multiply by 2",
            input_schema=Input,
            output_schema=Output,
            executor=multiplier,
        )
        
        agent1 = Agent(role="Adder Agent", skills=[skill1])
        agent2 = Agent(role="Multiplier Agent", skills=[skill2])
        
        task1 = Task(description="Add task", agent=agent1)
        task2 = Task(description="Multiply task", agent=agent2)
        
        team = Team(
            agents=[agent1, agent2],
            tasks=[task1, task2]
        )
        
        assert len(team.agents) == 2
        assert len(team.agents[0].skills) == 1
        assert len(team.agents[1].skills) == 1

    def test_team_empty_agents(self):
        """Test team with no agents."""
        team = Team(agents=[], tasks=[])
        
        assert len(team.agents) == 0

    def test_team_multiple_agents(self):
        """Test team with multiple agents."""
        from houyi.runtime.task import Task
        
        agents = [
            Agent(role=f"Agent {i}")
            for i in range(5)
        ]
        tasks = [
            Task(description=f"Task {i}", agent=agents[i])
            for i in range(5)
        ]
        
        team = Team(agents=agents, tasks=tasks)
        
        assert len(team.agents) == 5
        for i, agent in enumerate(team.agents):
            assert agent.role == f"Agent {i}"

    def test_team_agent_access(self):
        """Test accessing individual agents from team."""
        from houyi.runtime.task import Task
        
        agent1 = Agent(role="Leader")
        agent2 = Agent(role="Worker")
        
        task1 = Task(description="Lead", agent=agent1)
        task2 = Task(description="Work", agent=agent2)
        
        team = Team(agents=[agent1, agent2], tasks=[task1, task2])
        
        # Access by index
        assert team.agents[0].role == "Leader"
        assert team.agents[1].role == "Worker"

    def test_team_observability(self):
        """Test team observability configuration."""
        team = Team(agents=[], tasks=[], observability={"enabled": True})
        
        assert team.observability["enabled"] is True

    def test_team_with_different_llms(self):
        """Test team with agents using different LLMs."""
        from houyi.runtime.task import Task
        
        agent1 = Agent(role="GPT4 Agent", llm="gpt-4")
        agent2 = Agent(role="GPT3 Agent", llm="gpt-3.5-turbo")
        
        task1 = Task(description="Task 1", agent=agent1)
        task2 = Task(description="Task 2", agent=agent2)
        
        team = Team(agents=[agent1, agent2], tasks=[task1, task2])
        
        assert team.agents[0].spec.policies["llm"] == "gpt-4"
        assert team.agents[1].spec.policies["llm"] == "gpt-3.5-turbo"

    def test_team_agents_with_memory(self):
        """Test team with agents having memory enabled."""
        from houyi.runtime.task import Task
        
        agent1 = Agent(role="Memory Agent", memory=True)
        agent2 = Agent(role="Stateless Agent", memory=False)
        
        task1 = Task(description="Task 1", agent=agent1)
        task2 = Task(description="Task 2", agent=agent2)
        
        team = Team(agents=[agent1, agent2], tasks=[task1, task2])
        
        assert team.agents[0].spec.policies["memory"] is True
        assert team.agents[1].spec.policies["memory"] is False
