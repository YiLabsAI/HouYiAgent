"""Integration tests covering multiple modules end-to-end."""

from pydantic import BaseModel

from houyi import AgentSpec, SkillSpec
from houyi.evaluation.runner import evaluate
from houyi.runtime.agent import Agent
from houyi.runtime.task import Task
from houyi.runtime.team import Team


class TestEndToEndIntegration:
    """Test end-to-end workflows."""

    def test_agent_with_skill_execution(self):
        """Test complete agent workflow with skill execution."""

        class Input(BaseModel):
            x: int

        class Output(BaseModel):
            result: int

        def calculator(input: Input) -> Output:
            return Output(result=input.x * 2)

        skill = SkillSpec(
            name="calculator",
            description="Calculate",
            input_schema=Input,
            output_schema=Output,
            executor=calculator,
        )

        agent = Agent(role="Calculator Agent", skills=[skill], llm="gpt-4")

        assert agent.role == "Calculator Agent"
        assert len(agent.skills) == 1

    def test_team_with_multiple_agents(self):
        """Test team with multiple agents."""
        agent1 = Agent(role="Agent 1")
        agent2 = Agent(role="Agent 2")

        task1 = Task(description="Task 1", agent=agent1)
        task2 = Task(description="Task 2", agent=agent2)

        team = Team(agents=[agent1, agent2], tasks=[task1, task2])

        assert len(team.agents) == 2
        assert len(team.tasks) == 2

    def test_evaluation_workflow(self):
        """Test evaluation workflow."""

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            answer: str

        def simple_qa(input: Input) -> Output:
            return Output(answer="42")

        skill = SkillSpec(
            name="qa",
            description="QA",
            input_schema=Input,
            output_schema=Output,
            executor=simple_qa,
        )

        agent = AgentSpec(role="QA Agent", skills=[skill])

        results = evaluate(
            agent=agent,
            test_cases=[{"input": "What is the answer?", "expected_output": "42"}],
            evaluators=["accuracy"],
        )

        assert results is not None
        assert len(results.results) > 0

    def test_agent_system_prompt_generation(self):
        """Test agent system prompt generation."""

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            result: str

        def search(input: Input) -> Output:
            return Output(result="found")

        skill = SkillSpec(
            name="search",
            description="Search the web",
            input_schema=Input,
            output_schema=Output,
            executor=search,
        )

        agent = Agent(role="Research Agent", skills=[skill])

        prompt = agent.spec.to_system_prompt()

        assert "Research Agent" in prompt
        assert "search" in prompt.lower()

    def test_agent_tool_schemas(self):
        """Test agent tool schema generation."""

        class Input(BaseModel):
            text: str

        class Output(BaseModel):
            length: int

        def counter(input: Input) -> Output:
            return Output(length=len(input.text))

        skill = SkillSpec(
            name="counter",
            description="Count characters",
            input_schema=Input,
            output_schema=Output,
            executor=counter,
        )

        agent = Agent(role="Counter Agent", skills=[skill])

        schemas = agent.spec.get_tool_schemas()

        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "counter"

    def test_multiple_skills_agent(self):
        """Test agent with multiple skills."""

        class Input1(BaseModel):
            x: int

        class Output1(BaseModel):
            result: int

        class Input2(BaseModel):
            text: str

        class Output2(BaseModel):
            upper: str

        def adder(input: Input1) -> Output1:
            return Output1(result=input.x + 10)

        def uppercaser(input: Input2) -> Output2:
            return Output2(upper=input.text.upper())

        skill1 = SkillSpec(
            name="adder",
            description="Add 10",
            input_schema=Input1,
            output_schema=Output1,
            executor=adder,
        )

        skill2 = SkillSpec(
            name="uppercaser",
            description="Uppercase",
            input_schema=Input2,
            output_schema=Output2,
            executor=uppercaser,
        )

        agent = Agent(role="Multi-Skill Agent", skills=[skill1, skill2])

        assert len(agent.skills) == 2
        assert agent.skills[0].name == "adder"
        assert agent.skills[1].name == "uppercaser"

    def test_task_creation(self):
        """Test task creation."""
        agent = Agent(role="Worker")
        task = Task(description="Complete the work", expected_output="Done", agent=agent)

        assert task.description == "Complete the work"
        assert task.expected_output == "Done"
        assert task.agent == agent

    def test_agent_with_custom_system_prompt(self):
        """Test agent with custom system prompt."""
        agent = Agent(role="Custom Agent", system_prompt="You are a specialized assistant.")

        prompt = agent.spec.to_system_prompt()
        assert prompt == "You are a specialized assistant."

    def test_agent_with_memory_enabled(self):
        """Test agent with memory configuration."""
        agent = Agent(role="Memory Agent", memory=True)

        assert agent.spec.policies["memory"] is True

    def test_agent_with_different_llm(self):
        """Test agent with different LLM."""
        agent = Agent(role="GPT-3.5 Agent", llm="gpt-3.5-turbo")

        assert agent.spec.policies["llm"] == "gpt-3.5-turbo"
