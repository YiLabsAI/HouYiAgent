"""Tests for runtime Agent class."""

from pydantic import BaseModel

from houyi.core.skill import SkillSpec
from houyi.runtime.agent import Agent


class TestAgent:
    """Test Agent runtime class."""

    def test_agent_initialization(self):
        """Test basic agent initialization."""
        agent = Agent(
            role="Test Agent",
            llm="gpt-4",
            memory=False
        )

        assert agent.role == "Test Agent"
        assert agent.spec.role == "Test Agent"
        assert agent.spec.policies["llm"] == "gpt-4"
        assert agent.spec.policies["memory"] is False
        assert agent.state is not None

    def test_agent_with_skills(self):
        """Test agent initialization with skills."""
        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            result: str

        def search(input: Input) -> Output:
            return Output(result="test")

        skill = SkillSpec(
            name="search",
            description="Search",
            input_schema=Input,
            output_schema=Output,
            executor=search,
        )

        agent = Agent(
            role="Search Agent",
            skills=[skill]
        )

        assert len(agent.skills) == 1
        assert agent.skills[0].name == "search"

    def test_agent_with_system_prompt(self):
        """Test agent with custom system prompt."""
        agent = Agent(
            role="Custom Agent",
            system_prompt="You are a helpful assistant."
        )

        assert agent.spec.system_prompt == "You are a helpful assistant."

    def test_agent_properties(self):
        """Test agent property accessors."""
        class Input(BaseModel):
            x: int

        class Output(BaseModel):
            y: int

        def doubler(input: Input) -> Output:
            return Output(y=input.x * 2)

        skill = SkillSpec(
            name="doubler",
            description="Double",
            input_schema=Input,
            output_schema=Output,
            executor=doubler,
        )

        agent = Agent(
            role="Math Agent",
            skills=[skill],
            llm="gpt-3.5-turbo"
        )

        # Test properties
        assert agent.role == "Math Agent"
        assert len(agent.skills) == 1
        assert agent.skills[0].name == "doubler"

    def test_agent_observability_config(self):
        """Test agent observability configuration."""
        agent = Agent(
            role="Test Agent",
            observability={"enabled": True, "export_to": "console"}
        )

        assert agent.observability_config["enabled"] is True
        assert agent.observability_config["export_to"] == "console"

    def test_agent_default_observability(self):
        """Test agent default observability settings."""
        agent = Agent(role="Test Agent")

        assert agent.observability_config is not None
        assert agent.observability_config["enabled"] is True

    def test_agent_session_state(self):
        """Test agent session state initialization."""
        agent = Agent(role="Test Agent")

        assert agent.state is not None
        assert agent.state.session_id is not None
        assert agent.state.agent_id is not None
        assert "Test Agent" in agent.state.agent_id or "agent_" in agent.state.agent_id

    def test_agent_multiple_skills(self):
        """Test agent with multiple skills."""
        class Input1(BaseModel):
            x: int

        class Output1(BaseModel):
            result: int

        class Input2(BaseModel):
            text: str

        class Output2(BaseModel):
            length: int

        def adder(input: Input1) -> Output1:
            return Output1(result=input.x + 10)

        def counter(input: Input2) -> Output2:
            return Output2(length=len(input.text))

        skill1 = SkillSpec(
            name="adder",
            description="Add 10",
            input_schema=Input1,
            output_schema=Output1,
            executor=adder,
        )

        skill2 = SkillSpec(
            name="counter",
            description="Count chars",
            input_schema=Input2,
            output_schema=Output2,
            executor=counter,
        )

        agent = Agent(
            role="Multi-Skill Agent",
            skills=[skill1, skill2]
        )

        assert len(agent.skills) == 2
        assert agent.skills[0].name == "adder"
        assert agent.skills[1].name == "counter"

    def test_agent_empty_skills(self):
        """Test agent with no skills."""
        agent = Agent(role="No-Skill Agent")

        assert len(agent.skills) == 0
        assert agent.skills == []

    def test_agent_memory_enabled(self):
        """Test agent with memory enabled."""
        agent = Agent(
            role="Memory Agent",
            memory=True
        )

        assert agent.spec.policies["memory"] is True

    def test_agent_memory_disabled(self):
        """Test agent with memory disabled."""
        agent = Agent(
            role="Stateless Agent",
            memory=False
        )

        assert agent.spec.policies["memory"] is False
