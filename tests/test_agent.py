"""Tests for core/agent.py - AgentSpec class."""

from pydantic import BaseModel

from houyi.core.agent import AgentSpec
from houyi.core.skill import SkillSpec


class TestAgentSpec:
    """Test AgentSpec class."""

    def test_agent_spec_basic(self):
        """Test basic AgentSpec creation."""
        agent = AgentSpec(role="assistant")

        assert agent.role == "assistant"
        assert agent.skills == []
        assert agent.system_prompt is None
        assert agent.policies == {}

    def test_agent_spec_with_skills(self):
        """Test AgentSpec with skills."""

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            result: str

        def search(input: Input) -> Output:
            return Output(result="test")

        skill = SkillSpec(
            name="search",
            description="Search skill",
            input_schema=Input,
            output_schema=Output,
            executor=search,
        )

        agent = AgentSpec(role="researcher", skills=[skill])

        assert len(agent.skills) == 1
        assert agent.skills[0].name == "search"

    def test_to_system_prompt_basic(self):
        """Test to_system_prompt() with basic agent."""
        agent = AgentSpec(role="assistant")

        prompt = agent.to_system_prompt()

        assert "assistant" in prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_to_system_prompt_with_custom_prompt(self):
        """Test to_system_prompt() with custom system prompt."""
        custom_prompt = "You are a helpful AI assistant."
        agent = AgentSpec(role="assistant", system_prompt=custom_prompt)

        prompt = agent.to_system_prompt()

        assert prompt == custom_prompt

    def test_to_system_prompt_with_skills(self):
        """Test to_system_prompt() includes skills."""

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            result: str

        def search(input: Input) -> Output:
            return Output(result="test")

        skill1 = SkillSpec(
            name="search",
            description="Search the web",
            input_schema=Input,
            output_schema=Output,
            executor=search,
        )

        skill2 = SkillSpec(
            name="calculate",
            description="Perform calculations",
            input_schema=Input,
            output_schema=Output,
            executor=search,
        )

        agent = AgentSpec(role="researcher", skills=[skill1, skill2])

        prompt = agent.to_system_prompt()

        assert "researcher" in prompt
        assert "search" in prompt.lower()
        assert "calculate" in prompt.lower()
        assert "Search the web" in prompt
        assert "Perform calculations" in prompt

    def test_get_tool_schemas(self):
        """Test get_tool_schemas() method."""

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            result: str

        def search(input: Input) -> Output:
            return Output(result="test")

        skill = SkillSpec(
            name="search",
            description="Search skill",
            input_schema=Input,
            output_schema=Output,
            executor=search,
        )

        agent = AgentSpec(role="assistant", skills=[skill])

        schemas = agent.get_tool_schemas()

        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "search"

    def test_get_tool_schemas_multiple_skills(self):
        """Test get_tool_schemas() with multiple skills."""

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            result: str

        def func(input: Input) -> Output:
            return Output(result="test")

        skill1 = SkillSpec(
            name="skill1",
            description="Skill 1",
            input_schema=Input,
            output_schema=Output,
            executor=func,
        )

        skill2 = SkillSpec(
            name="skill2",
            description="Skill 2",
            input_schema=Input,
            output_schema=Output,
            executor=func,
        )

        agent = AgentSpec(role="assistant", skills=[skill1, skill2])

        schemas = agent.get_tool_schemas()

        assert len(schemas) == 2
        assert schemas[0]["function"]["name"] == "skill1"
        assert schemas[1]["function"]["name"] == "skill2"

    def test_get_tool_schemas_empty(self):
        """Test get_tool_schemas() with no skills."""
        agent = AgentSpec(role="assistant")

        schemas = agent.get_tool_schemas()

        assert schemas == []

    def test_agent_spec_with_policies(self):
        """Test AgentSpec with policies."""
        policies = {"llm": "gpt-4", "max_retries": 3, "timeout": 30}

        agent = AgentSpec(role="assistant", policies=policies)

        assert agent.policies["llm"] == "gpt-4"
        assert agent.policies["max_retries"] == 3
        assert agent.policies["timeout"] == 30

    def test_agent_spec_policies_default(self):
        """Test AgentSpec policies default to empty dict."""
        agent = AgentSpec(role="assistant")

        assert agent.policies == {}
        assert isinstance(agent.policies, dict)
