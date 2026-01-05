"""Tests for core abstractions."""

import pytest
from pydantic import BaseModel

from houyi import AgentSpec, AssertionSpec, SkillSpec


class TestSkillSpec:
    """Test SkillSpec."""

    def test_skill_creation(self) -> None:
        """Test creating a skill."""

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            result: str

        def executor(input: Input) -> Output:
            return Output(result=f"Processed: {input.query}")

        skill = SkillSpec(
            name="test_skill",
            description="A test skill",
            input_schema=Input,
            output_schema=Output,
            executor=executor,
        )

        assert skill.name == "test_skill"
        assert skill.description == "A test skill"

    def test_to_tool_schema(self) -> None:
        """Test converting skill to OpenAI tool schema."""

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            result: str

        def executor(input: Input) -> Output:
            return Output(result="test")

        skill = SkillSpec(
            name="search",
            description="Search the web",
            input_schema=Input,
            output_schema=Output,
            executor=executor,
        )

        schema = skill.to_tool_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert schema["function"]["description"] == "Search the web"
        assert "parameters" in schema["function"]


class TestAssertionSpec:
    """Test AssertionSpec."""

    def test_assertion_with_string_condition(self) -> None:
        """Test assertion with string condition."""
        assertion = AssertionSpec(
            name="cost_check",
            condition="cost < 1.0",
            on_failure="abort",
        )

        assert assertion.evaluate({"cost": 0.5}) is True
        assert assertion.evaluate({"cost": 1.5}) is False

    def test_assertion_with_callable(self) -> None:
        """Test assertion with callable condition."""

        def check_length(ctx: dict) -> bool:
            return len(ctx.get("text", "")) > 10

        assertion = AssertionSpec(
            name="length_check",
            condition=check_length,
            on_failure="retry",
        )

        assert assertion.evaluate({"text": "short"}) is False
        assert assertion.evaluate({"text": "this is a long text"}) is True


class TestAgentSpec:
    """Test AgentSpec."""

    def test_agent_creation(self) -> None:
        """Test creating an agent."""

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            result: str

        def search(input: Input) -> Output:
            return Output(result="test")

        agent = AgentSpec(
            role="Research Assistant",
            skills=[
                SkillSpec(
                    name="search",
                    description="Search the web",
                    input_schema=Input,
                    output_schema=Output,
                    executor=search,
                )
            ],
        )

        assert agent.role == "Research Assistant"
        assert len(agent.skills) == 1

    def test_system_prompt_generation(self) -> None:
        """Test system prompt generation."""

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            result: str

        def search(input: Input) -> Output:
            return Output(result="test")

        agent = AgentSpec(
            role="Research Assistant",
            skills=[
                SkillSpec(
                    name="search",
                    description="Search the web",
                    input_schema=Input,
                    output_schema=Output,
                    executor=search,
                )
            ],
        )

        prompt = agent.to_system_prompt()
        assert "Research Assistant" in prompt
        assert "search" in prompt
        assert "Search the web" in prompt

    def test_get_tool_schemas(self) -> None:
        """Test getting tool schemas."""

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            result: str

        def search(input: Input) -> Output:
            return Output(result="test")

        agent = AgentSpec(
            name="researcher",
            role="Research Assistant",
            goal="Research information",
            skills=[
                SkillSpec(
                    name="search",
                    description="Search the web",
                    input_schema=Input,
                    output_schema=Output,
                    executor=search,
                )
            ],
        )

        schemas = agent.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "search"
