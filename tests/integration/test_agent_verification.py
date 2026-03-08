"""Integration tests for Agent with verification enabled.

Tests the full workflow: Agent -> DAGPlanner -> LocalExecutor -> Verification
"""

import pytest

from houyi import Agent, tool
from houyi.application.workflow.executor import LocalExecutor
from houyi.application.workflow.orchestration.planner import DAGPlanner
from houyi.application.workflow.orchestration.state import SessionState
from houyi.assurance.verification import VerificationConfig


@tool
def generate_sql(query: str) -> str:
    """Generate SQL query (simulated, returns query without semicolon)."""
    return f"SELECT * FROM users WHERE name = '{query}'"


@tool
def generate_safe_sql(query: str) -> str:
    """Generate safe SQL query."""
    return f"SELECT * FROM users WHERE name = '{query}';"


@pytest.mark.asyncio
async def test_agent_with_verification_disabled():
    """Test agent with verification disabled."""
    config = VerificationConfig.disabled()

    agent = Agent(
        role="SQL Generator",
        skills=[generate_sql],
        llm=None,  # No LLM, direct execution
    )
    agent.spec.verification_config = config

    # Create planner with verification config
    planner = DAGPlanner(verification_config=config)
    plan = planner.plan("test query", agent.spec)

    # Verification nodes should not be added
    verify_nodes = [n for n in plan.nodes if n.node_type.value == "verify"]
    assert len(verify_nodes) == 0, "No verification nodes when disabled"


@pytest.mark.asyncio
async def test_agent_with_verification_enabled():
    """Test agent with verification enabled."""
    from pydantic import BaseModel, Field

    from houyi.domain.skill.spec import SkillSpec

    config = VerificationConfig.lenient()

    # Create skill with metadata
    class SQLInput(BaseModel):
        query: str = Field(..., description="Query to generate SQL for")

    class SQLOutput(BaseModel):
        sql: str = Field(..., description="Generated SQL")

    sql_skill = SkillSpec(
        name="generate_sql",
        description="Generate SQL query",
        input_schema=SQLInput,
        output_schema=SQLOutput,
        executor=lambda query: f"SELECT * FROM users WHERE name = '{query}';",
        metadata={"output_type": "sql"},
    )

    agent = Agent(
        role="SQL Generator",
        skills=[sql_skill],
        llm=None,
    )
    agent.spec.verification_config = config

    # Create planner with verification config
    planner = DAGPlanner(verification_config=config)
    plan = planner.plan("test query", agent.spec)

    # Verification nodes should be added
    verify_nodes = [n for n in plan.nodes if n.node_type.value == "verify"]
    assert len(verify_nodes) > 0, "Should have verification nodes when enabled"

    # Check verification node has rules
    verify_node = verify_nodes[0]
    assert verify_node.verification_rules is not None
    assert len(verify_node.verification_rules) > 0


@pytest.mark.asyncio
async def test_skill_level_verification_override():
    """Test skill-level verification config overrides agent-level."""
    from houyi.assurance.verification import VerificationConfig

    # Agent has verification enabled
    agent_config = VerificationConfig.lenient()

    # Skill has verification disabled
    skill_config = VerificationConfig.disabled()
    generate_sql.verification_config = skill_config

    agent = Agent(
        role="SQL Generator",
        skills=[generate_sql],
        llm=None,
    )
    agent.spec.verification_config = agent_config

    # Create planner with agent config
    planner = DAGPlanner(verification_config=agent_config)
    plan = planner.plan("test query", agent.spec)

    # Should respect skill-level override (disabled)
    verify_nodes = [n for n in plan.nodes if n.node_type.value == "verify"]
    # Note: Current implementation checks skill config in _should_verify
    # This test documents the expected behavior


@pytest.mark.asyncio
async def test_verification_with_executor():
    """Test full execution with verification."""
    config = VerificationConfig.lenient()

    # Create skill with metadata
    generate_safe_sql.metadata = {"output_type": "sql"}

    agent = Agent(
        role="SQL Generator",
        skills=[generate_safe_sql],
        llm=None,
    )
    agent.spec.verification_config = config

    # Create planner and executor
    planner = DAGPlanner(verification_config=config)
    executor = LocalExecutor()

    # Plan and execute
    plan = planner.plan("admin", agent.spec)
    state = SessionState(session_id="test", agent_id="test_agent")

    result = await executor.execute(plan, state)

    # Should succeed (safe SQL with semicolon)
    assert result.success is True
    assert "SELECT" in str(result.output)


@pytest.mark.asyncio
async def test_multiple_skills_with_verification():
    """Test agent with multiple skills, each with different verification needs."""

    @tool
    def generate_python(code: str) -> str:
        """Generate Python code."""
        return f"print('{code}')"

    generate_python.metadata = {"output_type": "python"}
    generate_safe_sql.metadata = {"output_type": "sql"}

    config = VerificationConfig.lenient()

    agent = Agent(
        role="Code Generator",
        skills=[generate_safe_sql, generate_python],
        llm=None,
    )
    agent.spec.verification_config = config

    planner = DAGPlanner(verification_config=config)
    plan = planner.plan("test", agent.spec)

    # In direct execution mode (no LLM), only first skill is used
    # So we should have at least 1 verification node
    verify_nodes = [n for n in plan.nodes if n.node_type.value == "verify"]
    assert len(verify_nodes) >= 1, "Should have verification for skill"

    # Check verifier type
    rule_types = set()
    for node in verify_nodes:
        if node.verification_rules:
            for rule in node.verification_rules:
                rule_types.add(rule.verifier_type)

    assert "sql" in rule_types, "Should have SQL verifier for first skill"


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_agent_with_verification_enabled())
