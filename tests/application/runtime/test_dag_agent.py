"""Tests for AGENT node execution in DAG and Agent team_agents API."""

from __future__ import annotations

import pytest

from houyi.application.workflow.executor import LocalExecutor
from houyi.application.workflow.orchestration.plan import (
    ExecutionPlan,
    IRNode,
    NodeType,
)
from houyi.application.workflow.orchestration.state import SessionState


class TestAgentNodeType:
    def test_agent_enum_exists(self):
        assert NodeType.AGENT.value == "agent"

    def test_irnode_agent_fields(self):
        node = IRNode(
            node_id="n1",
            node_type=NodeType.AGENT,
            agent_id="researcher",
            handoff_to="writer",
        )
        assert node.agent_id == "researcher"
        assert node.handoff_to == "writer"

    def test_irnode_backward_compat(self):
        node = IRNode(node_id="n2", node_type=NodeType.LLM)
        assert node.agent_id is None
        assert node.handoff_to is None


class TestAgentNodeExecution:
    @pytest.mark.asyncio
    async def test_execute_agent_node(self):
        plan = ExecutionPlan(
            plan_id="p1",
            nodes=[
                IRNode(
                    node_id="a1",
                    node_type=NodeType.AGENT,
                    agent_id="sub_researcher",
                    inputs={"task": "find facts"},
                    outputs={"result": "$agent_output"},
                )
            ],
            entry_node="a1",
            metadata={"task": "research task"},
        )
        state = SessionState(session_id="s1", agent_id="main")
        executor = LocalExecutor()
        result = await executor.execute(plan, state)
        assert result.success

    @pytest.mark.asyncio
    async def test_agent_handoff(self):
        plan = ExecutionPlan(
            plan_id="p2",
            nodes=[
                IRNode(
                    node_id="a1",
                    node_type=NodeType.AGENT,
                    agent_id="explorer",
                    handoff_to="writer",
                    inputs={"task": "explore"},
                    outputs={"result": "$explore_result"},
                ),
            ],
            entry_node="a1",
            metadata={"task": "explore and write"},
        )
        state = SessionState(session_id="s2", agent_id="main")
        executor = LocalExecutor()
        result = await executor.execute(plan, state)
        assert result.success


class TestAgentTeamAgentsAPI:
    def test_team_agents(self):
        from houyi.application.runtime.agent import Agent
        from houyi.domain.agent.spec import AgentTeamConfig

        agent = Agent(
            role="orchestrator",
            team_agents=[
                AgentTeamConfig(role="researcher"),
                AgentTeamConfig(role="writer"),
            ],
            mode="delegate",
        )
        assert len(agent.spec.team_agents) == 2
        assert agent.mode == "delegate"

    def test_no_team_agents_compat(self):
        from houyi.application.runtime.agent import Agent

        agent = Agent(role="simple")
        assert agent.spec.team_agents == []
        assert agent.mode is None
