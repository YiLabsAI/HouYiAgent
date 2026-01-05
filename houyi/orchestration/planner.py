"""DAG planner for task decomposition."""

from __future__ import annotations

from houyi.core.agent import AgentSpec
from houyi.orchestration.plan import ExecutionPlan, IRNode, NodeType
from houyi.orchestration.state import SessionState


class DAGPlanner:
    """Generate execution plans from high-level tasks.

    Decomposes tasks into DAG of IR nodes (LLM, TOOL, VERIFY, etc.).
    """

    def plan(
        self,
        task: str,
        agent: AgentSpec,
        session_state: SessionState | None = None,
    ) -> ExecutionPlan:
        """Generate execution plan for a task.

        Args:
            task: Task description
            agent: Agent specification
            session_state: Optional session state

        Returns:
            ExecutionPlan with DAG of IR nodes
        """
        plan_id = f"plan_{id(task)}"
        nodes = []

        # Analyze if task needs skills
        needs_skills = len(agent.skills) > 0

        if needs_skills:
            # Create nodes for skill-based execution
            # 1. LLM node to decide which skill to use
            llm_node = IRNode(
                node_id="llm_decide",
                node_type=NodeType.LLM,
                inputs={"task": task, "available_skills": [s.name for s in agent.skills]},
                outputs={"skill_name": "$skill_to_use", "skill_input": "$skill_params"},
                dependencies=[],
                metadata={"purpose": "decide_skill"},
            )
            nodes.append(llm_node)

            # 2. TOOL nodes for each skill (one will be selected)
            for skill in agent.skills:
                tool_node = IRNode(
                    node_id=f"tool_{skill.name}",
                    node_type=NodeType.TOOL,
                    skill_ref=skill,
                    inputs={"params": "$skill_params"},
                    outputs={"result": f"$tool_{skill.name}_result"},
                    dependencies=["llm_decide"],
                    metadata={"skill_name": skill.name},
                )
                nodes.append(tool_node)

            # 3. LLM node to synthesize final answer
            final_node = IRNode(
                node_id="llm_synthesize",
                node_type=NodeType.LLM,
                inputs={
                    "task": task,
                    "tool_results": "$tool_results",
                },
                outputs={"final_answer": "$answer"},
                dependencies=[f"tool_{s.name}" for s in agent.skills],
                metadata={"purpose": "synthesize"},
            )
            nodes.append(final_node)

            entry_node = "llm_decide"
        else:
            # Simple LLM-only execution
            llm_node = IRNode(
                node_id="llm_main",
                node_type=NodeType.LLM,
                inputs={"task": task},
                outputs={"answer": "$answer"},
                dependencies=[],
                metadata={"purpose": "direct_answer"},
            )
            nodes.append(llm_node)
            entry_node = "llm_main"

        return ExecutionPlan(
            plan_id=plan_id,
            nodes=nodes,
            entry_node=entry_node,
            metadata={
                "task": task,
                "agent_role": agent.role,
                "num_skills": len(agent.skills),
            },
        )
