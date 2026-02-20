"""DAG planner for task decomposition."""

from __future__ import annotations

from typing import Any

from houyi.core.agent import AgentSpec
from houyi.orchestration.plan import ExecutionPlan, IRNode, NodeType
from houyi.orchestration.state import SessionState
from houyi.verification import VerificationConfig, VerificationRule


class DAGPlanner:
    """Generate execution plans from high-level tasks.

    Decomposes tasks into DAG of IR nodes (LLM, TOOL, VERIFY, etc.).
    """

    def __init__(self, verification_config: VerificationConfig | None = None):
        """Initialize planner.

        Args:
            verification_config: Global verification configuration
        """
        self.verification_config = verification_config

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
        has_llm = agent.policies.get("llm") is not None

        if needs_skills and not has_llm:
            # Fallback: Direct skill execution without LLM
            # Use simple heuristic to match task to skill and extract parameters
            return self._plan_direct_skill_execution(task, agent, plan_id)

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

                # Insert verification node if enabled
                if self._should_verify(skill):
                    verify_node = self._create_verify_node(
                        skill, f"tool_{skill.name}", f"$tool_{skill.name}_result"
                    )
                    nodes.append(verify_node)

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

    def _should_verify(self, skill: Any) -> bool:
        """Check if skill output should be verified.

        Args:
            skill: Skill specification

        Returns:
            True if verification should be added
        """
        if not self.verification_config or not self.verification_config.enabled:
            return False

        # Check skill-level config
        skill_config = getattr(skill, "verification_config", None)
        if skill_config is not None:
            return skill_config.enabled

        # Use global config
        return True

    def _create_verify_node(self, skill: Any, tool_node_id: str, output_var: str) -> IRNode:
        """Create verification node for skill output.

        Args:
            skill: Skill specification
            tool_node_id: ID of the tool node to verify
            output_var: Variable containing tool output

        Returns:
            Verification IR node
        """
        # Determine verification rules based on skill metadata
        rules = []
        skill_meta = getattr(skill, "metadata", {})
        output_type = skill_meta.get("output_type", "unknown")

        if output_type == "sql" or "sql" in skill.name.lower():
            rules.append(
                VerificationRule(
                    rule_id=f"verify_{skill.name}_sql",
                    verifier_type="sql",
                    rule_spec={
                        "check_syntax": True,
                        "check_injection": True,
                    },
                )
            )
        elif output_type == "python" or "python" in skill.name.lower():
            rules.append(
                VerificationRule(
                    rule_id=f"verify_{skill.name}_python",
                    verifier_type="python",
                    rule_spec={
                        "check_syntax": True,
                        "check_imports": True,
                    },
                )
            )

        return IRNode(
            node_id=f"verify_{skill.name}",
            node_type=NodeType.VERIFY,
            verification_rules=rules if rules else None,
            inputs={"output": output_var},
            outputs={"verified": f"$verified_{skill.name}"},
            dependencies=[tool_node_id],
            metadata={
                "skill_name": skill.name,
                "verification_mode": self.verification_config.mode.value
                if self.verification_config
                else "lenient",
            },
        )

    def _plan_direct_skill_execution(
        self, task: str, agent: AgentSpec, plan_id: str
    ) -> ExecutionPlan:
        """Plan direct skill execution without LLM.

        Args:
            task: Task description
            agent: Agent specification
            plan_id: Plan identifier

        Returns:
            ExecutionPlan with direct skill execution
        """
        nodes = []

        # Use first skill (simple heuristic)
        skill = agent.skills[0]

        # Create TOOL node with direct execution
        tool_node = IRNode(
            node_id=f"tool_{skill.name}",
            node_type=NodeType.TOOL,
            skill_ref=skill,
            inputs={"params": {"task": task}},
            outputs={"result": f"$tool_{skill.name}_result"},
            dependencies=[],
            metadata={"skill_name": skill.name, "direct_execution": True},
        )
        nodes.append(tool_node)

        # Insert verification node if enabled
        if self._should_verify(skill):
            verify_node = self._create_verify_node(
                skill, f"tool_{skill.name}", f"$tool_{skill.name}_result"
            )
            nodes.append(verify_node)

        return ExecutionPlan(
            plan_id=plan_id,
            nodes=nodes,
            entry_node=f"tool_{skill.name}",
            metadata={
                "task": task,
                "agent_role": agent.role,
                "num_skills": len(agent.skills),
                "direct_execution": True,
            },
        )
