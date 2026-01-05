"""Local executor with DAG execution support."""

from __future__ import annotations

import asyncio
from typing import Any

from houyi.orchestration.plan import ExecutionPlan, IRNode, NodeType
from houyi.orchestration.state import SessionState


class ExecutionResult:
    """Result of execution."""

    def __init__(
        self,
        success: bool,
        output: Any,
        final_state: SessionState,
        metadata: dict[str, Any] | None = None
    ):
        self.success = success
        self.output = output
        self.final_state = final_state
        self.metadata = metadata or {}


class LocalExecutor:
    """Local executor with DAG execution support.

    Executes ExecutionPlan using topological sort and async concurrency.
    """

    def __init__(self, trace_manager: Any = None):
        self.context: dict[str, Any] = {}
        self.trace_manager = trace_manager

    async def execute(
        self,
        plan: ExecutionPlan,
        initial_state: SessionState
    ) -> ExecutionResult:
        """Execute the plan.

        Args:
            plan: Execution plan (DAG)
            initial_state: Initial session state

        Returns:
            ExecutionResult with output and final state
        """
        # Initialize
        completed_node_ids: set[str] = set()
        self.context = {"task": plan.metadata.get("task", "")}

        # Execute DAG
        while len(completed_node_ids) < len(plan.nodes):
            # Get ready nodes
            ready_nodes = plan.get_ready_nodes(completed_node_ids)

            if not ready_nodes:
                # Check for circular dependency
                if len(completed_node_ids) < len(plan.nodes):
                    raise RuntimeError(
                        f"Circular dependency detected. "
                        f"Completed: {len(completed_node_ids)}/{len(plan.nodes)}"
                    )
                break

            # Execute ready nodes concurrently
            results = await asyncio.gather(*[
                self._execute_node(node, self.context)
                for node in ready_nodes
            ])

            # Update context and mark complete
            for node, result in zip(ready_nodes, results, strict=False):
                # Store result in context
                for output_key, var_name in node.outputs.items():
                    if var_name.startswith("$"):
                        self.context[var_name[1:]] = result.get(output_key)

                completed_node_ids.add(node.node_id)

        # Build final result
        final_output = self.context.get("answer", self.context)

        # Update state with execution results
        updated_state = SessionState(
            session_id=initial_state.session_id,
            agent_id=initial_state.agent_id,
            current_plan_id=plan.plan_id,
            memory_stack=initial_state.memory_stack + [self.context],
            execution_pointer=None,  # Execution complete
            parent_state_id=initial_state.session_id,
            metadata={**initial_state.metadata, "completed": True}
        )

        return ExecutionResult(
            success=True,
            output=final_output,
            final_state=updated_state,
            metadata={
                "nodes_executed": len(completed_node_ids),
                "context": self.context,
            }
        )

    async def _execute_node(
        self,
        node: IRNode,
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a single node.

        Args:
            node: IR node to execute
            context: Execution context

        Returns:
            Node execution result
        """
        # Resolve inputs
        inputs = node.get_input_values(context)

        # Create span if trace manager available
        if self.trace_manager:
            span_name = f"node.{node.node_type.value}"
            with self.trace_manager.start_span(
                span_name,
                attributes={
                    "node.id": node.node_id,
                    "node.type": node.node_type.value,
                }
            ) as span:
                result = await self._execute_node_impl(node, inputs)
                span.set_attribute("node.success", True)
                return result
        else:
            return await self._execute_node_impl(node, inputs)

    async def _execute_node_impl(
        self,
        node: IRNode,
        inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute node implementation.

        Args:
            node: IR node to execute
            inputs: Resolved inputs

        Returns:
            Node execution result
        """
        # Execute based on node type
        if node.node_type == NodeType.LLM:
            return await self._execute_llm_node(node, inputs)
        elif node.node_type == NodeType.TOOL:
            return await self._execute_tool_node(node, inputs)
        elif node.node_type == NodeType.VERIFY:
            return await self._execute_verify_node(node, inputs)
        else:
            raise ValueError(f"Unsupported node type: {node.node_type}")

    async def _execute_llm_node(
        self,
        node: IRNode,
        inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute LLM node.

        Args:
            node: LLM node
            inputs: Resolved inputs

        Returns:
            LLM response
        """
        # Get LLM adapter from node metadata or use default
        use_real_llm = node.metadata.get("use_real_llm", False)
        node.metadata.get("model", "gpt-4")

        if use_real_llm:
            # Use real LLM adapter
            from houyi.llm.base import LLMMessage, MessageRole
            from houyi.llm.openai_adapter import OpenAIAdapter

            try:
                adapter = OpenAIAdapter()

                # Build messages
                task = inputs.get("task", "")
                messages = [
                    LLMMessage(role=MessageRole.USER, content=task)
                ]

                # Call LLM
                response = await adapter.chat(messages)

                return {"answer": response.content}
            except Exception as e:
                print(f"LLM call failed: {e}, using mock response")
                return {
                    "answer": f"Mock LLM response (LLM unavailable): {inputs.get('task', '')}"
                }

        # Mock implementation when LLM is not configured
        task = inputs.get("task", inputs.get("prompt", ""))
        purpose = node.metadata.get("purpose", "reasoning")

        return {
            "answer": f"Mock {purpose} response: {task}"
        }

    async def _execute_tool_node(
        self,
        node: IRNode,
        inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute TOOL node.

        Args:
            node: TOOL node
            inputs: Resolved inputs

        Returns:
            Tool execution result
        """
        if not node.skill_ref:
            raise ValueError(f"TOOL node {node.node_id} has no skill_ref")

        skill = node.skill_ref

        # Use SkillExecutor if skill has executor
        if skill.executor:
            from houyi.execution.skill_executor import SkillExecutor

            executor = SkillExecutor()
            try:
                result = await executor.execute(skill, inputs.get("params", {}))
                return {"result": result}
            except Exception as e:
                # Fallback to placeholder on error
                print(f"Skill execution failed: {e}, using placeholder")

        # Placeholder implementation
        return {
            "result": f"Result from {skill.name}"
        }

    async def _execute_verify_node(
        self,
        node: IRNode,
        inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute VERIFY node.

        Args:
            node: VERIFY node
            inputs: Resolved inputs

        Returns:
            Verification result
        """
        # TODO: Implement assertion verification
        return {
            "verified": True
        }
