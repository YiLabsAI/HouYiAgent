"""Local executor with DAG execution support."""

from __future__ import annotations

import asyncio
from typing import Any

from houyi.core.skill import SkillSpec
from houyi.orchestration.plan import ExecutionPlan, IRNode, NodeType
from houyi.orchestration.state import SessionState


class ExecutionResult:
    """Result of execution."""

    def __init__(
        self,
        success: bool,
        output: Any,
        final_state: SessionState,
        metadata: dict[str, Any] | None = None,
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

    async def execute(self, plan: ExecutionPlan, initial_state: SessionState) -> ExecutionResult:
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
            results = await asyncio.gather(
                *[self._execute_node(node, self.context) for node in ready_nodes]
            )

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
            metadata={**initial_state.metadata, "completed": True},
        )

        return ExecutionResult(
            success=True,
            output=final_output,
            final_state=updated_state,
            metadata={
                "nodes_executed": len(completed_node_ids),
                "context": self.context,
            },
        )

    async def _execute_node(self, node: IRNode, context: dict[str, Any]) -> dict[str, Any]:
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
                },
            ) as span:
                result = await self._execute_node_impl(node, inputs)
                span.set_attribute("node.success", True)
                return result
        else:
            return await self._execute_node_impl(node, inputs)

    async def _execute_node_impl(self, node: IRNode, inputs: dict[str, Any]) -> dict[str, Any]:
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

    async def _execute_llm_node(self, node: IRNode, inputs: dict[str, Any]) -> dict[str, Any]:
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
                messages = [LLMMessage(role=MessageRole.USER, content=task)]

                # Call LLM
                response = await adapter.chat(messages)  # type: ignore[arg-type]

                return {"answer": response.content}
            except Exception as e:
                print(f"LLM call failed: {e}, using mock response")
                return {"answer": f"Mock LLM response (LLM unavailable): {inputs.get('task', '')}"}

        # Mock implementation when LLM is not configured
        task = inputs.get("task", inputs.get("prompt", ""))
        purpose = node.metadata.get("purpose", "reasoning")

        return {"answer": f"Mock {purpose} response: {task}"}

    async def _execute_tool_node(self, node: IRNode, inputs: dict[str, Any]) -> dict[str, Any]:
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

            # Check if this is direct execution mode (no LLM)
            is_direct_mode = node.metadata.get("direct_execution", False)

            if is_direct_mode:
                # Extract parameters from task string using skill schema
                params = self._extract_params_from_task(inputs.get("task", ""), skill)
            else:
                # Use params from LLM decision
                params = inputs.get("params", {})

            try:
                result = await executor.execute(skill, params)
                return {"result": result}
            except Exception as e:
                # Fallback to placeholder on error
                print(f"Skill execution failed: {e}, using placeholder")

        # Placeholder implementation
        return {"result": f"Result from {skill.name}"}

    def _extract_params_from_task(self, task: str, skill: SkillSpec) -> dict[str, Any]:
        """Extract skill parameters from task string.

        Simple heuristic-based parameter extraction for fallback mode.

        Args:
            task: Task description string
            skill: Skill specification

        Returns:
            Dictionary of extracted parameters
        """
        import inspect
        from typing import get_type_hints

        params = {}

        # Get skill input schema fields
        if hasattr(skill.input_schema, "model_fields"):
            fields = skill.input_schema.model_fields

            # Simple heuristic: use task as first string parameter
            for field_name, field_info in fields.items():
                field_type = field_info.annotation

                # Handle string parameters
                if field_type is str or str(field_type) == "<class 'str'>":
                    params[field_name] = task
                    break
                # Handle list parameters
                if hasattr(field_type, "__origin__") and field_type.__origin__ is list:  # type: ignore[union-attr]
                    params[field_name] = [task]  # type: ignore[assignment]
                    break

        # If no params extracted, try using original function signature
        if not params and hasattr(skill, "_original_func"):
            sig = inspect.signature(skill._original_func)
            hints = get_type_hints(skill._original_func)

            for param_name, _param in sig.parameters.items():
                if param_name == "self":
                    continue

                param_type = hints.get(param_name, str)

                if param_type is str:
                    params[param_name] = task
                    break
                if hasattr(param_type, "__origin__") and param_type.__origin__ is list:
                    params[param_name] = [task]  # type: ignore[assignment]
                    break

        return params

    async def _execute_verify_node(self, node: IRNode, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute VERIFY node.

        Args:
            node: VERIFY node
            inputs: Resolved inputs

        Returns:
            Verification result
        """
        from houyi.verification import ConstraintChecker, PythonVerifier, SQLVerifier

        if not node.verification_rules:
            return {"verified": True}

        # Get output to verify
        output = inputs.get("output")
        if output is None:
            return {"verified": False, "error": "No output to verify"}

        # Run all verification rules
        all_passed = True
        errors = []

        for rule in node.verification_rules:
            # Select verifier based on type
            verifier: SQLVerifier | PythonVerifier | ConstraintChecker | None = None
            if rule.verifier_type == "sql":
                verifier = SQLVerifier()
            elif rule.verifier_type == "python":
                verifier = PythonVerifier()
            elif rule.verifier_type == "constraint":
                verifier = ConstraintChecker()

            if verifier:
                result = await verifier.verify(output, rule)
                if not result.passed:
                    all_passed = False
                    errors.append(
                        {
                            "rule_id": result.rule_id,
                            "error_type": result.error_type,
                            "error_message": result.error_message,
                        }
                    )

        return {
            "verified": all_passed,
            "errors": errors if not all_passed else None,
        }
