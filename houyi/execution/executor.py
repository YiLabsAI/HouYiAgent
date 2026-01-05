"""Simplified executor for MVP."""

from __future__ import annotations

from typing import Any

from houyi.core.agent import AgentSpec


class SimplifiedExecutor:
    """Simplified executor (MVP implementation).
    
    This is a placeholder executor that doesn't implement full DAG execution.
    It will be replaced with LocalExecutor in Phase 2.
    """
    
    def execute(
        self,
        task_description: str,
        agent_spec: AgentSpec,
        expected_output: str | None = None
    ) -> Any:
        """Execute a task (simplified).
        
        Args:
            task_description: Task description
            agent_spec: Agent specification
            expected_output: Expected output format
            
        Returns:
            Execution result (placeholder)
        """
        # TODO: Implement actual LLM call and skill execution
        # For now, return placeholder result
        
        system_prompt = agent_spec.to_system_prompt()
        tool_schemas = agent_spec.get_tool_schemas()
        
        return {
            "task": task_description,
            "expected_output": expected_output,
            "system_prompt": system_prompt,
            "available_tools": [schema["function"]["name"] for schema in tool_schemas],
            "result": f"Placeholder result for: {task_description}",
            "note": "SimplifiedExecutor - actual LLM execution not yet implemented"
        }
