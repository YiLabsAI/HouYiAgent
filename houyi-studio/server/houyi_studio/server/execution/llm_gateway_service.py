"""Gateway service for LLM execution helpers."""

from __future__ import annotations

from typing import Any

from houyi.protocol.ir import ExecutionIR, NodeExecutionIR

from .llm_execution_flow import LLMExecutionFlow


class LLMGatewayService:
    """Service for delegating LLM execution helpers."""

    def __init__(self, llm_execution_flow: LLMExecutionFlow) -> None:
        self._llm_execution_flow = llm_execution_flow

    def record_llm_call(
        self,
        *,
        execution_id: str,
        node_id: str,
        model: str,
        prompt: str | list[dict[str, Any]],
        response: str,
        metadata: dict | None = None,
    ) -> None:
        """Record an LLM call for deterministic replay."""
        self._llm_execution_flow.record_llm_call(
            execution_id=execution_id,
            node_id=node_id,
            model=model,
            prompt=prompt,
            response=response,
            metadata=metadata,
        )

    def get_recorded_llm_response(self, execution_id: str, node_id: str) -> str | None:
        """Get recorded LLM response for deterministic replay."""
        return self._llm_execution_flow.get_recorded_llm_response(execution_id, node_id)

    async def execute_llm_real(
        self,
        *,
        session_id: str,
        execution: ExecutionIR,
        node_id: str,
        node_exec: NodeExecutionIR,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        enable_reasoning: bool = False,
        thinking_budget: int | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        enable_tool_calls: bool = False,
        tool_names: list[str] | None = None,
        tool_choice: Any | None = None,
        max_tool_calls: int = 6,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        prompt_cache_key: str | None = None,
    ) -> None:
        await self._llm_execution_flow.execute_llm_real(
            session_id=session_id,
            execution=execution,
            node_id=node_id,
            node_exec=node_exec,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            enable_reasoning=enable_reasoning,
            thinking_budget=thinking_budget,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            enable_tool_calls=enable_tool_calls,
            tool_names=tool_names,
            tool_choice=tool_choice,
            max_tool_calls=max_tool_calls,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            prompt_cache_key=prompt_cache_key,
        )

    async def execute_llm_tool_calls(
        self,
        *,
        session_id: str,
        execution: ExecutionIR,
        node_id: str,
        node_exec: NodeExecutionIR,
        prompt: str,
        system_prompt: str | None,
        user_prompt: str | None,
        model: str | None,
        tool_names: list[str],
        tool_choice: Any | None,
        max_tool_calls: int,
        max_tokens: int | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        prompt_cache_key: str | None = None,
    ) -> bool:
        return await self._llm_execution_flow.execute_llm_tool_calls(
            session_id=session_id,
            execution=execution,
            node_id=node_id,
            node_exec=node_exec,
            prompt=prompt,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            tool_names=tool_names,
            tool_choice=tool_choice,
            max_tool_calls=max_tool_calls,
            max_tokens=max_tokens,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            prompt_cache_key=prompt_cache_key,
        )

    async def execute_llm_mock(
        self,
        *,
        session_id: str,
        execution: ExecutionIR,
        node_id: str,
        node_exec: NodeExecutionIR,
    ) -> None:
        await self._llm_execution_flow.execute_llm_mock(
            session_id=session_id,
            execution=execution,
            node_id=node_id,
            node_exec=node_exec,
        )
