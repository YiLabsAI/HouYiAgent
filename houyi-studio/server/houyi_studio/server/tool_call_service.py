"""Tool-call service for console execution engine."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from houyi.core.skill_registry import DEFAULT_SKILL_REGISTRY, SkillRegistry
from houyi.core.tool_call_adapter import normalize_adapter_error
from houyi.core.tool_call_adapter_registry import ToolCallAdapterRegistry, ToolCallAdapterRequest
from houyi.execution.tool_call_messages import build_fast_path_prompt, build_tool_call_messages
from houyi.execution.tool_call_orchestrator import (
    build_chat_kwargs,
    choose_tool_cache,
    wrap_tool_choice,
)
from houyi.execution.tool_call_runner_service import ToolCallRunnerService
from houyi.execution.tool_call_web_search_hooks import build_web_search_tool_hooks
from houyi.protocol.ir import ExecutionIR, NodeExecutionIR
from houyi.protocol.ir.tooling_ir import LLMToolCallOutputIR

from .tool_call_response import ConsoleToolCallResponseAssembler, ToolCallContext

logger = logging.getLogger(__name__)


class ToolCallService:
    """Encapsulates tool-calling execution with internal-first policy."""

    def __init__(
        self,
        connection_manager: Any,
        record_llm_call: Callable[..., None],
        tool_call_cache: dict[str, dict[str, Any]],
        llm_tool_call_cache: dict[str, Any],
        skill_registry: SkillRegistry | None = None,
    ) -> None:
        self.connection_manager = connection_manager
        self.record_llm_call = record_llm_call
        self.tool_call_cache = tool_call_cache
        self.llm_tool_call_cache = llm_tool_call_cache
        self.skill_registry = skill_registry or DEFAULT_SKILL_REGISTRY
        self.tool_call_runner = ToolCallRunnerService()

    async def execute_tool_calls(
        self,
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
        from houyi.execution.skill_executor import SkillExecutor
        from houyi.llm.openai_adapter import OpenAIAdapter

        skills = self.tool_call_runner.select_skills(tool_names)
        if not skills:
            logger.warning("[%s] Tool-calling enabled but no tools are registered", node_id)
            return False

        tools = [skill.to_tool_schema() for skill in skills]
        api_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("SILICONFLOW_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        tool_model = (
            os.getenv("HOUYI_TOOLCALL_MODEL")
            or model
            or os.getenv("DEEPSEEK_MODEL")
            or "deepseek-ai/DeepSeek-V3"
        )
        toolcall_adapter = (os.getenv("HOUYI_TOOLCALL_ADAPTER") or "real").strip().lower()
        toolcall_max_tokens = os.getenv("HOUYI_TOOLCALL_MAX_TOKENS")
        if toolcall_max_tokens:
            try:
                max_tokens = max(1, int(toolcall_max_tokens))
            except ValueError:
                logger.warning("Invalid HOUYI_TOOLCALL_MAX_TOKENS=%s", toolcall_max_tokens)

        fast_path_flag = (os.getenv("HOUYI_TOOLCALL_FAST_PATH") or "").strip().lower()
        fast_path_enabled = fast_path_flag in {"1", "true", "yes", "on"}
        fast_path_prompt = build_fast_path_prompt(tool_names=tool_names, skills=skills)

        message_result = build_tool_call_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt=prompt,
            fast_path_enabled=fast_path_enabled,
            fast_path_prompt=fast_path_prompt,
            tool_choice=tool_choice,
        )
        messages = message_result.messages
        user_content = message_result.user_content
        tool_choice = message_result.tool_choice

        run_settings = (
            execution.metadata.get("run_settings", {})
            if isinstance(execution.metadata, dict)
            else {}
        )
        web_search_provider = None
        if isinstance(run_settings, dict):
            web_search_provider = run_settings.get("web_search_provider")

        replay_mode = (
            execution.metadata.get("replay_mode") if isinstance(execution.metadata, dict) else None
        )
        allow_fresh_web_search_cache = (
            os.getenv("HOUYI_FRESH_REPLAY_USE_WEB_SEARCH_CACHE") or ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        tool_hooks = build_web_search_tool_hooks(
            web_search_provider=web_search_provider
            if isinstance(web_search_provider, str)
            else None,
            replay_mode=replay_mode if isinstance(replay_mode, str) else None,
            allow_fresh_web_search_cache=allow_fresh_web_search_cache,
        )

        def _set_tool_call_error(error: str) -> None:
            tool_call_rounds = len([message for message in messages if message.get("tool_calls")])
            output_payload = LLMToolCallOutputIR(
                content="",
                tool_calls=[],
                finish_reason=None,
                tool_finish_reason=None,
                tool_call_rounds=tool_call_rounds,
                max_rounds_reached=False,
                tool_errors=[],
                messages=messages,
                error=error,
            )
            node_exec.outputs = output_payload.model_dump(by_alias=True)

        adapter_registry = ToolCallAdapterRegistry()

        def _build_openai_adapter() -> Any:
            if not api_key:
                raise ValueError("Tool-calling requires SILICONFLOW_API_KEY or OPENAI_API_KEY")
            return OpenAIAdapter(api_key=api_key, model=tool_model, base_url=base_url)

        adapter = None
        base_adapter = None
        try:
            if toolcall_adapter != "real":
                tool_sequence = tool_names or [skill.name for skill in skills]
                request = ToolCallAdapterRequest(
                    adapter_name=toolcall_adapter,
                    tool_names=tool_names or [],
                    skills=skills,
                    tool_sequence=tool_sequence,
                    parallel_tool_calls=parallel_tool_calls,
                    now=datetime.now(timezone.utc),
                )
                adapter = adapter_registry.resolve(
                    request,
                    fallback_factory=_build_openai_adapter,
                )
                base_adapter = getattr(adapter, "inner", adapter)
            else:
                adapter = adapter_registry.resolve(
                    ToolCallAdapterRequest(
                        adapter_name="real",
                        tool_names=tool_names or [],
                        skills=skills,
                        tool_sequence=tool_names or [skill.name for skill in skills],
                        parallel_tool_calls=parallel_tool_calls,
                        now=datetime.now(timezone.utc),
                    ),
                    fallback_factory=_build_openai_adapter,
                )
                base_adapter = getattr(adapter, "inner", adapter)
        except Exception as exc:
            normalized_error = normalize_adapter_error(exc)
            logger.warning(
                "[%s] Tool-calling adapter init failed (%s): %s",
                node_id,
                toolcall_adapter,
                normalized_error.message,
            )
            node_exec.error = (
                "Tool-calling adapter init failed: "
                f"{normalized_error.error_type}: {normalized_error.message}"
            )
            _set_tool_call_error(node_exec.error)
            return True

        adapter = wrap_tool_choice(adapter=adapter, tool_choice=tool_choice)

        message_preview = [
            {
                "role": message.get("role"),
                "content": (message.get("content") or "")[:200],
            }
            for message in messages
        ]
        logger.info(
            "[%s] Tool-calling request: model=%s tool_choice=%s temperature=%s parallel_tool_calls=%s messages=%s",
            node_id,
            tool_model,
            tool_choice,
            temperature,
            parallel_tool_calls,
            message_preview,
        )

        chat_kwargs = build_chat_kwargs(
            max_tokens=max_tokens,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            prompt_cache_key=prompt_cache_key,
        )

        max_retries = 3
        retry_env = os.getenv("HOUYI_TOOLCALL_MAX_RETRIES")
        if retry_env:
            try:
                max_retries = max(1, int(retry_env))
            except ValueError:
                logger.warning("Invalid HOUYI_TOOLCALL_MAX_RETRIES=%s", retry_env)
        timeout = 30.0
        timeout_env = os.getenv("HOUYI_TOOLCALL_TIMEOUT")
        if timeout_env:
            try:
                timeout = float(timeout_env)
            except ValueError:
                logger.warning("Invalid HOUYI_TOOLCALL_TIMEOUT=%s", timeout_env)

        executor = SkillExecutor(max_retries=max_retries, timeout=timeout)

        allow_fresh_tool_cache = (
            os.getenv("HOUYI_FRESH_REPLAY_USE_TOOL_CACHE") or ""
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        tool_cache = choose_tool_cache(
            execution=execution,
            tool_cache=self.tool_call_cache,
            allow_fresh_tool_cache=allow_fresh_tool_cache,
        )
        try:
            response, tool_trace = await self.tool_call_runner.run_tool_calls(
                adapter=adapter,
                messages=messages,
                tools=tools,
                skills=skills,
                executor=executor,
                max_rounds=max_tool_calls,
                chat_kwargs=chat_kwargs,
                tool_hooks=tool_hooks,
                allow_tool_replace=False,
                tool_cache=tool_cache,
                llm_cache=self.llm_tool_call_cache,
            )
        except Exception as exc:
            node_exec.error = str(exc)
            _set_tool_call_error(node_exec.error)
            return True

        assembler = ConsoleToolCallResponseAssembler(
            connection_manager=self.connection_manager,
            record_llm_call=self.record_llm_call,
        )
        context = ToolCallContext(
            session_id=session_id,
            execution=execution,
            node_id=node_id,
            node_exec=node_exec,
            messages=messages,
            response=response,
            tool_trace=tool_trace,
            base_adapter=base_adapter,
            tool_model=tool_model,
            prompt=prompt,
            user_content=user_content,
            max_tool_calls=max_tool_calls,
            skills=skills,
            final_chat_kwargs=chat_kwargs,
            prompt_cache_key=prompt_cache_key,
            llm_cache=self.llm_tool_call_cache,
            created_at=datetime.now(timezone.utc),
        )
        await assembler.assemble(context)

        return True


__all__ = [
    "ToolCallService",
]
