"""Skill executor for workflow skill execution with validation and retries."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from houyi.domain.errors import DependencyMissingError
from houyi.domain.skill.exceptions import SkillExecutionError
from houyi.domain.skill.spec import SkillSpec

try:
    from houyi.infrastructure.observability import (
        Span as _Span,
    )
    from houyi.infrastructure.observability import (
        SpanType as _SpanType,
    )
    from houyi.infrastructure.observability import (
        TraceContext as _TraceContext,
    )

    _HAS_OBSERVABILITY = True
except ImportError:
    _HAS_OBSERVABILITY = False

logger = logging.getLogger(__name__)


class SkillExecutor:
    """Executor for skills with validation and error handling."""

    def __init__(
        self,
        max_retries: int = 3,
        timeout: float = 15.0,
    ):
        self.max_retries = max_retries
        self.timeout = timeout
        self._on_retry_span: Callable | None = None

    async def execute(
        self,
        skill: SkillSpec,
        input_data: dict[str, Any],
    ) -> Any:
        """Execute a skill with validation and error handling."""
        if not skill.executor:
            raise SkillExecutionError(skill.name, "Skill has no executor function bound")

        try:
            validated_input = skill.input_schema(**input_data)
        except ValidationError as e:
            raise SkillExecutionError(skill.name, f"Input validation failed: {e}", e) from e

        last_error = None
        for attempt in range(self.max_retries):
            try:
                result = await self._execute_with_timeout(skill.executor, validated_input)
                return self._validate_and_merge_output(skill, result)
            except SkillExecutionError:
                raise
            except Exception as e:
                last_error = e
                logger.warning(
                    "[%s] attempt %d/%d failed: %s: %s",
                    skill.name,
                    attempt + 1,
                    self.max_retries,
                    type(e).__name__,
                    e,
                )
                self._emit_retry_span(skill.name, attempt + 1, self.max_retries, e)
                if isinstance(
                    e,
                    (
                        ImportError,
                        ModuleNotFoundError,
                        TypeError,
                        ValueError,
                        DependencyMissingError,
                    ),
                ):
                    break
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue

        retry_message = f"Execution failed after {self.max_retries} retries"
        if last_error is not None:
            retry_message = f"{retry_message}: {last_error}"
        raise SkillExecutionError(skill.name, retry_message, last_error)

    def _emit_retry_span(
        self, skill_name: str, attempt: int, max_retries: int, exc: Exception
    ) -> None:
        if not _HAS_OBSERVABILITY:
            return
        parent = _TraceContext.current()
        if parent is None:
            return
        span = _Span(
            name=f"retry.{skill_name}",
            span_type=_SpanType.RETRY,
            parent=parent,
            attributes={
                "retry.attempt": attempt,
                "retry.max_retries": max_retries,
                "retry.error": f"{type(exc).__name__}: {exc}",
                "retry.skill": skill_name,
            },
        )
        span.set_status("error", str(exc))
        span.end()
        if self._on_retry_span is not None:
            self._on_retry_span(span)

    def _validate_and_merge_output(self, skill: SkillSpec, result: Any) -> dict[str, Any]:
        try:
            validated_output = skill.output_schema(**result)
        except ValidationError as e:
            raise SkillExecutionError(skill.name, f"Output validation failed: {e}", e) from e

        dumped = validated_output.model_dump()
        if isinstance(result, dict) and isinstance(dumped, dict):
            for key, value in result.items():
                if key not in dumped:
                    dumped[key] = value
        return dumped

    @staticmethod
    def _build_executor_kwargs(
        executor: Callable[..., Any], input_data: Any
    ) -> tuple[dict[str, Any], bool]:
        if not hasattr(input_data, "model_dump"):
            return {}, False
        dumped = input_data.model_dump()
        if not isinstance(dumped, dict):
            return {}, False
        try:
            signature = inspect.signature(executor)
        except (TypeError, ValueError):
            return dumped, True

        params = [
            p
            for p in signature.parameters.values()
            if p.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        ]
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
        )
        if accepts_kwargs:
            return dumped, True
        allowed = {p.name for p in params}
        return {k: v for k, v in dumped.items() if k in allowed}, True

    @staticmethod
    def _should_pass_model(executor: Callable[..., Any], input_data: Any) -> bool:
        if not isinstance(input_data, BaseModel):
            return True
        try:
            signature = inspect.signature(executor)
        except (TypeError, ValueError):
            return True
        params = list(signature.parameters.values())
        if len(params) != 1:
            return False
        annotation = params[0].annotation
        if annotation is inspect._empty:
            return False
        try:
            return isinstance(annotation, type) and issubclass(annotation, BaseModel)
        except TypeError:
            return False

    def _call_executor_sync(self, executor: Callable[..., Any], input_data: Any) -> Any:
        if isinstance(input_data, BaseModel) and not self._should_pass_model(executor, input_data):
            kwargs, ok = self._build_executor_kwargs(executor, input_data)
            if ok and kwargs:
                return executor(**kwargs)
        return executor(input_data)

    async def _call_executor_async(self, executor: Callable[..., Any], input_data: Any) -> Any:
        if isinstance(input_data, BaseModel) and not self._should_pass_model(executor, input_data):
            kwargs, ok = self._build_executor_kwargs(executor, input_data)
            if ok and kwargs:
                return await executor(**kwargs)
        return await executor(input_data)

    async def _execute_with_timeout(
        self,
        executor: callable,
        input_data: Any,
    ) -> dict[str, Any]:
        if asyncio.iscoroutinefunction(executor):
            result = await asyncio.wait_for(
                self._call_executor_async(executor, input_data),
                timeout=self.timeout,
            )
        else:
            loop = asyncio.get_running_loop()
            call = functools.partial(self._call_executor_sync, executor, input_data)
            result = await asyncio.wait_for(loop.run_in_executor(None, call), timeout=self.timeout)

        if hasattr(result, "model_dump"):
            return result.model_dump()
        if isinstance(result, dict):
            return result
        return {"result": result}


__all__ = ["SkillExecutionError", "SkillExecutor"]
