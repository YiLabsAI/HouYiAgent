"""Skill executor for executing skills with validation and error handling."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from houyi.core.errors import DependencyMissingError
from houyi.core.skill import SkillSpec

try:
    from houyi.observability.context import TraceContext as _TraceContext
    from houyi.observability.trace_manager import Span as _Span
    from houyi.observability.types import SpanType as _SpanType

    _HAS_OBSERVABILITY = True
except ImportError:
    _HAS_OBSERVABILITY = False

logger = logging.getLogger(__name__)


class SkillExecutionError(Exception):
    """Error during skill execution."""

    def __init__(self, skill_name: str, message: str, original_error: Exception | None = None):
        self.skill_name = skill_name
        self.message = message
        self.original_error = original_error
        super().__init__(f"Skill '{skill_name}' execution failed: {message}")


class SkillExecutor:
    """Executor for skills with validation and error handling.

    Handles:
    - Input validation
    - Skill execution (sync or async)
    - Output validation
    - Error handling and retries
    - Timeout control
    """

    def __init__(
        self,
        max_retries: int = 3,
        timeout: float = 15.0,
    ):
        """Initialize skill executor.

        Args:
            max_retries: Maximum number of retries on failure
            timeout: Execution timeout in seconds
        """
        self.max_retries = max_retries
        self.timeout = timeout
        self._on_retry_span: Callable | None = None

    async def execute(
        self,
        skill: SkillSpec,
        input_data: dict[str, Any],
    ) -> Any:
        """Execute a skill with validation and error handling.

        Args:
            skill: Skill specification
            input_data: Input data (will be validated against input_schema)

        Returns:
            Skill execution result (validated against output_schema)

        Raises:
            SkillExecutionError: If execution fails after retries
        """
        if not skill.executor:
            raise SkillExecutionError(skill.name, "Skill has no executor function bound")

        # Validate input
        try:
            validated_input = skill.input_schema(**input_data)
        except ValidationError as e:
            raise SkillExecutionError(skill.name, f"Input validation failed: {e}", e) from e

        # Execute with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Execute skill (with timeout)
                result = await self._execute_with_timeout(skill.executor, validated_input)

                return self._validate_and_merge_output(skill, result)

            except SkillExecutionError:
                # Re-raise validation errors immediately
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
                # Don't retry deterministic errors
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
                    # Wait before retry (exponential backoff)
                    await asyncio.sleep(2**attempt)
                    continue

        # All retries failed
        raise SkillExecutionError(
            skill.name, f"Execution failed after {self.max_retries} retries", last_error
        )

    def _emit_retry_span(
        self, skill_name: str, attempt: int, max_retries: int, exc: Exception
    ) -> None:
        """Create a RETRY span for a failed attempt (observability)."""
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

    async def _execute_with_timeout(
        self,
        executor: callable,
        input_data: Any,
    ) -> dict[str, Any]:
        """Execute skill function with timeout.

        Args:
            executor: Skill executor function
            input_data: Validated input data

        Returns:
            Execution result

        Raises:
            asyncio.TimeoutError: If execution exceeds timeout
        """

        def _build_kwargs() -> tuple[dict[str, Any], bool]:
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

        def _should_pass_model() -> bool:
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

        def _call_executor_sync() -> Any:
            if isinstance(input_data, BaseModel) and not _should_pass_model():
                kwargs, ok = _build_kwargs()
                if ok and kwargs:
                    return executor(**kwargs)  # type: ignore[misc]
            return executor(input_data)  # type: ignore[misc]

        async def _call_executor_async() -> Any:
            if isinstance(input_data, BaseModel) and not _should_pass_model():
                kwargs, ok = _build_kwargs()
                if ok and kwargs:
                    return await executor(**kwargs)  # type: ignore[misc]
            return await executor(input_data)  # type: ignore[misc]

        # Check if executor is async
        if asyncio.iscoroutinefunction(executor):
            # Async executor
            result = await asyncio.wait_for(_call_executor_async(), timeout=self.timeout)
        else:
            # Sync executor - run in thread pool
            loop = asyncio.get_running_loop()
            call = functools.partial(_call_executor_sync)
            result = await asyncio.wait_for(loop.run_in_executor(None, call), timeout=self.timeout)

        # Ensure result is a dict
        if hasattr(result, "model_dump"):
            # Pydantic model
            return result.model_dump()
        elif isinstance(result, dict):
            return result
        else:
            # Wrap non-dict results
            return {"result": result}
