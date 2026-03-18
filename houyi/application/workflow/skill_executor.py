"""Skill executor for workflow skill execution with validation and retries."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
import shlex
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from houyi.domain.errors import DependencyMissingError
from houyi.domain.skill.exceptions import SkillExecutionError
from houyi.domain.skill.spec import SkillSpec
from houyi.infrastructure.config.env_config import ENV_SHELL_CURL_TIMEOUT

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
_DEFAULT_SHELL_CURL_TIMEOUT = 3.0


def _summarize_retry_input(input_data: Any) -> str:
    if hasattr(input_data, "model_dump"):
        try:
            dumped = input_data.model_dump()
        except Exception:
            dumped = None
        if isinstance(dumped, dict):
            command = dumped.get("command")
            cwd = dumped.get("cwd")
            parts: list[str] = []
            if isinstance(command, str) and command.strip():
                compact = " ".join(command.strip().split())
                parts.append(f"command={compact[:160]}")
            if isinstance(cwd, str) and cwd.strip():
                parts.append(f"cwd={cwd}")
            if parts:
                return " (" + ", ".join(parts) + ")"
    return ""


def _extract_retry_input_fields(input_data: Any) -> dict[str, Any]:
    if not hasattr(input_data, "model_dump"):
        return {}
    try:
        dumped = input_data.model_dump()
    except Exception:
        return {}
    if not isinstance(dumped, dict):
        return {}
    fields: dict[str, Any] = {}
    command = dumped.get("command")
    cwd = dumped.get("cwd")
    timeout_seconds = dumped.get("timeout_seconds")
    if isinstance(command, str) and command.strip():
        fields["retry.command"] = " ".join(command.strip().split())[:240]
    if isinstance(cwd, str) and cwd.strip():
        fields["retry.cwd"] = cwd
    if isinstance(timeout_seconds, int | float):
        fields["retry.tool_timeout_seconds"] = float(timeout_seconds)
    return fields


def _resolve_executor_timeout(default_timeout: float, input_data: Any) -> float:
    command = _extract_command(input_data)
    if not _is_curl_command(command):
        return default_timeout
    configured = os.getenv(ENV_SHELL_CURL_TIMEOUT, "").strip()
    if configured:
        try:
            value = float(configured)
            if value > 0:
                return min(default_timeout, value)
        except ValueError:
            return default_timeout
    command_timeout = _extract_curl_max_time_seconds(command)
    input_timeout = _extract_input_timeout_seconds(input_data)
    effective_timeout = command_timeout or input_timeout or default_timeout
    if command_timeout and input_timeout:
        effective_timeout = min(command_timeout, input_timeout)
    if effective_timeout <= 0:
        return min(default_timeout, _DEFAULT_SHELL_CURL_TIMEOUT)
    return min(default_timeout, effective_timeout + 1.0)


def _extract_command(input_data: Any) -> str | None:
    if not hasattr(input_data, "model_dump"):
        return None
    try:
        dumped = input_data.model_dump()
    except Exception:
        return None
    if not isinstance(dumped, dict):
        return None
    command = dumped.get("command")
    return command if isinstance(command, str) and command.strip() else None


def _extract_input_timeout_seconds(input_data: Any) -> float | None:
    if not hasattr(input_data, "model_dump"):
        return None
    try:
        dumped = input_data.model_dump()
    except Exception:
        return None
    if not isinstance(dumped, dict):
        return None
    timeout_seconds = dumped.get("timeout_seconds")
    if isinstance(timeout_seconds, int | float) and float(timeout_seconds) > 0:
        return float(timeout_seconds)
    return None


def _extract_curl_max_time_seconds(command: str | None) -> float | None:
    if not _is_curl_command(command):
        return None
    try:
        parts = shlex.split(str(command or ""))
    except ValueError:
        parts = str(command or "").split()
    for index, part in enumerate(parts):
        if part in {"--max-time", "-m"} and index + 1 < len(parts):
            try:
                value = float(parts[index + 1])
            except ValueError:
                return None
            return value if value > 0 else None
        if part.startswith("--max-time="):
            try:
                value = float(part.split("=", 1)[1])
            except ValueError:
                return None
            return value if value > 0 else None
    return None


def _is_curl_command(command: str | None) -> bool:
    if not isinstance(command, str) or not command.strip():
        return False
    with_tokens = command.strip()
    with_tokens = with_tokens.removeprefix("timeout ").strip()
    try:
        parts = shlex.split(with_tokens)
    except ValueError:
        parts = with_tokens.split()
    return bool(parts) and parts[0] == "curl"


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

    @staticmethod
    def _effective_timeout_for_input(input_data: Any, default_timeout: float) -> float:
        return _resolve_executor_timeout(default_timeout, input_data)

    @staticmethod
    def _should_retry_after_timeout(skill: SkillSpec, validated_input: Any) -> bool:
        if skill.name != "houyi_shell_exec":
            return True
        command = _extract_command(validated_input)
        return not _is_curl_command(command)

    async def execute(
        self,
        skill: SkillSpec,
        input_data: dict[str, Any],
    ) -> Any:
        """Execute a skill with validation and error handling."""
        executor = skill.executor
        if not executor:
            raise SkillExecutionError(skill.name, "Skill has no executor function bound")

        validated_input = self._validate_input(skill, input_data)
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            attempt_number = attempt + 1
            try:
                return await self._execute_attempt(skill, executor, validated_input)
            except SkillExecutionError:
                raise
            except TimeoutError as exc:
                last_error = exc
                if await self._handle_timeout(skill, validated_input, attempt_number, exc):
                    continue
                break
            except Exception as exc:
                last_error = exc
                if await self._handle_execution_error(skill, validated_input, attempt_number, exc):
                    continue
                break

        raise SkillExecutionError(
            skill.name,
            self._build_retry_message(last_error),
            last_error,
        )

    def _validate_input(self, skill: SkillSpec, input_data: dict[str, Any]) -> Any:
        try:
            return skill.input_schema(**input_data)
        except ValidationError as exc:
            raise SkillExecutionError(
                skill.name,
                f"Input validation failed: {exc}",
                exc,
            ) from exc

    async def _execute_attempt(
        self,
        skill: SkillSpec,
        executor: Callable[..., Any],
        validated_input: Any,
    ) -> dict[str, Any]:
        result = await self._execute_with_timeout(executor, validated_input)
        return self._validate_and_merge_output(skill, result)

    async def _handle_timeout(
        self,
        skill: SkillSpec,
        validated_input: Any,
        attempt_number: int,
        exc: TimeoutError,
    ) -> bool:
        effective_timeout = self._effective_timeout_for_input(validated_input, self.timeout)
        logger.warning(
            "[%s] attempt %d/%d executor timeout after %.2fs%s",
            skill.name,
            attempt_number,
            self.max_retries,
            effective_timeout,
            _summarize_retry_input(validated_input),
        )
        self._emit_retry_span(
            skill.name,
            attempt_number,
            self.max_retries,
            exc,
            extra_attributes={
                "retry.timeout_seconds": effective_timeout,
                **_extract_retry_input_fields(validated_input),
            },
        )
        if not self._should_retry_after_timeout(skill, validated_input):
            return False
        return await self._sleep_before_retry(attempt_number)

    async def _handle_execution_error(
        self,
        skill: SkillSpec,
        validated_input: Any,
        attempt_number: int,
        exc: Exception,
    ) -> bool:
        logger.warning(
            "[%s] attempt %d/%d failed%s: %s: %s",
            skill.name,
            attempt_number,
            self.max_retries,
            _summarize_retry_input(validated_input),
            type(exc).__name__,
            exc,
        )
        self._emit_retry_span(skill.name, attempt_number, self.max_retries, exc)
        if isinstance(
            exc,
            (
                ImportError,
                ModuleNotFoundError,
                TypeError,
                ValueError,
                DependencyMissingError,
            ),
        ):
            return False
        return await self._sleep_before_retry(attempt_number)

    async def _sleep_before_retry(self, attempt_number: int) -> bool:
        if attempt_number >= self.max_retries:
            return False
        await asyncio.sleep(2 ** (attempt_number - 1))
        return True

    def _build_retry_message(self, last_error: Exception | None) -> str:
        retry_message = f"Execution failed after {self.max_retries} retries"
        if last_error is not None:
            retry_message = f"{retry_message}: {last_error}"
        return retry_message

    def _emit_retry_span(
        self,
        skill_name: str,
        attempt: int,
        max_retries: int,
        exc: Exception,
        *,
        extra_attributes: dict[str, Any] | None = None,
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
                **(extra_attributes or {}),
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
        timeout = self._effective_timeout_for_input(input_data, self.timeout)
        if asyncio.iscoroutinefunction(executor):
            result = await asyncio.wait_for(
                self._call_executor_async(executor, input_data),
                timeout=timeout,
            )
        else:
            loop = asyncio.get_running_loop()
            call = functools.partial(self._call_executor_sync, executor, input_data)
            result = await asyncio.wait_for(loop.run_in_executor(None, call), timeout=timeout)

        if hasattr(result, "model_dump"):
            return result.model_dump()
        if isinstance(result, dict):
            return result
        return {"result": result}


__all__ = ["SkillExecutionError", "SkillExecutor"]
