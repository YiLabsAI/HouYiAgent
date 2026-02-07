"""Tests for SkillExecutor - core skill execution engine."""

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from houyi.core.skill import SkillSpec
from houyi.execution.skill_executor import SkillExecutionError, SkillExecutor
from houyi.observability.context import TraceContext
from houyi.observability.trace_manager import Span
from houyi.observability.types import SpanType


class TestSkillExecutor:
    """Test SkillExecutor core functionality."""

    @pytest.mark.asyncio
    async def test_execute_sync_skill(self) -> None:
        """Test executing a synchronous skill."""

        class Input(BaseModel):
            x: int
            y: int

        class Output(BaseModel):
            result: int

        def add(input_data: Input) -> Output:
            return Output(result=input_data.x + input_data.y)

        skill = SkillSpec(
            name="add",
            description="Add two numbers",
            input_schema=Input,
            output_schema=Output,
            executor=add,
        )

        executor = SkillExecutor()
        result = await executor.execute(skill, {"x": 3, "y": 5})

        assert result["result"] == 8

    @pytest.mark.asyncio
    async def test_execute_kwargs_executor(self) -> None:
        class Input(BaseModel):
            a: int
            b: int

        class Output(BaseModel):
            total: int

        def add(a: int, b: int) -> Output:
            return Output(total=a + b)

        skill = SkillSpec(
            name="add_kwargs",
            description="Add two numbers with kwargs",
            input_schema=Input,
            output_schema=Output,
            executor=add,
        )

        executor = SkillExecutor()
        result = await executor.execute(skill, {"a": 2, "b": 4})
        assert result["total"] == 6

    @pytest.mark.asyncio
    async def test_execute_preserves_extra_output_fields(self) -> None:
        class Input(BaseModel):
            x: int
            y: int

        class Output(BaseModel):
            total: int

        def add_with_metadata(input_data: Input) -> dict[str, Any]:
            return {
                "total": input_data.x + input_data.y,
                "metadata": {"source": "unit_test"},
            }

        skill = SkillSpec(
            name="add_with_metadata",
            description="Add numbers and attach metadata",
            input_schema=Input,
            output_schema=Output,
            executor=add_with_metadata,
        )

        executor = SkillExecutor()
        result = await executor.execute(skill, {"x": 1, "y": 2})

        assert result["total"] == 3
        assert result["metadata"] == {"source": "unit_test"}

    @pytest.mark.asyncio
    async def test_execute_async_skill(self) -> None:
        """Test executing an asynchronous skill."""

        class Input(BaseModel):
            value: int

        class Output(BaseModel):
            doubled: int

        async def async_doubler(input_data: Input) -> Output:
            await asyncio.sleep(0.01)  # Simulate async work
            return Output(doubled=input_data.value * 2)

        skill = SkillSpec(
            name="async_doubler",
            description="Double a number asynchronously",
            input_schema=Input,
            output_schema=Output,
            executor=async_doubler,
        )

        executor = SkillExecutor()
        result = await executor.execute(skill, {"value": 10})

        assert result["doubled"] == 20

    @pytest.mark.asyncio
    async def test_input_validation(self) -> None:
        """Test input validation catches invalid inputs."""

        class Input(BaseModel):
            value: int

        class Output(BaseModel):
            result: int

        def process(input_data: Input) -> Output:
            return Output(result=input_data.value * 2)

        skill = SkillSpec(
            name="process",
            description="Process a number",
            input_schema=Input,
            output_schema=Output,
            executor=process,
        )

        executor = SkillExecutor()

        # Invalid input type
        with pytest.raises((SkillExecutionError, ValidationError)):
            await executor.execute(skill, {"value": "not_an_int"})

    @pytest.mark.asyncio
    async def test_output_validation(self) -> None:
        """Test output validation catches invalid outputs."""

        class Input(BaseModel):
            value: int

        class Output(BaseModel):
            result: int

        def bad_executor(input_data: Input) -> dict:
            # Returns dict instead of Output model
            return {"wrong_key": input_data.value}

        skill = SkillSpec(
            name="bad_skill",
            description="Skill with bad output",
            input_schema=Input,
            output_schema=Output,
            executor=bad_executor,
        )

        executor = SkillExecutor()

        with pytest.raises((SkillExecutionError, ValidationError, TypeError)):
            await executor.execute(skill, {"value": 5})

    @pytest.mark.asyncio
    async def test_execution_timeout(self) -> None:
        """Test execution timeout control."""

        class Input(BaseModel):
            duration: float

        class Output(BaseModel):
            completed: bool

        async def slow_task(input_data: Input) -> Output:
            await asyncio.sleep(input_data.duration)
            return Output(completed=True)

        skill = SkillSpec(
            name="slow_task",
            description="A slow task",
            input_schema=Input,
            output_schema=Output,
            executor=slow_task,
        )

        # Set very short timeout
        executor = SkillExecutor(timeout=0.1)

        with pytest.raises((SkillExecutionError, asyncio.TimeoutError)):
            await executor.execute(skill, {"duration": 1.0})

    @pytest.mark.asyncio
    async def test_retry_on_failure(self) -> None:
        """Test retry logic on transient failures."""

        class Input(BaseModel):
            fail_count: int

        class Output(BaseModel):
            attempts: int

        attempt_counter = {"count": 0}

        def flaky_executor(input_data: Input) -> Output:
            attempt_counter["count"] += 1
            if attempt_counter["count"] <= input_data.fail_count:
                raise RuntimeError(f"Attempt {attempt_counter['count']} failed")
            return Output(attempts=attempt_counter["count"])

        skill = SkillSpec(
            name="flaky_skill",
            description="A flaky skill",
            input_schema=Input,
            output_schema=Output,
            executor=flaky_executor,
        )

        executor = SkillExecutor(max_retries=3)

        # Should succeed on 3rd attempt
        result = await executor.execute(skill, {"fail_count": 2})
        assert result["attempts"] == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self) -> None:
        """Test that retries are exhausted on persistent failures."""

        class Input(BaseModel):
            value: int

        class Output(BaseModel):
            result: int

        def always_fails(input_data: Input) -> Output:
            raise RuntimeError("Persistent failure")

        skill = SkillSpec(
            name="failing_skill",
            description="Always fails",
            input_schema=Input,
            output_schema=Output,
            executor=always_fails,
        )

        executor = SkillExecutor(max_retries=2)

        with pytest.raises(SkillExecutionError):
            await executor.execute(skill, {"value": 1})

    @pytest.mark.asyncio
    async def test_error_message_contains_skill_name(self) -> None:
        """Test that error messages include skill name for debugging."""

        class Input(BaseModel):
            value: int

        class Output(BaseModel):
            result: int

        def failing(input_data: Input) -> Output:
            raise ValueError("Something went wrong")

        skill = SkillSpec(
            name="debug_skill",
            description="For debugging",
            input_schema=Input,
            output_schema=Output,
            executor=failing,
        )

        executor = SkillExecutor(max_retries=0)

        try:
            await executor.execute(skill, {"value": 1})
            assert False, "Should have raised SkillExecutionError"
        except SkillExecutionError as e:
            assert "debug_skill" in str(e)
            assert e.skill_name == "debug_skill"


class TestSkillExecutorRetrySpans:
    """Test that SkillExecutor emits retry spans (Phase 5c v2)."""

    @pytest.mark.asyncio
    async def test_retry_creates_retry_spans(self) -> None:
        """Each failed attempt should create a RETRY span."""

        class Input(BaseModel):
            fail_count: int

        class Output(BaseModel):
            attempts: int

        attempt_counter = {"count": 0}

        def flaky(input_data: Input) -> Output:
            attempt_counter["count"] += 1
            if attempt_counter["count"] <= input_data.fail_count:
                raise RuntimeError(f"Attempt {attempt_counter['count']} failed")
            return Output(attempts=attempt_counter["count"])

        skill = SkillSpec(
            name="flaky_skill",
            description="Flaky",
            input_schema=Input,
            output_schema=Output,
            executor=flaky,
        )

        # Set up a trace context so retry spans have a parent
        root = Span(name="execution", span_type=SpanType.EXECUTION)
        token = TraceContext.push(root)

        collected_spans: list[Span] = []
        executor = SkillExecutor(max_retries=3)
        executor._on_retry_span = collected_spans.append  # type: ignore[attr-defined]

        try:
            result = await executor.execute(skill, {"fail_count": 2})
        finally:
            TraceContext.pop(token)

        assert result["attempts"] == 3
        # 2 failed attempts → 2 retry spans
        assert len(collected_spans) == 2
        for span in collected_spans:
            assert span.span_type == SpanType.RETRY
            assert span.status == "error"
            assert span.end_time is not None
            assert "retry.attempt" in span.attributes
            assert span.parent_id == root.span_id

    @pytest.mark.asyncio
    async def test_no_retry_spans_on_first_success(self) -> None:
        """No retry spans when skill succeeds on first attempt."""

        class Input(BaseModel):
            value: int

        class Output(BaseModel):
            result: int

        def ok_skill(input_data: Input) -> Output:
            return Output(result=input_data.value)

        skill = SkillSpec(
            name="ok_skill",
            description="OK",
            input_schema=Input,
            output_schema=Output,
            executor=ok_skill,
        )

        root = Span(name="execution", span_type=SpanType.EXECUTION)
        token = TraceContext.push(root)

        collected: list[Span] = []
        executor = SkillExecutor(max_retries=3)
        executor._on_retry_span = collected.append  # type: ignore[attr-defined]

        try:
            await executor.execute(skill, {"value": 42})
        finally:
            TraceContext.pop(token)

        assert len(collected) == 0

    @pytest.mark.asyncio
    async def test_retry_spans_without_trace_context(self) -> None:
        """Retry spans are skipped when no TraceContext is active."""

        class Input(BaseModel):
            fail_count: int

        class Output(BaseModel):
            attempts: int

        counter = {"n": 0}

        def flaky(input_data: Input) -> Output:
            counter["n"] += 1
            if counter["n"] <= input_data.fail_count:
                raise RuntimeError("fail")
            return Output(attempts=counter["n"])

        skill = SkillSpec(
            name="flaky_no_ctx",
            description="Flaky without context",
            input_schema=Input,
            output_schema=Output,
            executor=flaky,
        )

        collected: list[Span] = []
        executor = SkillExecutor(max_retries=3)
        executor._on_retry_span = collected.append  # type: ignore[attr-defined]

        result = await executor.execute(skill, {"fail_count": 1})
        assert result["attempts"] == 2
        # No trace context → no retry spans created
        assert len(collected) == 0
