"""Tests for SkillExecutor - core skill execution engine."""

import asyncio

import pytest
from pydantic import BaseModel, ValidationError

from houyi.core.skill import SkillSpec
from houyi.execution.skill_executor import SkillExecutionError, SkillExecutor


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
            executor=add
        )

        executor = SkillExecutor()
        result = await executor.execute(skill, {"x": 3, "y": 5})

        assert result["result"] == 8

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
            executor=async_doubler
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
            executor=process
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
            executor=bad_executor
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
            executor=slow_task
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
            executor=flaky_executor
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
            executor=always_fails
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
            executor=failing
        )

        executor = SkillExecutor(max_retries=0)

        try:
            await executor.execute(skill, {"value": 1})
            assert False, "Should have raised SkillExecutionError"
        except SkillExecutionError as e:
            assert "debug_skill" in str(e)
            assert e.skill_name == "debug_skill"
