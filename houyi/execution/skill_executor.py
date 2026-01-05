"""Skill executor for executing skills with validation and error handling."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import ValidationError

from houyi.core.skill import SkillSpec


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
        timeout: float = 30.0,
    ):
        """Initialize skill executor.
        
        Args:
            max_retries: Maximum number of retries on failure
            timeout: Execution timeout in seconds
        """
        self.max_retries = max_retries
        self.timeout = timeout
    
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
            raise SkillExecutionError(
                skill.name,
                "Skill has no executor function bound"
            )
        
        # Validate input
        try:
            validated_input = skill.input_schema(**input_data)
        except ValidationError as e:
            raise SkillExecutionError(
                skill.name,
                f"Input validation failed: {e}",
                e
            )
        
        # Execute with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Execute skill (with timeout)
                result = await self._execute_with_timeout(
                    skill.executor,
                    validated_input
                )
                
                # Validate output
                try:
                    validated_output = skill.output_schema(**result)
                    return validated_output.model_dump()
                except ValidationError as e:
                    raise SkillExecutionError(
                        skill.name,
                        f"Output validation failed: {e}",
                        e
                    )
                
            except SkillExecutionError:
                # Re-raise validation errors immediately
                raise
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # Wait before retry (exponential backoff)
                    await asyncio.sleep(2 ** attempt)
                    continue
        
        # All retries failed
        raise SkillExecutionError(
            skill.name,
            f"Execution failed after {self.max_retries} retries",
            last_error
        )
    
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
        # Check if executor is async
        if asyncio.iscoroutinefunction(executor):
            # Async executor
            result = await asyncio.wait_for(
                executor(input_data),
                timeout=self.timeout
            )
        else:
            # Sync executor - run in thread pool
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, executor, input_data),
                timeout=self.timeout
            )
        
        # Ensure result is a dict
        if hasattr(result, 'model_dump'):
            # Pydantic model
            return result.model_dump()
        elif isinstance(result, dict):
            return result
        else:
            # Wrap non-dict results
            return {"result": result}
