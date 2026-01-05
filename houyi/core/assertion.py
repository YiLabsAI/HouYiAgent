"""Assertion specification for verification."""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AssertionSpec(BaseModel):
    """Specification for a verifiable assertion/constraint.

    Assertions enable formal verification of execution results,
    converting soft goals into hard checks.
    """

    name: str = Field(..., description="Assertion identifier")
    condition: str | Callable[..., bool] = Field(
        ...,
        description="Logic condition (e.g., 'cost < 1.0' or Python function)",
    )
    on_failure: str = Field(
        default="abort",
        description="Behavior on failure: 'retry', 'abort', or 'human'",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the assertion",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Evaluate the assertion against a context.

        Args:
            context: Execution context with variables for evaluation

        Returns:
            True if assertion passes, False otherwise
        """
        if callable(self.condition):
            return self.condition(context)

        # Safe expression evaluation with limited operators
        # Only allow basic comparisons and logical operators
        try:
            # Create safe namespace with only allowed operations
            safe_namespace = {
                "__builtins__": {},
                # Allow basic comparisons
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "abs": abs,
                "min": min,
                "max": max,
                # Add context variables
                **context
            }
            
            # Validate expression doesn't contain dangerous operations
            dangerous_keywords = ['import', 'exec', 'eval', '__', 'open', 'file']
            if any(keyword in self.condition for keyword in dangerous_keywords):
                raise ValueError(f"Dangerous operation in assertion: {self.condition}")
            
            # Evaluate in safe namespace
            result = eval(self.condition, safe_namespace, {})
            return bool(result)
            
        except Exception as e:
            print(f"Assertion evaluation failed: {e}")
            return False
