"""Assertion specification for verification."""

import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class AssertionSpec(BaseModel):
    """Specification for a verifiable assertion/constraint."""

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
        """Evaluate the assertion against a context."""
        if callable(self.condition):
            return self.condition(context)

        try:
            safe_namespace = {
                "__builtins__": {},
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "abs": abs,
                "min": min,
                "max": max,
                **context,
            }

            dangerous_keywords = ["import", "exec", "eval", "__", "open", "file"]
            if any(keyword in self.condition for keyword in dangerous_keywords):
                raise ValueError(f"Dangerous operation in assertion: {self.condition}")

            result = eval(self.condition, safe_namespace, {})
            return bool(result)
        except Exception as e:
            logger.warning("Assertion evaluation failed: %s", e)
            return False
