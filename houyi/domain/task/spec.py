"""Task specification."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TaskSpec(BaseModel):
    """Specification for a task."""

    description: str = Field(..., description="Task description")
    expected_output: str | None = Field(
        default=None, description="Expected output format or criteria"
    )
    agent: Any | None = Field(default=None, description="Agent to execute this task")
    context: list[int] | None = Field(default=None, description="Task dependencies (indices)")

    model_config = ConfigDict(arbitrary_types_allowed=True)


__all__ = ["TaskSpec"]
