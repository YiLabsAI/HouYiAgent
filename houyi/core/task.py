"""Task specification."""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskSpec(BaseModel):
    """Specification for a task.
    
    A task defines what needs to be done, expected output format,
    and which agent should execute it.
    """

    description: str = Field(..., description="Task description")
    expected_output: Optional[str] = Field(default=None, description="Expected output format or criteria")
    agent: Optional[Any] = Field(default=None, description="Agent to execute this task")
    context: Optional[list[int]] = Field(default=None, description="Task dependencies (indices)")

    model_config = ConfigDict(arbitrary_types_allowed=True)
