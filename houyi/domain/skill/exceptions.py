"""Shared exception types for skill runtime and execution failures."""

from __future__ import annotations


class SkillExecutionError(Exception):
    """Error during skill execution."""

    def __init__(self, skill_name: str, message: str, original_error: Exception | None = None):
        self.skill_name = skill_name
        self.message = message
        self.original_error = original_error
        super().__init__(f"Skill '{skill_name}' execution failed: {message}")


__all__ = ["SkillExecutionError"]
