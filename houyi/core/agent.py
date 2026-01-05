"""Agent specification and runtime."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from houyi.core.skill import SkillSpec


class AgentSpec(BaseModel):
    """Specification for an agent.

    An agent is defined by its role, capabilities (skills),
    and policies (constraints, retry logic, etc.).
    """

    role: str = Field(..., description="Agent role (e.g., 'Researcher', 'Analyst')")
    skills: list[SkillSpec] = Field(default_factory=list, description="Available skills")
    system_prompt: str | None = Field(default=None, description="Optional custom system prompt")
    policies: dict[str, Any] = Field(
        default_factory=dict,
        description="Policy configuration (LLM, retry, timeout, cost budget, etc.)",
    )

    def to_system_prompt(self) -> str:
        """Generate system prompt from role and skills."""
        if self.system_prompt:
            return self.system_prompt
        
        prompt = f"You are a {self.role}.\n\n"
        
        if self.skills:
            skill_descriptions = "\n".join(
                f"- {skill.name}: {skill.description}" for skill in self.skills
            )
            prompt += f"Available skills:\n{skill_descriptions}"
        
        return prompt

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI function calling schemas for all skills."""
        return [skill.to_tool_schema() for skill in self.skills]

