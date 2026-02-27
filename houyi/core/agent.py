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
        description="Agent policies (LLM, memory, etc.)",
    )
    verification_config: Any = Field(
        default=None,
        description="Agent-level verification configuration",
    )

    def to_system_prompt(self) -> str:
        """Generate system prompt from role and skills.

        When core skills are present, appends a TOOL ROUTING POLICY section
        with a few-shot example to guide the LLM toward using official core
        tools over any third-party extensions.
        """
        if self.system_prompt:
            return self.system_prompt

        prompt = f"You are a {self.role}.\n\n"

        if self.skills:
            skill_descriptions = "\n".join(
                f"- {skill.name}: {skill.description}" for skill in self.skills
            )
            prompt += f"Available skills:\n{skill_descriptions}"

        core_skills = [s for s in self.skills if getattr(s, "is_core", False)]
        if core_skills:
            core_names = ", ".join(s.name for s in core_skills)
            prompt += (
                "\n\n"
                "TOOL ROUTING POLICY:\n"
                "This agent has CORE OFFICIAL TOOLS (marked with [CORE OFFICIAL TOOL] "
                "in their description). Always prefer [CORE OFFICIAL TOOL] over "
                "[THIRD-PARTY EXTENSION] tools for the same task.\n"
                f"Core tools in this agent: {core_names}.\n"
                "Example: if asked to search the web, use [CORE OFFICIAL TOOL] "
                "web_search, NOT ext__web_search (if present)."
            )

        return prompt

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI function calling schemas for all skills.

        Core tools (``is_core=True``) are placed first to leverage LLM
        position bias, ensuring official tools appear before extensions.
        """
        sorted_skills = sorted(
            self.skills, key=lambda s: (0 if getattr(s, "is_core", False) else 1)
        )
        return [skill.to_tool_schema() for skill in sorted_skills]
