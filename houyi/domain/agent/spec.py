"""Agent specification and runtime."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from houyi.domain.skill.spec import SkillSpec


class AgentTeamConfig(BaseModel):
    """Lightweight config for declaring an agent within a team orchestration.

    Used by both DELEGATE (supervisor-worker) and AUTONOMOUS (peer)
    topologies.
    """

    role: str = Field(..., description="Agent role name within the team")
    skills: list[SkillSpec] = Field(default_factory=list)
    system_prompt: str | None = None
    max_turns: int = 20
    tools: list[str] = Field(default_factory=list, description="Tool names available to this agent")
    model: str | None = Field(
        default=None,
        description="Override LLM model for this agent (None = inherit from parent)",
    )
    memory_scope: str = Field(
        default="isolated",
        description="Memory scope: 'isolated' (default) or 'shared'",
    )
    spawn_depth_limit: int = Field(
        default=1,
        description="Max nesting depth for recursive agent spawning (0 = leaf)",
    )


class AgentSpec(BaseModel):
    """Specification for an agent.

    Phase 2 fields (``team_agents``, ``max_turns``, ``context_strategy``,
    ``memory_config``) are backward-compatible—all default to ``None`` or
    safe defaults so that existing callers are unaffected.
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

    # --- Phase 2 fields (Sprint 2) ---
    team_agents: list[AgentTeamConfig] = Field(
        default_factory=list,
        description="Declared team agents for orchestration (DELEGATE / AUTONOMOUS)",
    )
    max_turns: int = Field(
        default=50,
        description="Maximum tool-loop turns before forced stop",
    )
    context_strategy: dict[str, Any] | None = Field(
        default=None,
        description="Context truncation / compression strategy config",
    )
    memory_config: dict[str, Any] | None = Field(
        default=None,
        description="Memory engine configuration (scope, policy, etc.)",
    )

    def to_system_prompt(self) -> str:
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
        sorted_skills = sorted(
            self.skills, key=lambda s: (0 if getattr(s, "is_core", False) else 1)
        )
        return [skill.to_tool_schema() for skill in sorted_skills]


__all__ = ["AgentSpec", "AgentTeamConfig"]
