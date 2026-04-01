"""Built-in deep_research skill.

Allows users to trigger lightweight deep research from the Chat interface.
The skill creates a ResearchSession, executes it, and returns an inline
summary with a link to the full Workspace session.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from houyi.domain.skill.policy import (
    ExecutionMode,
    InvocationPolicy,
    NetworkPerm,
    Permissions,
    SideEffect,
)
from houyi.domain.skill.spec import SkillSpec


class DeepResearchInput(BaseModel):
    query: str = Field(..., description="The research question to investigate.")
    depth: str = Field(
        default="quick",
        description="Research depth: 'quick', 'standard', or 'deep'.",
    )


class DeepResearchOutput(BaseModel):
    session_id: str
    summary: str
    report_url: str | None = None
    quality_score: float | None = None
    sources_count: int = 0


async def _deep_research_executor(
    *,
    query: str,
    depth: str = "quick",
) -> dict:
    """Execute a lightweight research session.

    In the server context, this is wired to the ResearchService. In
    standalone SDK usage, it creates a session directly. The Chat
    integration layer (ChatToolBridge) is responsible for injecting
    the actual ResearchService instance.
    """
    return {
        "session_id": "",
        "summary": f"Research requested: {query} (depth={depth}). "
        "Connect to ResearchService for full execution.",
        "report_url": None,
        "quality_score": None,
        "sources_count": 0,
    }


def build_deep_research_skill() -> SkillSpec:
    """Build the deep_research skill spec for ChatToolBridge registration."""
    return SkillSpec(
        name="deep_research",
        description=(
            "Conduct deep research on a topic. Generates a structured research "
            "plan, searches multiple sources, and produces a report with citations."
        ),
        input_schema=DeepResearchInput,
        output_schema=DeepResearchOutput,
        executor=_deep_research_executor,
        execution_mode=ExecutionMode.PLUGIN,
        invocation_policy=InvocationPolicy.default_for_side_effect(SideEffect.NETWORK),
        permissions=Permissions(network=NetworkPerm(enabled=True)),
    )
