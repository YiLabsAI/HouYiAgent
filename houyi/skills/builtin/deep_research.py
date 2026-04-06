"""Built-in deep_research skill.

Allows users to trigger lightweight deep research from the Chat interface.
The skill creates a research run, executes it, and returns an inline
summary with a link to the full Workspace run.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from houyi.domain.skill.policy import (
    ExecutionMode,
    InvocationPolicy,
    NetworkPerm,
    Permissions,
    SideEffect,
)
from houyi.domain.skill.spec import SkillSpec

logger = logging.getLogger(__name__)


class DeepResearchInput(BaseModel):
    query: str = Field(..., description="The research question to investigate.")
    depth: str = Field(
        default="quick",
        description="Research depth: 'quick', 'standard', or 'deep'.",
    )


class DeepResearchOutput(BaseModel):
    run_id: str
    summary: str
    report_url: str | None = None
    quality_score: float | None = None
    sources_count: int = 0


_research_service_ref: Any = None


def set_research_service(svc: Any) -> None:
    """Inject the server-side ResearchService so the skill can run real research."""
    global _research_service_ref
    _research_service_ref = svc


async def execute_deep_research(
    *,
    query: str,
    depth: str = "quick",
) -> dict:
    """Execute a research run via the injected ResearchService.

    Falls back to a placeholder response when no service is available
    (e.g., standalone SDK usage without a server).
    """
    svc = _research_service_ref
    if svc is None:
        return {
            "run_id": "",
            "summary": f"Research requested: {query} (depth={depth}). "
            "No ResearchService available — run inside HouYi server.",
            "report_url": None,
            "quality_score": None,
            "sources_count": 0,
        }

    from houyi.application.research.types import ResearchDepth, ResearchSettings

    try:
        depth_map = {
            "quick": ResearchDepth.QUICK,
            "standard": ResearchDepth.STANDARD,
            "deep": ResearchDepth.DEEP,
        }
        resolved_depth = depth_map.get(depth, ResearchDepth.QUICK)
        settings = ResearchSettings(depth=resolved_depth)
        runtime, _plan = await svc.create_run(
            query=query,
            settings=settings,
        )
        run_id = runtime.run_id

        await svc.launch_run(run_id)

        report = await svc.get_report(run_id)

        sections_text = "\n\n".join(f"## {s.title}\n{s.content}" for s in report.sections)
        summary = f"# {report.title}\n\n{sections_text}"
        if len(summary) > 4000:
            summary = summary[:3900] + "\n\n... [truncated, see full report]"

        quality = None
        if report.quality_score:
            quality = (report.quality_score.race_overall + report.quality_score.fact_overall) / 2

        return {
            "run_id": run_id,
            "summary": summary,
            "report_url": f"#/research/{run_id}",
            "quality_score": quality,
            "sources_count": len(report.references),
        }
    except Exception:
        logger.exception("deep_research skill execution failed for query: %s", query)
        return {
            "run_id": "",
            "summary": f"Research failed for: {query}. Please try again.",
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
        executor=execute_deep_research,
        execution_mode=ExecutionMode.PLUGIN,
        invocation_policy=InvocationPolicy.default_for_side_effect(SideEffect.NETWORK),
        permissions=Permissions(network=NetworkPerm(enabled=True)),
    )
