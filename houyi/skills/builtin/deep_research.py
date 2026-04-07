"""Built-in deep_research skill.

Allows users to trigger lightweight deep research from the Chat interface.
The skill creates a research run, executes it, and returns an inline
summary with a link to the full Workspace run.
"""

from __future__ import annotations

import logging
import os
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
from houyi.infrastructure.config.env_config import (
    ENV_RESEARCH_MAX_AGENTS,
    ENV_RESEARCH_ORCHESTRATION_MODE,
)

logger = logging.getLogger(__name__)

_MODE_DIRECT = "direct"
_MODE_DELEGATE = "delegate"
_MODE_AUTONOMOUS = "autonomous"
_VALID_ORCHESTRATION_MODES = frozenset({_MODE_DIRECT, _MODE_DELEGATE, _MODE_AUTONOMOUS})
_DEFAULT_ORCHESTRATION_MODE = _MODE_DELEGATE
_DEFAULT_MAX_AGENTS = 3


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


def _default_orchestration_mode() -> str:
    raw = os.getenv(ENV_RESEARCH_ORCHESTRATION_MODE, _DEFAULT_ORCHESTRATION_MODE).strip().lower()
    if raw in _VALID_ORCHESTRATION_MODES:
        return raw
    logger.warning(
        "Invalid %s=%r, fallback to %s",
        ENV_RESEARCH_ORCHESTRATION_MODE,
        raw,
        _DEFAULT_ORCHESTRATION_MODE,
    )
    return _DEFAULT_ORCHESTRATION_MODE


def _default_max_agents() -> int:
    raw = os.getenv(ENV_RESEARCH_MAX_AGENTS, str(_DEFAULT_MAX_AGENTS)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "Invalid %s=%r, fallback to %d", ENV_RESEARCH_MAX_AGENTS, raw, _DEFAULT_MAX_AGENTS
        )
        return _DEFAULT_MAX_AGENTS


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

    from houyi.application.research.types import OrchestrationMode, ResearchDepth, ResearchSettings

    try:
        depth_map = {
            "quick": ResearchDepth.QUICK,
            "standard": ResearchDepth.STANDARD,
            "deep": ResearchDepth.DEEP,
        }
        mode_map = {
            _MODE_DIRECT: OrchestrationMode.DIRECT,
            _MODE_DELEGATE: OrchestrationMode.DELEGATE,
            _MODE_AUTONOMOUS: OrchestrationMode.AUTONOMOUS,
        }
        resolved_depth = depth_map.get(depth, ResearchDepth.QUICK)
        resolved_mode = mode_map[_default_orchestration_mode()]
        settings = ResearchSettings(
            depth=resolved_depth,
            orchestration_mode=resolved_mode,
            max_agents=_default_max_agents(),
        )
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
