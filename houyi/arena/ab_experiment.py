"""A/B Experiment Framework — compare orchestration modes.

Runs the same query through two research configurations (A = DELEGATE,
B = AUTONOMOUS) and produces a comparison report using RACE and FACT metrics.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from pydantic import BaseModel

from houyi.adapters.llm.base import LLMAdapter
from houyi.application.research.session import ResearchSession
from houyi.application.research.types import (
    OrchestrationMode,
    ResearchSettings,
)
from houyi.skills.web_search.service import WebSearchService

logger = logging.getLogger(__name__)


class ExperimentArm(BaseModel):
    """Result from one arm of the A/B experiment."""

    arm_id: str
    mode: str
    session_id: str = ""
    quality_race: float | None = None
    quality_fact: float | None = None
    source_count: int = 0
    duration_seconds: float = 0.0
    error: str | None = None


class ExperimentReport(BaseModel):
    """Side-by-side comparison of two research approaches."""

    query: str
    arm_a: ExperimentArm | None = None
    arm_b: ExperimentArm | None = None
    winner: str | None = None
    margin: float = 0.0
    recommendation: str = ""


class ABExperiment:
    """Run the same query through two orchestration modes and compare."""

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        web_search: WebSearchService,
        **llm_kwargs: Any,
    ) -> None:
        self._llm = llm_adapter
        self._web_search = web_search
        self._llm_kwargs = llm_kwargs

    async def run(
        self,
        query: str,
        *,
        settings_a: ResearchSettings | None = None,
        settings_b: ResearchSettings | None = None,
    ) -> ExperimentReport:
        """Execute both arms and produce a comparison report."""
        a_settings = settings_a or ResearchSettings(
            orchestration_mode=OrchestrationMode.DELEGATE,
            max_sub_questions=3,
        )
        b_settings = settings_b or ResearchSettings(
            orchestration_mode=OrchestrationMode.AUTONOMOUS,
            max_sub_questions=3,
            max_agents=3,
        )

        arm_a_task = self._run_arm("arm_a", query, a_settings)
        arm_b_task = self._run_arm("arm_b", query, b_settings)

        arm_a, arm_b = await asyncio.gather(arm_a_task, arm_b_task)

        winner, margin, rec = _determine_winner(arm_a, arm_b)

        return ExperimentReport(
            query=query,
            arm_a=arm_a,
            arm_b=arm_b,
            winner=winner,
            margin=margin,
            recommendation=rec,
        )

    async def _run_arm(self, arm_id: str, query: str, settings: ResearchSettings) -> ExperimentArm:
        start = time.monotonic()
        session = ResearchSession(
            llm_adapter=self._llm,
            web_search=self._web_search,
            settings=settings,
            **self._llm_kwargs,
        )
        try:
            await session.start(query)
            await session.confirm_plan()
            await session.execute()
            report = await session.get_report()

            race = session.quality_score.race.overall if session.quality_score else None
            fact = session.quality_score.fact.citation_accuracy if session.quality_score else None

            return ExperimentArm(
                arm_id=arm_id,
                mode=settings.orchestration_mode.value,
                session_id=session.session_id,
                quality_race=race,
                quality_fact=fact,
                source_count=report.metadata.source_count,
                duration_seconds=round(time.monotonic() - start, 2),
            )
        except Exception as exc:
            return ExperimentArm(
                arm_id=arm_id,
                mode=settings.orchestration_mode.value,
                session_id=session.session_id,
                duration_seconds=round(time.monotonic() - start, 2),
                error=str(exc),
            )


def _determine_winner(a: ExperimentArm, b: ExperimentArm) -> tuple[str | None, float, str]:
    """Compare arms by weighted RACE + FACT score."""
    if a.error and b.error:
        return None, 0.0, "Both arms failed."
    if a.error:
        return "arm_b", 0.0, f"Arm A failed: {a.error}"
    if b.error:
        return "arm_a", 0.0, f"Arm B failed: {b.error}"

    score_a = _combined_score(a.quality_race, a.quality_fact)
    score_b = _combined_score(b.quality_race, b.quality_fact)
    margin = abs(score_a - score_b)

    if margin < 2.0:
        winner = None
        rec = f"Tie (A={score_a:.1f}, B={score_b:.1f}). Consider speed: A={a.duration_seconds:.0f}s, B={b.duration_seconds:.0f}s."
    elif score_a > score_b:
        winner = "arm_a"
        rec = f"Arm A ({a.mode}) wins by {margin:.1f} points."
    else:
        winner = "arm_b"
        rec = f"Arm B ({b.mode}) wins by {margin:.1f} points."

    return winner, margin, rec


def _combined_score(race: float | None, fact: float | None) -> float:
    r = race or 0.0
    f = fact or 0.0
    return r * 0.6 + f * 0.4
