from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from houyi.application.research.planner import (
    ResearchPlanner,
    apply_plan_edits,
    validate_research_plan,
)
from houyi.application.research.types import (
    ClarificationResult,
    PlanEdit,
    PlanStatus,
    ResearchPlan,
    ResearchSettings,
)

logger = logging.getLogger(__name__)

_CLARIFICATION_REPLAN_CONFIDENCE = 0.65


def is_soft_plan_validation_error(error: str) -> bool:
    """Return True when *error* from :func:`validate_research_plan` represents
    a soft quality contract that the runtime can degrade on instead of
    crashing the user-visible confirm/execute flow.

    The sub-question floor (``_MIN_SUB_QUESTIONS_BY_DEPTH``) is driven by the
    planner prompt; bench runs always observed ≥floor outputs, so downgrading
    floor shortages to warnings preserves the leaderboard happy path while
    making the edge case recoverable.
    """
    return error.startswith("Planner returned fewer than ")


def format_soft_plan_warning(plan: ResearchPlan, error: str) -> str:
    """Render :func:`is_soft_plan_validation_error` into an operator-readable
    message that names the depth mode, actual count, and expected floor so
    the cause is obvious without grepping the validator source.
    """
    depth = plan.settings.depth.value if plan.settings else "unknown"
    count = len(plan.sub_questions)
    # Parse the floor out of the validator message (e.g. "fewer than 5 sub-questions")
    # with a safe fallback so a future wording change does not mask the warning.
    expected = error.removeprefix("Planner returned fewer than ").split(" ", 1)[0]
    return (
        f"Planner returned only {count} sub-questions for '{depth}' mode "
        f"(minimum expected: {expected}). Report breadth may be reduced; "
        "proceeding with the degraded plan."
    )


@dataclass(slots=True)
class PlanningRequest:
    query: str
    settings: ResearchSettings
    memory_context: str | None = None


@dataclass(slots=True)
class PlanningResult:
    plan: ResearchPlan
    clarification: ClarificationResult | None
    effective_query: str


class PlanningCoordinator:
    def __init__(
        self,
        planner: ResearchPlanner,
        emit: Callable[..., Awaitable[None]],
    ) -> None:
        self._planner = planner
        self._emit = emit

    async def start(self, request: PlanningRequest) -> PlanningResult:
        effective_query = request.query
        draft = await self._planner.generate_plan_draft(
            request.query,
            settings=request.settings,
            memory_context=request.memory_context,
        )
        clarification = draft.clarification
        if request.settings.depth in ("standard", "deep") and _should_replan(
            request.query, clarification
        ):
            assert clarification is not None
            assert clarification.refined_query is not None
            effective_query = clarification.refined_query.strip()
            await self._emit(
                "research.query_refined",
                original=request.query,
                refined=effective_query,
                issues=clarification.issues,
            )
            draft = await self._planner.generate_plan_draft(
                effective_query,
                settings=request.settings,
                memory_context=request.memory_context,
            )
        plan = draft.plan
        await self._emit("research.plan_generated", plan=plan.model_dump())
        return PlanningResult(
            plan=plan,
            clarification=clarification,
            effective_query=effective_query,
        )

    def edit(self, plan: ResearchPlan, edits: list[PlanEdit]) -> ResearchPlan:
        return apply_plan_edits(plan, edits)

    async def confirm(self, plan: ResearchPlan) -> ResearchPlan:
        validation_error = validate_research_plan(plan)
        if validation_error is not None:
            if is_soft_plan_validation_error(validation_error):
                # Planner may return fewer sub-questions than the declared
                # depth floor on difficult queries even after the in-draft
                # retry.  Historically this crashed the houyi-studio confirm
                # flow; warn and proceed so execution can still run.  The
                # execute() path deliberately does not re-log — confirm is
                # the single source of truth for this warning.
                logger.warning(format_soft_plan_warning(plan, validation_error))
            else:
                raise ValueError(validation_error)
        plan.status = PlanStatus.CONFIRMED
        await self._emit("research.plan_confirmed", plan=plan.model_dump())
        return plan


def _should_replan(query: str, clarification: ClarificationResult | None) -> bool:
    if clarification is None or not clarification.refined_query:
        return False
    if clarification.refined_query.strip() == query.strip():
        return False
    return (
        clarification.needs_clarification
        and clarification.confidence < _CLARIFICATION_REPLAN_CONFIDENCE
    )
