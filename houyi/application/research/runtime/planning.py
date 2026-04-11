from __future__ import annotations

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

_CLARIFICATION_REPLAN_CONFIDENCE = 0.65


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
