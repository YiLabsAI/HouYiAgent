from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from houyi.application.research.planner import ResearchPlanner, apply_plan_edits
from houyi.application.research.runtime.clarification import ClarificationAgent, ClarificationResult
from houyi.application.research.types import PlanEdit, PlanStatus, ResearchPlan, ResearchSettings


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
        clarifier: ClarificationAgent,
        emit: Callable[..., Awaitable[None]],
    ) -> None:
        self._planner = planner
        self._clarifier = clarifier
        self._emit = emit

    async def start(self, request: PlanningRequest) -> PlanningResult:
        effective_query = request.query
        clarification: ClarificationResult | None = None
        if request.settings.depth in ("standard", "deep"):
            clarification = await self._clarifier.analyze(request.query)
            if clarification.refined_query and clarification.confidence < 0.7:
                effective_query = clarification.refined_query
                await self._emit(
                    "research.query_refined",
                    original=request.query,
                    refined=effective_query,
                    issues=clarification.issues,
                )
        plan = await self._planner.generate_plan(
            effective_query,
            settings=request.settings,
            memory_context=request.memory_context,
        )
        await self._emit("research.plan_generated", plan=plan.model_dump())
        return PlanningResult(
            plan=plan,
            clarification=clarification,
            effective_query=effective_query,
        )

    def edit(self, plan: ResearchPlan, edits: list[PlanEdit]) -> ResearchPlan:
        return apply_plan_edits(plan, edits)

    async def confirm(self, plan: ResearchPlan) -> ResearchPlan:
        plan.status = PlanStatus.CONFIRMED
        await self._emit("research.plan_confirmed", plan=plan.model_dump())
        return plan
