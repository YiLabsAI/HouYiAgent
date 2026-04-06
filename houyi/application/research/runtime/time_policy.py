from __future__ import annotations

from houyi.application.research.types import OrchestrationMode, ResearchPlan, ResearchSettings


class TimeBudgetPolicy:
    _PER_QUESTION_BUDGET_SECONDS = 120
    _AGENT_TIMEOUT_SECONDS = 300
    _REPORT_BUDGET_BY_DEPTH: dict[str, int] = {"quick": 600, "standard": 1200, "deep": 1500}

    def report_budget_seconds(self, settings: ResearchSettings) -> int:
        depth = settings.depth
        key = depth.value if hasattr(depth, "value") else str(depth)
        return self._REPORT_BUDGET_BY_DEPTH.get(key, 1200)

    def runtime_timeout_seconds(
        self,
        settings: ResearchSettings,
        plan: ResearchPlan | None,
        checkpoint_count: int,
    ) -> float:
        report_budget = float(self.report_budget_seconds(settings))
        total = len(plan.sub_questions) if plan else 3
        remaining = max(0, total - checkpoint_count)
        mode = settings.orchestration_mode
        extra = 60 if mode == OrchestrationMode.AUTONOMOUS else 0
        if mode in (OrchestrationMode.DELEGATE, OrchestrationMode.AUTONOMOUS):
            if remaining == 0:
                return report_budget + extra
            batches = max(1, -(-remaining // settings.max_agents))
            return batches * self._AGENT_TIMEOUT_SECONDS + report_budget + extra
        if remaining == 0:
            return report_budget
        return remaining * self._PER_QUESTION_BUDGET_SECONDS + report_budget
