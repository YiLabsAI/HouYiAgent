"""Plan service for managing session execution plans."""

from __future__ import annotations

from houyi.protocol.ir import PlanIR

from .stores import PlanStore


class PlanService:
    """Service for managing session-scoped execution plans."""

    def __init__(self, plan_store: PlanStore) -> None:
        self._plan_store = plan_store

    def get_current_plan(self, session_id: str) -> PlanIR | None:
        """Return the current plan for a session."""
        return self._plan_store.get(session_id)

    def set_current_plan(self, session_id: str, plan: PlanIR, *, persist: bool = False) -> None:
        """Store the current plan for a session."""
        self._plan_store.set(session_id, plan, persist=persist)

    def get_plan_for_session(self, session_id: str, fallback_plan: PlanIR) -> PlanIR:
        """Return the latest plan for a session, falling back if missing."""
        return self._plan_store.get(session_id) or fallback_plan

    def get_cached_plan(self, session_id: str) -> PlanIR | None:
        """Return cached plan without loading from disk."""
        return self._plan_store.get_cached(session_id)

    def save_plan_to_file(self, session_id: str, plan: PlanIR) -> None:
        """Persist a session plan for later retrieval."""
        self._plan_store.save_to_file(session_id, plan)

    def load_plan_from_file(self, session_id: str) -> PlanIR | None:
        """Load a plan from storage for the session."""
        return self._plan_store.load_from_file(session_id)
