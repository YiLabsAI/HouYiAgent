from __future__ import annotations

from typing import Any

from houyi.application.context.session_context import SessionContextStateManager
from houyi.application.context.token_estimator import TokenEstimator
from houyi.application.context.types import SessionContextState


class SessionContextResolver:
    """Resolves session context state from neutral message payloads."""

    def __init__(self, *, state_manager: SessionContextStateManager) -> None:
        self._state_manager = state_manager

    def estimate_units(self, message_payload: dict[str, Any], *, model: str) -> int:
        estimator = TokenEstimator(model=model)
        return max(0, int(estimator.count_message(message_payload) or 0))

    def build_initial_state(
        self,
        session_id: str,
        *,
        now: float | None = None,
    ) -> SessionContextState:
        return self._state_manager.build_initial_state(session_id, now=now)

    def recover_state(
        self,
        *,
        session_id: str,
        message_payloads: list[dict[str, Any]],
        model: str,
        previous_state: SessionContextState | None = None,
        updated_at: float | None = None,
    ) -> SessionContextState:
        used_units = min(
            self._state_manager.rolling_capacity,
            sum(
                self.estimate_units(message_payload, model=model)
                for message_payload in message_payloads
            ),
        )
        return self._state_manager.recover_state(
            session_id=session_id,
            used_units=used_units,
            previous_state=previous_state,
            updated_at=updated_at,
        )

    def normalize_state(
        self,
        *,
        session_id: str,
        state: SessionContextState,
        updated_at: float | None = None,
    ) -> SessionContextState:
        return self._state_manager.normalize_state(
            session_id=session_id,
            state=state,
            updated_at=updated_at,
        )

    def apply_delta(
        self,
        *,
        state: SessionContextState,
        added_units: int = 0,
        released_units: int = 0,
        compacted_at: float | None = None,
        compaction_delta: int | None = None,
        compacted_message_count: int | None = None,
        now: float | None = None,
    ) -> SessionContextState:
        return self._state_manager.apply_delta(
            state=state,
            added_units=added_units,
            released_units=released_units,
            compacted_at=compacted_at,
            compaction_delta=compaction_delta,
            compacted_message_count=compacted_message_count,
            now=now,
        )
