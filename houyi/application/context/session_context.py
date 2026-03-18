from __future__ import annotations

import time
from typing import Literal

from houyi.application.context.types import SessionContextState

SessionContextLevel = Literal["healthy", "elevated", "near_compaction", "compacted_recently"]


class SessionContextStateManager:
    """Pure session-level rolling-state logic reusable across runtimes."""

    def __init__(self, *, rolling_capacity: int) -> None:
        self._rolling_capacity = max(1, int(rolling_capacity))

    @property
    def rolling_capacity(self) -> int:
        return self._rolling_capacity

    def build_initial_state(
        self,
        session_id: str,
        *,
        now: float | None = None,
    ) -> SessionContextState:
        return SessionContextState(
            session_id=session_id,
            used_units=0,
            max_units=self._rolling_capacity,
            state="healthy",
            updated_at=now or time.time(),
        )

    def recover_state(
        self,
        *,
        session_id: str,
        used_units: int,
        previous_state: SessionContextState | None = None,
        updated_at: float | None = None,
    ) -> SessionContextState:
        normalized_used_units = max(0, min(int(used_units or 0), self._rolling_capacity))
        return SessionContextState(
            session_id=session_id,
            used_units=normalized_used_units,
            max_units=self._rolling_capacity,
            state=derive_session_context_state(
                used_units=normalized_used_units,
                max_units=self._rolling_capacity,
            ),
            last_compacted_at=(
                previous_state.last_compacted_at
                if isinstance(previous_state, SessionContextState)
                else None
            ),
            last_compaction_delta=None,
            last_compacted_message_count=(
                previous_state.last_compacted_message_count
                if isinstance(previous_state, SessionContextState)
                else None
            ),
            updated_at=updated_at or time.time(),
        )

    def normalize_state(
        self,
        *,
        session_id: str,
        state: SessionContextState,
        updated_at: float | None = None,
    ) -> SessionContextState:
        normalized_state = state.model_copy(
            update={
                "session_id": session_id,
                "max_units": self._rolling_capacity,
                "used_units": max(0, min(int(state.used_units or 0), self._rolling_capacity)),
                "last_compacted_at": state.last_compacted_at,
                "last_compaction_delta": (
                    max(0, int(state.last_compaction_delta))
                    if state.last_compaction_delta is not None
                    else None
                ),
                "last_compacted_message_count": (
                    max(0, int(state.last_compacted_message_count))
                    if state.last_compacted_message_count is not None
                    else None
                ),
                "updated_at": updated_at or state.updated_at or time.time(),
            }
        )
        normalized_state.state = derive_session_context_state(
            used_units=normalized_state.used_units,
            max_units=normalized_state.max_units,
            last_compaction_delta=normalized_state.last_compaction_delta,
        )
        return normalized_state

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
        used_units = max(
            0,
            min(
                state.max_units,
                int(state.used_units or 0) + max(0, int(added_units)) - max(0, int(released_units)),
            ),
        )
        next_state = state.model_copy(
            update={
                "used_units": used_units,
                "last_compacted_at": compacted_at
                if compacted_at is not None
                else state.last_compacted_at,
                "last_compaction_delta": (
                    max(0, int(compaction_delta)) if compaction_delta is not None else None
                ),
                "last_compacted_message_count": (
                    max(0, int(compacted_message_count))
                    if compacted_message_count is not None
                    else state.last_compacted_message_count
                ),
                "updated_at": now or time.time(),
            }
        )
        next_state.state = derive_session_context_state(
            used_units=next_state.used_units,
            max_units=next_state.max_units,
            last_compaction_delta=next_state.last_compaction_delta,
        )
        return next_state


def derive_session_context_state(
    *,
    used_units: int,
    max_units: int,
    last_compaction_delta: int | None = None,
) -> SessionContextLevel:
    if last_compaction_delta and last_compaction_delta > 0:
        return "compacted_recently"
    if max_units <= 0:
        return "healthy"
    ratio = used_units / max_units
    if ratio >= 0.9:
        return "near_compaction"
    if ratio >= 0.7:
        return "elevated"
    return "healthy"
