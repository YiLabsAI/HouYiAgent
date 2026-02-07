"""Storage abstractions for execution engine state."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from houyi.protocol.ir import CheckpointIR, ExecutionIR, PlanIR

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExecutionStore:
    executions: dict[str, ExecutionIR] = field(default_factory=dict)

    def get(self, execution_id: str) -> ExecutionIR | None:
        return self.executions.get(execution_id)

    def save(self, execution: ExecutionIR) -> None:
        self.executions[execution.execution_id] = execution


@dataclass(slots=True)
class CheckpointStore:
    checkpoints: dict[str, list[CheckpointIR]] = field(default_factory=dict)

    def get(self, execution_id: str) -> list[CheckpointIR]:
        return self.checkpoints.get(execution_id, [])

    def init_execution(self, execution_id: str) -> None:
        self.checkpoints.setdefault(execution_id, [])

    def add(self, execution_id: str, checkpoint: CheckpointIR) -> None:
        self.checkpoints.setdefault(execution_id, []).append(checkpoint)


# Plan state is in-memory only.  HouYi Studio is a single-user local IDE;
# server restart triggers a frontend reload (via boot_id detection), which
# generates a new session_id and starts with an empty canvas.  File-based
# persistence was removed because it caused stale plans to reappear after
# restart/refresh.


@dataclass(slots=True)
class PlanStore:
    plans_dir: Path
    plans: dict[str, PlanIR] = field(default_factory=dict)
    session_plans: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._cleanup_legacy_files()

    def get(self, session_id: str) -> PlanIR | None:
        return self.plans.get(session_id)

    def get_cached(self, session_id: str) -> PlanIR | None:
        """Return cached plan without loading from disk."""
        return self.plans.get(session_id)

    def set(self, session_id: str, plan: PlanIR, persist: bool = True) -> None:
        self.plans[session_id] = plan
        self.session_plans[session_id] = plan.plan_id

    def save_to_file(self, session_id: str, plan: PlanIR) -> None:
        """No-op: plan persistence removed (in-memory only)."""

    def load_from_file(self, session_id: str) -> PlanIR | None:
        """No-op: plan persistence removed (in-memory only)."""
        return None

    # -- internal helpers --

    def _cleanup_legacy_files(self) -> None:
        """Remove stale plan files from previous versions."""
        try:
            for pattern in ("session_*.json", "current_plan.json"):
                for f in self.plans_dir.glob(pattern):
                    f.unlink()
                    logger.info("Cleaned up legacy plan file: %s", f.name)
        except Exception as exc:
            logger.warning("Failed to clean up legacy plan files: %s", exc)
