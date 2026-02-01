"""Storage abstractions for execution engine state."""

from __future__ import annotations

import json
import logging
import os
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


@dataclass(slots=True)
class PlanStore:
    plans_dir: Path
    plans: dict[str, PlanIR] = field(default_factory=dict)
    session_plans: dict[str, str] = field(default_factory=dict)

    def get(self, session_id: str) -> PlanIR | None:
        if session_id in self.plans:
            return self.plans[session_id]
        plan = self.load_from_file(session_id)
        if plan:
            self.plans[session_id] = plan
            logger.info("Loaded plan from file for session: %s", session_id)
        return plan

    def get_cached(self, session_id: str) -> PlanIR | None:
        """Return cached plan without loading from disk."""
        return self.plans.get(session_id)

    def set(self, session_id: str, plan: PlanIR, persist: bool = True) -> None:
        self.plans[session_id] = plan
        self.session_plans[session_id] = plan.plan_id
        if persist:
            self.save_to_file(session_id, plan)

    def save_to_file(self, session_id: str, plan: PlanIR) -> None:
        if os.getenv("HOUYI_DISABLE_PLAN_PERSISTENCE") == "1":
            logger.info("Plan persistence disabled for E2E.")
            return
        try:
            plan_file = self.plans_dir / f"{session_id}.json"
            # Backward compatibility: legacy callers may mutate node.position directly.
            # PlanLayoutIR is the authoritative storage for layout, so we sync positions
            # from nodes before serialization to avoid losing layout during persistence.
            for node in plan.nodes:
                if not isinstance(getattr(node, "position", None), dict):
                    continue
                plan.set_node_position(node.node_id, node.position)
            plan_dict = plan.model_dump(mode="json")
            with open(plan_file, "w", encoding="utf-8") as f:
                json.dump(plan_dict, f, indent=2, ensure_ascii=False)
            logger.info(
                "Saved plan to file: %s (%d nodes, %d edges)",
                plan_file,
                len(plan.nodes),
                len(plan.edges),
            )
        except Exception as exc:
            logger.error("Failed to save plan to file: %s", exc, exc_info=True)

    def load_from_file(self, session_id: str) -> PlanIR | None:
        try:
            plan_file = self.plans_dir / f"{session_id}.json"
            if not plan_file.exists():
                return None
            with open(plan_file, encoding="utf-8") as f:
                plan_dict = json.load(f)
            plan = PlanIR.model_validate(plan_dict)
            logger.info(
                "Loaded plan from file: %s (%d nodes, %d edges)",
                plan_file,
                len(plan.nodes),
                len(plan.edges),
            )
            return plan
        except Exception as exc:
            logger.error("Failed to load plan from file: %s", exc, exc_info=True)
            return None
