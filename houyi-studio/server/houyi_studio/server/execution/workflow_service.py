"""Workflow service for managing reusable workflow definitions."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from houyi.application.workflow.workflow_utils import (
    plan_to_workflow_dict,
    sanitize_workflow_name,
    workflow_dict_to_plan,
)
from houyi.interface.protocol.ir import PlanIR

logger = logging.getLogger(__name__)


class WorkflowService:
    """Service for persisting reusable workflow definitions."""

    def __init__(self, workflows_dir: Path) -> None:
        self._workflows_dir = workflows_dir

    def save_workflow(self, workflow_name: str, plan: PlanIR) -> bool:
        """Save a workflow with a user-defined name."""
        try:
            safe_name = sanitize_workflow_name(workflow_name)
            if not safe_name:
                logger.error("Invalid workflow name: %s", workflow_name)
                return False

            workflow_file = self._workflows_dir / f"{safe_name}.json"
            # Backward compatibility: callers may mutate node.position directly.
            # PlanLayoutIR is the authoritative storage for layout, so we sync positions
            # from nodes before serialization to avoid losing layout during persistence.
            plan_dict = plan_to_workflow_dict(workflow_name=workflow_name, plan=plan)

            layout_positions_count = 0
            try:
                layout_positions_count = len((plan_dict.get("layout") or {}).get("positions") or {})
            except Exception:
                layout_positions_count = 0

            logger.info(
                "Saving workflow '%s' with layout.positions=%d (nodes=%d, edges=%d)",
                workflow_name,
                layout_positions_count,
                len(plan.nodes),
                len(plan.edges),
            )
            with open(workflow_file, "w", encoding="utf-8") as file:
                json.dump(plan_dict, file, indent=2, ensure_ascii=False)

            logger.info(
                "Saved workflow '%s' to file: %s (%d nodes, %d edges)",
                workflow_name,
                workflow_file,
                len(plan.nodes),
                len(plan.edges),
            )
            return True
        except Exception as exc:
            logger.error("Failed to save workflow '%s': %s", workflow_name, exc, exc_info=True)
            return False

    def load_workflow(self, workflow_name: str) -> PlanIR | None:
        """Load a workflow by name."""
        try:
            safe_name = sanitize_workflow_name(workflow_name)
            workflow_file = self._workflows_dir / f"{safe_name}.json"

            if not workflow_file.exists():
                logger.warning("Workflow not found: %s", workflow_name)
                return None

            with open(workflow_file, encoding="utf-8") as file:
                plan_dict = json.load(file)

            plan = workflow_dict_to_plan(plan_dict)
            logger.info(
                "Loaded workflow '%s' from file: %s (%d nodes, %d edges)",
                workflow_name,
                workflow_file,
                len(plan.nodes),
                len(plan.edges),
            )
            return plan
        except Exception as exc:
            logger.error("Failed to load workflow '%s': %s", workflow_name, exc, exc_info=True)
            return None

    def list_workflows(self) -> list[dict[str, Any]]:
        """List all saved workflows."""
        workflows: list[dict[str, Any]] = []
        try:
            for workflow_file in self._workflows_dir.glob("*.json"):
                try:
                    with open(workflow_file, encoding="utf-8") as file:
                        plan_dict = json.load(file)

                    workflows.append(
                        {
                            "name": plan_dict.get("workflow_name", workflow_file.stem),
                            "saved_at": plan_dict.get("saved_at", ""),
                            "node_count": len(plan_dict.get("nodes", [])),
                            "edge_count": len(plan_dict.get("edges", [])),
                        }
                    )
                except Exception as exc:
                    logger.error("Failed to read workflow file %s: %s", workflow_file, exc)

            workflows.sort(key=lambda item: item.get("saved_at", ""), reverse=True)
            logger.debug("Listed %d workflows", len(workflows))
        except Exception as exc:
            logger.error("Failed to list workflows: %s", exc, exc_info=True)

        return workflows
