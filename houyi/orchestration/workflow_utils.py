from __future__ import annotations

from datetime import datetime
from typing import Any

from houyi.protocol.ir import PlanIR


def sanitize_workflow_name(workflow_name: str) -> str:
    safe_name = "".join(ch for ch in workflow_name if ch.isalnum() or ch in (" ", "-", "_")).strip()
    return safe_name.replace(" ", "_")


def plan_to_workflow_dict(*, workflow_name: str, plan: PlanIR) -> dict[str, Any]:
    for node in plan.nodes:
        if not isinstance(getattr(node, "position", None), dict):
            continue
        plan.set_node_position(node.node_id, node.position)

    plan_dict = plan.model_dump(mode="json")

    if "layout" not in plan_dict:
        layout = getattr(plan, "layout", None)
        if layout is not None:
            plan_dict["layout"] = layout.model_dump(mode="json")

    plan_dict["workflow_name"] = workflow_name
    plan_dict["saved_at"] = datetime.now().isoformat()
    return plan_dict


def workflow_dict_to_plan(plan_dict: dict[str, Any]) -> PlanIR:
    plan_dict.pop("workflow_name", None)
    plan_dict.pop("saved_at", None)
    return PlanIR.model_validate(plan_dict)
