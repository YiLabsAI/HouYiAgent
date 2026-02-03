from __future__ import annotations

import re

from houyi.orchestration.plan import NodeType
from houyi.orchestration.workflow_utils import (
    plan_to_workflow_dict,
    sanitize_workflow_name,
    workflow_dict_to_plan,
)
from houyi.protocol.ir import EdgeIR, NodeIR, PlanIR


def test_sanitize_workflow_name_keeps_safe_chars_and_normalizes_spaces() -> None:
    assert sanitize_workflow_name("My Workflow 01") == "My_Workflow_01"
    assert sanitize_workflow_name("  weird@@name!!  ") == "weirdname"
    assert sanitize_workflow_name("a-b_c") == "a-b_c"


def test_workflow_dict_roundtrip_preserves_plan_and_strips_metadata_fields() -> None:
    plan = PlanIR(
        plan_id="plan_1",
        version=1,
        nodes=[
            NodeIR(
                node_id="n1",
                node_type=NodeType.TOOL,
                position={"x": 12, "y": 34},
                config={},
                inputs={},
                outputs={},
                metadata={},
            ),
        ],
        edges=[EdgeIR(edge_id="e_n1_n1", source_node_id="n1", target_node_id="n1")],
        entry_node_id="n1",
        metadata={},
    )

    workflow = plan_to_workflow_dict(workflow_name="wf_1", plan=plan)

    assert workflow["workflow_name"] == "wf_1"
    assert re.match(r"^\d{4}-\d{2}-\d{2}T", workflow["saved_at"])
    assert workflow["plan_id"] == "plan_1"
    assert "layout" in workflow

    restored = workflow_dict_to_plan(dict(workflow))
    assert restored.plan_id == plan.plan_id
    assert restored.entry_node_id == plan.entry_node_id
    assert restored.layout is not None
    assert "workflow_name" not in restored.model_dump()
    assert "saved_at" not in restored.model_dump()
