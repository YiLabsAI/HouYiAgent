"""Tests for plan patch application behavior."""

from __future__ import annotations

import sys
from pathlib import Path

from houyi.interface.protocol.ir import NodeIR, NodeType, PlanIR

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STUDIO_SERVER_ROOT = _REPO_ROOT / "houyi-studio" / "server"
sys.path.insert(0, str(_STUDIO_SERVER_ROOT))

from houyi_studio.server.gateway.app import _apply_plan_patches  # noqa: E402
from houyi_studio.server.gateway.commands import PlanPatch  # noqa: E402


def test_apply_patches_fields() -> None:
    """update_node patches should persist inputs/outputs/metadata."""

    plan = PlanIR(
        plan_id="plan_test",
        version=1,
        nodes=[
            NodeIR(
                node_id="tool_1",
                node_type=NodeType.TOOL,
                position={"x": 0, "y": 0},
                config={"tool_name": "web_search"},
                inputs={"query": "old"},
                outputs={"result": "old"},
                metadata={"label": "tool_1"},
            )
        ],
        edges=[],
        entry_node_id="tool_1",
    )

    patches = [
        PlanPatch(
            action="update_node",
            node_id="tool_1",
            node={
                "inputs": {"query": "new"},
                "outputs": {"result": "new"},
                "metadata": {"label": "t"},
            },
        )
    ]

    modified = _apply_plan_patches(plan, patches)

    assert modified is True
    updated = plan.get_node("tool_1")
    assert updated is not None
    assert updated.inputs == {"query": "new"}
    assert updated.outputs == {"result": "new"}
    assert updated.metadata == {"label": "t"}


def test_patch_position_layout() -> None:
    plan = PlanIR(
        plan_id="plan_test",
        version=1,
        nodes=[
            NodeIR(
                node_id="tool_1",
                node_type=NodeType.TOOL,
                position={"x": 1, "y": 2},
                config={"tool_name": "web_search"},
            )
        ],
        edges=[],
        entry_node_id="tool_1",
    )

    patches = [
        PlanPatch(
            action="update_node",
            node_id="tool_1",
            node={"position": {"x": 10, "y": 20}},
        )
    ]

    modified = _apply_plan_patches(plan, patches)
    assert modified is True
    assert plan.get_node_position("tool_1") == {"x": 10.0, "y": 20.0}
    updated = plan.get_node("tool_1")
    assert updated is not None
    assert updated.position == {"x": 10.0, "y": 20.0}


def test_add_node_layout() -> None:
    plan = PlanIR(
        plan_id="plan_test",
        version=1,
        nodes=[],
        edges=[],
        entry_node_id="",
    )

    patches = [
        PlanPatch(
            action="add_node",
            node={
                "node_id": "node_new",
                "node_type": "tool",
                "position": {"x": 123, "y": 456},
                "config": {"tool_name": "web_search"},
            },
        )
    ]

    modified = _apply_plan_patches(plan, patches)
    assert modified is True
    assert plan.get_node_position("node_new") == {"x": 123.0, "y": 456.0}
    new_node = plan.get_node("node_new")
    assert new_node is not None
    assert new_node.position == {"x": 123.0, "y": 456.0}
