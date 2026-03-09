from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

from houyi.domain.skill.registry import DEFAULT_SKILL_REGISTRY
from houyi.domain.skill.spec import SkillSpec
from houyi.interface.protocol.ir import ExecutionIR, PlanIR

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STUDIO_SERVER_ROOT = _REPO_ROOT / "houyi-studio" / "server"
sys.path.insert(0, str(_STUDIO_SERVER_ROOT))

from houyi_studio.server.execution.context import ExecutionContext  # noqa: E402
from houyi_studio.server.execution.node_execution_flow import NodeExecutionFlow  # noqa: E402
from houyi_studio.server.execution.node_executor_factory import NodeExecutorFactory  # noqa: E402
from houyi_studio.server.execution.observation_service import ObservationService  # noqa: E402
from houyi_studio.server.gateway.event_bus import EventBus  # noqa: E402

from houyi.application.workflow.config_service import ConfigService  # noqa: E402
from houyi.interface.protocol.ir import NodeIR, NodeType  # noqa: E402


class _DummyConnectionManager:
    async def send_event(self, _session_id: str, _event: object) -> None:
        return None


async def _noop_notify(*_args: object, **_kwargs: object) -> None:
    return None


@pytest.mark.asyncio
async def test_tool_system_5node_route_disables_fallback_when_verified() -> None:
    class InputSchema(BaseModel):
        query: str
        max_results: int = 10
        provider: str | None = None

    class OutputSchema(BaseModel):
        provider: str
        results: list[dict]

    async def executor(input_data: InputSchema) -> OutputSchema:
        if input_data.provider == "ddg":
            return OutputSchema(
                provider="ddg",
                results=[{"title": "t", "url": "u"}, {"title": "t2", "url": "u2"}],
            )
        return OutputSchema(
            provider=input_data.provider or "serper", results=[{"title": "x", "url": "y"}]
        )

    skill = SkillSpec(
        name="web_search",
        description="Web search",
        input_schema=InputSchema,
        output_schema=OutputSchema,
    )
    skill.bind_executor(executor)
    DEFAULT_SKILL_REGISTRY.register(skill, overwrite=True)

    plan = PlanIR(
        plan_id="plan_1",
        version=1,
        entry_node_id="tool_primary",
        nodes=[
            NodeIR(
                node_id="tool_primary",
                node_type=NodeType.TOOL,
                position={"x": 0, "y": 0},
                config={"tool_name": "web_search"},
                inputs={"query": "q", "max_results": 1, "provider": "ddg"},
                outputs={"result": "primary"},
                metadata={},
            ),
            NodeIR(
                node_id="verify_1",
                node_type=NodeType.VERIFY,
                position={"x": 0, "y": 0},
                config={
                    "raise_on_failure": False,
                    "verification_rules": [
                        {
                            "rule_id": "require_results",
                            "verifier_type": "constraint",
                            "rule_spec": {"require_keys": ["results"]},
                            "severity": "error",
                            "auto_fixable": False,
                        },
                        {
                            "rule_id": "min_results",
                            "verifier_type": "constraint",
                            "rule_spec": {"min_items_path": "results", "min_items": 2},
                            "severity": "error",
                            "auto_fixable": False,
                        },
                    ],
                },
                inputs={"output": "$primary"},
                outputs={"verified": "verified"},
                metadata={},
            ),
            NodeIR(
                node_id="route_1",
                node_type=NodeType.ROUTE,
                position={"x": 0, "y": 0},
                config={"disable_nodes_on_true": ["tool_fallback"], "disable_nodes_on_false": []},
                inputs={"verified": "$verified"},
                outputs={"disabled_nodes": "disabled_nodes"},
                metadata={},
            ),
            NodeIR(
                node_id="tool_fallback",
                node_type=NodeType.TOOL,
                position={"x": 0, "y": 0},
                config={"tool_name": "web_search"},
                inputs={"query": "q", "max_results": 3, "provider": "serper"},
                outputs={"result": "fallback"},
                metadata={},
            ),
            NodeIR(
                node_id="logic_1",
                node_type=NodeType.LOGIC,
                position={"x": 0, "y": 0},
                config={"template": "verified={verified}"},
                inputs={"verified": "$verified"},
                outputs={"result": "final"},
                metadata={},
            ),
        ],
        edges=[
            {
                "edge_id": "tool_primary-verify_1",
                "source_node_id": "tool_primary",
                "target_node_id": "verify_1",
                "metadata": {},
            },
            {
                "edge_id": "verify_1-route_1",
                "source_node_id": "verify_1",
                "target_node_id": "route_1",
                "metadata": {},
            },
            {
                "edge_id": "route_1-tool_fallback",
                "source_node_id": "route_1",
                "target_node_id": "tool_fallback",
                "metadata": {},
            },
            {
                "edge_id": "route_1-logic_1",
                "source_node_id": "route_1",
                "target_node_id": "logic_1",
                "metadata": {},
            },
            {
                "edge_id": "tool_fallback-logic_1",
                "source_node_id": "tool_fallback",
                "target_node_id": "logic_1",
                "metadata": {},
            },
        ],
    )

    execution = ExecutionIR(
        execution_id="exec_1",
        plan_id="plan_1",
        context={},
        metadata={"run_settings": {}},
    )

    observation_service = ObservationService(
        connection_manager=_DummyConnectionManager(),
        event_bus=EventBus(),
    )

    registry = NodeExecutorFactory(
        config_service=ConfigService(),
        execute_llm_real=lambda *_args, **_kwargs: asyncio.sleep(0),
        execute_llm_mock=lambda *_args, **_kwargs: asyncio.sleep(0),
    ).build_registry()

    flow = NodeExecutionFlow(
        node_executor_registry=registry,
        observation_service=observation_service,
        notify_lifecycle=_noop_notify,
        context_factory=lambda session_id, exec_ir, plan_ir: ExecutionContext(
            session_id=session_id,
            execution=exec_ir,
            plan=plan_ir,
            run_settings=exec_ir.metadata.get("run_settings") or {},
        ),
    )

    session_id = "session_1"
    context = ExecutionContext(
        session_id=session_id, execution=execution, plan=plan, run_settings={}
    )

    order = ["tool_primary", "verify_1", "route_1", "tool_fallback", "logic_1"]
    for node_id in order:
        node = plan.get_node(node_id)
        if node is None:
            continue
        if node.deleted_at is not None:
            continue
        await flow.execute(context, node_id=node_id)

    assert execution.context["verified"] is True
    assert "final" in execution.context

    fallback = plan.get_node("tool_fallback")
    assert fallback is not None
    assert fallback.deleted_at is not None

    verify_exec = execution.node_executions.get("verify_1")
    assert verify_exec is not None
    assert verify_exec.outputs.get("verified") is True
    assert isinstance(verify_exec.outputs.get("results"), list)
    assert all(
        isinstance(item, dict) and "rule_id" in item and "passed" in item
        for item in (verify_exec.outputs.get("results") or [])
    )
    assert verify_exec.outputs.get("errors") is None


@pytest.mark.asyncio
async def test_tool_system_5node_route_rules_first_match() -> None:
    plan = PlanIR(
        plan_id="plan_1",
        version=1,
        entry_node_id="route_1",
        nodes=[
            NodeIR(
                node_id="route_1",
                node_type=NodeType.ROUTE,
                position={"x": 0, "y": 0},
                config={
                    "route_rules": [
                        {"label": "first", "when": True, "then": {"disable_nodes": ["node_a"]}},
                        {"label": "second", "when": True, "then": {"disable_nodes": ["node_b"]}},
                    ],
                    "disable_nodes_on_true": [],
                    "disable_nodes_on_false": [],
                },
                inputs={"verified": True},
                outputs={"disabled_nodes": "disabled"},
                metadata={},
            ),
            NodeIR(
                node_id="node_a",
                node_type=NodeType.TOOL,
                position={"x": 0, "y": 0},
                config={"tool_name": "web_search"},
                inputs={"query": "q"},
                outputs={"result": "a"},
                metadata={},
            ),
            NodeIR(
                node_id="node_b",
                node_type=NodeType.TOOL,
                position={"x": 0, "y": 0},
                config={"tool_name": "web_search"},
                inputs={"query": "q"},
                outputs={"result": "b"},
                metadata={},
            ),
        ],
        edges=[],
    )

    execution = ExecutionIR(
        execution_id="exec_1", plan_id="plan_1", context={}, metadata={"run_settings": {}}
    )
    observation_service = ObservationService(
        connection_manager=_DummyConnectionManager(), event_bus=EventBus()
    )

    registry = NodeExecutorFactory(
        config_service=ConfigService(),
        execute_llm_real=lambda *_args, **_kwargs: asyncio.sleep(0),
        execute_llm_mock=lambda *_args, **_kwargs: asyncio.sleep(0),
    ).build_registry()

    flow = NodeExecutionFlow(
        node_executor_registry=registry,
        observation_service=observation_service,
        notify_lifecycle=_noop_notify,
        context_factory=lambda session_id, exec_ir, plan_ir: ExecutionContext(
            session_id=session_id,
            execution=exec_ir,
            plan=plan_ir,
            run_settings=exec_ir.metadata.get("run_settings") or {},
        ),
    )

    context = ExecutionContext(
        session_id="session_1", execution=execution, plan=plan, run_settings={}
    )

    await flow.execute(context, node_id="route_1")

    # First rule should win: node_a disabled, node_b still enabled.
    assert plan.get_node("node_a").deleted_at is not None
    assert plan.get_node("node_b").deleted_at is None
    route_exec = execution.node_executions.get("route_1")
    assert route_exec is not None
    assert route_exec.outputs.get("route_mode") == "rules"
    assert route_exec.outputs.get("matched_rule", {}).get("label") == "first"
