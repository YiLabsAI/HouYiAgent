"""Tests for ToolNodeExecutor behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

from houyi.core.skill import SkillSpec
from houyi.core.skill_registry import DEFAULT_SKILL_REGISTRY
from houyi.protocol.ir import ExecutionIR, NodeExecutionIR, NodeIR, NodeType, PlanIR

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STUDIO_SERVER_ROOT = _REPO_ROOT / "houyi-studio" / "server"
sys.path.insert(0, str(_STUDIO_SERVER_ROOT))

from houyi_studio.server.execution_context import ExecutionContext  # noqa: E402
from houyi_studio.server.node_executors import ToolNodeExecutor  # noqa: E402


def _build_context(*, execution: ExecutionIR, plan: PlanIR) -> ExecutionContext:
    return ExecutionContext(session_id="session_1", execution=execution, plan=plan)


@pytest.mark.asyncio
async def test_tool_node_executor_normalizes_tool_name() -> None:
    """Tool executor should normalize names like 'Web Search' to 'web_search'."""

    class InputSchema(BaseModel):
        query: str

    class OutputSchema(BaseModel):
        result: str

    async def executor(query: str) -> OutputSchema:
        return OutputSchema(result=f"ok:{query}")

    skill = SkillSpec(
        name="web_search",
        description="Web search",
        input_schema=InputSchema,
        output_schema=OutputSchema,
    )
    skill.bind_executor(executor)
    DEFAULT_SKILL_REGISTRY.register(skill, overwrite=True)

    node = NodeIR(
        node_id="tool_1",
        node_type=NodeType.TOOL,
        position={"x": 0, "y": 0},
        config={"tool_name": "Web Search"},
        inputs={"query": "hi"},
        outputs={},
        metadata={},
    )
    node_exec = NodeExecutionIR(node_id="tool_1")

    plan = PlanIR(plan_id="plan_1", version=1, nodes=[node], edges=[], entry_node_id="tool_1")
    execution = ExecutionIR(execution_id="exec_1", plan_id="plan_1")
    context = _build_context(execution=execution, plan=plan)

    executor_instance = ToolNodeExecutor()
    await executor_instance.execute(context, node, node_exec)

    assert node_exec.error is None
    assert node_exec.outputs["is_error"] is False
    assert node_exec.outputs["metadata"]["tool_name"] == "web_search"


@pytest.mark.asyncio
async def test_tool_node_executor_surfaces_cache_hit_from_result_metadata() -> None:
    class InputSchema(BaseModel):
        query: str

    class OutputSchema(BaseModel):
        result: str
        metadata: dict | None = None

    async def executor(query: str) -> dict:
        return {
            "result": f"ok:{query}",
            "metadata": {"cache_hit": True, "cache_key": "k1"},
        }

    skill = SkillSpec(
        name="web_search",
        description="Web search",
        input_schema=InputSchema,
        output_schema=OutputSchema,
    )
    skill.bind_executor(executor)
    DEFAULT_SKILL_REGISTRY.register(skill, overwrite=True)

    node = NodeIR(
        node_id="tool_1",
        node_type=NodeType.TOOL,
        position={"x": 0, "y": 0},
        config={"tool_name": "web_search"},
        inputs={"query": "hi"},
        outputs={},
        metadata={},
    )
    node_exec = NodeExecutionIR(node_id="tool_1")

    plan = PlanIR(plan_id="plan_1", version=1, nodes=[node], edges=[], entry_node_id="tool_1")
    execution = ExecutionIR(execution_id="exec_1", plan_id="plan_1")
    context = _build_context(execution=execution, plan=plan)

    executor_instance = ToolNodeExecutor()
    await executor_instance.execute(context, node, node_exec)

    assert node_exec.error is None
    assert node_exec.outputs["metadata"]["cache_hit"] is True
    assert node_exec.outputs["metadata"]["cache_key"] == "k1"
    assert node_exec.outputs["output"]["metadata"]["cache_hit"] is True
    assert node_exec.outputs["output"]["metadata"]["cache_key"] == "k1"


@pytest.mark.asyncio
async def test_tool_node_executor_surfaces_cache_hit_from_raw_metadata() -> None:
    class InputSchema(BaseModel):
        query: str

    class OutputSchema(BaseModel):
        result: str
        raw: dict | None = None
        metadata: dict | None = None

    async def executor(query: str) -> dict:
        return {
            "result": f"ok:{query}",
            "raw": {"result": {"ok": True}, "metadata": {"cache_hit": True, "cache_key": "k2"}},
        }

    skill = SkillSpec(
        name="web_search",
        description="Web search",
        input_schema=InputSchema,
        output_schema=OutputSchema,
    )
    skill.bind_executor(executor)
    DEFAULT_SKILL_REGISTRY.register(skill, overwrite=True)

    node = NodeIR(
        node_id="tool_1",
        node_type=NodeType.TOOL,
        position={"x": 0, "y": 0},
        config={"tool_name": "web_search"},
        inputs={"query": "hi"},
        outputs={},
        metadata={},
    )
    node_exec = NodeExecutionIR(node_id="tool_1")

    plan = PlanIR(plan_id="plan_1", version=1, nodes=[node], edges=[], entry_node_id="tool_1")
    execution = ExecutionIR(execution_id="exec_1", plan_id="plan_1")
    context = _build_context(execution=execution, plan=plan)

    executor_instance = ToolNodeExecutor()
    await executor_instance.execute(context, node, node_exec)

    assert node_exec.error is None
    assert node_exec.outputs["metadata"]["cache_hit"] is True
    assert node_exec.outputs["metadata"]["cache_key"] == "k2"
    assert node_exec.outputs["output"]["metadata"]["cache_hit"] is True
    assert node_exec.outputs["output"]["metadata"]["cache_key"] == "k2"
    assert node_exec.outputs["output"]["raw"]["metadata"]["cache_hit"] is True
    assert node_exec.outputs["output"]["raw"]["metadata"]["cache_key"] == "k2"


@pytest.mark.asyncio
async def test_tool_node_executor_web_search_provider_from_run_settings() -> None:
    """Tool executor should inject provider from run_settings for web_search."""

    class InputSchema(BaseModel):
        query: str
        provider: str | None = None

    class OutputSchema(BaseModel):
        provider: str

    async def executor(input_data: InputSchema) -> OutputSchema:
        return OutputSchema(provider=input_data.provider or "ddg")

    skill = SkillSpec(
        name="web_search",
        description="Web search",
        input_schema=InputSchema,
        output_schema=OutputSchema,
    )
    skill.bind_executor(executor)
    DEFAULT_SKILL_REGISTRY.register(skill, overwrite=True)

    node = NodeIR(
        node_id="tool_1",
        node_type=NodeType.TOOL,
        position={"x": 0, "y": 0},
        config={"tool_name": "web_search"},
        inputs={"query": "hi"},
        outputs={},
        metadata={},
    )
    node_exec = NodeExecutionIR(node_id="tool_1")

    plan = PlanIR(plan_id="plan_1", version=1, nodes=[node], edges=[], entry_node_id="tool_1")
    execution = ExecutionIR(
        execution_id="exec_1",
        plan_id="plan_1",
        metadata={"run_settings": {"web_search_provider": "serper"}},
    )
    context = ExecutionContext(
        session_id="session_1",
        execution=execution,
        plan=plan,
        run_settings=execution.metadata.get("run_settings") or {},
    )

    executor_instance = ToolNodeExecutor()
    await executor_instance.execute(context, node, node_exec)

    assert node_exec.error is None
    assert node_exec.inputs["provider"] == "serper"
    assert node_exec.inputs["query"] == "hi"
    assert node_exec.outputs["output"]["provider"] == "serper"


@pytest.mark.asyncio
async def test_tool_node_executor_web_search_preserves_explicit_provider() -> None:
    """Explicit provider in node inputs should not be overwritten by run settings."""

    class InputSchema(BaseModel):
        query: str
        provider: str | None = None

    class OutputSchema(BaseModel):
        provider: str

    async def executor(input_data: InputSchema) -> OutputSchema:
        return OutputSchema(provider=input_data.provider or "ddg")

    skill = SkillSpec(
        name="web_search",
        description="Web search",
        input_schema=InputSchema,
        output_schema=OutputSchema,
    )
    skill.bind_executor(executor)
    DEFAULT_SKILL_REGISTRY.register(skill, overwrite=True)

    node = NodeIR(
        node_id="tool_1",
        node_type=NodeType.TOOL,
        position={"x": 0, "y": 0},
        config={"tool_name": "web_search"},
        inputs={"query": "hi", "provider": "ddg"},
        outputs={},
        metadata={},
    )
    node_exec = NodeExecutionIR(node_id="tool_1")

    plan = PlanIR(plan_id="plan_1", version=1, nodes=[node], edges=[], entry_node_id="tool_1")
    execution = ExecutionIR(
        execution_id="exec_1",
        plan_id="plan_1",
        metadata={"run_settings": {"web_search_provider": "serper"}},
    )
    context = ExecutionContext(
        session_id="session_1",
        execution=execution,
        plan=plan,
        run_settings=execution.metadata.get("run_settings") or {},
    )

    executor_instance = ToolNodeExecutor()
    await executor_instance.execute(context, node, node_exec)

    assert node_exec.error is None
    assert node_exec.inputs["provider"] == "ddg"
    assert node_exec.inputs["query"] == "hi"
    assert node_exec.outputs["output"]["provider"] == "ddg"
