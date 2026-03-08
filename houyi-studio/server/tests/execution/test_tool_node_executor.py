"""Tests for ToolNodeExecutor behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

from houyi.domain.skill.registry import DEFAULT_SKILL_REGISTRY
from houyi.domain.skill.spec import SkillSpec
from houyi.interface.protocol.ir import ExecutionIR, NodeExecutionIR, NodeIR, NodeType, PlanIR

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STUDIO_SERVER_ROOT = _REPO_ROOT / "houyi-studio" / "server"
sys.path.insert(0, str(_STUDIO_SERVER_ROOT))

from houyi_studio.server.execution.context import ExecutionContext  # noqa: E402
from houyi_studio.server.execution.node_executors import ToolNodeExecutor  # noqa: E402


def _build_context(
    *, execution: ExecutionIR, plan: PlanIR, observation_service=None
) -> ExecutionContext:
    return ExecutionContext(
        session_id="session_1",
        execution=execution,
        plan=plan,
        observation_service=observation_service,
    )


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


@pytest.mark.asyncio
async def test_tool_node_executor_emits_span_update_events() -> None:
    """Regression for Bug4c: ToolNodeExecutor must emit SpanUpdateEvent start+end.

    Without this, tool spans never appear in the frontend Timeline waterfall,
    and cache_hit status is invisible to the user.
    """
    from unittest.mock import AsyncMock

    from houyi_studio.server.gateway.events import SpanUpdateEvent

    from houyi.infrastructure.observability import Span, SpanType, TraceContext

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

    # Mock observation service to capture emitted events
    mock_obs = AsyncMock()
    context = _build_context(execution=execution, plan=plan, observation_service=mock_obs)

    # Push a parent span so TraceContext.current() returns non-None
    parent_span = Span(name="node.tool", span_type=SpanType.NODE, node_id="tool_1")
    token = TraceContext.push(parent_span)

    try:
        executor_instance = ToolNodeExecutor()
        await executor_instance.execute(context, node, node_exec)
    finally:
        TraceContext.pop(token)

    assert node_exec.error is None

    # Verify observation_service.emit was called with SpanUpdateEvent
    emitted = [call.args[0] for call in mock_obs.emit.call_args_list]
    span_updates = [e for e in emitted if isinstance(e, SpanUpdateEvent)]

    # Must have at least 2 span updates: start (no end_time) and end (with end_time)
    assert len(span_updates) >= 2, f"Expected >=2 SpanUpdateEvents, got {len(span_updates)}"

    # First emission should be the start (no end_time)
    start_event = span_updates[0]
    assert start_event.span_type == "tool"
    assert start_event.execution_id == "exec_1"

    # Last emission should be the end (with end_time and cache_hit)
    end_event = span_updates[-1]
    assert end_event.span_type == "tool"
    assert end_event.end_time is not None
    assert end_event.cache_hit is True


# ── Tool cache tests (Issue 5) ──────────────────────────────────


def _register_counting_skill() -> list[int]:
    """Register a web_search skill that counts invocations."""

    class InputSchema(BaseModel):
        query: str

    class OutputSchema(BaseModel):
        result: str

    call_count: list[int] = [0]

    async def executor(query: str) -> dict:
        call_count[0] += 1
        return {"result": f"ok:{query}"}

    skill = SkillSpec(
        name="web_search",
        description="Web search",
        input_schema=InputSchema,
        output_schema=OutputSchema,
    )
    skill.bind_executor(executor)
    DEFAULT_SKILL_REGISTRY.register(skill, overwrite=True)
    return call_count


@pytest.mark.asyncio
async def test_tool_cache_hit_skips_execution() -> None:
    """When tool_cache contains a matching entry, SkillExecutor should NOT be called."""
    call_count = _register_counting_skill()

    node = NodeIR(
        node_id="tool_1",
        node_type=NodeType.TOOL,
        position={"x": 0, "y": 0},
        config={"tool_name": "web_search"},
        inputs={"query": "cached_query"},
        outputs={},
        metadata={},
    )
    node_exec = NodeExecutionIR(node_id="tool_1")

    plan = PlanIR(plan_id="plan_1", version=1, nodes=[node], edges=[], entry_node_id="tool_1")
    execution = ExecutionIR(execution_id="exec_1", plan_id="plan_1")

    # Pre-populate tool cache with a matching entry
    import json

    cache_key = json.dumps(
        {"tool": "web_search", "args": {"query": "cached_query"}, "version": None}, sort_keys=True
    )
    tool_cache: dict[str, dict] = {cache_key: {"result": "cached_result", "metadata": {}}}

    context = ExecutionContext(
        session_id="session_1",
        execution=execution,
        plan=plan,
        tool_cache=tool_cache,
    )

    executor_instance = ToolNodeExecutor()
    await executor_instance.execute(context, node, node_exec)

    assert node_exec.error is None
    assert call_count[0] == 0, "SkillExecutor should NOT have been called on cache hit"
    assert node_exec.outputs["metadata"]["cache_hit"] is True
    assert node_exec.outputs["metadata"]["cache_key"] == cache_key


@pytest.mark.asyncio
async def test_tool_cache_miss_populates_cache() -> None:
    """On cache miss, result should be stored in tool_cache for future lookups."""
    call_count = _register_counting_skill()

    node = NodeIR(
        node_id="tool_1",
        node_type=NodeType.TOOL,
        position={"x": 0, "y": 0},
        config={"tool_name": "web_search"},
        inputs={"query": "new_query"},
        outputs={},
        metadata={},
    )
    node_exec = NodeExecutionIR(node_id="tool_1")

    plan = PlanIR(plan_id="plan_1", version=1, nodes=[node], edges=[], entry_node_id="tool_1")
    execution = ExecutionIR(execution_id="exec_1", plan_id="plan_1")

    tool_cache: dict[str, dict] = {}
    context = ExecutionContext(
        session_id="session_1",
        execution=execution,
        plan=plan,
        tool_cache=tool_cache,
    )

    executor_instance = ToolNodeExecutor()
    await executor_instance.execute(context, node, node_exec)

    assert node_exec.error is None
    assert call_count[0] == 1, "SkillExecutor should have been called on cache miss"
    assert len(tool_cache) == 1, "Cache should have one entry after miss"


@pytest.mark.asyncio
async def test_tool_cache_skipped_for_fresh_replay() -> None:
    """Fresh replay should bypass tool_cache by default (no cache hit)."""
    call_count = _register_counting_skill()

    node = NodeIR(
        node_id="tool_1",
        node_type=NodeType.TOOL,
        position={"x": 0, "y": 0},
        config={"tool_name": "web_search"},
        inputs={"query": "cached_query"},
        outputs={},
        metadata={},
    )
    node_exec = NodeExecutionIR(node_id="tool_1")

    plan = PlanIR(plan_id="plan_1", version=1, nodes=[node], edges=[], entry_node_id="tool_1")
    execution = ExecutionIR(
        execution_id="exec_1",
        plan_id="plan_1",
        metadata={"replay_mode": "fresh"},
    )

    # Pre-populate cache — should be ignored for fresh replay
    import json

    cache_key = json.dumps(
        {"tool": "web_search", "args": {"query": "cached_query"}, "version": None}, sort_keys=True
    )
    tool_cache: dict[str, dict] = {cache_key: {"result": "cached_result", "metadata": {}}}

    context = ExecutionContext(
        session_id="session_1",
        execution=execution,
        plan=plan,
        tool_cache=tool_cache,
    )

    executor_instance = ToolNodeExecutor()
    await executor_instance.execute(context, node, node_exec)

    assert node_exec.error is None
    assert call_count[0] == 1, "SkillExecutor SHOULD be called for fresh replay (cache bypassed)"


@pytest.mark.asyncio
async def test_emit_child_spans_emits_internal_sub_spans() -> None:
    """_emit_child_spans should recursively emit all child spans to observation service."""
    from unittest.mock import AsyncMock

    from houyi.infrastructure.observability import Span, SpanType

    # Build a tool span with nested children
    tool_span = Span(name="tool.web_search", span_type=SpanType.TOOL)
    child1 = Span(name="provider.ddg", parent=tool_span, span_type=SpanType.INTERNAL)
    child1.set_status("error", "timeout")
    child1.end()
    child2 = Span(name="provider.serper", parent=tool_span, span_type=SpanType.INTERNAL)
    child2.end()
    # Grandchild of child2
    grandchild = Span(name="fetch.jina", parent=child2, span_type=SpanType.INTERNAL)
    grandchild.end()

    mock_obs = AsyncMock()
    await ToolNodeExecutor._emit_child_spans(
        tool_span, mock_obs, session_id="s1", execution_id="e1"
    )

    # Should have emitted 3 spans: child1, child2, grandchild
    assert mock_obs.emit.call_count == 3
    emitted = [call.args[0] for call in mock_obs.emit.call_args_list]
    emitted_names = [e.name for e in emitted]
    assert "provider.ddg" in emitted_names
    assert "provider.serper" in emitted_names
    assert "fetch.jina" in emitted_names

    # Verify error span has error status
    ddg_event = next(e for e in emitted if e.name == "provider.ddg")
    assert ddg_event.status == "error"


@pytest.mark.asyncio
async def test_emit_child_spans_noop_without_children() -> None:
    """_emit_child_spans should be a no-op when span has no children."""
    from unittest.mock import AsyncMock

    from houyi.infrastructure.observability import Span, SpanType

    tool_span = Span(name="tool.web_search", span_type=SpanType.TOOL)
    mock_obs = AsyncMock()
    await ToolNodeExecutor._emit_child_spans(
        tool_span, mock_obs, session_id="s1", execution_id="e1"
    )
    assert mock_obs.emit.call_count == 0
