"""Tests for OpenTelemetry span instrumentation in ToolCallRunner.

Validates that ToolCallRunner.run() creates the correct span hierarchy:
  execution (root)
    ├── llm.call (per round)
    └── tool.execute (per tool call, with parallel_group_id)

Also covers SkillExecutor.execute() emitting a tool span, and
error status propagation on tool failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel

from houyi.application.tool_calling.runner import ToolCallRunner
from houyi.domain.skill.exceptions import SkillExecutionError
from houyi.domain.skill.spec import SkillSpec
from houyi.infrastructure.observability.trace_manager import Span, TraceManager
from houyi.infrastructure.observability.types import SpanType

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeResponse:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def model_copy(self, deep: bool = False) -> _FakeResponse:
        if not deep:
            return _FakeResponse(
                content=self.content,
                tool_calls=list(self.tool_calls),
                metadata=dict(self.metadata),
            )
        return _FakeResponse(
            content=self.content,
            tool_calls=json.loads(json.dumps(self.tool_calls)),
            metadata=json.loads(json.dumps(self.metadata)),
        )


class _FakeAdapter:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: int = 0
        self.model = "fake-model"
        self.base_url = "http://fake.local"

    async def chat(
        self, _messages: list[Any], tools: list[dict[str, Any]] | None = None, **_kw: Any
    ) -> _FakeResponse:
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return _FakeResponse(content="done", tool_calls=[])


class _EmptyInput(BaseModel):
    pass


class _SimpleOutput(BaseModel):
    ok: bool = True


class _DummyExecutor:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.max_retries = 1
        self.timeout = 0.01

    async def execute(self, skill: SkillSpec, args: dict[str, Any]) -> dict[str, Any]:
        if self.fail:
            raise SkillExecutionError(skill.name, "failed", original_error=RuntimeError("boom"))
        return {"ok": True}


def _make_skill(name: str) -> SkillSpec:
    return SkillSpec(
        name=name,
        description=f"test skill {name}",
        input_schema=_EmptyInput,
        output_schema=_SimpleOutput,
        executor=lambda _: _SimpleOutput(ok=True),
    )


def _collect_spans(root: Span) -> list[Span]:
    """Flatten span tree depth-first."""
    result = [root]
    for child in root.children:
        result.extend(_collect_spans(child))
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolCallRunnerSpans:
    """Verify ToolCallRunner emits correct OTel span hierarchy."""

    @pytest.mark.asyncio
    async def test_root_execution_span_created(self) -> None:
        """run() should wrap the entire loop in an EXECUTION span."""
        tm = TraceManager(enabled=True, exporters=[])
        runner = ToolCallRunner(trace_manager=tm)

        adapter = _FakeAdapter([_FakeResponse(content="hi", tool_calls=[])])
        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
            skills=[],
            executor=_DummyExecutor(),
            max_rounds=1,
        )

        assert len(tm.root_spans) == 1
        root = tm.root_spans[0]
        assert root.name == "tool_call_runner.run"
        assert root.span_type == SpanType.EXECUTION
        assert root.status == "ok"
        assert root.end_time is not None

    @pytest.mark.asyncio
    async def test_llm_call_span_per_round(self) -> None:
        """Each LLM adapter.chat() round should create a child LLM span."""
        skill = _make_skill("echo")
        tm = TraceManager(enabled=True, exporters=[])
        runner = ToolCallRunner(trace_manager=tm)

        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": "{}"},
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )
        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=_DummyExecutor(),
            max_rounds=3,
        )

        root = tm.root_spans[0]
        llm_spans = [c for c in root.children if c.span_type == SpanType.LLM]
        # Two rounds -> two LLM spans
        assert len(llm_spans) == 2
        for s in llm_spans:
            assert s.name == "llm.call"
            assert s.end_time is not None
            assert s.status == "ok"

    @pytest.mark.asyncio
    async def test_tool_execute_span_per_tool_call(self) -> None:
        """Each tool execution should create a child TOOL span."""
        skill = _make_skill("echo")
        tm = TraceManager(enabled=True, exporters=[])
        runner = ToolCallRunner(trace_manager=tm)

        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": "{}"},
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )
        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=_DummyExecutor(),
            max_rounds=3,
        )

        root = tm.root_spans[0]
        tool_spans = [c for c in root.children if c.span_type == SpanType.TOOL]
        assert len(tool_spans) == 1
        assert tool_spans[0].name == "tool.echo"
        assert tool_spans[0].tool_name == "echo"
        assert tool_spans[0].end_time is not None

    @pytest.mark.asyncio
    async def test_tool_span_carries_parallel_group_id(self) -> None:
        """Parallel tool spans should carry group_id attribute."""
        skills = [_make_skill(f"t{i}") for i in range(3)]
        tool_calls = [
            {"id": f"c{i}", "type": "function", "function": {"name": f"t{i}", "arguments": "{}"}}
            for i in range(3)
        ]
        tm = TraceManager(enabled=True, exporters=[])
        runner = ToolCallRunner(trace_manager=tm)

        adapter = _FakeAdapter([_FakeResponse(content="", tool_calls=tool_calls)])
        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[s.to_tool_schema() for s in skills],
            skills=skills,
            executor=_DummyExecutor(),
            max_rounds=1,
            chat_kwargs={"parallel_tool_calls": True},
        )

        root = tm.root_spans[0]
        tool_spans = [c for c in root.children if c.span_type == SpanType.TOOL]
        assert len(tool_spans) == 3
        for ts in tool_spans:
            assert ts.group_id == "round_1"

    @pytest.mark.asyncio
    async def test_tool_error_sets_span_status_error(self) -> None:
        """Failed tool execution should set TOOL span status to error."""
        skill = _make_skill("fail_tool")
        tm = TraceManager(enabled=True, exporters=[])
        runner = ToolCallRunner(trace_manager=tm)

        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "fail_tool", "arguments": "{}"},
                        }
                    ],
                ),
            ]
        )
        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=_DummyExecutor(fail=True),
            max_rounds=1,
        )

        root = tm.root_spans[0]
        tool_spans = [c for c in root.children if c.span_type == SpanType.TOOL]
        assert len(tool_spans) == 1
        assert tool_spans[0].status == "error"

    @pytest.mark.asyncio
    async def test_execution_span_status_ok_on_success(self) -> None:
        """Root execution span should be 'ok' when loop succeeds."""
        tm = TraceManager(enabled=True, exporters=[])
        runner = ToolCallRunner(trace_manager=tm)

        adapter = _FakeAdapter([_FakeResponse(content="done", tool_calls=[])])
        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            skills=[],
            executor=_DummyExecutor(),
            max_rounds=1,
        )

        root = tm.root_spans[0]
        assert root.status == "ok"

    @pytest.mark.asyncio
    async def test_span_trace_id_propagated(self) -> None:
        """All spans in a run should share the same trace_id."""
        skill = _make_skill("echo")
        tm = TraceManager(enabled=True, exporters=[])
        runner = ToolCallRunner(trace_manager=tm)

        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": "{}"},
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )
        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=_DummyExecutor(),
            max_rounds=3,
        )

        root = tm.root_spans[0]
        all_spans = _collect_spans(root)
        trace_ids = {s.trace_id for s in all_spans}
        assert len(trace_ids) == 1, f"Expected single trace_id, got {trace_ids}"

    @pytest.mark.asyncio
    async def test_llm_span_records_model_attribute(self) -> None:
        """LLM spans should record the adapter model name."""
        tm = TraceManager(enabled=True, exporters=[])
        runner = ToolCallRunner(trace_manager=tm)

        adapter = _FakeAdapter([_FakeResponse(content="done", tool_calls=[])])
        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            skills=[],
            executor=_DummyExecutor(),
            max_rounds=1,
        )

        root = tm.root_spans[0]
        llm_spans = [c for c in root.children if c.span_type == SpanType.LLM]
        assert len(llm_spans) == 1
        assert llm_spans[0].model == "fake-model"

    @pytest.mark.asyncio
    async def test_no_spans_when_trace_manager_is_none(self) -> None:
        """When trace_manager is None, run() should not create any spans."""
        runner = ToolCallRunner(trace_manager=None)
        adapter = _FakeAdapter([_FakeResponse(content="done", tool_calls=[])])

        # Should not raise
        response, trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            skills=[],
            executor=_DummyExecutor(),
            max_rounds=1,
        )
        assert response.content == "done"

    @pytest.mark.asyncio
    async def test_tool_span_attributes_include_tool_call_id(self) -> None:
        """TOOL spans should include tool_call_id in attributes."""
        skill = _make_skill("echo")
        tm = TraceManager(enabled=True, exporters=[])
        runner = ToolCallRunner(trace_manager=tm)

        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {"name": "echo", "arguments": "{}"},
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )
        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=_DummyExecutor(),
            max_rounds=3,
        )

        root = tm.root_spans[0]
        tool_spans = [c for c in root.children if c.span_type == SpanType.TOOL]
        assert tool_spans[0].attributes.get("tool.call_id") == "call_abc123"

    @pytest.mark.asyncio
    async def test_llm_cache_hit_sets_span_cache_hit(self) -> None:
        """When LLM cache is hit, the LLM span should have cache_hit=True."""
        tm = TraceManager(enabled=True, exporters=[])
        runner = ToolCallRunner(trace_manager=tm)

        adapter = _FakeAdapter([_FakeResponse(content="should_not_be_called")])

        # Pre-populate LLM cache
        cached_resp = _FakeResponse(content="cached")
        llm_cache: dict[str, Any] = {}
        key = runner._build_llm_cache_key(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            chat_kwargs={},
        )
        assert key is not None
        llm_cache[key] = cached_resp

        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            skills=[],
            executor=_DummyExecutor(),
            max_rounds=1,
            llm_cache=llm_cache,
        )

        root = tm.root_spans[0]
        llm_spans = [c for c in root.children if c.span_type == SpanType.LLM]
        assert len(llm_spans) == 1
        assert llm_spans[0].cache_hit is True
        assert llm_spans[0].attributes.get("llm.cache_hit") is True
