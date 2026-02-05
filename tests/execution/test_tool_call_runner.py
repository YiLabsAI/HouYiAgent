"""Tests for ToolCallRunner.

These tests focus on core tool-calling loop behavior, including:
- Normal tool execution
- Tool cache hits
- Tool not found / missing tool name errors
- LLM cache hits
- Fast-path placeholder resolution
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel

from houyi.core.skill import SkillSpec
from houyi.execution.skill_executor import SkillExecutionError
from houyi.execution.tool_call_runner import ToolCallRunner


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
        self, _messages: list[Any], tools: list[dict[str, Any]] | None = None, **_kwargs: Any
    ) -> _FakeResponse:
        self.calls += 1
        assert tools is None or isinstance(tools, list)
        if self._responses:
            return self._responses.pop(0)
        return _FakeResponse(content="done", tool_calls=[])


class _DummyExecutor:
    def __init__(self, *, fail: bool = False, timeout: bool = False) -> None:
        self.fail = fail
        self.simulate_timeout = timeout
        self.max_retries = 1
        self.timeout = 0.01

    async def execute(self, skill: SkillSpec, args: dict[str, Any]) -> dict[str, Any]:
        if self.simulate_timeout:
            raise SkillExecutionError(skill.name, "timeout", original_error=TimeoutError("timeout"))
        if self.fail:
            raise SkillExecutionError(skill.name, "failed", original_error=RuntimeError("boom"))
        if skill.name == "tool1":
            return {"value": 1}
        if skill.name == "tool2":
            return {"received": args.get("x")}
        return {"ok": True, **args}


class TestToolCallRunner:
    @pytest.mark.asyncio
    async def test_run_returns_immediately_without_tool_calls(self) -> None:
        adapter = _FakeAdapter([_FakeResponse(content="hello", tool_calls=[])])
        runner = ToolCallRunner()

        response, tool_trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            skills=[],
            executor=_DummyExecutor(),
            max_rounds=3,
        )

        assert response.content == "hello"
        assert tool_trace == []
        assert adapter.calls == 1

    @pytest.mark.asyncio
    async def test_tool_cache_hit_sets_metadata(self) -> None:
        class Input(BaseModel):
            q: int

        class Output(BaseModel):
            ok: bool

        skill = SkillSpec(
            name="echo",
            description="echo",
            input_schema=Input,
            output_schema=Output,
            executor=lambda input_data: Output(ok=True),
            metadata={"version": "v1"},
        )

        cached = {
            "call_id": "call_1",
            "raw": {"ok": True, "metadata": {}},
            "content": json.dumps({"ok": True}, ensure_ascii=True, sort_keys=True),
            "is_error": False,
            "metadata": {},
        }

        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": json.dumps({"q": 1})},
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )

        tool_cache: dict[str, dict[str, Any]] = {}
        runner = ToolCallRunner(trace_manager=None)
        cache_key = runner._build_tool_cache_key("echo", {"q": 1}, skill)
        assert cache_key is not None
        tool_cache[cache_key] = cached

        response, tool_trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=_DummyExecutor(),
            max_rounds=3,
            tool_cache=tool_cache,
        )

        assert response.content == "done"
        assert len(tool_trace) == 1
        result = tool_trace[0]["result"]
        assert result["metadata"]["cache_hit"] is True
        assert result["raw"]["metadata"]["cache_hit"] is True

    @pytest.mark.asyncio
    async def test_tool_not_found_and_tool_name_missing(self) -> None:
        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": None, "arguments": "{}"},
                        },
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {"name": "missing_tool", "arguments": "{}"},
                        },
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )

        runner = ToolCallRunner()
        response, tool_trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            skills=[],
            executor=_DummyExecutor(),
            max_rounds=2,
        )

        assert response.content == "done"
        assert len(tool_trace) == 2
        assert tool_trace[0]["result"]["raw"]["error"] == "tool_name_missing"
        assert tool_trace[1]["result"]["raw"]["error"].startswith("tool_not_found")

    @pytest.mark.asyncio
    async def test_llm_cache_hit_skips_adapter_chat(self) -> None:
        # First response is cached; adapter.chat should not be called.
        cached_response = _FakeResponse(content="cached", tool_calls=[])
        adapter = _FakeAdapter([_FakeResponse(content="should_not_be_used", tool_calls=[])])
        runner = ToolCallRunner()

        llm_cache: dict[str, Any] = {}
        key = runner._build_llm_cache_key(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            chat_kwargs={},
        )
        assert key is not None
        llm_cache[key] = cached_response

        response, tool_trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            skills=[],
            executor=_DummyExecutor(),
            max_rounds=1,
            llm_cache=llm_cache,
        )

        assert response.content == "cached"
        assert tool_trace == []
        assert adapter.calls == 0

    @pytest.mark.asyncio
    async def test_fast_path_placeholder_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOUYI_TOOLCALL_FAST_PATH", "1")

        class Input1(BaseModel):
            pass

        class Output1(BaseModel):
            value: int

        class Input2(BaseModel):
            x: int

        class Output2(BaseModel):
            received: int

        skill1 = SkillSpec(
            name="tool1",
            description="first tool",
            input_schema=Input1,
            output_schema=Output1,
            executor=lambda _input: Output1(value=1),
        )
        skill2 = SkillSpec(
            name="tool2",
            description="second tool",
            input_schema=Input2,
            output_schema=Output2,
            executor=lambda input_data: Output2(received=input_data.x),
        )

        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "tool1", "arguments": "{}"},
                        },
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {
                                "name": "tool2",
                                "arguments": json.dumps({"x": "$tool.tool1.value"}),
                            },
                        },
                    ],
                )
            ]
        )

        runner = ToolCallRunner()
        response, tool_trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[skill1.to_tool_schema(), skill2.to_tool_schema()],
            skills=[skill1, skill2],
            executor=_DummyExecutor(),
            max_rounds=2,
            chat_kwargs={"parallel_tool_calls": False},
        )

        assert response.tool_calls
        assert len(tool_trace) == 2
        # Placeholder should be resolved to 1.
        assert tool_trace[1]["args"]["x"] == 1


class _EmptyInput(BaseModel):
    """Empty input schema for testing."""
    pass


class _SimpleOutput(BaseModel):
    """Simple output schema for testing."""
    ok: bool = True


class TestToolCallRunnerMetrics:
    """Tests for ToolCallRunner metrics integration."""

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_success(self) -> None:
        """Test that metrics are recorded on successful tool execution."""
        from houyi.core.skill.metrics import MetricsStore

        skill = SkillSpec(
            name="test_tool",
            description="test",
            input_schema=_EmptyInput,
            output_schema=_SimpleOutput,
            executor=lambda _: _SimpleOutput(ok=True),
        )
        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {"id": "c1", "type": "function", "function": {"name": "test_tool", "arguments": "{}"}}
                    ],
                )
            ]
        )

        metrics_store = MetricsStore()
        runner = ToolCallRunner(metrics_store=metrics_store)

        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=_DummyExecutor(),
            max_rounds=1,
        )

        # Check metrics were recorded
        metrics = metrics_store.aggregate("test_tool")
        assert metrics is not None
        assert metrics.reliability.success_count >= 1
        assert metrics.reliability.error_count == 0
        assert metrics.latency.samples >= 1

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_failure(self) -> None:
        """Test that metrics are recorded on failed tool execution."""
        from houyi.core.skill.metrics import MetricsStore

        skill = SkillSpec(
            name="fail_tool",
            description="test",
            input_schema=_EmptyInput,
            output_schema=_SimpleOutput,
            executor=lambda _: _SimpleOutput(ok=True),
        )
        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {"id": "c1", "type": "function", "function": {"name": "fail_tool", "arguments": "{}"}}
                    ],
                )
            ]
        )

        metrics_store = MetricsStore()
        runner = ToolCallRunner(metrics_store=metrics_store)

        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=_DummyExecutor(fail=True),
            max_rounds=1,
        )

        # Check error metrics were recorded
        metrics = metrics_store.aggregate("fail_tool")
        assert metrics is not None
        assert metrics.reliability.error_count >= 1
        assert metrics.reliability.success_count == 0

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_timeout(self) -> None:
        """Test that timeout metrics are recorded correctly."""
        from houyi.core.skill.metrics import MetricsStore

        skill = SkillSpec(
            name="timeout_tool",
            description="test",
            input_schema=_EmptyInput,
            output_schema=_SimpleOutput,
            executor=lambda _: _SimpleOutput(ok=True),
        )
        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {"id": "c1", "type": "function", "function": {"name": "timeout_tool", "arguments": "{}"}}
                    ],
                )
            ]
        )

        metrics_store = MetricsStore()
        runner = ToolCallRunner(metrics_store=metrics_store)

        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=_DummyExecutor(timeout=True),
            max_rounds=1,
        )

        # Check timeout metrics were recorded
        metrics = metrics_store.aggregate("timeout_tool")
        assert metrics is not None
        assert metrics.reliability.timeout_count >= 1

    @pytest.mark.asyncio
    async def test_get_skill_metrics(self) -> None:
        """Test getting aggregated metrics for a skill."""
        from houyi.core.skill.metrics import MetricsStore

        skill = SkillSpec(
            name="get_metrics_tool",
            description="test",
            input_schema=_EmptyInput,
            output_schema=_SimpleOutput,
            executor=lambda _: _SimpleOutput(ok=True),
        )
        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {"id": "c1", "type": "function", "function": {"name": "get_metrics_tool", "arguments": "{}"}}
                    ],
                )
            ]
        )

        metrics_store = MetricsStore()
        runner = ToolCallRunner(metrics_store=metrics_store)

        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=_DummyExecutor(),
            max_rounds=1,
        )

        # Use runner's public API to get metrics
        metrics = runner.get_skill_metrics("get_metrics_tool")
        assert metrics is not None
        assert metrics.skill_name == "get_metrics_tool"

    @pytest.mark.asyncio
    async def test_get_all_skill_metrics(self) -> None:
        """Test getting metrics for all skills."""
        from houyi.core.skill.metrics import MetricsStore

        class Output1(BaseModel):
            value: int

        skill1 = SkillSpec(
            name="tool1",
            description="test",
            input_schema=_EmptyInput,
            output_schema=Output1,
            executor=lambda _: Output1(value=1),
        )
        skill2 = SkillSpec(
            name="tool2",
            description="test",
            input_schema=_EmptyInput,
            output_schema=_SimpleOutput,
            executor=lambda _: _SimpleOutput(ok=True),
        )
        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {"id": "c1", "type": "function", "function": {"name": "tool1", "arguments": "{}"}},
                        {"id": "c2", "type": "function", "function": {"name": "tool2", "arguments": "{}"}},
                    ],
                )
            ]
        )

        metrics_store = MetricsStore()
        runner = ToolCallRunner(metrics_store=metrics_store)

        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[skill1.to_tool_schema(), skill2.to_tool_schema()],
            skills=[skill1, skill2],
            executor=_DummyExecutor(),
            max_rounds=1,
        )

        all_metrics = runner.get_all_skill_metrics()
        assert "tool1" in all_metrics
        assert "tool2" in all_metrics

    def test_no_metrics_store_returns_none(self) -> None:
        """Test that methods return None/empty when no metrics_store is configured."""
        runner = ToolCallRunner()
        assert runner.get_skill_metrics("any_tool") is None
        assert runner.get_all_skill_metrics() == {}
