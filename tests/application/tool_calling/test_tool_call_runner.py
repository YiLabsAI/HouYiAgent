"""Tests for ToolCallRunner.

These tests focus on core tool-calling loop behavior, including:
- Normal tool execution
- Tool cache hits
- Tool not found / missing tool name errors
- LLM cache hits
- Fast-path placeholder resolution
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel

from houyi.application.tool_calling.budget import prepare_tool_loop_messages
from houyi.application.tool_calling.runner import ToolCallRunner
from houyi.domain.skill.exceptions import SkillExecutionError
from houyi.domain.skill.spec import SkillSpec
from houyi.infrastructure.config.env_config import (
    ENV_TOOLCALL_LOOP_MAX_MESSAGE_CHARS,
    ENV_TOOLCALL_LOOP_MAX_TOTAL_CHARS,
    ENV_TOOLCALL_RESULT_SUMMARY_ENABLED,
    ENV_TOOLCALL_RESULT_SUMMARY_MAX_CHARS,
    ENV_TOOLCALL_RESULT_SUMMARY_MAX_ITEMS,
    ENV_TOOLCALL_TIMING,
)


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
        self.chat_payloads: list[list[dict[str, Any]]] = []

    async def chat(
        self, _messages: list[Any], tools: list[dict[str, Any]] | None = None, **_kwargs: Any
    ) -> _FakeResponse:
        self.calls += 1
        self.chat_payloads.append(json.loads(json.dumps(_messages)))
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
    async def test_enforces_loop_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_TOOLCALL_LOOP_MAX_MESSAGE_CHARS, "200")
        monkeypatch.setenv(ENV_TOOLCALL_LOOP_MAX_TOTAL_CHARS, "500")

        class Input(BaseModel):
            q: str

        class Output(BaseModel):
            ok: bool

        skill = SkillSpec(
            name="echo",
            description="echo",
            input_schema=Input,
            output_schema=Output,
            executor=lambda input_data: Output(ok=bool(input_data.q)),
        )

        huge_args = json.dumps({"q": "x" * 5000})
        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": huge_args},
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )

        runner = ToolCallRunner()
        await runner.run(
            adapter=adapter,
            messages=[
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "u" * 800},
            ],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=_DummyExecutor(),
            max_rounds=2,
        )

        assert len(adapter.chat_payloads) == 2
        for payload in adapter.chat_payloads:
            total_chars = 0
            for msg in payload:
                content = str(msg.get("content") or "")
                assert len(content) <= 220
                total_chars += len(content)
                tool_calls = msg.get("tool_calls")
                if not isinstance(tool_calls, list):
                    continue
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    fn = call.get("function")
                    if not isinstance(fn, dict):
                        continue
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        assert len(args) <= 220
                        total_chars += len(args)
            assert total_chars <= 500

    @pytest.mark.asyncio
    async def test_preserves_history(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_TOOLCALL_LOOP_MAX_MESSAGE_CHARS, "200")
        monkeypatch.setenv(ENV_TOOLCALL_LOOP_MAX_TOTAL_CHARS, "500")

        class Input(BaseModel):
            q: str

        class Output(BaseModel):
            ok: bool

        skill = SkillSpec(
            name="echo",
            description="echo",
            input_schema=Input,
            output_schema=Output,
            executor=lambda input_data: Output(ok=bool(input_data.q)),
        )

        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": '{"q":"skill.md"}'},
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )

        messages = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "u" * 800},
        ]

        runner = ToolCallRunner()
        await runner.run(
            adapter=adapter,
            messages=messages,
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=_DummyExecutor(),
            max_rounds=2,
        )

        assert any(
            msg.get("role") == "assistant" and isinstance(msg.get("tool_calls"), list)
            for msg in messages
        )
        assert any(msg.get("role") == "tool" for msg in messages)

    @pytest.mark.asyncio
    async def test_uses_model_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_TOOLCALL_LOOP_MAX_MESSAGE_CHARS, raising=False)
        monkeypatch.delenv(ENV_TOOLCALL_LOOP_MAX_TOTAL_CHARS, raising=False)

        adapter = _FakeAdapter([_FakeResponse(content="done", tool_calls=[])])
        adapter.model = "unknown-model"

        runner = ToolCallRunner()
        await runner.run(
            adapter=adapter,
            messages=[
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "u" * 20_000},
            ],
            tools=[],
            skills=[],
            executor=_DummyExecutor(),
            max_rounds=1,
        )

        assert len(adapter.chat_payloads) == 1
        payload = adapter.chat_payloads[0]
        total_chars = sum(len(str(msg.get("content") or "")) for msg in payload)
        # unknown model -> DEFAULT_CONTEXT_WINDOW auto budget
        assert total_chars <= 9_000
        user_message = next(msg for msg in payload if msg.get("role") == "user")
        assert len(str(user_message.get("content") or "")) <= 1_050

    @pytest.mark.asyncio
    async def test_uses_prefixed_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_TOOLCALL_LOOP_MAX_MESSAGE_CHARS, raising=False)
        monkeypatch.delenv(ENV_TOOLCALL_LOOP_MAX_TOTAL_CHARS, raising=False)

        adapter = _FakeAdapter([_FakeResponse(content="done", tool_calls=[])])
        adapter.model = "Pro/moonshotai/Kimi-K2.5"

        runner = ToolCallRunner()
        await runner.run(
            adapter=adapter,
            messages=[
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "u" * 20_000},
            ],
            tools=[],
            skills=[],
            executor=_DummyExecutor(),
            max_rounds=1,
        )

        assert len(adapter.chat_payloads) == 1
        payload = adapter.chat_payloads[0]
        total_chars = sum(len(str(msg.get("content") or "")) for msg in payload)
        assert total_chars > 9_000

    @pytest.mark.asyncio
    async def test_records_tool_duration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_TOOLCALL_TIMING, raising=False)

        class Input(BaseModel):
            q: str

        class Output(BaseModel):
            ok: bool

        skill = SkillSpec(
            name="echo",
            description="echo",
            input_schema=Input,
            output_schema=Output,
            executor=lambda input_data: Output(ok=bool(input_data.q)),
        )
        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": '{"q":"hello"}'},
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )

        response, tool_trace = await ToolCallRunner().run(
            adapter=adapter,
            messages=[{"role": "user", "content": "run echo"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=_DummyExecutor(),
            max_rounds=2,
        )

        assert response.content == "done"
        assert len(tool_trace) == 1
        assert tool_trace[0]["tool_call_id"] == "call_1"
        assert isinstance(tool_trace[0].get("duration_ms"), (int, float))
        assert tool_trace[0]["duration_ms"] > 0

    @pytest.mark.asyncio
    async def test_summarizes_large_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_TOOLCALL_LOOP_MAX_MESSAGE_CHARS, "20000")
        monkeypatch.setenv(ENV_TOOLCALL_LOOP_MAX_TOTAL_CHARS, "100000")
        monkeypatch.setenv(ENV_TOOLCALL_RESULT_SUMMARY_ENABLED, "1")
        monkeypatch.setenv(ENV_TOOLCALL_RESULT_SUMMARY_MAX_CHARS, "600")
        monkeypatch.setenv(ENV_TOOLCALL_RESULT_SUMMARY_MAX_ITEMS, "5")

        class Input(BaseModel):
            q: str

        class Output(BaseModel):
            ok: bool

        skill = SkillSpec(
            name="echo",
            description="echo",
            input_schema=Input,
            output_schema=Output,
            executor=lambda input_data: Output(ok=bool(input_data.q)),
        )

        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": json.dumps({"q": "x"})},
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )

        class _LargeResultExecutor(_DummyExecutor):
            async def execute(self, skill: SkillSpec, args: dict[str, Any]) -> dict[str, Any]:
                _ = (skill, args)
                return {
                    "data": {
                        "items": [{"idx": i, "payload": "y" * 500} for i in range(200)],
                    }
                }

        runner = ToolCallRunner()
        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=_LargeResultExecutor(),
            max_rounds=2,
        )

        assert len(adapter.chat_payloads) == 2
        second_round_payload = adapter.chat_payloads[-1]
        tool_message = next(msg for msg in second_round_payload if msg.get("role") == "tool")
        assert len(tool_message["content"]) < 5000
        assert (
            "[truncated" in tool_message["content"]
            or '"_truncated": true' in tool_message["content"].lower()
        )


def test_prepare_drops_group() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "demo_a", "arguments": "{}"},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "demo_b", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "content": '{"ok":true}',
            "tool_call_id": "call_1",
            "name": "demo_a",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "demo_c", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"cwd":"/tmp"}',
            "tool_call_id": "call_3",
            "name": "demo_c",
        },
    ]

    prepared = prepare_tool_loop_messages(messages, max_message_chars=12_000, max_total_chars=8_000)

    assert len(prepared) == 2
    assert prepared[0]["role"] == "assistant"
    assert prepared[0]["tool_calls"][0]["id"] == "call_3"
    assert prepared[1]["role"] == "tool"
    assert prepared[1]["tool_call_id"] == "call_3"


class TestToolCallRunnerBehavior:
    @pytest.mark.asyncio
    async def test_run_returns(self) -> None:
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
    async def test_tool_cache_sets(self) -> None:
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
    async def test_tool_missing_name(self) -> None:
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
    async def test_llm_cache_skips(self) -> None:
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
    async def test_fast_path_resolves(self, monkeypatch: pytest.MonkeyPatch) -> None:
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

    @pytest.mark.asyncio
    async def test_parallel_respects_limit(self) -> None:
        class _TrackingExecutor:
            def __init__(self) -> None:
                self.in_flight = 0
                self.peak_in_flight = 0
                self._lock = asyncio.Lock()

            async def execute(self, _skill: SkillSpec, _args: dict[str, Any]) -> dict[str, Any]:
                async with self._lock:
                    self.in_flight += 1
                    self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
                try:
                    await asyncio.sleep(0.02)
                    return {"ok": True}
                finally:
                    async with self._lock:
                        self.in_flight -= 1

        skills: list[SkillSpec] = []
        tool_calls: list[dict[str, Any]] = []
        for idx in range(3):
            name = f"tool_{idx}"
            skills.append(
                SkillSpec(
                    name=name,
                    description="parallel test",
                    input_schema=_EmptyInput,
                    output_schema=_SimpleOutput,
                    executor=lambda _: _SimpleOutput(ok=True),
                )
            )
            tool_calls.append(
                {
                    "id": f"c{idx}",
                    "type": "function",
                    "function": {"name": name, "arguments": "{}"},
                }
            )

        adapter = _FakeAdapter([_FakeResponse(content="", tool_calls=tool_calls)])
        tracker = _TrackingExecutor()
        runner = ToolCallRunner()

        _, tool_trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[skill.to_tool_schema() for skill in skills],
            skills=skills,
            executor=tracker,
            max_rounds=1,
            chat_kwargs={"parallel_tool_calls": True, "max_parallel_calls": 1},
        )

        assert tracker.peak_in_flight == 1
        assert len(tool_trace) == 3
        assert {entry.get("parallel_group_id") for entry in tool_trace} == {"round_1"}

    @pytest.mark.asyncio
    async def test_invalid_parallel_default(self) -> None:
        class _TrackingExecutor:
            def __init__(self) -> None:
                self.in_flight = 0
                self.peak_in_flight = 0
                self._lock = asyncio.Lock()

            async def execute(self, _skill: SkillSpec, _args: dict[str, Any]) -> dict[str, Any]:
                async with self._lock:
                    self.in_flight += 1
                    self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
                try:
                    await asyncio.sleep(0.02)
                    return {"ok": True}
                finally:
                    async with self._lock:
                        self.in_flight -= 1

        skills: list[SkillSpec] = []
        tool_calls: list[dict[str, Any]] = []
        for idx in range(3):
            name = f"tool_invalid_{idx}"
            skills.append(
                SkillSpec(
                    name=name,
                    description="parallel test",
                    input_schema=_EmptyInput,
                    output_schema=_SimpleOutput,
                    executor=lambda _: _SimpleOutput(ok=True),
                )
            )
            tool_calls.append(
                {
                    "id": f"cx{idx}",
                    "type": "function",
                    "function": {"name": name, "arguments": "{}"},
                }
            )

        adapter = _FakeAdapter([_FakeResponse(content="", tool_calls=tool_calls)])
        tracker = _TrackingExecutor()
        runner = ToolCallRunner()

        _, tool_trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[skill.to_tool_schema() for skill in skills],
            skills=skills,
            executor=tracker,
            max_rounds=1,
            chat_kwargs={"parallel_tool_calls": True, "max_parallel_calls": "bad-value"},
        )

        # Fallback default is 5, with 3 tool calls all can run concurrently.
        assert tracker.peak_in_flight >= 2
        assert len(tool_trace) == 3
        assert {entry.get("parallel_group_id") for entry in tool_trace} == {"round_1"}

    @pytest.mark.asyncio
    async def test_before_hook_patches(self) -> None:
        class _PatchArgsHook:
            async def before_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
                return {"args": {"patched": tool_call["args"]["value"] + 1}}

        class _CapturingExecutor(_DummyExecutor):
            def __init__(self) -> None:
                super().__init__()
                self.calls: list[dict[str, Any]] = []

            async def execute(self, skill: SkillSpec, args: dict[str, Any]) -> dict[str, Any]:
                self.calls.append({"skill": skill.name, "args": dict(args)})
                return {"patched": args["patched"]}

        class _Input(BaseModel):
            patched: int

        class _Output(BaseModel):
            patched: int

        skill = SkillSpec(
            name="patch_args",
            description="patch args",
            input_schema=_Input,
            output_schema=_Output,
            executor=lambda input_data: _Output(patched=input_data.patched),
        )
        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_patch",
                            "type": "function",
                            "function": {
                                "name": "patch_args",
                                "arguments": json.dumps({"value": 2}),
                            },
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )

        executor = _CapturingExecutor()
        runner = ToolCallRunner()
        _, tool_trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=executor,
            max_rounds=2,
            tool_hooks=[_PatchArgsHook()],
        )

        assert executor.calls == [{"skill": "patch_args", "args": {"patched": 3}}]
        assert tool_trace[0]["args"] == {"patched": 3}

    @pytest.mark.asyncio
    async def test_before_hook_records(self) -> None:
        class _ReplaceHook:
            async def before_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
                return {"tool_name": "tool2"}

        class _CapturingExecutor(_DummyExecutor):
            def __init__(self) -> None:
                super().__init__()
                self.executed_skills: list[str] = []

            async def execute(self, skill: SkillSpec, args: dict[str, Any]) -> dict[str, Any]:
                self.executed_skills.append(skill.name)
                return {"executed_skill": skill.name, **args}

        class _Input(BaseModel):
            pass

        class _Output(BaseModel):
            source: str

        skill1 = SkillSpec(
            name="tool1",
            description="tool1",
            input_schema=_Input,
            output_schema=_Output,
            executor=lambda _input: _Output(source="tool1"),
        )
        skill2 = SkillSpec(
            name="tool2",
            description="tool2",
            input_schema=_Input,
            output_schema=_Output,
            executor=lambda _input: _Output(source="tool2"),
        )
        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_replace_blocked",
                            "type": "function",
                            "function": {"name": "tool1", "arguments": "{}"},
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )

        executor = _CapturingExecutor()
        runner = ToolCallRunner()
        _, tool_trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[skill1.to_tool_schema(), skill2.to_tool_schema()],
            skills=[skill1, skill2],
            executor=executor,
            max_rounds=2,
            tool_hooks=[_ReplaceHook()],
            allow_tool_replace=False,
        )

        assert executor.executed_skills == ["tool1"]
        assert tool_trace[0]["tool_name"] == "tool1"
        assert tool_trace[0]["tool_override"] == {
            "from": "tool1",
            "to": "tool2",
            "allowed": False,
            "applied": False,
        }
        assert tool_trace[0]["result"]["raw"]["executed_skill"] == "tool1"

    @pytest.mark.asyncio
    async def test_before_hook_replaces(self) -> None:
        class _ReplaceHook:
            async def before_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
                return {"tool_name": "tool2", "args": {"from_hook": True}}

        class _CapturingExecutor(_DummyExecutor):
            def __init__(self) -> None:
                super().__init__()
                self.executed_skills: list[str] = []

            async def execute(self, skill: SkillSpec, args: dict[str, Any]) -> dict[str, Any]:
                self.executed_skills.append(skill.name)
                return {"executed_skill": skill.name, **args}

        class _Input(BaseModel):
            from_hook: bool = False

        class _Output(BaseModel):
            source: str

        skill1 = SkillSpec(
            name="tool1",
            description="tool1",
            input_schema=_Input,
            output_schema=_Output,
            executor=lambda _input: _Output(source="tool1"),
        )
        skill2 = SkillSpec(
            name="tool2",
            description="tool2",
            input_schema=_Input,
            output_schema=_Output,
            executor=lambda _input: _Output(source="tool2"),
        )
        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_replace_allowed",
                            "type": "function",
                            "function": {"name": "tool1", "arguments": "{}"},
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )

        executor = _CapturingExecutor()
        runner = ToolCallRunner()
        _, tool_trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[skill1.to_tool_schema(), skill2.to_tool_schema()],
            skills=[skill1, skill2],
            executor=executor,
            max_rounds=2,
            tool_hooks=[_ReplaceHook()],
            allow_tool_replace=True,
        )

        assert executor.executed_skills == ["tool2"]
        assert tool_trace[0]["tool_name"] == "tool2"
        assert tool_trace[0]["args"] == {"from_hook": True}
        assert tool_trace[0]["tool_override"] == {
            "from": "tool1",
            "to": "tool2",
            "allowed": True,
            "applied": True,
        }
        assert tool_trace[0]["result"]["raw"]["executed_skill"] == "tool2"

    @pytest.mark.asyncio
    async def test_pre_hook_fires(self) -> None:
        from houyi.domain.skill.hooks import HookEvent, HookType, SkillHook, SkillHooksManager

        seen: list[dict[str, Any]] = []

        async def on_pre_tool_use(ctx: Any) -> dict[str, Any]:
            seen.append(
                {
                    "tool_name": ctx.tool_name,
                    "tool_args": dict(ctx.tool_args),
                    "skill_name": ctx.skill_name,
                }
            )
            return {"success": True, "output": "noted"}

        hooks = SkillHooksManager()
        hooks.register_hooks(
            SkillSpec(
                name="hooked",
                description="hooked skill hooks",
                input_schema=_EmptyInput,
                output_schema=_SimpleOutput,
                hooks=[
                    SkillHook(
                        event=HookEvent.PRE_TOOL_USE,
                        hook_type=HookType.HANDLER,
                        handler=on_pre_tool_use,
                    )
                ],
            )
        )

        skill = SkillSpec(
            name="hooked",
            description="hooked tool",
            input_schema=_EmptyInput,
            output_schema=_SimpleOutput,
            executor=lambda _: _SimpleOutput(ok=True),
        )
        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_pre_tool_use",
                            "type": "function",
                            "function": {"name": "hooked", "arguments": "{}"},
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )

        runner = ToolCallRunner(skill_hooks_manager=hooks)
        response, tool_trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=_DummyExecutor(),
            max_rounds=2,
        )

        assert response.content == "done"
        assert len(tool_trace) == 1
        assert seen == [{"tool_name": "hooked", "tool_args": {}, "skill_name": "hooked"}]


class _EmptyInput(BaseModel):
    """Empty input schema for testing."""

    pass


class _SimpleOutput(BaseModel):
    """Simple output schema for testing."""

    ok: bool = True


class TestToolCallRunnerMetrics:
    """Tests for ToolCallRunner metrics integration."""

    @pytest.mark.asyncio
    async def test_metrics_success(self) -> None:
        """Test that metrics are recorded on successful tool execution."""
        from houyi.domain.skill.metrics import MetricsStore

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
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "test_tool", "arguments": "{}"},
                        }
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
    async def test_metrics_failure(self) -> None:
        """Test that metrics are recorded on failed tool execution."""
        from houyi.domain.skill.metrics import MetricsStore

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
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "fail_tool", "arguments": "{}"},
                        }
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
    async def test_metrics_timeout(self) -> None:
        """Test that timeout metrics are recorded correctly."""
        from houyi.domain.skill.metrics import MetricsStore

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
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "timeout_tool", "arguments": "{}"},
                        }
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
        from houyi.domain.skill.metrics import MetricsStore

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
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "get_metrics_tool", "arguments": "{}"},
                        }
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
    async def test_get_all_metrics(self) -> None:
        """Test getting metrics for all skills."""
        from houyi.domain.skill.metrics import MetricsStore

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
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "tool1", "arguments": "{}"},
                        },
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {"name": "tool2", "arguments": "{}"},
                        },
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

    def test_no_metrics_store(self) -> None:
        """Test that methods return None/empty when no metrics_store is configured."""
        runner = ToolCallRunner()
        assert runner.get_skill_metrics("any_tool") is None
        assert runner.get_all_skill_metrics() == {}
