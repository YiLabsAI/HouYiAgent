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
import inspect
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from houyi.application.tool_calling.budget import (
    MessageBudget,
    prepare_tool_loop_messages,
    resolve_tool_loop_budget_chars,
)
from houyi.application.tool_calling.runner import ToolCallRunner
from houyi.application.tool_calling.tool_bridge import ToolBridge
from houyi.domain.skill.exceptions import SkillExecutionError
from houyi.domain.skill.registry import SkillRegistry
from houyi.domain.skill.spec import SkillSpec
from houyi.infrastructure.config.env_config import (
    ENV_TOOLCALL_LOOP_MAX_MESSAGE_CHARS,
    ENV_TOOLCALL_LOOP_MAX_TOTAL_CHARS,
    ENV_TOOLCALL_RESULT_SUMMARY_ENABLED,
    ENV_TOOLCALL_RESULT_SUMMARY_MAX_CHARS,
    ENV_TOOLCALL_RESULT_SUMMARY_MAX_ITEMS,
    ENV_TOOLCALL_TIMING,
)
from houyi.skills.builtin import local_tools


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
        self.tool_payloads: list[list[dict[str, Any]] | None] = []

    async def chat(
        self, _messages: list[Any], tools: list[dict[str, Any]] | None = None, **_kwargs: Any
    ) -> _FakeResponse:
        self.calls += 1
        self.chat_payloads.append(json.loads(json.dumps(_messages)))
        self.tool_payloads.append(
            json.loads(json.dumps(tools)) if isinstance(tools, list) else None
        )
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


class _DirectSkillExecutor:
    def __init__(self) -> None:
        self.max_retries = 1
        self.timeout = 10.0

    async def execute(self, skill: SkillSpec, args: dict[str, Any]) -> dict[str, Any]:
        result = skill.executor(**args)
        if inspect.isawaitable(result):
            result = await result
        return result


class _BudgetAdapter:
    def __init__(self, model: str) -> None:
        self.model = model


def _serialized_chars(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _collect_trace_metrics(
    *, tool_trace: list[dict[str, Any]], tool_payloads: list[list[dict[str, Any]] | None]
) -> dict[str, float]:
    schema_chars_total = sum(
        _serialized_chars(payload) for payload in tool_payloads if isinstance(payload, list)
    )
    prompt_payload_chars_total = sum(
        _serialized_chars(payload) for payload in tool_payloads if isinstance(payload, list)
    )
    completion_payload_chars_total = 0
    tool_duration_values = [
        float(entry.get("duration_ms") or 0.0)
        for entry in tool_trace
        if isinstance(entry.get("duration_ms"), (int, float))
    ]
    for payload in tool_payloads:
        if not isinstance(payload, list):
            continue
        assistant_messages = [
            message
            for message in payload
            if isinstance(message, dict) and message.get("role") == "assistant"
        ]
        if assistant_messages:
            completion_payload_chars_total += _serialized_chars(assistant_messages[-1])
    return {
        "llm_round_count": len(tool_payloads),
        "tool_round_count": len(
            {
                entry.get("round_index")
                for entry in tool_trace
                if isinstance(entry.get("round_index"), int)
            }
        ),
        "tool_call_count": len(tool_trace),
        "raw_payload_chars_total": sum(
            int(entry.get("raw_payload_chars") or 0) for entry in tool_trace
        ),
        "presented_content_chars_total": sum(
            int(entry.get("presented_content_chars") or 0) for entry in tool_trace
        ),
        "schema_chars_total": schema_chars_total,
        "prompt_payload_chars_total": prompt_payload_chars_total,
        "completion_payload_chars_total": completion_payload_chars_total,
        "prompt_token_proxy": prompt_payload_chars_total / 4.0,
        "completion_token_proxy": completion_payload_chars_total / 4.0,
        "total_token_proxy": (prompt_payload_chars_total + completion_payload_chars_total) / 4.0,
        "tool_duration_ms_total": sum(tool_duration_values),
        "tool_duration_ms_avg": (
            sum(tool_duration_values) / len(tool_duration_values) if tool_duration_values else 0.0
        ),
    }


def _projected_cli_bundle(schema_exposure: str) -> tuple[list[dict[str, Any]], list[SkillSpec]]:
    registry = SkillRegistry()
    for skill in local_tools.build_builtin_local_tools():
        if skill.name == "houyi_local_cli":
            registry.register(skill)
            break
    bridge = ToolBridge(registry)
    skills = bridge.collect_skills(
        skill_filter=["houyi_local_cli"],
        schema_exposure=schema_exposure,
    )
    tools = bridge.collect_tool_schemas(
        skill_filter=["houyi_local_cli"],
        schema_exposure=schema_exposure,
    )
    return tools, skills


def _bridge_skill_bundle(
    *, skill_name: str, schema_exposure: str
) -> tuple[list[dict[str, Any]], list[SkillSpec]]:
    registry = SkillRegistry()
    for skill in local_tools.build_builtin_local_tools():
        if skill.name == skill_name:
            registry.register(skill)
            break
    bridge = ToolBridge(registry)
    skills = bridge.collect_skills(
        skill_filter=[skill_name],
        schema_exposure=schema_exposure,
    )
    tools = bridge.collect_tool_schemas(
        skill_filter=[skill_name],
        schema_exposure=schema_exposure,
    )
    return tools, skills


class TestToolCallRunner:
    async def _run_pipeline_lane(
        self,
        *,
        runner: ToolCallRunner,
        adapter: _FakeAdapter,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        skills: list[SkillSpec],
    ) -> tuple[_FakeResponse, list[dict[str, Any]], dict[str, float]]:
        started_at = time.perf_counter()
        response, tool_trace = await runner.run(
            adapter=adapter,
            messages=messages,
            tools=tools,
            skills=skills,
            executor=_DirectSkillExecutor(),
            max_rounds=5,
        )
        metrics = _collect_trace_metrics(tool_trace=tool_trace, tool_payloads=adapter.tool_payloads)
        metrics["wall_time_ms"] = (time.perf_counter() - started_at) * 1000.0
        return response, tool_trace, metrics

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

    def test_budget_tightens_deepseek(self) -> None:
        generic = resolve_tool_loop_budget_chars(_BudgetAdapter("gpt-4o"), None, None, "gpt-4o")
        deepseek = resolve_tool_loop_budget_chars(
            _BudgetAdapter("deepseek-ai/DeepSeek-R1"),
            None,
            None,
            "deepseek-ai/DeepSeek-R1",
        )

        assert deepseek[0] < generic[0]
        assert deepseek[1] < generic[1]

    def test_budget_keeps_override(self) -> None:
        message_chars, total_chars = resolve_tool_loop_budget_chars(
            _BudgetAdapter("deepseek-ai/DeepSeek-R1"),
            7_500,
            90_000,
            "deepseek-ai/DeepSeek-R1",
        )

        assert message_chars == 7_500
        assert total_chars == 90_000

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
        assert tool_trace[0]["status"] == "ok"
        assert tool_trace[0]["raw_payload_chars"] > 0
        assert tool_trace[0]["presented_content_chars"] > 0
        assert tool_trace[0]["presentation"]["footer_attached"] is True

    @pytest.mark.asyncio
    async def test_get_final_answer(self) -> None:
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
                _FakeResponse(
                    content="Found the local skill.md file and added the related web details.",
                    tool_calls=[],
                ),
            ]
        )

        messages = [
            {"role": "user", "content": "Find the local skill.md file and add related web details"}
        ]
        response, tool_trace = await ToolCallRunner().run(
            adapter=adapter,
            messages=messages,
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=_DummyExecutor(),
            max_rounds=2,
        )

        assert adapter.calls == 2
        assert response.tool_calls == []
        assert (
            response.content == "Found the local skill.md file and added the related web details."
        )
        assert len(tool_trace) == 1
        assert any(msg.get("role") == "tool" for msg in messages)

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
        assert len(str(tool_message["content"])) < 5000
        assert (
            "[truncated" in str(tool_message["content"])
            or '"_truncated": true' in str(tool_message["content"]).lower()
        )

    @pytest.mark.asyncio
    async def test_local_cli_large_read(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HOUYI_WORKSPACE_ROOT", str(tmp_path))
        monkeypatch.setenv(ENV_TOOLCALL_LOOP_MAX_MESSAGE_CHARS, "40000")
        monkeypatch.setenv(ENV_TOOLCALL_LOOP_MAX_TOTAL_CHARS, "120000")
        monkeypatch.setenv(ENV_TOOLCALL_RESULT_SUMMARY_ENABLED, "1")
        monkeypatch.setenv(ENV_TOOLCALL_RESULT_SUMMARY_MAX_CHARS, "600")
        monkeypatch.setenv(ENV_TOOLCALL_RESULT_SUMMARY_MAX_ITEMS, "5")

        target = tmp_path / "docs" / "large.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(f"line-{idx}: {'x' * 120}\n" for idx in range(2500)), encoding="utf-8"
        )

        local_cli_skill = next(
            skill
            for skill in local_tools.build_builtin_local_tools()
            if skill.name == "houyi_local_cli"
        )
        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_local_read_large",
                            "type": "function",
                            "function": {
                                "name": "houyi_local_cli",
                                "arguments": json.dumps(
                                    {"command": "read", "path": "docs/large.txt"}
                                ),
                            },
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )

        response, tool_trace = await ToolCallRunner().run(
            adapter=adapter,
            messages=[{"role": "user", "content": "read the large local file"}],
            tools=[local_cli_skill.to_tool_schema()],
            skills=[local_cli_skill],
            executor=_DirectSkillExecutor(),
            max_rounds=2,
        )

        assert response.content == "done"
        assert len(tool_trace) == 1
        assert tool_trace[0]["tool_name"] == "houyi_local_cli"
        assert tool_trace[0]["status"] == "ok"
        assert tool_trace[0]["result_summarized"] is True
        assert tool_trace[0]["result_artifact_candidate"] is True
        assert tool_trace[0]["presentation"]["result_summarized"] is True
        assert tool_trace[0]["presentation"]["result_artifact_candidate"] is True
        assert tool_trace[0]["raw_payload_chars"] > tool_trace[0]["presented_content_chars"]

        second_round_payload = adapter.chat_payloads[-1]
        tool_message = next(msg for msg in second_round_payload if msg.get("role") == "tool")
        assert len(str(tool_message["content"])) < 2000
        assert (
            "[truncated" in str(tool_message["content"])
            or '"_truncated": true' in str(tool_message["content"]).lower()
        )

    @pytest.mark.asyncio
    async def test_local_cli_invalid_grep(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HOUYI_WORKSPACE_ROOT", str(tmp_path))

        local_cli_skill = next(
            skill
            for skill in local_tools.build_builtin_local_tools()
            if skill.name == "houyi_local_cli"
        )
        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_local_grep_invalid",
                            "type": "function",
                            "function": {
                                "name": "houyi_local_cli",
                                "arguments": json.dumps({"command": "grep", "path": "."}),
                            },
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )

        response, tool_trace = await ToolCallRunner().run(
            adapter=adapter,
            messages=[{"role": "user", "content": "grep the workspace"}],
            tools=[local_cli_skill.to_tool_schema()],
            skills=[local_cli_skill],
            executor=_DirectSkillExecutor(),
            max_rounds=2,
        )

        assert response.content == "done"
        assert len(tool_trace) == 1
        assert tool_trace[0]["result"]["raw"]["success"] is False
        assert "query is required" in tool_trace[0]["result"]["raw"]["message"]
        assert tool_trace[0]["status"] == "ok"
        assert tool_trace[0]["presentation"]["error_detail_attached"] is False
        assert tool_trace[0]["presentation"]["recovery_guidance_attached"] is False

    @pytest.mark.asyncio
    async def test_local_cli_missing_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HOUYI_WORKSPACE_ROOT", str(tmp_path))

        local_cli_skill = next(
            skill
            for skill in local_tools.build_builtin_local_tools()
            if skill.name == "houyi_local_cli"
        )
        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_local_read_missing",
                            "type": "function",
                            "function": {
                                "name": "houyi_local_cli",
                                "arguments": json.dumps(
                                    {"command": "read", "path": "docs/missing.txt"}
                                ),
                            },
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )

        response, tool_trace = await ToolCallRunner().run(
            adapter=adapter,
            messages=[{"role": "user", "content": "read a missing file"}],
            tools=[local_cli_skill.to_tool_schema()],
            skills=[local_cli_skill],
            executor=_DirectSkillExecutor(),
            max_rounds=2,
        )

        assert response.content == "done"
        assert len(tool_trace) == 1
        assert tool_trace[0]["result"]["raw"]["success"] is False
        assert "File not found" in tool_trace[0]["result"]["raw"]["message"]
        assert tool_trace[0]["status"] == "ok"
        assert tool_trace[0]["presentation"]["error_detail_attached"] is False
        assert tool_trace[0]["presentation"]["recovery_guidance_attached"] is False

    @pytest.mark.asyncio
    async def test_local_cli_large_grep(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HOUYI_WORKSPACE_ROOT", str(tmp_path))
        monkeypatch.setenv(ENV_TOOLCALL_LOOP_MAX_MESSAGE_CHARS, "40000")
        monkeypatch.setenv(ENV_TOOLCALL_LOOP_MAX_TOTAL_CHARS, "120000")
        monkeypatch.setenv(ENV_TOOLCALL_RESULT_SUMMARY_ENABLED, "1")
        monkeypatch.setenv(ENV_TOOLCALL_RESULT_SUMMARY_MAX_CHARS, "600")
        monkeypatch.setenv(ENV_TOOLCALL_RESULT_SUMMARY_MAX_ITEMS, "5")

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(60):
            (docs_dir / f"match_{idx}.txt").write_text(
                "".join(f"alpha result line {line_idx} {'y' * 80}\n" for line_idx in range(8)),
                encoding="utf-8",
            )

        local_cli_skill = next(
            skill
            for skill in local_tools.build_builtin_local_tools()
            if skill.name == "houyi_local_cli"
        )
        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_local_grep_large",
                            "type": "function",
                            "function": {
                                "name": "houyi_local_cli",
                                "arguments": json.dumps(
                                    {"command": "grep", "path": "docs", "query": "alpha"}
                                ),
                            },
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )

        response, tool_trace = await ToolCallRunner().run(
            adapter=adapter,
            messages=[{"role": "user", "content": "grep a lot of local files"}],
            tools=[local_cli_skill.to_tool_schema()],
            skills=[local_cli_skill],
            executor=_DirectSkillExecutor(),
            max_rounds=2,
        )

        assert response.content == "done"
        assert len(tool_trace) == 1
        assert tool_trace[0]["status"] == "ok"
        assert tool_trace[0]["result"]["raw"]["success"] is True
        assert tool_trace[0]["result"]["raw"]["data"]["truncated"] is True
        assert tool_trace[0]["result_summarized"] is True
        assert tool_trace[0]["result_artifact_candidate"] is True
        assert tool_trace[0]["raw_payload_chars"] > tool_trace[0]["presented_content_chars"]

        second_round_payload = adapter.chat_payloads[-1]
        tool_message = next(msg for msg in second_round_payload if msg.get("role") == "tool")
        assert len(str(tool_message["content"])) < 2000
        assert (
            "[truncated" in str(tool_message["content"])
            or '"_truncated": true' in str(tool_message["content"]).lower()
        )

    @pytest.mark.asyncio
    async def test_pipeline_list(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HOUYI_WORKSPACE_ROOT", str(tmp_path))

        skills_dir = tmp_path / "houyi" / "skills"
        (skills_dir / "web_search").mkdir(parents=True, exist_ok=True)
        (skills_dir / "planning").mkdir(parents=True, exist_ok=True)
        (skills_dir / "weather").mkdir(parents=True, exist_ok=True)
        (skills_dir / "web_search" / "SKILL.md").write_text(
            "# Web Search\nweb search local pipeline target\nline-3\nline-4\n",
            encoding="utf-8",
        )
        (skills_dir / "planning" / "SKILL.md").write_text(
            "# Planning\nplanning skill\n",
            encoding="utf-8",
        )
        (skills_dir / "weather" / "SKILL.md").write_text(
            "# Weather\nweather skill\n",
            encoding="utf-8",
        )

        skill_map = {skill.name: skill for skill in local_tools.build_builtin_local_tools()}
        typed_skills = [
            skill_map["houyi_list_dir"],
            skill_map["houyi_find_files"],
            skill_map["houyi_grep"],
            skill_map["houyi_read_file"],
        ]
        local_cli_skill = skill_map["houyi_local_cli"]
        projected_tools, projected_skills = _projected_cli_bundle("projected")
        projected_min_tools, projected_min_skills = _projected_cli_bundle("projected_minimal")

        typed_adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "typed_list",
                            "type": "function",
                            "function": {
                                "name": "houyi_list_dir",
                                "arguments": json.dumps({"path": "houyi/skills"}),
                            },
                        }
                    ],
                ),
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "typed_find",
                            "type": "function",
                            "function": {
                                "name": "houyi_find_files",
                                "arguments": json.dumps(
                                    {
                                        "root_path": "houyi/skills",
                                        "pattern": "SKILL.md",
                                        "search_mode": "exact",
                                    }
                                ),
                            },
                        }
                    ],
                ),
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "typed_grep",
                            "type": "function",
                            "function": {
                                "name": "houyi_grep",
                                "arguments": json.dumps(
                                    {"path": "houyi/skills", "query": "web search"}
                                ),
                            },
                        }
                    ],
                ),
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "typed_read",
                            "type": "function",
                            "function": {
                                "name": "houyi_read_file",
                                "arguments": json.dumps(
                                    {
                                        "path": "houyi/skills/web_search/SKILL.md",
                                        "start_line": 1,
                                        "end_line": 20,
                                    }
                                ),
                            },
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )
        local_cli_adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "cli_list",
                            "type": "function",
                            "function": {
                                "name": "houyi_local_cli",
                                "arguments": json.dumps(
                                    {"command": "list", "path": "houyi/skills"}
                                ),
                            },
                        }
                    ],
                ),
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "cli_find",
                            "type": "function",
                            "function": {
                                "name": "houyi_local_cli",
                                "arguments": json.dumps(
                                    {
                                        "command": "find",
                                        "path": "houyi/skills",
                                        "pattern": "SKILL.md",
                                        "search_mode": "exact",
                                    }
                                ),
                            },
                        }
                    ],
                ),
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "cli_grep",
                            "type": "function",
                            "function": {
                                "name": "houyi_local_cli",
                                "arguments": json.dumps(
                                    {
                                        "command": "grep",
                                        "path": "houyi/skills",
                                        "query": "web search",
                                    }
                                ),
                            },
                        }
                    ],
                ),
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "cli_read",
                            "type": "function",
                            "function": {
                                "name": "houyi_local_cli",
                                "arguments": json.dumps(
                                    {
                                        "command": "read",
                                        "path": "houyi/skills/web_search/SKILL.md",
                                        "start_line": 1,
                                        "end_line": 20,
                                    }
                                ),
                            },
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )
        projected_adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "projected_list",
                            "type": "function",
                            "function": {
                                "name": "houyi_local_cli_list",
                                "arguments": json.dumps({"path": "houyi/skills"}),
                            },
                        }
                    ],
                ),
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "projected_find",
                            "type": "function",
                            "function": {
                                "name": "houyi_local_cli_find",
                                "arguments": json.dumps(
                                    {
                                        "path": "houyi/skills",
                                        "pattern": "SKILL.md",
                                        "search_mode": "exact",
                                    }
                                ),
                            },
                        }
                    ],
                ),
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "projected_grep",
                            "type": "function",
                            "function": {
                                "name": "houyi_local_cli_grep",
                                "arguments": json.dumps(
                                    {"path": "houyi/skills", "query": "web search"}
                                ),
                            },
                        }
                    ],
                ),
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "projected_read",
                            "type": "function",
                            "function": {
                                "name": "houyi_local_cli_read",
                                "arguments": json.dumps(
                                    {
                                        "path": "houyi/skills/web_search/SKILL.md",
                                        "start_line": 1,
                                        "end_line": 20,
                                    }
                                ),
                            },
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )
        projected_min_adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "projected_min_list",
                            "type": "function",
                            "function": {
                                "name": "houyi_local_cli_list",
                                "arguments": json.dumps({"path": "houyi/skills"}),
                            },
                        }
                    ],
                ),
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "projected_min_find",
                            "type": "function",
                            "function": {
                                "name": "houyi_local_cli_find",
                                "arguments": json.dumps(
                                    {
                                        "path": "houyi/skills",
                                        "pattern": "SKILL.md",
                                        "search_mode": "exact",
                                    }
                                ),
                            },
                        }
                    ],
                ),
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "projected_min_grep",
                            "type": "function",
                            "function": {
                                "name": "houyi_local_cli_grep",
                                "arguments": json.dumps(
                                    {"path": "houyi/skills", "query": "web search"}
                                ),
                            },
                        }
                    ],
                ),
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "projected_min_read",
                            "type": "function",
                            "function": {
                                "name": "houyi_local_cli_read",
                                "arguments": json.dumps(
                                    {
                                        "path": "houyi/skills/web_search/SKILL.md",
                                        "start_line": 1,
                                        "end_line": 20,
                                    }
                                ),
                            },
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )

        runner = ToolCallRunner()
        typed_response, typed_trace, typed_metrics = await self._run_pipeline_lane(
            runner=runner,
            adapter=typed_adapter,
            messages=[
                {"role": "user", "content": "find the local web search skill and preview it"}
            ],
            tools=[skill.to_tool_schema() for skill in typed_skills],
            skills=typed_skills,
        )
        local_cli_response, local_cli_trace, local_cli_metrics = await self._run_pipeline_lane(
            runner=runner,
            adapter=local_cli_adapter,
            messages=[
                {"role": "user", "content": "find the local web search skill and preview it"}
            ],
            tools=[local_cli_skill.to_tool_schema()],
            skills=[local_cli_skill],
        )
        projected_response, projected_trace, projected_metrics = await self._run_pipeline_lane(
            runner=runner,
            adapter=projected_adapter,
            messages=[
                {"role": "user", "content": "find the local web search skill and preview it"}
            ],
            tools=projected_tools,
            skills=projected_skills,
        )
        (
            projected_min_response,
            projected_min_trace,
            projected_min_metrics,
        ) = await self._run_pipeline_lane(
            runner=runner,
            adapter=projected_min_adapter,
            messages=[
                {"role": "user", "content": "find the local web search skill and preview it"}
            ],
            tools=projected_min_tools,
            skills=projected_min_skills,
        )

        assert typed_response.content == "done"
        assert local_cli_response.content == "done"
        assert projected_response.content == "done"
        assert projected_min_response.content == "done"
        assert typed_metrics["llm_round_count"] == 5
        assert local_cli_metrics["llm_round_count"] == 5
        assert projected_metrics["llm_round_count"] == 5
        assert projected_min_metrics["llm_round_count"] == 5
        assert typed_metrics["tool_round_count"] == 4
        assert local_cli_metrics["tool_round_count"] == 4
        assert projected_metrics["tool_round_count"] == 4
        assert projected_min_metrics["tool_round_count"] == 4
        assert typed_metrics["tool_call_count"] == 4
        assert local_cli_metrics["tool_call_count"] == 4
        assert projected_metrics["tool_call_count"] == 4
        assert projected_min_metrics["tool_call_count"] == 4
        assert local_cli_metrics["schema_chars_total"] < typed_metrics["schema_chars_total"]
        assert projected_metrics["schema_chars_total"] > local_cli_metrics["schema_chars_total"]
        assert (
            projected_min_metrics["schema_chars_total"] <= projected_metrics["schema_chars_total"]
        )
        assert projected_min_metrics["schema_chars_total"] < typed_metrics["schema_chars_total"]
        assert local_cli_metrics["presented_content_chars_total"] <= (
            typed_metrics["presented_content_chars_total"] + 400
        )
        assert projected_metrics["presented_content_chars_total"] <= (
            typed_metrics["presented_content_chars_total"] + 400
        )
        assert projected_min_metrics["presented_content_chars_total"] <= (
            typed_metrics["presented_content_chars_total"] + 400
        )
        assert local_cli_metrics["raw_payload_chars_total"] <= (
            typed_metrics["raw_payload_chars_total"] + 400
        )
        assert projected_metrics["raw_payload_chars_total"] <= (
            typed_metrics["raw_payload_chars_total"] + 400
        )
        assert projected_min_metrics["raw_payload_chars_total"] <= (
            typed_metrics["raw_payload_chars_total"] + 400
        )
        assert local_cli_metrics["prompt_token_proxy"] < typed_metrics["prompt_token_proxy"]
        assert projected_metrics["prompt_token_proxy"] > local_cli_metrics["prompt_token_proxy"]
        assert (
            projected_min_metrics["prompt_token_proxy"] <= projected_metrics["prompt_token_proxy"]
        )
        assert projected_min_metrics["prompt_token_proxy"] < typed_metrics["prompt_token_proxy"]
        assert local_cli_metrics["completion_token_proxy"] <= (
            typed_metrics["completion_token_proxy"] + 100
        )
        assert projected_metrics["completion_token_proxy"] <= (
            typed_metrics["completion_token_proxy"] + 100
        )
        assert projected_min_metrics["completion_token_proxy"] <= (
            typed_metrics["completion_token_proxy"] + 100
        )
        assert local_cli_metrics["total_token_proxy"] < typed_metrics["total_token_proxy"]
        assert projected_min_metrics["total_token_proxy"] < typed_metrics["total_token_proxy"]
        assert typed_metrics["tool_duration_ms_total"] > 0
        assert local_cli_metrics["tool_duration_ms_total"] > 0
        assert projected_metrics["tool_duration_ms_total"] > 0
        assert projected_min_metrics["tool_duration_ms_total"] > 0
        assert typed_metrics["tool_duration_ms_avg"] > 0
        assert local_cli_metrics["tool_duration_ms_avg"] > 0
        assert projected_metrics["tool_duration_ms_avg"] > 0
        assert projected_min_metrics["tool_duration_ms_avg"] > 0
        assert typed_metrics["wall_time_ms"] > 0
        assert local_cli_metrics["wall_time_ms"] > 0
        assert projected_metrics["wall_time_ms"] > 0
        assert projected_min_metrics["wall_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_chain_baseline(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HOUYI_WORKSPACE_ROOT", str(tmp_path))

        skills_dir = tmp_path / "houyi" / "skills" / "web_search"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / "SKILL.md").write_text(
            "# Web Search\nweb search local chain target\nline-3\n",
            encoding="utf-8",
        )

        chain_skill = next(
            skill
            for skill in local_tools.build_builtin_local_tools()
            if skill.name == "houyi_local_cli_chain"
        )
        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "chain_flow",
                            "type": "function",
                            "function": {
                                "name": "houyi_local_cli_chain",
                                "arguments": json.dumps(
                                    {
                                        "workflow": (
                                            "find path=houyi/skills pattern=SKILL.md search_mode=exact "
                                            "| grep query='web search' | read start_line=1 end_line=2"
                                        )
                                    }
                                ),
                            },
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )

        runner = ToolCallRunner()
        response, tool_trace = await runner.run(
            adapter=adapter,
            messages=[
                {"role": "user", "content": "find the local web search skill and preview it"}
            ],
            tools=[chain_skill.to_tool_schema()],
            skills=[chain_skill],
            executor=_DirectSkillExecutor(),
            max_rounds=3,
        )

        assert response.content == "done"
        assert len(tool_trace) == 1
        raw = tool_trace[0]["result"]["raw"]
        assert raw["success"] is True
        assert len(raw["data"]["steps"]) == 3
        assert raw["data"]["steps"][0]["command"] == "find"
        assert raw["data"]["steps"][1]["command"] == "grep"
        assert raw["data"]["steps"][2]["command"] == "read"
        assert tool_trace[0]["tool_name"] == "houyi_local_cli_chain"

    @pytest.mark.asyncio
    async def test_chain_fallback(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HOUYI_WORKSPACE_ROOT", str(tmp_path))

        skills_dir = tmp_path / "houyi" / "skills" / "web_search"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / "SKILL.md").write_text(
            "# Web Search\nweb search local chain target\nline-3\n",
            encoding="utf-8",
        )

        chain_skill = next(
            skill
            for skill in local_tools.build_builtin_local_tools()
            if skill.name == "houyi_local_cli_chain"
        )
        adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "chain_fallback",
                            "type": "function",
                            "function": {
                                "name": "houyi_local_cli_chain",
                                "arguments": json.dumps(
                                    {
                                        "workflow": (
                                            "find path=houyi/skills pattern=missing.md search_mode=exact "
                                            "|| find path=houyi/skills pattern=SKILL.md search_mode=exact "
                                            "| read start_line=1 end_line=1"
                                        )
                                    }
                                ),
                            },
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )

        runner = ToolCallRunner()
        response, tool_trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "recover and preview the matching skill"}],
            tools=[chain_skill.to_tool_schema()],
            skills=[chain_skill],
            executor=_DirectSkillExecutor(),
            max_rounds=3,
        )

        assert response.content == "done"
        raw = tool_trace[0]["result"]["raw"]
        steps = raw["data"]["steps"]
        assert steps[0]["success"] is False
        assert steps[1]["success"] is True
        assert steps[2]["success"] is True

    @pytest.mark.asyncio
    async def test_chain_comparative_workflow_surface(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HOUYI_WORKSPACE_ROOT", str(tmp_path))

        skills_dir = tmp_path / "houyi" / "skills" / "web_search"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / "SKILL.md").write_text(
            "# Web Search\nweb search local chain target\nline-3\n",
            encoding="utf-8",
        )

        projected_tools, projected_skills = _projected_cli_bundle("projected")
        chain_tools, chain_skills = _bridge_skill_bundle(
            skill_name="houyi_local_cli_chain",
            schema_exposure="full",
        )
        chain_min_tools, chain_min_skills = _bridge_skill_bundle(
            skill_name="houyi_local_cli_chain",
            schema_exposure="minimal",
        )

        projected_adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "projected_find",
                            "type": "function",
                            "function": {
                                "name": "houyi_local_cli_find",
                                "arguments": json.dumps(
                                    {
                                        "path": "houyi/skills",
                                        "pattern": "SKILL.md",
                                        "search_mode": "exact",
                                    }
                                ),
                            },
                        }
                    ],
                ),
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "projected_grep",
                            "type": "function",
                            "function": {
                                "name": "houyi_local_cli_grep",
                                "arguments": json.dumps(
                                    {
                                        "path": "houyi/skills/web_search/SKILL.md",
                                        "query": "web search",
                                    }
                                ),
                            },
                        }
                    ],
                ),
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "projected_read",
                            "type": "function",
                            "function": {
                                "name": "houyi_local_cli_read",
                                "arguments": json.dumps(
                                    {
                                        "path": "houyi/skills/web_search/SKILL.md",
                                        "start_line": 1,
                                        "end_line": 2,
                                    }
                                ),
                            },
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )
        chain_adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "chain_workflow",
                            "type": "function",
                            "function": {
                                "name": "houyi_local_cli_chain",
                                "arguments": json.dumps(
                                    {
                                        "workflow": (
                                            "find path=houyi/skills pattern=SKILL.md search_mode=exact "
                                            "| grep query='web search' "
                                            "| read start_line=1 end_line=2"
                                        )
                                    }
                                ),
                            },
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )
        chain_min_adapter = _FakeAdapter(
            [
                _FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "chain_min_workflow",
                            "type": "function",
                            "function": {
                                "name": "houyi_local_cli_chain",
                                "arguments": json.dumps(
                                    {
                                        "workflow": (
                                            "find path=houyi/skills pattern=SKILL.md search_mode=exact "
                                            "| grep query='web search' "
                                            "| read start_line=1 end_line=2"
                                        )
                                    }
                                ),
                            },
                        }
                    ],
                ),
                _FakeResponse(content="done", tool_calls=[]),
            ]
        )

        runner = ToolCallRunner()
        projected_response, projected_trace, projected_metrics = await self._run_pipeline_lane(
            runner=runner,
            adapter=projected_adapter,
            messages=[
                {"role": "user", "content": "find the local web search skill and preview it"}
            ],
            tools=projected_tools,
            skills=projected_skills,
        )
        chain_response, chain_trace, chain_metrics = await self._run_pipeline_lane(
            runner=runner,
            adapter=chain_adapter,
            messages=[
                {"role": "user", "content": "find the local web search skill and preview it"}
            ],
            tools=chain_tools,
            skills=chain_skills,
        )
        chain_min_response, chain_min_trace, chain_min_metrics = await self._run_pipeline_lane(
            runner=runner,
            adapter=chain_min_adapter,
            messages=[
                {"role": "user", "content": "find the local web search skill and preview it"}
            ],
            tools=chain_min_tools,
            skills=chain_min_skills,
        )

        assert projected_response.content == "done"
        assert chain_response.content == "done"
        assert chain_min_response.content == "done"
        assert len(projected_trace) == 3
        assert len(chain_trace) == 1
        assert len(chain_min_trace) == 1
        assert projected_metrics["tool_round_count"] == 3
        assert chain_metrics["tool_round_count"] == 1
        assert chain_min_metrics["tool_round_count"] == 1
        assert projected_metrics["tool_call_count"] == 3
        assert chain_metrics["tool_call_count"] == 1
        assert chain_min_metrics["tool_call_count"] == 1
        assert chain_metrics["llm_round_count"] < projected_metrics["llm_round_count"]
        assert chain_min_metrics["llm_round_count"] < projected_metrics["llm_round_count"]
        assert chain_metrics["schema_chars_total"] < projected_metrics["schema_chars_total"]
        assert chain_min_metrics["schema_chars_total"] <= chain_metrics["schema_chars_total"]
        assert chain_metrics["prompt_token_proxy"] < projected_metrics["prompt_token_proxy"]
        assert chain_min_metrics["prompt_token_proxy"] <= chain_metrics["prompt_token_proxy"]
        assert chain_metrics["wall_time_ms"] > 0
        assert chain_min_metrics["wall_time_ms"] > 0
        chain_raw = chain_trace[0]["result"]["raw"]
        assert chain_raw["success"] is True
        assert len(chain_raw["data"]["steps"]) == 3
        assert chain_raw["data"]["steps"][0]["command"] == "find"
        assert chain_raw["data"]["steps"][1]["command"] == "grep"
        assert chain_raw["data"]["steps"][2]["command"] == "read"


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


def test_preserves_missing_content() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "demo", "arguments": {"q": 1}},
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"ok":true}',
            "tool_call_id": "call_1",
        },
    ]

    prepared = prepare_tool_loop_messages(messages, max_message_chars=12_000, max_total_chars=8_000)

    assert prepared[0]["role"] == "assistant"
    assert "content" not in prepared[0]
    assert prepared[0]["tool_calls"][0]["function"]["arguments"] == '{"q": 1}'
    assert prepared[1] == {
        "role": "tool",
        "content": '{"ok":true}',
        "tool_call_id": "call_1",
    }


def test_caps_large_history() -> None:
    messages = [{"role": "system", "content": "sys"}]
    for index in range(60):
        messages.append({"role": "user", "content": f"u-{index}"})
        messages.append({"role": "assistant", "content": f"a-{index}"})

    prepared = prepare_tool_loop_messages(
        messages,
        max_message_chars=12_000,
        max_total_chars=1_000_000,
    )

    assert prepared[0] == {"role": "system", "content": "sys"}
    assert len(prepared) == 49
    assert prepared[1]["content"] == "u-36"
    assert prepared[-1]["content"] == "a-59"


def test_latest_assistant_tool() -> None:
    messages = [{"role": "system", "content": "sys"}]
    for index in range(47):
        messages.append({"role": "user", "content": f"u-{index}"})
    messages.extend(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_latest",
                        "type": "function",
                        "function": {"name": "demo", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": '{"ok": true}',
                "tool_call_id": "call_latest",
            },
        ]
    )

    prepared = prepare_tool_loop_messages(
        messages,
        max_message_chars=12_000,
        max_total_chars=1_000_000,
    )

    assert prepared[0] == {"role": "system", "content": "sys"}
    assert prepared[-2]["role"] == "assistant"
    assert prepared[-2]["tool_calls"][0]["id"] == "call_latest"
    assert prepared[-1] == {
        "role": "tool",
        "content": '{"ok": true}',
        "tool_call_id": "call_latest",
    }


def test_prioritizes_recent_toolcontext() -> None:
    messages = [{"role": "system", "content": "sys"}]
    for index in range(60):
        messages.append({"role": "user", "content": f"u-{index}"})
        messages.append({"role": "assistant", "content": f"a-{index}"})
    messages.extend(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_latest",
                        "type": "function",
                        "function": {"name": "demo", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": '{"ok": true}',
                "tool_call_id": "call_latest",
            },
        ]
    )

    prepared = prepare_tool_loop_messages(
        messages,
        max_message_chars=12_000,
        max_total_chars=1_000_000,
    )

    non_system = [message for message in prepared if message["role"] != "system"]
    assert non_system[0] == {"role": "user", "content": "u-52"}
    assert non_system[1] == {"role": "assistant", "content": "a-52"}
    assert non_system[-2]["role"] == "assistant"
    assert non_system[-2]["tool_calls"][0]["id"] == "call_latest"
    assert non_system[-1] == {
        "role": "tool",
        "content": '{"ok": true}',
        "tool_call_id": "call_latest",
    }


def test_drops_orphan_assistant() -> None:
    messages = [{"role": "system", "content": "sys"}]
    for index in range(54):
        messages.append({"role": "user", "content": f"u-{index}"})
        messages.append({"role": "assistant", "content": f"a-{index}"})
    messages.append({"role": "user", "content": "latest"})

    prepared = prepare_tool_loop_messages(
        messages,
        max_message_chars=12_000,
        max_total_chars=1_000_000,
    )

    non_system = [message for message in prepared if message["role"] != "system"]
    assert non_system[0] == {"role": "user", "content": "u-31"}
    assert non_system[1] == {"role": "assistant", "content": "a-31"}
    assert non_system[-1] == {"role": "user", "content": "latest"}
    assert all(
        not (message["role"] == "assistant" and index == 0)
        for index, message in enumerate(non_system)
    )


def test_keeps_tool_context() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_repo",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": {"path": "repo.py"}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_repo",
            "content": '{"path":"repo.py","symbols":["A","B","C"],"lines":[1,2,3]}',
        },
        {"role": "user", "content": "latest request"},
        {"role": "assistant", "content": "latest answer"},
    ]

    prepared = prepare_tool_loop_messages(
        messages,
        max_message_chars=12_000,
        max_total_chars=120,
    )

    assert prepared[0] == {"role": "system", "content": "sys"}
    assert any(
        message.get("role") == "assistant"
        and isinstance(message.get("tool_calls"), list)
        and message["tool_calls"][0]["id"] == "call_repo"
        for message in prepared
    )
    assert any(
        message.get("role") == "tool" and message.get("tool_call_id") == "call_repo"
        for message in prepared
    )
    assert any(message.get("content") == "latest request" for message in prepared)
    assert not any(message.get("content") == "earlier answer" for message in prepared)


def test_budget_keep_system() -> None:
    prepared = prepare_tool_loop_messages(
        [
            {"role": "system", "content": "s" * 300},
            {"role": "user", "content": "u" * 80},
        ],
        max_message_chars=200,
        max_total_chars=100,
    )

    assert len(prepared) == 1
    assert prepared[0]["role"] == "system"


def test_result_summary_list() -> None:
    content = json.dumps([{"idx": idx, "value": "x" * 40} for idx in range(20)])

    summarized, truncated = MessageBudget.summarize_tool_result(
        content,
        max_chars=400,
        max_items=3,
    )

    assert truncated is True
    assert '"_truncated": true' in summarized
    assert '"_original_count": 20' in summarized


def test_result_summary_text() -> None:
    summarized, truncated = MessageBudget.summarize_tool_result("x" * 300, max_chars=80)

    assert truncated is True
    assert summarized.endswith("[truncated, original length: 300]")


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
        assert tool_trace[0]["result"]["raw"]["recovery_guidance"]["code"] == "missing_tool"
        assert tool_trace[1]["result"]["raw"]["recovery_guidance"]["similar_tools"] == []

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

        response, tool_trace = await runner.run(
            adapter=_FakeAdapter(
                [
                    _FakeResponse(
                        content="",
                        tool_calls=[
                            {
                                "id": "c2",
                                "type": "function",
                                "function": {"name": "timeout_tool", "arguments": "{}"},
                            }
                        ],
                    )
                ]
            ),
            messages=[{"role": "user", "content": "hi"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=_DummyExecutor(timeout=True),
            max_rounds=1,
        )

        assert response is not None
        assert tool_trace[0]["result"]["raw"]["recovery_guidance"]["code"] == "execution_timeout"
        assert tool_trace[0]["presentation"]["recovery_guidance_attached"] is True

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
