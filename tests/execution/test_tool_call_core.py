from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from houyi.execution.tool_call_core import assemble_tool_call_output, build_llm_cache_key


@dataclass
class _Execution:
    metadata: dict[str, Any]
    execution_id: str = "exec_1"


@dataclass
class _NodeExec:
    outputs: dict[str, Any] | None = None
    streaming_output: str = ""


class _Response:
    def __init__(
        self,
        *,
        content: str,
        finish_reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.content = content
        self.finish_reason = finish_reason
        self.metadata = metadata if metadata is not None else {}

    def model_copy(self, deep: bool = False) -> _Response:
        copied = _Response(
            content=self.content, finish_reason=self.finish_reason, metadata=dict(self.metadata)
        )
        return copied


class _BaseAdapter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.model = "test-model"
        self.base_url = "http://base"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> _Response:
        self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        return _Response(content="final", finish_reason="stop", metadata={})


class TestToolCallCore:
    @pytest.mark.asyncio
    async def test_assemble_no_tool_trace_uses_initial_response(self) -> None:
        base_adapter = _BaseAdapter()
        execution = _Execution(metadata={})
        node_exec = _NodeExec()
        response = _Response(content="hello", finish_reason="stop")

        result = await assemble_tool_call_output(
            session_id="s1",
            execution=execution,  # type: ignore[arg-type]
            node_id="n1",
            node_exec=node_exec,  # type: ignore[arg-type]
            messages=[{"role": "user", "content": "hi"}],
            response=response,
            tool_trace=[],
            base_adapter=base_adapter,
            tool_model="m",
            prompt="p",
            user_content="u",
            max_tool_calls=6,
            skills=[],  # type: ignore[arg-type]
            final_chat_kwargs=None,
            prompt_cache_key=None,
            llm_cache=None,
        )

        assert result.content == "hello"
        assert result.output_payload.content == "hello"
        assert result.output_payload.tool_calls == []
        assert result.output_payload.tool_errors == []
        assert result.output_payload.metadata == {}
        assert base_adapter.calls == []

    @pytest.mark.asyncio
    async def test_assemble_tool_trace_cache_hit_skips_chat(self) -> None:
        base_adapter = _BaseAdapter()
        execution = _Execution(metadata={})
        node_exec = _NodeExec()

        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1"}]},
        ]
        final_chat_kwargs = {"max_tokens": 10, "prompt_cache_key": "pk"}

        cache_key = build_llm_cache_key(
            adapter=base_adapter,
            messages=messages,
            tools=None,
            chat_kwargs={"max_tokens": 10},
        )
        assert cache_key
        llm_cache: dict[str, Any] = {cache_key: _Response(content="cached", finish_reason="stop")}

        result = await assemble_tool_call_output(
            session_id="s1",
            execution=execution,  # type: ignore[arg-type]
            node_id="n1",
            node_exec=node_exec,  # type: ignore[arg-type]
            messages=messages,
            response=_Response(content="", finish_reason=None),
            tool_trace=[
                {
                    "tool_name": "t",
                    "tool_call_id": "tc1",
                    "args": {"x": 1},
                    "result": {"raw": {"ok": True}, "is_error": False},
                }
            ],
            base_adapter=base_adapter,
            tool_model="m",
            prompt="p",
            user_content="u",
            max_tool_calls=6,
            skills=[],  # type: ignore[arg-type]
            final_chat_kwargs=final_chat_kwargs,
            prompt_cache_key="pk",
            llm_cache=llm_cache,
        )

        assert result.content == "cached"
        assert result.output_payload.content == "cached"
        assert result.output_payload.metadata.get("llm_cache_hit") is True
        assert result.output_payload.metadata.get("llm_cache_key") == cache_key
        assert len(result.output_payload.tool_calls) == 1
        assert base_adapter.calls == []

    @pytest.mark.asyncio
    async def test_assemble_tool_trace_cache_miss_calls_chat_and_stores(self) -> None:
        base_adapter = _BaseAdapter()
        execution = _Execution(metadata={})
        node_exec = _NodeExec()

        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1"}]},
        ]
        final_chat_kwargs = {"max_tokens": 10, "prompt_cache_key": "pk"}
        llm_cache: dict[str, Any] = {}

        result = await assemble_tool_call_output(
            session_id="s1",
            execution=execution,  # type: ignore[arg-type]
            node_id="n1",
            node_exec=node_exec,  # type: ignore[arg-type]
            messages=messages,
            response=_Response(content="", finish_reason=None),
            tool_trace=[
                {
                    "tool_name": "t",
                    "tool_call_id": "tc1",
                    "args": {"x": 1},
                    "result": {"raw": {"ok": True}, "is_error": False},
                }
            ],
            base_adapter=base_adapter,
            tool_model="m",
            prompt="p",
            user_content="u",
            max_tool_calls=6,
            skills=[],  # type: ignore[arg-type]
            final_chat_kwargs=final_chat_kwargs,
            prompt_cache_key="pk",
            llm_cache=llm_cache,
        )

        assert result.content == "final"
        assert len(base_adapter.calls) == 1
        # Should have stored a cached response under the computed cache key.
        cache_key = build_llm_cache_key(
            adapter=base_adapter,
            messages=messages,
            tools=None,
            chat_kwargs={"max_tokens": 10},
        )
        assert cache_key in llm_cache

    @pytest.mark.asyncio
    async def test_normalize_tool_trace_collects_errors(self) -> None:
        base_adapter = _BaseAdapter()
        execution = _Execution(metadata={})
        node_exec = _NodeExec()

        result = await assemble_tool_call_output(
            session_id="s1",
            execution=execution,  # type: ignore[arg-type]
            node_id="n1",
            node_exec=node_exec,  # type: ignore[arg-type]
            messages=[{"role": "user", "content": "hi"}],
            response=_Response(content="hello", finish_reason="stop"),
            tool_trace=[
                {
                    "tool_name": "t",
                    "requested_tool_name": "t",
                    "tool_call_id": "tc1",
                    "args": {"x": 1},
                    "result": {"raw": "boom", "is_error": True},
                }
            ],
            base_adapter=base_adapter,
            tool_model="m",
            prompt="p",
            user_content="u",
            max_tool_calls=6,
            skills=[],  # type: ignore[arg-type]
            final_chat_kwargs=None,
            prompt_cache_key=None,
            llm_cache=None,
        )

        assert len(result.output_payload.tool_calls) == 1
        assert len(result.output_payload.tool_errors) == 1
        assert result.output_payload.tool_errors[0].error == {"result": "boom"}
