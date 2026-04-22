from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from houyi.application.tool_calling.tool_call_output import (
    assemble_tool_call_output,
    build_llm_cache_key,
)
from houyi.application.workflow.serialization import to_wire_data


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
    async def test_no_trace_uses_initial(self) -> None:
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
        assert to_wire_data(result.output_payload)["content"] == "hello"
        assert result.messages_for_log == [{"role": "user", "content": "hi"}]
        assert result.tool_call_rounds == 0
        assert result.normalized_tool_trace == []
        assert result.tool_errors == []
        assert base_adapter.calls == []

    @pytest.mark.asyncio
    async def test_cache_hit_skips_chat(self) -> None:
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
        assert result.messages_for_log == messages
        assert result.tool_call_rounds == 1
        assert len(result.normalized_tool_trace) == 1
        assert result.normalized_tool_trace[0].tool_call_id == "tc1"
        assert result.tool_errors == []
        assert to_wire_data(result.normalized_tool_trace[0])["tool_call_id"] == "tc1"
        assert base_adapter.calls == []


class TestExecutionSerialization:
    @pytest.mark.asyncio
    async def test_wire_data_nested_payload(self) -> None:
        result = await assemble_tool_call_output(
            session_id="s1",
            execution=_Execution(metadata={}),
            node_id="n1",
            node_exec=_NodeExec(),
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1"}]},
            ],
            response=_Response(content="final", finish_reason="stop"),
            tool_trace=[
                {
                    "tool_name": "search",
                    "tool_call_id": "tc1",
                    "args": {"q": "hello"},
                    "result": {"raw": {"ok": True}, "is_error": False},
                }
            ],
            base_adapter=_BaseAdapter(),
            tool_model="m",
            prompt="p",
            user_content="u",
            max_tool_calls=6,
            skills=[],
            final_chat_kwargs=None,
            prompt_cache_key=None,
            llm_cache=None,
        )

        payload = to_wire_data(result.output_payload)
        assert payload["type"] == "llm_response"
        assert payload["tool_calls"][0]["tool_call_id"] == "tc1"
        assert payload["tool_calls"][0]["result"]["raw"] == {"ok": True}

    @pytest.mark.asyncio
    async def test_trace_cache_miss_stores(self) -> None:
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
    async def test_normalize_collects_errors(self) -> None:
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


class TestNormalizeToolTrace:
    """Tests for _normalize_tool_trace helper."""

    def test_parallel_group_id_propagated(self) -> None:
        """parallel_group_id from raw trace entry should survive normalization."""
        from houyi.application.tool_calling.tool_call_output import _normalize_tool_trace

        raw_trace = [
            {
                "tool_name": "search",
                "requested_tool_name": "search",
                "tool_call_id": "c1",
                "parallel_group_id": "round_1",
                "args": {"q": "test"},
                "result": {"raw": {"ok": True}, "is_error": False},
            },
            {
                "tool_name": "read",
                "tool_call_id": "c2",
                "parallel_group_id": "round_1",
                "args": {},
                "result": {"raw": {"ok": True}, "is_error": False},
            },
        ]
        calls, errors = _normalize_tool_trace(raw_trace)
        assert len(calls) == 2
        assert calls[0].parallel_group_id == "round_1"
        assert calls[1].parallel_group_id == "round_1"
        assert errors == []

    def test_group_id_none_sequential(self) -> None:
        from houyi.application.tool_calling.tool_call_output import _normalize_tool_trace

        raw_trace = [
            {
                "tool_name": "t",
                "tool_call_id": "c1",
                "args": {},
                "result": {"raw": {"ok": True}, "is_error": False},
            }
        ]
        calls, _ = _normalize_tool_trace(raw_trace)
        assert calls[0].parallel_group_id is None

    def test_error_carries_group_id(self) -> None:
        from houyi.application.tool_calling.tool_call_output import _normalize_tool_trace

        raw_trace = [
            {
                "tool_name": "fail",
                "tool_call_id": "c1",
                "parallel_group_id": "round_2",
                "args": {},
                "result": {"raw": {"error": "boom"}, "is_error": True},
            }
        ]
        calls, errors = _normalize_tool_trace(raw_trace)
        assert calls[0].parallel_group_id == "round_2"
        assert calls[0].result.is_error is True
        assert len(errors) == 1


class TestAssembleOutputMetadata:
    """Tests for trace_id and usage metadata in assemble_tool_call_output."""

    @pytest.mark.asyncio
    async def test_trace_id_from_execution(self) -> None:
        """trace_id should come from execution.execution_id."""
        base_adapter = _BaseAdapter()
        execution = _Execution(metadata={}, execution_id="trace_abc")
        node_exec = _NodeExec()

        result = await assemble_tool_call_output(
            session_id="s1",
            execution=execution,
            node_id="n1",
            node_exec=node_exec,
            messages=[{"role": "user", "content": "hi"}],
            response=_Response(content="done"),
            tool_trace=[
                {
                    "tool_name": "t",
                    "tool_call_id": "tc1",
                    "args": {},
                    "result": {"raw": {"ok": True}, "is_error": False},
                }
            ],
            base_adapter=base_adapter,
            tool_model="m",
            prompt="p",
            user_content="u",
            max_tool_calls=6,
            skills=[],
            final_chat_kwargs=None,
            prompt_cache_key=None,
            llm_cache=None,
        )
        assert result.output_payload.metadata["trace_id"] == "trace_abc"

    @pytest.mark.asyncio
    async def test_usage_from_response(self) -> None:
        """usage dict should be included in metadata when response has usage."""

        class _AdapterWithUsage(_BaseAdapter):
            async def chat(self, messages: Any, tools: Any = None, **kwargs: Any) -> _Response:
                self.calls.append({"messages": messages, "tools": tools})
                return _Response(
                    content="done",
                    metadata={
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                    },
                )

        adapter = _AdapterWithUsage()
        result = await assemble_tool_call_output(
            session_id="s1",
            execution=_Execution(metadata={}),
            node_id="n1",
            node_exec=_NodeExec(),
            messages=[{"role": "user", "content": "hi"}],
            response=_Response(content=""),
            tool_trace=[
                {
                    "tool_name": "t",
                    "tool_call_id": "tc1",
                    "args": {},
                    "result": {"raw": {"ok": True}, "is_error": False},
                }
            ],
            base_adapter=adapter,
            tool_model="m",
            prompt="p",
            user_content="u",
            max_tool_calls=6,
            skills=[],
            final_chat_kwargs={"max_tokens": 10},
            prompt_cache_key=None,
            llm_cache=None,
        )
        assert result.output_payload.metadata.get("usage") == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    @pytest.mark.asyncio
    async def test_usage_from_metadata(self) -> None:
        """usage should be extracted from response.metadata when usage attr is None."""
        usage = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
        resp = _Response(content="done", metadata={"usage": usage})
        result = await assemble_tool_call_output(
            session_id="s1",
            execution=_Execution(metadata={}),
            node_id="n1",
            node_exec=_NodeExec(),
            messages=[{"role": "user", "content": "hi"}],
            response=resp,
            tool_trace=[
                {
                    "tool_name": "t",
                    "tool_call_id": "tc1",
                    "args": {},
                    "result": {"raw": {"ok": True}, "is_error": False},
                }
            ],
            base_adapter=_BaseAdapter(),
            tool_model="m",
            prompt="p",
            user_content="u",
            max_tool_calls=6,
            skills=[],
            final_chat_kwargs=None,
            prompt_cache_key=None,
            llm_cache=None,
        )
        assert result.output_payload.metadata.get("usage") == usage

    @pytest.mark.asyncio
    async def test_no_trace_empty_execution(self) -> None:
        result = await assemble_tool_call_output(
            session_id="s1",
            execution=_Execution(metadata={}, execution_id=""),
            node_id="n1",
            node_exec=_NodeExec(),
            messages=[{"role": "user", "content": "hi"}],
            response=_Response(content="done"),
            tool_trace=[
                {
                    "tool_name": "t",
                    "tool_call_id": "tc1",
                    "args": {},
                    "result": {"raw": {"ok": True}, "is_error": False},
                }
            ],
            base_adapter=_BaseAdapter(),
            tool_model="m",
            prompt="p",
            user_content="u",
            max_tool_calls=6,
            skills=[],
            final_chat_kwargs=None,
            prompt_cache_key=None,
            llm_cache=None,
        )
        assert "trace_id" not in result.output_payload.metadata
