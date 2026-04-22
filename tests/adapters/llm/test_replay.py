from __future__ import annotations

from datetime import datetime

from houyi.adapters.llm.replay import (
    ReplayDecisionKind,
    build_prompt_cache_key,
    decide_replay,
    get_cached_response,
    get_recorded_llm_response,
    get_recorded_tool_call_output,
    is_deterministic_replay,
    record_llm_call,
)
from houyi.interface.protocol.ir.checkpoint_ir import LLMCallLog


def test_deterministic_requires_flag() -> None:
    assert is_deterministic_replay(execution_metadata=None) is False
    assert is_deterministic_replay(execution_metadata={"replay_mode": "best_effort"}) is False
    assert is_deterministic_replay(execution_metadata={"replay_mode": "deterministic"}) is True


def test_build_cache_key() -> None:
    assert build_prompt_cache_key(model="test-model", prompt_cache_key=None) is None
    assert build_prompt_cache_key(model="test-model", prompt_cache_key="") is None
    assert build_prompt_cache_key(model="test-model", prompt_cache_key="p1") == "test-model:p1"
    assert build_prompt_cache_key(model=None, prompt_cache_key="p1") == ":p1"


def test_get_cached_requires_key() -> None:
    llm_cache = {"test-model:p1": "cached response"}
    assert get_cached_response(llm_cache=llm_cache, cache_key=None) is None
    assert get_cached_response(llm_cache=llm_cache, cache_key="missing") is None
    assert get_cached_response(llm_cache=llm_cache, cache_key="test-model:p1") == "cached response"


def test_record_appends_and_ids() -> None:
    llm_call_logs: dict[str, list[LLMCallLog]] = {}

    first = record_llm_call(
        llm_call_logs=llm_call_logs,
        execution_id="exec_1",
        node_id="node_1",
        model="test-model",
        prompt="hello",
        response="world",
        metadata={"source": "test"},
    )
    second = record_llm_call(
        llm_call_logs=llm_call_logs,
        execution_id="exec_1",
        node_id="node_1",
        model="test-model",
        prompt=[{"role": "user", "content": "again"}],
        response="done",
    )

    assert first.call_id == "llm_0_node_1"
    assert second.call_id == "llm_1_node_1"
    assert llm_call_logs["exec_1"] == [first, second]
    assert first.metadata == {"source": "test"}


def test_decide_returns_recorded_text() -> None:
    llm_call_logs = {
        "exec_1": [
            LLMCallLog(
                call_id="llm_0_node_1",
                node_id="node_1",
                timestamp=datetime.now(),
                model="test-model",
                prompt="hello",
                response="recorded response",
                metadata={},
            )
        ]
    }

    decision = decide_replay(
        execution_metadata={"replay_mode": "deterministic"},
        llm_call_logs=llm_call_logs,
        execution_id="exec_1",
        node_id="node_1",
        llm_cache={},
        model="test-model",
        prompt_cache_key="p1",
    )

    assert decision.kind == ReplayDecisionKind.RECORDED_LLM_TEXT
    assert decision.llm_text == "recorded response"
    assert decision.tool_output is None


def test_decide_returns_recorded_tool() -> None:
    llm_call_logs = {
        "exec_1": [
            LLMCallLog(
                call_id="llm_0_node_1",
                node_id="node_1",
                timestamp=datetime.now(),
                model="test-model",
                prompt="hello",
                response="",
                metadata={
                    "tool_call_output": {
                        "content": "",
                        "tool_calls": [{"id": "call_1", "type": "function"}],
                        "messages": [],
                    }
                },
            )
        ]
    }

    decision = decide_replay(
        execution_metadata={"replay_mode": "deterministic"},
        llm_call_logs=llm_call_logs,
        execution_id="exec_1",
        node_id="node_1",
        llm_cache={},
        model="test-model",
        prompt_cache_key="p1",
    )

    assert decision.kind == ReplayDecisionKind.RECORDED_TOOL_OUTPUT
    assert decision.tool_output is not None
    assert decision.tool_output.get("tool_calls")


def test_decide_cached_on_hit() -> None:
    decision = decide_replay(
        execution_metadata={},
        llm_call_logs={},
        execution_id="exec_1",
        node_id="node_1",
        llm_cache={"test-model:p1": "cached response"},
        model="test-model",
        prompt_cache_key="p1",
    )

    assert decision.kind == ReplayDecisionKind.CACHED_LLM_TEXT
    assert decision.llm_text == "cached response"
    assert decision.cache_key == "test-model:p1"


def test_decide_none_without_sources() -> None:
    decision = decide_replay(
        execution_metadata={},
        llm_call_logs={},
        execution_id="exec_1",
        node_id="node_1",
        llm_cache={},
        model="test-model",
        prompt_cache_key="p1",
    )

    assert decision.kind == ReplayDecisionKind.NONE
    assert decision.llm_text is None
    assert decision.tool_output is None


def test_get_recorded_skips_tool() -> None:
    llm_call_logs = {
        "exec_1": [
            LLMCallLog(
                call_id="llm_0_node_1",
                node_id="node_1",
                timestamp=datetime.now(),
                model="test-model",
                prompt="hello",
                response="tool placeholder",
                metadata={"tool_calls": [{"id": "call_1"}]},
            ),
            LLMCallLog(
                call_id="llm_1_node_1",
                node_id="node_1",
                timestamp=datetime.now(),
                model="test-model",
                prompt="hello",
                response="real response",
                metadata={},
            ),
        ]
    }

    assert (
        get_recorded_llm_response(
            llm_call_logs=llm_call_logs,
            execution_id="exec_1",
            node_id="node_1",
        )
        == "real response"
    )


def test_reconstructs_from_response() -> None:
    llm_call_logs = {
        "exec_1": [
            LLMCallLog(
                call_id="llm_0_node_1",
                node_id="node_1",
                timestamp=datetime.now(),
                model="test-model",
                prompt=[{"role": "user", "content": "hello"}],
                response="tool reasoning",
                metadata={
                    "tool_calls": [{"id": "call_1", "type": "function"}],
                    "finish_reason": "tool_calls",
                    "tool_finish_reason": "completed",
                    "tool_call_rounds": 2,
                    "max_rounds_reached": True,
                    "tool_errors": [
                        {
                            "tool_name": "demo",
                            "requested_tool_name": None,
                            "tool_call_id": None,
                            "error": {"message": "boom"},
                        }
                    ],
                },
            )
        ]
    }

    result = get_recorded_tool_call_output(
        llm_call_logs=llm_call_logs,
        execution_id="exec_1",
        node_id="node_1",
    )

    assert result is not None
    assert len(result["tool_calls"]) == 1
    assert result["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "tool reasoning"},
    ]
    assert result["finish_reason"] == "tool_calls"
    assert result["tool_finish_reason"] == "completed"
    assert result["tool_call_rounds"] == 2
    assert result["max_rounds_reached"] is True
    assert result["tool_errors"] == [
        {
            "tool_name": "demo",
            "requested_tool_name": None,
            "tool_call_id": None,
            "error": {"message": "boom"},
        }
    ]


def test_builds_user_from_string() -> None:
    llm_call_logs = {
        "exec_1": [
            LLMCallLog(
                call_id="llm_0_node_1",
                node_id="node_1",
                timestamp=datetime.now(),
                model="test-model",
                prompt="hello",
                response="",
                metadata={"tool_calls": [{"id": "call_1", "type": "function"}]},
            )
        ]
    }

    result = get_recorded_tool_call_output(
        llm_call_logs=llm_call_logs,
        execution_id="exec_1",
        node_id="node_1",
    )

    assert result is not None
    assert result["messages"] == [{"role": "user", "content": "hello"}]
