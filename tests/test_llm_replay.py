from __future__ import annotations

from datetime import datetime

from houyi.llm.replay import ReplayDecisionKind, decide_replay
from houyi.protocol.ir.checkpoint_ir import LLMCallLog


def test_decide_replay_returns_recorded_llm_text_when_deterministic() -> None:
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


def test_decide_replay_returns_recorded_tool_output_when_deterministic() -> None:
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


def test_decide_replay_returns_cached_llm_text_when_cache_hit() -> None:
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


def test_decide_replay_returns_none_when_no_sources_available() -> None:
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
