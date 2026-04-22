from __future__ import annotations

from houyi.application.workflow.llm_node_utils import build_llm_node_inputs


def test_converts_tokens_uses_resolved() -> None:
    resolved_called: dict[str, int] = {"count": 0}

    def resolve_tool_settings(config: dict) -> dict:
        resolved_called["count"] += 1
        return {
            "tool_names": ["web_search"],
            "tool_choice": "auto",
            "max_tool_calls": 3,
            "temperature": 0.2,
            "parallel_tool_calls": True,
            "prompt_cache_key": "k1",
        }

    inputs = build_llm_node_inputs(
        config={"prompt": "p", "max_tokens": "128"},
        run_settings=None,
        resolve_tool_settings=resolve_tool_settings,
    )

    assert resolved_called["count"] == 1
    assert inputs["max_tokens"] == 128
    assert inputs["enable_tool_calls"] is True
    assert inputs["tool_names"] == ["web_search"]
    assert inputs["max_tool_calls"] == 3
    assert inputs["prompt_cache_key"] == "k1"


def test_run_settings_handles_invalid() -> None:
    def resolve_tool_settings(_config: dict) -> dict:
        raise AssertionError("resolver should not be called when run_settings is provided")

    inputs = build_llm_node_inputs(
        config={"max_tokens": "oops"},
        run_settings={"enable_tool_calls": False, "tool_names": [], "max_tool_calls": 1},
        resolve_tool_settings=resolve_tool_settings,
    )

    assert inputs["max_tokens"] is None
    assert inputs["enable_tool_calls"] is False
    assert inputs["max_tool_calls"] == 1
