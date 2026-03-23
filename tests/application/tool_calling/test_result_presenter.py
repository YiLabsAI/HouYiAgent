from __future__ import annotations

from houyi.application.tool_calling.result_presenter import _ToolCallResultPresenter
from houyi.application.tool_calling.runner_models import (
    _BlockedToolCallPresentationRequest,
    _ToolCallPresentationRequest,
)
from houyi.application.tool_calling.tool_results import ToolResultBuilder


class TestToolCallResultPresenter:
    def test_blocked_policy(self) -> None:
        presenter = _ToolCallResultPresenter()

        trace_entry, tool_message = presenter.build_blocked_trace_and_message(
            _BlockedToolCallPresentationRequest(
                tool_name="dangerous_tool",
                args={"path": "/tmp/a"},
                tool_call_id="call_blocked",
                error_code="policy_blocked",
                message="blocked by policy",
                block_reason="deny",
            )
        )

        assert trace_entry["tool_name"] == "dangerous_tool"
        assert trace_entry["requested_tool_name"] == "dangerous_tool"
        assert trace_entry["tool_call_id"] == "call_blocked"
        assert trace_entry["args"] == {"path": "/tmp/a"}
        assert trace_entry["policy_blocked"] is True
        assert trace_entry["block_reason"] == "deny"

        result = trace_entry["result"]
        assert isinstance(result, dict)
        assert result["raw"]["error"] == "policy_blocked"
        assert result["raw"]["message"] == "blocked by policy"
        assert result["metadata"]["policy_blocked"] is True
        assert result["metadata"]["tool_name"] == "dangerous_tool"

        assert tool_message["role"] == "tool"
        assert tool_message["tool_call_id"] == "call_blocked"
        assert tool_message["name"] == "dangerous_tool"
        assert tool_message["content"] == ToolResultBuilder.format(result)

    def test_omit_override(self) -> None:
        presenter = _ToolCallResultPresenter()
        result = ToolResultBuilder.build({"ok": True}, call_id="call_1")

        trace_entry, tool_message = presenter.build_trace_and_message(
            _ToolCallPresentationRequest(
                tool_name="tool1",
                requested_tool_name="tool1",
                tool_call_id="call_1",
                round_index_value=1,
                parallel_group_id=None,
                duration_ms=125.0,
                args={"x": 1},
                result=result,
                attempted_tool_name=None,
                allow_tool_replace=False,
                tool_result_summary_enabled=False,
                tool_result_summary_max_chars=200,
                tool_result_summary_max_items=5,
            )
        )

        assert trace_entry["tool_name"] == "tool1"
        assert trace_entry["requested_tool_name"] == "tool1"
        assert trace_entry["round_index"] == 1
        assert trace_entry["parallel_group_id"] is None
        assert trace_entry["duration_ms"] == 125.0
        assert trace_entry["args"] == {"x": 1}
        assert trace_entry["tool_override"] is None
        assert trace_entry["presentation"]["footer_attached"] is True
        assert trace_entry["presentation"]["is_binary"] is False
        assert tool_message["name"] == "tool1"
        assert "duration_ms=125.00" in str(tool_message["content"])
        assert tool_message["metadata"]["duration_ms"] == 125.0
        assert tool_message["metadata"]["presentation"]["footer_attached"] is True

    def test_unapplied_override(self) -> None:
        presenter = _ToolCallResultPresenter()
        result = ToolResultBuilder.build({"executed_skill": "tool1"}, call_id="call_replace")

        trace_entry, _tool_message = presenter.build_trace_and_message(
            _ToolCallPresentationRequest(
                tool_name="tool1",
                requested_tool_name="tool1",
                tool_call_id="call_replace",
                round_index_value=1,
                parallel_group_id="round_1",
                duration_ms=240.0,
                args={},
                result=result,
                attempted_tool_name="tool2",
                allow_tool_replace=False,
                tool_result_summary_enabled=False,
                tool_result_summary_max_chars=200,
                tool_result_summary_max_items=5,
            )
        )

        override = trace_entry["tool_override"]
        assert isinstance(override, dict)
        assert override == {
            "from": "tool1",
            "to": "tool2",
            "allowed": False,
            "applied": False,
        }

    def test_applied_override(self) -> None:
        presenter = _ToolCallResultPresenter()
        result = ToolResultBuilder.build({"executed_skill": "tool2"}, call_id="call_replace")

        trace_entry, tool_message = presenter.build_trace_and_message(
            _ToolCallPresentationRequest(
                tool_name="tool2",
                requested_tool_name="tool1",
                tool_call_id="call_replace",
                round_index_value=2,
                parallel_group_id="round_2",
                duration_ms=320.0,
                args={"from_hook": True},
                result=result,
                attempted_tool_name="tool2",
                allow_tool_replace=True,
                tool_result_summary_enabled=False,
                tool_result_summary_max_chars=200,
                tool_result_summary_max_items=5,
            )
        )

        override = trace_entry["tool_override"]
        assert isinstance(override, dict)
        assert override == {
            "from": "tool1",
            "to": "tool2",
            "allowed": True,
            "applied": True,
        }
        assert tool_message["name"] == "tool2"
        assert tool_message["tool_call_id"] == "call_replace"

    def test_large_result_summary(self) -> None:
        presenter = _ToolCallResultPresenter()
        result = ToolResultBuilder.build(
            {"items": [{"idx": i, "payload": "y" * 200} for i in range(20)]},
            call_id="call_big",
        )
        original_content = ToolResultBuilder.format(result)

        trace_entry, tool_message = presenter.build_trace_and_message(
            _ToolCallPresentationRequest(
                tool_name="tool_big",
                requested_tool_name="tool_big",
                tool_call_id="call_big",
                round_index_value=1,
                parallel_group_id=None,
                duration_ms=None,
                args={},
                result=result,
                attempted_tool_name="tool_big",
                allow_tool_replace=True,
                tool_result_summary_enabled=True,
                tool_result_summary_max_chars=250,
                tool_result_summary_max_items=3,
            )
        )

        assert trace_entry["result"] is result
        assert isinstance(tool_message["content"], str)
        assert tool_message["content"] != original_content
        assert len(str(tool_message["content"])) < len(original_content)
        assert result["metadata"]["result_summarized"] is True
        assert result["metadata"]["result_summary_max_chars"] == 250
        assert result["metadata"]["result_summary_max_items"] == 3
        assert trace_entry["presentation"]["result_summarized"] is True
        assert tool_message["metadata"]["presentation"]["result_summarized"] is True

    def test_summary_disabled(self) -> None:
        presenter = _ToolCallResultPresenter()
        result = ToolResultBuilder.build(
            {"items": [{"idx": i, "payload": "y" * 200} for i in range(10)]},
            call_id="call_no_summary",
        )
        original_content = ToolResultBuilder.format(result)

        _trace_entry, tool_message = presenter.build_trace_and_message(
            _ToolCallPresentationRequest(
                tool_name="tool_big",
                requested_tool_name="tool_big",
                tool_call_id="call_no_summary",
                round_index_value=1,
                parallel_group_id=None,
                duration_ms=None,
                args={},
                result=result,
                attempted_tool_name="tool_big",
                allow_tool_replace=True,
                tool_result_summary_enabled=False,
                tool_result_summary_max_chars=250,
                tool_result_summary_max_items=3,
            )
        )

        assert original_content in str(tool_message["content"])
        assert "result_summarized" not in result.get("metadata", {})

    def test_error_detail(self) -> None:
        presenter = _ToolCallResultPresenter()
        result = ToolResultBuilder.build(
            {
                "error": "tool_execution_failed",
                "message": "validation failed",
                "cause": "field 'path' is required",
            },
            call_id="call_err",
        )

        trace_entry, tool_message = presenter.build_trace_and_message(
            _ToolCallPresentationRequest(
                tool_name="reader",
                requested_tool_name="reader",
                tool_call_id="call_err",
                round_index_value=1,
                parallel_group_id=None,
                duration_ms=10.0,
                args={},
                result=result,
                attempted_tool_name=None,
                allow_tool_replace=False,
                tool_result_summary_enabled=False,
                tool_result_summary_max_chars=500,
                tool_result_summary_max_items=5,
            )
        )

        assert "validation failed" in str(tool_message["content"])
        assert trace_entry["presentation"]["error_detail_attached"] is True

    def test_binary_like_result(self) -> None:
        presenter = _ToolCallResultPresenter()
        result = ToolResultBuilder.build({"blob": b"\x00\x01\x02"}, call_id="call_bin")

        trace_entry, tool_message = presenter.build_trace_and_message(
            _ToolCallPresentationRequest(
                tool_name="reader",
                requested_tool_name="reader",
                tool_call_id="call_bin",
                round_index_value=1,
                parallel_group_id=None,
                duration_ms=None,
                args={},
                result=result,
                attempted_tool_name=None,
                allow_tool_replace=False,
                tool_result_summary_enabled=False,
                tool_result_summary_max_chars=500,
                tool_result_summary_max_items=5,
            )
        )

        assert tool_message["content"] == "Binary-like tool result omitted from inline expansion."
        assert trace_entry["presentation"]["is_binary"] is True

    def test_footer_marks_large_result(self) -> None:
        presenter = _ToolCallResultPresenter()
        result = ToolResultBuilder.build(
            {"items": [{"idx": i, "payload": "z" * 80} for i in range(8)]},
            call_id="call_artifact",
        )

        trace_entry, tool_message = presenter.build_trace_and_message(
            _ToolCallPresentationRequest(
                tool_name="search",
                requested_tool_name="search",
                tool_call_id="call_artifact",
                round_index_value=2,
                parallel_group_id="grp-1",
                duration_ms=42.0,
                args={"q": "abc"},
                result=result,
                attempted_tool_name=None,
                allow_tool_replace=False,
                tool_result_summary_enabled=False,
                tool_result_summary_max_chars=120,
                tool_result_summary_max_items=3,
            )
        )

        assert trace_entry["presentation"]["result_artifact_candidate"] is True
        assert tool_message["metadata"]["presentation"]["result_artifact_candidate"] is True
        assert "parallel_group_id=grp-1" in str(tool_message["content"])
