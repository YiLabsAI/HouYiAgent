import pytest

from houyi.execution.tool_call_coordinator import ToolCallCoordinator


def test_tool_call_coordinator_is_not_available_in_this_build() -> None:
    with pytest.raises(RuntimeError) as exc:
        ToolCallCoordinator()

    assert "ToolCallCoordinator is not available in this build" in str(exc.value)
