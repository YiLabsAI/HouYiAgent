from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STUDIO_SERVER_ROOT = _REPO_ROOT / "houyi-studio" / "server"
sys.path.insert(0, str(_STUDIO_SERVER_ROOT))

from houyi_studio.server.execution.llm_execution_flow import LLMExecutionFlow  # noqa: E402


class TestLlmExecutionFlow:
    def test_normalize_usage(self):
        usage = LLMExecutionFlow._normalize_adapter_usage(
            {
                "input_tokens": 12,
                "completion_tokens": 8,
                "reasoning_tokens": 3,
            }
        )

        assert usage is not None
        assert usage["prompt_tokens"] == 12
        assert usage["completion_tokens"] == 8
        assert usage["total_tokens"] == 20

    def test_empty_usage(self):
        assert LLMExecutionFlow._normalize_adapter_usage(None) is None
