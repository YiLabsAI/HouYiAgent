from unittest.mock import patch

import pytest

from houyi.application.context.context_recovery import (
    ContextRecoveryPolicy,
    RenderRecoveryPolicy,
)
from houyi.application.context.token_estimator import TokenEstimator
from houyi.application.context.types import PlannedContextUsage


@pytest.fixture(autouse=True, scope="module")
def _mock_tiktoken():
    """Mock tiktoken to avoid real encoding load, saving ~0.1s per TokenEstimator()."""
    with patch.object(TokenEstimator, "_try_load_encoding", return_value=None):
        yield


class TestContextRecoveryPolicy:
    def test_recovery_returns_latest(self):
        policy = ContextRecoveryPolicy()
        estimator = TokenEstimator(model="gpt-4o-mini")
        history_messages = [
            {"role": "user", "content": "older"},
            {"role": "user", "content": "latest"},
        ]

        fallback = policy.build_latest_message_recovery(
            history_messages,
            estimator=estimator,
        )

        assert fallback is not None
        llm_messages, usage = fallback
        assert llm_messages == [{"role": "user", "content": "latest"}]
        assert usage["used_tokens"] > 0
        assert usage["available_input_tokens"] == 0
        assert usage["block_breakdown"] == {"recent": usage["used_tokens"]}


class TestRenderRecoveryPolicy:
    def test_recovery_updates_usage(self):
        policy = RenderRecoveryPolicy()
        estimator = TokenEstimator(model="gpt-4o-mini")
        history_messages = [
            {"role": "assistant", "content": "older"},
            {"role": "user", "content": "latest"},
        ]
        usage = PlannedContextUsage(
            model=estimator.model,
            max_context_tokens=estimator.context_window,
            used_tokens=120,
            reserved_output_tokens=100,
            available_tokens=80,
            planned_prompt_tokens=120,
            available_input_tokens=80,
            block_breakdown={"recent": 120},
            dropped_blocks=[],
            drop_reasons={},
        )

        fallback = policy.apply_empty_render_recovery(
            history_messages,
            estimator=estimator,
            usage=usage,
        )

        assert fallback is not None
        llm_messages, fallback_usage = fallback
        assert llm_messages == [{"role": "user", "content": "latest"}]
        assert fallback_usage["used_tokens"] > 0
        assert fallback_usage["planned_prompt_tokens"] == fallback_usage["used_tokens"]
        assert fallback_usage["available_input_tokens"] >= 0
        assert fallback_usage["block_breakdown"] == {"recent": fallback_usage["used_tokens"]}
