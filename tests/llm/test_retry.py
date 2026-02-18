"""Unit tests for houyi.llm.retry — shared retry / backoff helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from houyi.llm.retry import exponential_backoff, is_retryable_status


class TestRetryHelpers:
    """Test shared retry/backoff helper functions."""

    def test_retryable_status_codes(self):
        assert is_retryable_status(429)
        assert is_retryable_status(500)
        assert is_retryable_status(502)
        assert is_retryable_status(503)
        assert is_retryable_status(504)
        assert not is_retryable_status(400)
        assert not is_retryable_status(401)
        assert not is_retryable_status(403)
        assert not is_retryable_status(404)
        assert not is_retryable_status(200)

    @pytest.mark.asyncio
    async def test_exponential_backoff_sleeps(self):
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await exponential_backoff(0, base=1.0, cap=10.0)
            mock_sleep.assert_called_once()
            delay = mock_sleep.call_args[0][0]
            assert 0 <= delay <= 1.0  # attempt 0: max delay = min(1*2^0, 10) = 1.0

    @pytest.mark.asyncio
    async def test_exponential_backoff_caps(self):
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await exponential_backoff(10, base=1.0, cap=5.0)
            mock_sleep.assert_called_once()
            delay = mock_sleep.call_args[0][0]
            assert 0 <= delay <= 5.0  # capped at 5.0
