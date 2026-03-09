"""Unit tests for houyi.adapters.llm.retry — shared retry / backoff helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from houyi.adapters.llm.retry import (
    RETRY_BUCKET_CONNECT,
    RETRY_BUCKET_READ,
    RETRY_BUCKET_STATUS,
    RetryController,
    RetryPolicy,
    classify_transport_exception,
    exponential_backoff,
    is_retryable_status,
    parse_retry_after_seconds,
)


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

    def test_parse_retry_after_seconds_integer(self):
        assert parse_retry_after_seconds("3") == 3.0

    def test_parse_retry_after_seconds_http_date(self):
        future = datetime.now(UTC) + timedelta(seconds=5)
        value = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
        parsed = parse_retry_after_seconds(value)
        assert parsed is not None
        assert 0 <= parsed <= 6.0

    def test_classify_transport_exception(self):
        class ConnectBoom(Exception):
            pass

        class ReadTimeoutBoom(Exception):
            pass

        class RemoteProtocolBoom(Exception):
            pass

        assert classify_transport_exception(ConnectBoom()) == RETRY_BUCKET_CONNECT
        assert classify_transport_exception(ReadTimeoutBoom()) == RETRY_BUCKET_READ
        assert classify_transport_exception(RemoteProtocolBoom()) == RETRY_BUCKET_READ


class TestRetryController:
    def test_status_retry_respects_retry_after(self):
        controller = RetryController(
            RetryPolicy(total_retries=3, status_retries=3, backoff_base=0.1, backoff_cap=1.0)
        )
        decision = controller.on_status_code(
            429,
            method="POST",
            headers={"Retry-After": "2"},
        )
        assert decision.retry
        assert decision.bucket == RETRY_BUCKET_STATUS
        assert decision.delay_seconds == 2.0

    def test_connect_bucket_limit(self):
        class ConnectError(Exception):
            pass

        controller = RetryController(
            RetryPolicy(
                total_retries=3,
                connect_retries=1,
                read_retries=3,
                status_retries=3,
            )
        )
        first = controller.on_transport_exception(ConnectError(), method="POST")
        second = controller.on_transport_exception(ConnectError(), method="POST")
        assert first.retry is True
        assert second.retry is False

    def test_total_retry_limit(self):
        controller = RetryController(RetryPolicy(total_retries=1, status_retries=3))
        first = controller.on_status_code(429, method="POST", headers={})
        second = controller.on_status_code(503, method="POST", headers={})
        assert first.retry is True
        assert second.retry is False

    def test_method_not_allowed(self):
        controller = RetryController(
            RetryPolicy(total_retries=3, status_retries=3, allowed_methods=frozenset({"GET"}))
        )
        decision = controller.on_status_code(429, method="POST", headers={})
        assert decision.retry is False
