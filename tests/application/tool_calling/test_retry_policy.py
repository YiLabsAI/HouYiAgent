"""Tests for retry policy."""

from __future__ import annotations

from houyi.application.tool_calling.retry_policy import RetryPolicy, get_num_retries_from_policy


class TimeoutFailure(Exception):
    pass


class RateLimitFailure(Exception):
    pass


class AuthenticationFailure(Exception):
    pass


class BadRequestFailure(Exception):
    pass


class ContentPolicyFailure(Exception):
    pass


class InternalServerFailure(Exception):
    pass


def test_retry_policy_defaults_to_fallback() -> None:
    policy = RetryPolicy(default_retries=2)
    assert get_num_retries_from_policy(RuntimeError("boom"), policy) == 2


def test_retry_policy_uses_timeout_override() -> None:
    policy = RetryPolicy(default_retries=1, timeout_retries=3)
    assert get_num_retries_from_policy(TimeoutFailure("timeout"), policy) == 3


def test_retry_policy_uses_rate_limit_override() -> None:
    policy = RetryPolicy(default_retries=1, rate_limit_retries=4)
    assert get_num_retries_from_policy(RateLimitFailure("limit"), policy) == 4


def test_retry_policy_uses_auth_override() -> None:
    policy = RetryPolicy(default_retries=1, auth_retries=5)
    assert get_num_retries_from_policy(AuthenticationFailure("auth"), policy) == 5


def test_retry_policy_uses_bad_request_override() -> None:
    policy = RetryPolicy(default_retries=1, bad_request_retries=6)
    assert get_num_retries_from_policy(BadRequestFailure("bad"), policy) == 6


def test_retry_policy_uses_content_policy_override() -> None:
    policy = RetryPolicy(default_retries=1, content_policy_retries=7)
    assert get_num_retries_from_policy(ContentPolicyFailure("policy"), policy) == 7


def test_retry_policy_uses_internal_override() -> None:
    policy = RetryPolicy(default_retries=1, internal_error_retries=8)
    assert get_num_retries_from_policy(InternalServerFailure("internal"), policy) == 8
