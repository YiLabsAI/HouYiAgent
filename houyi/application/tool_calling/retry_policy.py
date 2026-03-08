"""Retry policy for tool-calling and execution retries."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RetryPolicy:
    """Configure retry counts per error type."""

    default_retries: int = 0
    timeout_retries: int | None = None
    rate_limit_retries: int | None = None
    auth_retries: int | None = None
    bad_request_retries: int | None = None
    content_policy_retries: int | None = None
    internal_error_retries: int | None = None

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any] | None) -> RetryPolicy:
        source = settings or {}
        return cls(
            default_retries=int(source.get("default_retries", 0)),
            timeout_retries=_coerce_optional_int(source.get("timeout_retries")),
            rate_limit_retries=_coerce_optional_int(source.get("rate_limit_retries")),
            auth_retries=_coerce_optional_int(source.get("auth_retries")),
            bad_request_retries=_coerce_optional_int(source.get("bad_request_retries")),
            content_policy_retries=_coerce_optional_int(source.get("content_policy_retries")),
            internal_error_retries=_coerce_optional_int(source.get("internal_error_retries")),
        )


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_num_retries_from_policy(exception: Exception, policy: RetryPolicy) -> int:
    """Return retries allowed for a given exception type."""

    name = exception.__class__.__name__.lower()
    if "timeout" in name and policy.timeout_retries is not None:
        return policy.timeout_retries
    if ("ratelimit" in name or "rate_limit" in name) and policy.rate_limit_retries is not None:
        return policy.rate_limit_retries
    if ("auth" in name or "authentication" in name) and policy.auth_retries is not None:
        return policy.auth_retries
    if ("badrequest" in name or "bad_request" in name) and policy.bad_request_retries is not None:
        return policy.bad_request_retries
    if (
        "contentpolicy" in name or "content_policy" in name
    ) and policy.content_policy_retries is not None:
        return policy.content_policy_retries
    if ("internal" in name or "server" in name) and policy.internal_error_retries is not None:
        return policy.internal_error_retries
    return policy.default_retries


def calculate_retry_delay(
    *,
    attempt: int,
    min_delay: float,
    max_delay: float,
    jitter: float,
) -> float:
    """Calculate exponential backoff delay with jitter."""

    base_delay = min_delay * (2.0**attempt)
    bounded_delay = min(max(base_delay, min_delay), max_delay)
    return bounded_delay + (jitter * random.random())
