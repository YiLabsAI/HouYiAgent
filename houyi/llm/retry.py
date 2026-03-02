"""Shared retry / backoff helpers for LLM adapters.

Provides exponential backoff with full-jitter strategy for transient
HTTP errors (429, 500, etc.).
"""

from __future__ import annotations

import asyncio
import email.utils
import logging
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retryable HTTP status codes
# ---------------------------------------------------------------------------

RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})

# ---------------------------------------------------------------------------
# Default retry parameters
# ---------------------------------------------------------------------------

DEFAULT_MAX_RETRIES: int = 3
DEFAULT_RETRY_BASE_DELAY: float = 1.0  # seconds
DEFAULT_RETRY_MAX_DELAY: float = 10.0  # seconds
DEFAULT_ALLOWED_RETRY_METHODS: frozenset[str] = frozenset(
    {"GET", "HEAD", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"}
)

RETRY_BUCKET_CONNECT = "connect"
RETRY_BUCKET_READ = "read"
RETRY_BUCKET_STATUS = "status"
RETRY_BUCKET_OTHER = "other"

# ---------------------------------------------------------------------------
# Vertex AI model limit
# ---------------------------------------------------------------------------

VERTEX_MAX_OUTPUT_TOKENS: int = 65_536

# ---------------------------------------------------------------------------
# SSE protocol (OpenAI-compatible streaming)
# ---------------------------------------------------------------------------

SSE_DATA_PREFIX = "data: "
SSE_DONE_SIGNAL = "[DONE]"

# ---------------------------------------------------------------------------
# OpenAI-compatible usage response keys
# ---------------------------------------------------------------------------

USAGE_KEY_PROMPT_TOKENS = "prompt_tokens"
USAGE_KEY_COMPLETION_TOKENS = "completion_tokens"
USAGE_KEY_TOTAL_TOKENS = "total_tokens"


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------


def is_retryable_status(status_code: int) -> bool:
    """Check if an HTTP status code is retryable."""
    return status_code in RETRYABLE_STATUS_CODES


def parse_retry_after_seconds(value: str | None) -> float | None:
    """Parse Retry-After header value into seconds.

    Supports integer seconds and HTTP-date formats.
    """
    if not value:
        return None

    stripped = value.strip()
    if not stripped:
        return None

    try:
        seconds = int(stripped)
        return float(max(0, seconds))
    except ValueError:
        pass

    try:
        dt = email.utils.parsedate_to_datetime(stripped)
    except (TypeError, ValueError, IndexError):
        return None

    if dt is None:
        return None

    now = time.time()
    timestamp = dt.timestamp()
    return max(0.0, timestamp - now)


def compute_backoff_delay(
    attempt: int,
    *,
    base: float = DEFAULT_RETRY_BASE_DELAY,
    cap: float = DEFAULT_RETRY_MAX_DELAY,
) -> float:
    """Compute full-jitter exponential backoff delay for a retry attempt."""
    delay = min(base * (2**attempt), cap)
    return random.uniform(0, delay)


@dataclass(slots=True)
class RetryPolicy:
    """Retry policy inspired by urllib3's category-based model."""

    total_retries: int = DEFAULT_MAX_RETRIES
    connect_retries: int | None = None
    read_retries: int | None = None
    status_retries: int | None = None
    other_retries: int | None = None
    status_forcelist: frozenset[int] = field(default_factory=lambda: RETRYABLE_STATUS_CODES)
    allowed_methods: frozenset[str] | None = field(
        default_factory=lambda: DEFAULT_ALLOWED_RETRY_METHODS
    )
    backoff_base: float = DEFAULT_RETRY_BASE_DELAY
    backoff_cap: float = DEFAULT_RETRY_MAX_DELAY
    respect_retry_after_header: bool = True

    def retries_for_bucket(self, bucket: str) -> int:
        if bucket == RETRY_BUCKET_CONNECT:
            specific = self.connect_retries
        elif bucket == RETRY_BUCKET_READ:
            specific = self.read_retries
        elif bucket == RETRY_BUCKET_STATUS:
            specific = self.status_retries
        else:
            specific = self.other_retries
        return self.total_retries if specific is None else max(0, int(specific))

    def allows_method(self, method: str | None) -> bool:
        if self.allowed_methods is None:
            return True
        if method is None:
            return True
        return method.upper() in self.allowed_methods


@dataclass(slots=True)
class RetryDecision:
    """Decision returned by RetryController for a failure event."""

    retry: bool
    delay_seconds: float = 0.0
    bucket: str = RETRY_BUCKET_OTHER
    reason: str = ""


def classify_transport_exception(exc: Exception) -> str:
    """Classify transport exception into retry buckets."""
    name = exc.__class__.__name__.lower()
    if "connect" in name:
        return RETRY_BUCKET_CONNECT
    if "timeout" in name or "read" in name or "protocol" in name:
        return RETRY_BUCKET_READ
    return RETRY_BUCKET_OTHER


class RetryController:
    """Stateful retry controller with total + per-bucket budgets."""

    def __init__(self, policy: RetryPolicy | None = None) -> None:
        self.policy = policy or RetryPolicy()
        self._total_retries_used = 0
        self._bucket_retries_used: dict[str, int] = {
            RETRY_BUCKET_CONNECT: 0,
            RETRY_BUCKET_READ: 0,
            RETRY_BUCKET_STATUS: 0,
            RETRY_BUCKET_OTHER: 0,
        }

    @property
    def retries_used(self) -> int:
        return self._total_retries_used

    def on_transport_exception(
        self,
        exc: Exception,
        *,
        method: str | None = None,
    ) -> RetryDecision:
        bucket = classify_transport_exception(exc)
        return self._evaluate_retry(bucket=bucket, method=method, retry_after_seconds=None)

    def on_status_code(
        self,
        status_code: int,
        *,
        method: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> RetryDecision:
        if status_code not in self.policy.status_forcelist:
            return RetryDecision(
                retry=False,
                bucket=RETRY_BUCKET_STATUS,
                reason=f"status {status_code} not retryable",
            )

        retry_after_seconds: float | None = None
        if headers and self.policy.respect_retry_after_header:
            retry_after_seconds = parse_retry_after_seconds(headers.get("Retry-After"))

        return self._evaluate_retry(
            bucket=RETRY_BUCKET_STATUS,
            method=method,
            retry_after_seconds=retry_after_seconds,
        )

    def _evaluate_retry(
        self,
        *,
        bucket: str,
        method: str | None,
        retry_after_seconds: float | None,
    ) -> RetryDecision:
        if not self.policy.allows_method(method):
            return RetryDecision(retry=False, bucket=bucket, reason="method not allowed")

        if self._total_retries_used >= self.policy.total_retries:
            return RetryDecision(retry=False, bucket=bucket, reason="total retries exhausted")

        bucket_limit = self.policy.retries_for_bucket(bucket)
        if self._bucket_retries_used[bucket] >= bucket_limit:
            return RetryDecision(retry=False, bucket=bucket, reason=f"{bucket} retries exhausted")

        if retry_after_seconds is not None:
            delay = retry_after_seconds
        else:
            delay = compute_backoff_delay(
                self._total_retries_used,
                base=self.policy.backoff_base,
                cap=self.policy.backoff_cap,
            )

        self._total_retries_used += 1
        self._bucket_retries_used[bucket] += 1
        return RetryDecision(
            retry=True, delay_seconds=delay, bucket=bucket, reason="retry scheduled"
        )


async def exponential_backoff(
    attempt: int,
    base: float = DEFAULT_RETRY_BASE_DELAY,
    cap: float = DEFAULT_RETRY_MAX_DELAY,
) -> None:
    """Sleep with exponential backoff + jitter (full-jitter strategy).

    Args:
        attempt: Zero-based retry attempt number.
        base: Base delay in seconds.
        cap: Maximum delay cap in seconds.
    """
    jitter = compute_backoff_delay(attempt, base=base, cap=cap)
    logger.info("Retry attempt %d: backing off %.2fs", attempt + 1, jitter)
    await asyncio.sleep(jitter)
