"""Shared retry / backoff helpers for LLM adapters.

Provides exponential backoff with full-jitter strategy for transient
HTTP errors (429, 500, etc.).
"""

from __future__ import annotations

import asyncio
import logging
import random

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
    delay = min(base * (2**attempt), cap)
    jitter = random.uniform(0, delay)
    logger.info("Retry attempt %d: backing off %.2fs", attempt + 1, jitter)
    await asyncio.sleep(jitter)
