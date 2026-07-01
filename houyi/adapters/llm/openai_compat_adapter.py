"""OpenAI-compatible adapter for OpenAI-style providers."""

from __future__ import annotations

import os

from houyi.adapters.llm.openai_compat_base import OpenAICompatAdapterBase
from houyi.infrastructure.config.env_config import (
    ENV_OPENAI_API_KEY,
    ENV_OPENAI_BASE_URL,
    ENV_OPENAI_ORG,
)


class OpenAICompatibleAdapter(OpenAICompatAdapterBase):
    """Adapter for OpenAI-compatible providers (OpenAI-style APIs)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4",
        base_url: str | None = None,
        organization: str | None = None,
        default_headers: dict[str, str] | None = None,
        strict_message_string_contract: bool = False,
    ) -> None:
        self.api_key = api_key or os.getenv(ENV_OPENAI_API_KEY)
        self.model = model
        self.base_url = base_url or os.getenv(ENV_OPENAI_BASE_URL)
        self.organization = organization or os.getenv(ENV_OPENAI_ORG)
        self.default_headers = default_headers or {}
        self.strict_message_string_contract = strict_message_string_contract
        self.last_usage: dict[str, object] | None = None
        self.last_finish_reason: str | None = None

        if not self.api_key:
            raise ValueError(
                "OpenAI-compatible API key not provided. "
                "Set OPENAI_API_KEY or pass api_key parameter."
            )

        try:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
                default_headers=self.default_headers or None,
                timeout=180.0,
            )
        except ImportError as exc:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai>=1.0.0"
            ) from exc
