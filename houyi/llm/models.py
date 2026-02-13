"""Model registry: defaults, context windows, and provider identifiers.

This file provides fallback constants for the SDK. In production, model
lists SHOULD be fetched dynamically from provider APIs (e.g. /v1/models)
rather than hardcoded here. The constants below serve as:
  - Default model when none is configured
  - Context window lookup for token estimation
  - Provider identifiers (canonical keys)

Usage:
    from houyi.llm.models import DEFAULT_MODEL, MODEL_CONTEXT_WINDOWS

    model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Well-known model identifiers (fallback defaults; prefer dynamic fetch)
# ---------------------------------------------------------------------------

# DeepSeek (SiliconFlow)
DEEPSEEK_V3 = "deepseek-ai/DeepSeek-V3"
DEEPSEEK_R1 = "deepseek-ai/DeepSeek-R1"

# OpenAI
GPT_4O = "gpt-4o"
GPT_4O_MINI = "gpt-4o-mini"
GPT_4_TURBO = "gpt-4-turbo"
GPT_35_TURBO = "gpt-3.5-turbo"

# Anthropic
CLAUDE_35_SONNET = "claude-3-5-sonnet"
CLAUDE_35_HAIKU = "claude-3-5-haiku"

# Google
GEMINI_25_PRO = "gemini-2.5-pro"

# ---------------------------------------------------------------------------
# Provider identifiers (canonical lowercase keys used in config/env/API)
#
# These are the single source of truth for provider identification.
# Display names ("SiliconFlow", "OpenAI", etc.) are a UI/frontend concern
# and defined in PROVIDER_DISPLAY_NAMES below.
#
# Google has two distinct providers:
#   - google_ai: Google AI Studio / Gemini API (API key auth, direct access)
#   - vertex:    Google Cloud Vertex AI (GCP project + service account auth)
# ---------------------------------------------------------------------------
PROVIDER_SILICONFLOW = "siliconflow"
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_DEEPSEEK = "deepseek"
PROVIDER_GOOGLE_AI = "google_ai"  # Gemini API (direct, API key)
PROVIDER_VERTEX = "vertex"  # Vertex AI (GCP project)

# Human-readable display names for UI rendering
PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    PROVIDER_SILICONFLOW: "SiliconFlow",
    PROVIDER_OPENAI: "OpenAI",
    PROVIDER_ANTHROPIC: "Anthropic",
    PROVIDER_DEEPSEEK: "DeepSeek",
    PROVIDER_GOOGLE_AI: "Google AI (Gemini)",
    PROVIDER_VERTEX: "Vertex AI",
}

# ---------------------------------------------------------------------------
# Default model used across the SDK when no model is specified
# ---------------------------------------------------------------------------
DEFAULT_MODEL: str = DEEPSEEK_V3

# ---------------------------------------------------------------------------
# Context window sizes (max input tokens).
# Conservative defaults; override via context_window_override if needed.
# ---------------------------------------------------------------------------
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    DEEPSEEK_V3: 65_536,
    DEEPSEEK_R1: 65_536,
    GPT_4O: 128_000,
    GPT_4O_MINI: 128_000,
    GPT_4_TURBO: 128_000,
    GPT_35_TURBO: 16_385,
    CLAUDE_35_SONNET: 200_000,
    CLAUDE_35_HAIKU: 200_000,
    GEMINI_25_PRO: 1_048_576,
}

# Default context window for unknown models
DEFAULT_CONTEXT_WINDOW: int = 8_192

# Default output token reservation
DEFAULT_OUTPUT_RESERVE: int = 4_096

# ---------------------------------------------------------------------------
# Fallback token estimation ratios (when tiktoken is unavailable)
# ---------------------------------------------------------------------------
# English: ~4 chars per token
# CJK: ~1.5 chars per token (domestic models may be more efficient)
# Blended estimate for mixed text
CHARS_PER_TOKEN_ENGLISH: float = 4.0
CHARS_PER_TOKEN_CJK: float = 1.5
CHARS_PER_TOKEN_BLENDED: float = 3.0
