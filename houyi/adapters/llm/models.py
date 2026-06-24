"""Model registry: defaults, context windows, and provider identifiers.

This file provides fallback constants for the SDK. In production, model
lists SHOULD be fetched dynamically from provider APIs (e.g. /v1/models)
rather than hardcoded here. The constants below serve as:
 - Default model when none is configured
 - Context window lookup for token estimation
 - Provider identifiers (canonical keys)

Usage:
 from houyi.adapters.llm.models import DEFAULT_MODEL, MODEL_CONTEXT_WINDOWS

 model = os.getenv(ENV_SILICONFLOW_MODEL, DEFAULT_MODEL)
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Well-known model identifiers (fallback defaults; prefer dynamic fetch)
# ---------------------------------------------------------------------------

# DeepSeek (SiliconFlow)
DEEPSEEK_V3 = "deepseek-ai/DeepSeek-V3"
DEEPSEEK_R1 = "deepseek-ai/DeepSeek-R1"
DEEPSEEK_V3_2 = "deepseek-ai/DeepSeek-V3.2"

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
GEMINI_3_PRO_PREVIEW = "gemini-3-pro-preview"
GEMINI_3_FLASH_PREVIEW = "gemini-3-flash-preview"
GEMINI_31_PRO_PREVIEW = "gemini-3.1-pro-preview"

# Moonshot (Pro/ prefix required on SiliconFlow for private-tier access)
KIMI_K2_5 = "Pro/moonshotai/Kimi-K2.5"

# Qwen
QWEN3_32B = "Qwen/Qwen3-32B"
QWQ_32B = "Qwen/QwQ-32B"

# MiniMax
MINIMAX_M25 = "MiniMax-M2.5"

# ZAI / GLM
GLM_47 = "GLM-4.7"
GLM_5 = "GLM-5"

# ---------------------------------------------------------------------------
# Provider identifiers (canonical lowercase keys used in config/env/API)
#
# These are the single source of truth for provider identification.
# Display names ("SiliconFlow", "OpenAI", etc.) are a UI/frontend concern
# and defined in PROVIDER_DISPLAY_NAMES below.
#
# Google has two distinct providers:
# - google_ai: Google AI Studio / Gemini API (API key auth, direct access)
# - vertex: Google Cloud Vertex AI (GCP project + service account auth)
# ---------------------------------------------------------------------------
PROVIDER_SILICONFLOW = "siliconflow"
PROVIDER_OPENAI = "openai"
PROVIDER_OPENAI_COMPAT = "openai_compat"  # generic OpenAI-compatible endpoint
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_DEEPSEEK = "deepseek"
PROVIDER_DASHSCOPE = "dashscope"  # Alibaba Cloud Bailian (OpenAI-compatible)
PROVIDER_GOOGLE_AI = "google_ai"  # Gemini API (direct, API key)
PROVIDER_VERTEX = "vertex"  # Vertex AI (GCP project)

# Adapter mode identifiers (tool-call subsystem)
ADAPTER_REAL = "real"  # use live LLM adapter
ADAPTER_FAKE = "fake"  # deterministic stub for E2E tests

# Human-readable display names for UI rendering
PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    PROVIDER_SILICONFLOW: "SiliconFlow",
    PROVIDER_OPENAI: "OpenAI",
    PROVIDER_OPENAI_COMPAT: "OpenAI Compatible",
    PROVIDER_ANTHROPIC: "Anthropic",
    PROVIDER_DEEPSEEK: "DeepSeek",
    PROVIDER_DASHSCOPE: "Bailian (DashScope)",
    PROVIDER_GOOGLE_AI: "Google AI (Gemini)",
    PROVIDER_VERTEX: "Vertex AI",
}

# ---------------------------------------------------------------------------
# Default model used across the SDK when no model is specified
# ---------------------------------------------------------------------------
DEFAULT_MODEL: str = DEEPSEEK_V3_2

# ---------------------------------------------------------------------------
# Context window sizes (max input tokens).
# Conservative defaults; override via context_window_override if needed.
# ---------------------------------------------------------------------------
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    DEEPSEEK_V3: 65_536,
    DEEPSEEK_R1: 65_536,
    DEEPSEEK_V3_2: 65_536,
    GPT_4O: 128_000,
    GPT_4O_MINI: 128_000,
    GPT_4_TURBO: 128_000,
    GPT_35_TURBO: 16_385,
    CLAUDE_35_SONNET: 200_000,
    CLAUDE_35_HAIKU: 200_000,
    GEMINI_25_PRO: 1_048_576,
    GEMINI_3_PRO_PREVIEW: 1_048_576,
    GEMINI_3_FLASH_PREVIEW: 1_048_576,
    GEMINI_31_PRO_PREVIEW: 1_048_576,
    KIMI_K2_5: 131_072,
    QWEN3_32B: 131_072,
    QWQ_32B: 131_072,
    MINIMAX_M25: 1_000_000,
    GLM_47: 128_000,
    GLM_5: 128_000,
}

MODEL_ID_ALIASES: dict[str, str] = {
    "moonshotai/kimi-k2.5".lower(): KIMI_K2_5,
    "qwen3-32b".lower(): QWEN3_32B,
    "qwen/qwen3-32b".lower(): QWEN3_32B,
    "qwq-32b".lower(): QWQ_32B,
    "qwen/qwq-32b".lower(): QWQ_32B,
    "zai-org/glm-4.7".lower(): GLM_47,
    "zai-org/glm-5".lower(): GLM_5,
}

MODEL_CONTEXT_WINDOW_PATTERNS: tuple[tuple[str, int], ...] = (
    (DEEPSEEK_V3.lower(), MODEL_CONTEXT_WINDOWS[DEEPSEEK_V3]),
    (DEEPSEEK_R1.lower(), MODEL_CONTEXT_WINDOWS[DEEPSEEK_R1]),
    (DEEPSEEK_V3_2.lower(), MODEL_CONTEXT_WINDOWS[DEEPSEEK_V3_2]),
    (GPT_4O_MINI.lower(), MODEL_CONTEXT_WINDOWS[GPT_4O_MINI]),
    (GPT_4O.lower(), MODEL_CONTEXT_WINDOWS[GPT_4O]),
    (GPT_4_TURBO.lower(), MODEL_CONTEXT_WINDOWS[GPT_4_TURBO]),
    (GPT_35_TURBO.lower(), MODEL_CONTEXT_WINDOWS[GPT_35_TURBO]),
    (CLAUDE_35_SONNET.lower(), MODEL_CONTEXT_WINDOWS[CLAUDE_35_SONNET]),
    (CLAUDE_35_HAIKU.lower(), MODEL_CONTEXT_WINDOWS[CLAUDE_35_HAIKU]),
    (GEMINI_31_PRO_PREVIEW.lower(), MODEL_CONTEXT_WINDOWS[GEMINI_31_PRO_PREVIEW]),
    (GEMINI_3_PRO_PREVIEW.lower(), MODEL_CONTEXT_WINDOWS[GEMINI_3_PRO_PREVIEW]),
    (GEMINI_3_FLASH_PREVIEW.lower(), MODEL_CONTEXT_WINDOWS[GEMINI_3_FLASH_PREVIEW]),
    (GEMINI_25_PRO.lower(), MODEL_CONTEXT_WINDOWS[GEMINI_25_PRO]),
    (KIMI_K2_5.lower(), MODEL_CONTEXT_WINDOWS[KIMI_K2_5]),
    (QWEN3_32B.lower(), MODEL_CONTEXT_WINDOWS[QWEN3_32B]),
    (QWQ_32B.lower(), MODEL_CONTEXT_WINDOWS[QWQ_32B]),
    (MINIMAX_M25.lower(), MODEL_CONTEXT_WINDOWS[MINIMAX_M25]),
    (GLM_47.lower(), MODEL_CONTEXT_WINDOWS[GLM_47]),
    (GLM_5.lower(), MODEL_CONTEXT_WINDOWS[GLM_5]),
)


def normalize_model_id(model: str) -> str:
    raw = str(model or "").strip()
    if not raw:
        return ""
    without_tag = raw.split(":", 1)[0].strip()
    alias = MODEL_ID_ALIASES.get(without_tag.lower())
    if alias:
        return alias
    normalized = without_tag
    if normalized.lower().startswith("pro/"):
        normalized = normalized.split("/", 1)[1].strip()
    alias = MODEL_ID_ALIASES.get(normalized.lower())
    if alias:
        return alias
    tail = normalized.rsplit("/", 1)[-1].strip() or normalized
    alias = MODEL_ID_ALIASES.get(tail.lower())
    if alias:
        return alias
    return tail


def _resolve_family_context_window(model_lower: str) -> int | None:
    if "gemini" in model_lower and (
        "2.5" in model_lower
        or "3.1" in model_lower
        or "3-pro" in model_lower
        or "3-flash" in model_lower
        or "flash" in model_lower
        or "pro" in model_lower
    ):
        return 1_048_576
    if "minimax" in model_lower:
        return 1_000_000
    if re.search(r"\bglm[-\s_/]*5\b", model_lower) or re.search(
        r"\bglm[-\s_/]*4\.7\b", model_lower
    ):
        return 128_000
    if "kimi-k2.5" in model_lower or ("kimi" in model_lower and "2.5" in model_lower):
        return 131_072
    if "qwen3-32b" in model_lower or "qwq-32b" in model_lower:
        return 131_072
    return None


def resolve_model_context_window(model: str) -> int | None:
    normalized = normalize_model_id(model)
    if not normalized:
        return None
    if normalized in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[normalized]
    model_lower = normalized.lower()
    alias = MODEL_ID_ALIASES.get(model_lower)
    if alias and alias in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[alias]
    for pattern, value in MODEL_CONTEXT_WINDOW_PATTERNS:
        if pattern in model_lower or model_lower in pattern:
            return value
    return _resolve_family_context_window(model_lower)


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
