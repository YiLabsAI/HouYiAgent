"""Unit tests for houyi.adapters.llm.models — model registry and provider identifiers.

Tests cover:
- All provider identifiers are unique and lowercase
- All model identifiers are non-empty strings
- MODEL_CONTEXT_WINDOWS keys match declared model constants
- PROVIDER_DISPLAY_NAMES covers all provider constants
- DEFAULT_MODEL is a valid model in MODEL_CONTEXT_WINDOWS
- Token estimation ratios are sane
"""

from __future__ import annotations

from houyi.adapters.llm.models import (
    # Token estimation
    CHARS_PER_TOKEN_BLENDED,
    CHARS_PER_TOKEN_CJK,
    CHARS_PER_TOKEN_ENGLISH,
    # Model identifiers
    CLAUDE_35_HAIKU,
    CLAUDE_35_SONNET,
    DEEPSEEK_R1,
    DEEPSEEK_V3,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_RESERVE,
    GEMINI_25_PRO,
    GPT_4_TURBO,
    GPT_4O,
    GPT_4O_MINI,
    GPT_35_TURBO,
    MODEL_CONTEXT_WINDOWS,
    # Provider identifiers
    PROVIDER_ANTHROPIC,
    PROVIDER_DASHSCOPE,
    PROVIDER_DEEPSEEK,
    PROVIDER_DISPLAY_NAMES,
    PROVIDER_GOOGLE_AI,
    PROVIDER_OPENAI,
    PROVIDER_OPENAI_COMPAT,
    PROVIDER_SILICONFLOW,
    PROVIDER_VERTEX,
)


class TestProviderIdentifiers:
    """Provider constants must be unique, lowercase, and have display names."""

    ALL_PROVIDERS = [
        PROVIDER_SILICONFLOW,
        PROVIDER_OPENAI,
        PROVIDER_OPENAI_COMPAT,
        PROVIDER_ANTHROPIC,
        PROVIDER_DEEPSEEK,
        PROVIDER_DASHSCOPE,
        PROVIDER_GOOGLE_AI,
        PROVIDER_VERTEX,
    ]

    def test_all_lowercase(self):
        for p in self.ALL_PROVIDERS:
            assert p == p.lower(), f"Provider '{p}' must be lowercase"

    def test_all_unique(self):
        assert len(set(self.ALL_PROVIDERS)) == len(self.ALL_PROVIDERS)

    def test_display_names_cover_providers(self):
        for p in self.ALL_PROVIDERS:
            assert p in PROVIDER_DISPLAY_NAMES, f"Missing display name for '{p}'"
            assert isinstance(PROVIDER_DISPLAY_NAMES[p], str)
            assert len(PROVIDER_DISPLAY_NAMES[p]) > 0

    def test_no_extra_display_names(self):
        for key in PROVIDER_DISPLAY_NAMES:
            assert key in self.ALL_PROVIDERS, f"Extra display name '{key}' has no provider constant"


class TestModelIdentifiers:
    """Model constants must be non-empty strings in MODEL_CONTEXT_WINDOWS."""

    ALL_MODELS = [
        DEEPSEEK_V3,
        DEEPSEEK_R1,
        GPT_4O,
        GPT_4O_MINI,
        GPT_4_TURBO,
        GPT_35_TURBO,
        CLAUDE_35_SONNET,
        CLAUDE_35_HAIKU,
        GEMINI_25_PRO,
    ]

    def test_all_non_empty_strings(self):
        for m in self.ALL_MODELS:
            assert isinstance(m, str) and len(m) > 0

    def test_all_in_context_windows(self):
        for m in self.ALL_MODELS:
            assert m in MODEL_CONTEXT_WINDOWS, f"Model '{m}' missing from MODEL_CONTEXT_WINDOWS"

    def test_context_windows_positive(self):
        for model, window in MODEL_CONTEXT_WINDOWS.items():
            assert isinstance(window, int) and window > 0, f"Invalid window for '{model}': {window}"

    def test_default_model_is_valid(self):
        assert DEFAULT_MODEL in MODEL_CONTEXT_WINDOWS

    def test_default_context_window_positive(self):
        assert DEFAULT_CONTEXT_WINDOW > 0

    def test_default_output_reserve_positive(self):
        assert DEFAULT_OUTPUT_RESERVE > 0
        assert DEFAULT_OUTPUT_RESERVE < DEFAULT_CONTEXT_WINDOW


class TestTokenEstimationRatios:
    """Token estimation ratios must be positive and ordered correctly."""

    def test_all_positive(self):
        assert CHARS_PER_TOKEN_ENGLISH > 0
        assert CHARS_PER_TOKEN_CJK > 0
        assert CHARS_PER_TOKEN_BLENDED > 0

    def test_cjk_less_than_english(self):
        assert CHARS_PER_TOKEN_CJK < CHARS_PER_TOKEN_ENGLISH

    def test_blended_between_cjk_english(self):
        assert CHARS_PER_TOKEN_CJK <= CHARS_PER_TOKEN_BLENDED <= CHARS_PER_TOKEN_ENGLISH
