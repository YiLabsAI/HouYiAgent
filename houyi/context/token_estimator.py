"""Token estimation for context window management.

Supports multiple estimation strategies:
- tiktoken-based (accurate, requires tiktoken package)
- Character-ratio fallback (fast, no dependencies)

Provides accurate token counting for context tracking.
Phase 2: Used by ContextCompressor to decide compression triggers.

"""

from __future__ import annotations

import logging
import unicodedata
from typing import Any

from houyi.llm.models import (
    CHARS_PER_TOKEN_CJK,
    CHARS_PER_TOKEN_ENGLISH,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_RESERVE,
    GPT_4_TURBO,
    GPT_4O,
    GPT_4O_MINI,
    GPT_35_TURBO,
    MODEL_CONTEXT_WINDOWS,
)

logger = logging.getLogger(__name__)


class TokenEstimator:
    """Estimates token counts for messages and strings.

    Attempts tiktoken first (accurate), falls back to character-ratio
    estimation if tiktoken is unavailable or the model is unsupported.

    Thread-safe: instances are stateless after __init__.
    """

    _tiktoken_available: bool | None = None

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        output_reserve: int = DEFAULT_OUTPUT_RESERVE,
        context_window_override: int | None = None,
    ):
        """Initialize token estimator.

        Args:
            model: Model identifier for tokenizer selection and context window lookup.
            output_reserve: Tokens reserved for model output.
            context_window_override: Override the default context window size.
        """
        self.model = model
        self.output_reserve = output_reserve
        self.context_window = context_window_override or self._lookup_context_window(model)
        self._encoding = self._try_load_encoding(model)

    @property
    def max_input_tokens(self) -> int:
        """Maximum tokens available for input (context_window - output_reserve)."""
        return max(0, self.context_window - self.output_reserve)

    def count_text(self, text: str) -> int:
        """Count tokens in a plain text string.

        Args:
            text: Input text.

        Returns:
            Estimated token count.
        """
        if not text:
            return 0
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        return self._fallback_count(text)

    def count_message(self, message: dict[str, Any]) -> int:
        """Count tokens in a single chat message dict.

        Follows OpenAI's token counting convention:
        each message = role tokens + content tokens + overhead (~4 tokens).

        Args:
            message: Dict with 'role' and 'content' keys.

        Returns:
            Estimated token count for this message.
        """
        overhead = 4  # <|im_start|>role\n...content...<|im_end|>\n
        role = message.get("role", "")
        content = message.get("content", "")
        if isinstance(content, list):
            # Multi-modal content (text + images); estimate text parts only
            text_parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            content = " ".join(text_parts)
        return overhead + self.count_text(role) + self.count_text(str(content))

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count total tokens across a list of messages.

        Args:
            messages: List of message dicts.

        Returns:
            Total estimated token count.
        """
        base_overhead = 3  # every reply is primed with <|im_start|>assistant<|message|>
        return base_overhead + sum(self.count_message(m) for m in messages)

    def would_exceed(self, messages: list[dict[str, Any]], threshold_ratio: float = 1.0) -> bool:
        """Check if messages would exceed the input token budget.

        Args:
            messages: List of message dicts.
            threshold_ratio: Fraction of max_input_tokens to use as threshold (default 1.0).
                             Use 0.7~0.8 for proactive compression triggers.

        Returns:
            True if estimated tokens exceed threshold.
        """
        threshold = int(self.max_input_tokens * threshold_ratio)
        return self.count_messages(messages) > threshold

    @staticmethod
    def _lookup_context_window(model: str) -> int:
        """Look up context window size for a model."""
        if model in MODEL_CONTEXT_WINDOWS:
            return MODEL_CONTEXT_WINDOWS[model]
        # Try partial match (e.g., "deepseek" in model name)
        model_lower = model.lower()
        for key, value in MODEL_CONTEXT_WINDOWS.items():
            if key.lower() in model_lower or model_lower in key.lower():
                return value
        logger.debug(
            "Unknown model '%s', using default %dk context window",
            model,
            DEFAULT_CONTEXT_WINDOW // 1024,
        )
        return DEFAULT_CONTEXT_WINDOW

    @classmethod
    def _try_load_encoding(cls, model: str) -> Any | None:
        """Try to load tiktoken encoding for the model."""
        if cls._tiktoken_available is False:
            return None
        try:
            import tiktoken

            cls._tiktoken_available = True
            # Map model names to tiktoken encoding names
            encoding_map = {
                GPT_4O: "o200k_base",
                GPT_4O_MINI: "o200k_base",
                GPT_4_TURBO: "cl100k_base",
                GPT_35_TURBO: "cl100k_base",
            }
            model_lower = model.lower()
            for key, enc_name in encoding_map.items():
                if key in model_lower:
                    return tiktoken.get_encoding(enc_name)
            # Default to cl100k_base (good approximation for most models)
            return tiktoken.get_encoding("cl100k_base")
        except ImportError:
            cls._tiktoken_available = False
            logger.info("tiktoken not available, using character-ratio estimation")
            return None
        except Exception as e:
            logger.warning("Failed to load tiktoken encoding: %s", e)
            return None

    @staticmethod
    def _fallback_count(text: str) -> int:
        """Estimate token count using character ratio.

        Uses a blended ratio that accounts for mixed CJK/Latin text.
        """
        if not text:
            return 0
        # Count CJK characters (roughly 1 token each)
        cjk_count = sum(1 for c in text if unicodedata.category(c).startswith("Lo"))
        non_cjk_len = len(text) - cjk_count
        cjk_tokens = int(cjk_count / CHARS_PER_TOKEN_CJK)
        eng_tokens = max(1, int(non_cjk_len / CHARS_PER_TOKEN_ENGLISH))
        return cjk_tokens + eng_tokens
