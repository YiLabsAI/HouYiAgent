"""Token estimation for context window management.

Supports multiple estimation strategies:
- tiktoken-based (accurate, requires tiktoken package)
- Character-ratio fallback (fast, no dependencies)

Provides accurate token counting for context tracking.
Phase 2: Used by ContextCompressor to decide compression triggers.

"""

from __future__ import annotations

import logging
import os
import unicodedata
from pathlib import Path
from typing import Any

from houyi.adapters.llm.models import (
    CHARS_PER_TOKEN_CJK,
    CHARS_PER_TOKEN_ENGLISH,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_RESERVE,
    GPT_4_TURBO,
    GPT_4O,
    GPT_4O_MINI,
    GPT_35_TURBO,
    resolve_model_context_window,
)

logger = logging.getLogger(__name__)

# Pin tiktoken's BPE cache to a stable, shared, non-temp directory (mirrors how
# local embeddings live under ~/.cache/huggingface). tiktoken's own default is
# tempfile.gettempdir()/data-gym-cache, which macOS wipes on reboot; a cold
# cache plus an unreachable download endpoint (openaipublic.blob SSL-fails
# here) then turns every TokenEstimator init into a network timeout. setdefault
# respects an explicit TIKTOKEN_CACHE_DIR from the environment when present.
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path.home() / ".cache" / "tiktoken"))


class TokenEstimator:
    """Estimates token counts for messages and strings.

    Attempts tiktoken first (accurate), falls back to character-ratio
    estimation if tiktoken is unavailable or the model is unsupported.

    Thread-safe: instances are stateless after __init__.
    """

    _tiktoken_available: bool | None = None
    # Per-encoding cache so each encoding is loaded at most once per process.
    # A failed load (e.g. the BPE file is not cached locally and the download
    # endpoint is unreachable) is also memoized, otherwise every TokenEstimator
    # instance would re-attempt the network call and pay its timeout again.
    _encodings: dict[str, Any] = {}
    _encoding_failures: set[str] = set()

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
        resolved = resolve_model_context_window(model)
        if resolved is not None:
            return resolved
        logger.debug(
            "Unknown model '%s', using default %dk context window",
            model,
            DEFAULT_CONTEXT_WINDOW // 1024,
        )
        return DEFAULT_CONTEXT_WINDOW

    @classmethod
    def _try_load_encoding(cls, model: str) -> Any | None:
        """Try to load tiktoken encoding for the model.

        Results are memoized per encoding name: a successful load is kept for
        the process, and a failed load is remembered so it is not retried on
        every subsequent instance (a missing local BPE cache plus an
        unreachable download endpoint would otherwise cost a full network
        timeout per TokenEstimator).
        """
        if cls._tiktoken_available is False:
            return None
        try:
            import tiktoken

            cls._tiktoken_available = True
        except ImportError:
            cls._tiktoken_available = False
            logger.info("tiktoken not available, using character-ratio estimation")
            return None

        enc_name = cls._encoding_name_for(model)
        if enc_name in cls._encoding_failures:
            return None
        if enc_name in cls._encodings:
            return cls._encodings[enc_name]
        try:
            encoding = tiktoken.get_encoding(enc_name)
        except Exception as e:
            cls._encoding_failures.add(enc_name)
            logger.warning("Failed to load tiktoken encoding %s: %s", enc_name, e)
            return None
        cls._encodings[enc_name] = encoding
        return encoding

    @staticmethod
    def _encoding_name_for(model: str) -> str:
        """Map a model id to its tiktoken encoding name (default cl100k_base)."""
        encoding_map = {
            GPT_4O: "o200k_base",
            GPT_4O_MINI: "o200k_base",
            GPT_4_TURBO: "cl100k_base",
            GPT_35_TURBO: "cl100k_base",
        }
        model_lower = model.lower()
        for key, enc_name in encoding_map.items():
            if key in model_lower:
                return enc_name
        return "cl100k_base"

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
