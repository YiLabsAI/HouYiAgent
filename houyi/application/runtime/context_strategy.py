"""Context management strategies for agent execution."""

from __future__ import annotations

from pydantic import BaseModel


class ContextStrategy(BaseModel):
    """Controls how conversation context is trimmed during agent tool-loops.

    Attributes:
        keep_tool_result: Number of recent tool-result messages to retain.
            ``-1`` means keep all (no truncation).
        context_compress_limit: Token count at which old messages are
            summarised into a compressed prefix.  ``0`` disables compression.
        max_context_tokens: Hard upper bound on total context window tokens.
        summary_on_limit: When *True*, generate a summary of dropped
            messages instead of silently discarding them.
    """

    keep_tool_result: int = -1
    context_compress_limit: int = 0
    max_context_tokens: int = 128_000
    summary_on_limit: bool = True
