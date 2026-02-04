"""Agentic mode for Houyi RAG.

Agentic mode uses LLM + Agent tools (grep, read_file) for retrieval,
without requiring pre-built vector indexes.
"""

from houyi.rag.agentic.mode import AgenticMode

__all__ = ["AgenticMode"]
