"""Indexed embedding providers, protocols, and factory helpers."""

from houyi.rag.indexed.embedding.api import APIEmbedder
from houyi.rag.indexed.embedding.base import BaseEmbedder, Embedder, ProgressCallback
from houyi.rag.indexed.embedding.factory import create_embedder
from houyi.rag.indexed.embedding.gemini import GeminiEmbedder
from houyi.rag.indexed.embedding.local import LocalEmbedder

__all__ = [
    "APIEmbedder",
    "BaseEmbedder",
    "Embedder",
    "GeminiEmbedder",
    "LocalEmbedder",
    "ProgressCallback",
    "create_embedder",
]
