"""Shared models for indexed RAG mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from houyi.rag.types import RetrievalStrategy


@dataclass
class RetrievalTaskResult:
    """Result of a single retrieval task."""

    strategy: RetrievalStrategy
    strategy_name: str
    results: list[Any] = field(default_factory=list)
    success: bool = True
    timed_out: bool = False
    error: str | None = None
    duration_ms: float = 0.0
