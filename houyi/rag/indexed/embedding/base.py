"""Embedding protocols and abstract base classes for indexed embedders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Protocol, runtime_checkable

ProgressCallback = Callable[[int, int, int], None]


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        raise NotImplementedError

    async def embed_batch(
        self,
        texts: list[str],
        progress_callback: ProgressCallback | None = None,
    ) -> list[list[float]]:
        """Embed multiple texts."""
        raise NotImplementedError


class BaseEmbedder(ABC):
    """Abstract base class for embedders."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        raise NotImplementedError

    async def embed_batch(
        self,
        texts: list[str],
        progress_callback: ProgressCallback | None = None,
    ) -> list[list[float]]:
        """Embed multiple texts (default: sequential)."""
        results = []
        for i, text in enumerate(texts):
            results.append(await self.embed(text))
            if progress_callback:
                progress_callback(i + 1, len(texts), 1)
        return results
