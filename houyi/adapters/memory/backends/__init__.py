"""Memory storage backend registry and factory."""

from __future__ import annotations

from typing import Any

from houyi.adapters.memory.backends.base import MemoryBackend
from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend

BACKEND_REGISTRY: dict[str, type[MemoryBackend]] = {
    "sqlite": SQLiteMemoryBackend,
}


def create_backend(name: str = "sqlite", **kwargs: Any) -> MemoryBackend:
    """Create a memory backend by registered name.

    Args:
        name: Backend identifier (default "sqlite").
        **kwargs: Passed to the backend constructor.

    Raises:
        KeyError: If *name* is not registered.
    """
    cls = BACKEND_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(BACKEND_REGISTRY))
        msg = f"Unknown memory backend {name!r}. Available: {available}"
        raise KeyError(msg)
    return cls(**kwargs)


__all__ = [
    "BACKEND_REGISTRY",
    "MemoryBackend",
    "SQLiteMemoryBackend",
    "create_backend",
]
