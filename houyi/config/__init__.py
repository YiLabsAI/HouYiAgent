"""Centralized configuration management for Houyi.

Usage::

    from houyi.config import env

    api_key = env.siliconflow_api_key
    knowledge_dir = env.rag_knowledge_dir
"""

from typing import Any

from houyi.config.env_config import EnvConfig


class _LazyEnv:
    """Lazy proxy that defers EnvConfig initialization until first attribute access."""

    def __getattr__(self, name: str) -> Any:
        return getattr(EnvConfig.get(), name)

    def __repr__(self) -> str:
        return repr(EnvConfig.get())


env: EnvConfig = _LazyEnv()  # type: ignore[assignment]

__all__ = ["EnvConfig", "env"]
