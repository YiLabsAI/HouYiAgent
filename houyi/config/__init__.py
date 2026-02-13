"""Centralized configuration management for Houyi.

Usage::

    from houyi.config import env

    api_key = env.siliconflow_api_key
    knowledge_dir = env.rag_knowledge_dir
"""

from houyi.config.env_config import EnvConfig

# Module-level convenience accessor (lazy singleton)
env: EnvConfig = EnvConfig.get()

__all__ = ["EnvConfig", "env"]
