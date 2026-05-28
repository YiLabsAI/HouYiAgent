"""Shared fixtures for integration tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from dotenv import load_dotenv

from houyi.infrastructure.config.env_config import EnvConfig

load_dotenv()


@pytest.fixture(autouse=True, scope="module")
def _reset_env_config_singleton() -> Iterator[None]:
    """Reload environment-backed config once per module for integration tests.

    Integration tests rely on real env / .env wiring. Module scope avoids
    repeated resets (~50-100ms each) while ensuring fresh config when switching
    between integration test modules.
    """
    EnvConfig._reset()
    yield
    EnvConfig._reset()
