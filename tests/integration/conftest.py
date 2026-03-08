"""Shared fixtures for integration tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from dotenv import load_dotenv

from houyi.infrastructure.config.env_config import EnvConfig

load_dotenv()


@pytest.fixture(autouse=True)
def _reset_env_config_singleton() -> Iterator[None]:
    """Reload environment-backed config for each integration test.

    Integration tests rely on real env / .env wiring, so they should observe the
    latest process environment instead of a previously cached ``EnvConfig`` snapshot.
    """
    EnvConfig._reset()
    yield
    EnvConfig._reset()
