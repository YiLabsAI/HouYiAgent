"""Shared test fixtures and safety guards.

SAFETY: Prevent any test from accidentally using the production
.houyi/knowledge storage directory.  All knowledge tests must use
KnowledgeService(storage_dir=tmp_path) for isolation.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from houyi.infrastructure.config.env_config import ENV_KNOWLEDGE_STORAGE

# Detect the real project root (workspace)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_PROD_KNOWLEDGE_DIR = os.path.join(_PROJECT_ROOT, ".houyi", "knowledge")


@pytest.fixture(autouse=True)
def _guard_production_knowledge_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure ENV_KNOWLEDGE_STORAGE never points to the production directory.

    If a test accidentally creates KnowledgeService() without an explicit
    storage_dir, this fixture redirects the default to a safe temp directory
    so that production data is never touched.
    """
    # If no env var is set, set one pointing to a temp directory that pytest
    # will clean up.  If one IS set, leave it alone (it may be from an
    # intentional test setup).
    if ENV_KNOWLEDGE_STORAGE not in os.environ:
        # Keep defaults isolated per test and aligned with the same .houyi layout
        # used by skill test fixtures.
        isolated_storage = tmp_path / ".houyi" / "knowledge"
        monkeypatch.setenv(ENV_KNOWLEDGE_STORAGE, str(isolated_storage))
