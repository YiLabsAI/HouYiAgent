"""Shared fixtures for skill-service tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from houyi.core.skill_registry import SkillRegistry

# Make sibling modules (_fakes) importable by test files.
_THIS_DIR = os.path.dirname(__file__)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _fakes import _FakePermissions, _FakePermKind, _FakePolicy, _FakeSkillSpec  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry():
    return SkillRegistry()


@pytest.fixture(autouse=True)
def isolate_skill_loader_project_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Route SkillLoader managed skill installs to test-local tmp roots.

    This keeps generated packages under ``<tmp>/.houyi/skills`` instead of the
    workspace root, so tests never leave artifact directories behind.
    """

    from houyi_studio.server.skill.loader import ENV_MANAGED_SKILLS_DIR

    monkeypatch.setenv(ENV_MANAGED_SKILLS_DIR, str(tmp_path / ".houyi" / "skills"))


@pytest.fixture
def skill_service(registry):
    from houyi_studio.server.skill.service import SkillService

    return SkillService(registry=registry)


@pytest.fixture
def populated_registry(registry):
    """Registry with two test skills."""
    s1 = _FakeSkillSpec(
        "web_search",
        display_name="Web Search",
        description="Search the web",
        permissions=_FakePermissions(
            network=_FakePermKind(enabled=True),
            descriptions=["Network: outbound access"],
        ),
        invocation_policy=_FakePolicy("allow"),
    )
    s2 = _FakeSkillSpec(
        "file_writer",
        display_name="File Writer",
        description="Write files",
        permissions=_FakePermissions(
            filesystem=_FakePermKind(write=True),
            descriptions=["Filesystem: write access"],
        ),
        invocation_policy=_FakePolicy("allow_with_consent"),
    )
    registry.register(s1, overwrite=True)
    registry.register(s2, overwrite=True)
    return registry


@pytest.fixture
def populated_service(populated_registry):
    from houyi_studio.server.skill.service import SkillService

    return SkillService(registry=populated_registry)
