"""End-to-end test for ext__planning-with-files lifecycle.

Validates that the external planning skill:
1. Loads from skills/planning-with-files/SKILL.md
2. Gets renamed to ext__planning-with-files (core protection)
3. Inherits core executor and schema via hydration
4. Reaches capability_tier=EXECUTABLE, runtime_status=READY
5. Can execute create/status/update/complete actions
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from houyi_studio.server.skill.startup_hooks import (
    _hydrate_external_runtime,
)

from houyi.core.skill.runtime_contract import CapabilityTier, RuntimeStatus
from houyi.core.skill_registry import SkillRegistry
from houyi.skills.planning import PlanningSkill

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def planning_registry(tmp_path: Path) -> tuple[SkillRegistry, Path]:
    """Set up a registry with core planning + external planning-with-files."""
    registry = SkillRegistry()

    # 1. Register core planning skill (as startup_hooks does)
    planning = PlanningSkill(workspace=tmp_path)
    core_spec = planning.to_spec()
    core_spec = core_spec.model_copy(update={"is_core": True})
    registry.register(core_spec, overwrite=True)

    # 2. Load external SKILL.md from skills/planning-with-files/
    project_root = Path(__file__).resolve().parents[4]
    external_skill_md = project_root / "skills" / "planning-with-files" / "SKILL.md"
    if not external_skill_md.exists():
        pytest.skip("skills/planning-with-files/SKILL.md not found")

    # register_from_skill_file will auto-rename to ext__ due to core conflict
    registered_name = registry.register_from_skill_file(
        str(external_skill_md),
        overwrite=False,
    )
    assert registered_name == "ext__planning-with-files"

    # 3. Hydrate: copy core executor + schema to external
    hydrated = _hydrate_external_runtime([registered_name], registry)
    assert "ext__planning-with-files" in hydrated

    return registry, tmp_path


# ── Integration level + runtime status ───────────────────────────────


class TestExtPlanningIntegration:
    """Verify ext__planning-with-files reaches EXECUTABLE/READY."""

    def test_capability_tier_is_executable(
        self,
        planning_registry: tuple[SkillRegistry, Path],
    ) -> None:
        registry, _ = planning_registry
        skill = registry.get("ext__planning-with-files")
        assert skill is not None
        assert skill.capability_tier == CapabilityTier.EXECUTABLE

    def test_runtime_status_is_ready(
        self,
        planning_registry: tuple[SkillRegistry, Path],
    ) -> None:
        registry, _ = planning_registry
        skill = registry.get("ext__planning-with-files")
        assert skill is not None
        assert skill.runtime_status == RuntimeStatus.READY

    def test_has_callable_executor(
        self,
        planning_registry: tuple[SkillRegistry, Path],
    ) -> None:
        registry, _ = planning_registry
        skill = registry.get("ext__planning-with-files")
        assert skill is not None
        assert callable(skill.executor)

    def test_has_real_input_schema(
        self,
        planning_registry: tuple[SkillRegistry, Path],
    ) -> None:
        registry, _ = planning_registry
        skill = registry.get("ext__planning-with-files")
        assert skill is not None
        schema = skill.input_schema.model_json_schema()
        assert schema.get("properties") or schema.get("required")


# ── Executor action tests (create / status / update / complete) ──────


class TestExtPlanningActions:
    """Verify ext__planning-with-files can execute all 4 actions."""

    def test_create_plan(
        self,
        planning_registry: tuple[SkillRegistry, Path],
    ) -> None:
        registry, workspace = planning_registry
        skill = registry.get("ext__planning-with-files")
        result = asyncio.get_event_loop().run_until_complete(
            skill.executor(action="create", task="Test task", subtasks=["Step 1", "Step 2"]),
        )
        assert result["success"] is True
        assert "plan_path" in result
        assert Path(result["plan_path"]).exists()

    def test_status_after_create(
        self,
        planning_registry: tuple[SkillRegistry, Path],
    ) -> None:
        registry, workspace = planning_registry
        skill = registry.get("ext__planning-with-files")
        loop = asyncio.get_event_loop()

        loop.run_until_complete(
            skill.executor(action="create", task="Status test", subtasks=["A", "B"]),
        )
        result = loop.run_until_complete(skill.executor(action="status"))
        assert result["success"] is True
        assert result["status"] == "in_progress"
        assert result["progress"]["total"] == 2
        assert result["progress"]["completed"] == 0

    def test_update_subtask(
        self,
        planning_registry: tuple[SkillRegistry, Path],
    ) -> None:
        registry, workspace = planning_registry
        skill = registry.get("ext__planning-with-files")
        loop = asyncio.get_event_loop()

        loop.run_until_complete(
            skill.executor(action="create", task="Update test", subtasks=["X", "Y"]),
        )
        result = loop.run_until_complete(
            skill.executor(action="update", subtask_index=0, completed=True),
        )
        assert result["success"] is True
        assert result["progress"]["completed"] == 1

    def test_complete_plan(
        self,
        planning_registry: tuple[SkillRegistry, Path],
    ) -> None:
        registry, workspace = planning_registry
        skill = registry.get("ext__planning-with-files")
        loop = asyncio.get_event_loop()

        loop.run_until_complete(
            skill.executor(action="create", task="Complete test", subtasks=["Only"]),
        )
        loop.run_until_complete(
            skill.executor(action="update", subtask_index=0, completed=True),
        )
        result = loop.run_until_complete(skill.executor(action="complete"))
        assert result["success"] is True
        assert result["status"] == "completed"

    def test_complete_fails_with_remaining_subtasks(
        self,
        planning_registry: tuple[SkillRegistry, Path],
    ) -> None:
        registry, workspace = planning_registry
        skill = registry.get("ext__planning-with-files")
        loop = asyncio.get_event_loop()

        loop.run_until_complete(
            skill.executor(action="create", task="Incomplete", subtasks=["A", "B"]),
        )
        result = loop.run_until_complete(skill.executor(action="complete"))
        assert result["success"] is False
        assert "remaining" in result["message"]
