"""Tests for PlanStore in-memory behavior.

PlanStore was refactored from file-based persistence to in-memory only.
These tests verify:
1. In-memory get/set works correctly
2. save_to_file and load_from_file are no-ops
3. Legacy file cleanup on startup
4. Session isolation (different sessions don't share plans)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from houyi_studio.server.stores import PlanStore

from houyi.protocol.ir import PlanIR


@pytest.fixture
def plans_dir(tmp_path: Path) -> Path:
    d = tmp_path / "plans"
    d.mkdir()
    return d


def _make_plan(plan_id: str = "plan_1") -> PlanIR:
    return PlanIR(plan_id=plan_id, nodes=[], edges=[], entry_node_id="")


class TestPlanStoreInMemory:
    """Core in-memory behavior."""

    def test_get_returns_none_for_unknown_session(self, plans_dir: Path) -> None:
        store = PlanStore(plans_dir=plans_dir)
        assert store.get("unknown_session") is None

    def test_set_and_get(self, plans_dir: Path) -> None:
        store = PlanStore(plans_dir=plans_dir)
        plan = _make_plan("p1")
        store.set("s1", plan)
        assert store.get("s1") is plan

    def test_get_cached_returns_none_for_unknown(self, plans_dir: Path) -> None:
        store = PlanStore(plans_dir=plans_dir)
        assert store.get_cached("unknown") is None

    def test_get_cached_returns_set_plan(self, plans_dir: Path) -> None:
        store = PlanStore(plans_dir=plans_dir)
        plan = _make_plan("p1")
        store.set("s1", plan)
        assert store.get_cached("s1") is plan

    def test_session_isolation(self, plans_dir: Path) -> None:
        """Different sessions should not share plans."""
        store = PlanStore(plans_dir=plans_dir)
        plan_a = _make_plan("pa")
        plan_b = _make_plan("pb")
        store.set("session_a", plan_a)
        store.set("session_b", plan_b)
        assert store.get("session_a") is plan_a
        assert store.get("session_b") is plan_b
        assert store.get("session_c") is None

    def test_set_overwrites_previous(self, plans_dir: Path) -> None:
        store = PlanStore(plans_dir=plans_dir)
        plan1 = _make_plan("p1")
        plan2 = _make_plan("p2")
        store.set("s1", plan1)
        store.set("s1", plan2)
        assert store.get("s1") is plan2


class TestPlanStorePersistenceRemoved:
    """Verify file persistence is a no-op."""

    def test_save_to_file_is_noop(self, plans_dir: Path) -> None:
        store = PlanStore(plans_dir=plans_dir)
        plan = _make_plan("p1")
        store.save_to_file("s1", plan)
        # No files should be created
        assert list(plans_dir.iterdir()) == []

    def test_load_from_file_returns_none(self, plans_dir: Path) -> None:
        store = PlanStore(plans_dir=plans_dir)
        assert store.load_from_file("any_session") is None

    def test_set_does_not_create_files(self, plans_dir: Path) -> None:
        store = PlanStore(plans_dir=plans_dir)
        plan = _make_plan("p1")
        store.set("s1", plan, persist=True)
        assert list(plans_dir.iterdir()) == []


class TestPlanStoreLegacyCleanup:
    """Verify legacy file cleanup on startup."""

    def test_cleans_session_json_files(self, plans_dir: Path) -> None:
        # Create legacy files
        (plans_dir / "session_abc123.json").write_text("{}")
        (plans_dir / "session_xyz789.json").write_text("{}")
        assert len(list(plans_dir.glob("session_*.json"))) == 2

        PlanStore(plans_dir=plans_dir)

        assert len(list(plans_dir.glob("session_*.json"))) == 0

    def test_cleans_current_plan_json(self, plans_dir: Path) -> None:
        (plans_dir / "current_plan.json").write_text("{}")
        assert (plans_dir / "current_plan.json").exists()

        PlanStore(plans_dir=plans_dir)

        assert not (plans_dir / "current_plan.json").exists()

    def test_preserves_unrelated_files(self, plans_dir: Path) -> None:
        (plans_dir / "other_file.txt").write_text("keep me")
        (plans_dir / "session_old.json").write_text("{}")

        PlanStore(plans_dir=plans_dir)

        assert (plans_dir / "other_file.txt").exists()
        assert not (plans_dir / "session_old.json").exists()
