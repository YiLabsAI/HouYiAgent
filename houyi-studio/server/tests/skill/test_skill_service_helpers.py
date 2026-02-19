"""Tests for SkillService helper functions and global service management."""

from __future__ import annotations

from _fakes import _FakePermissions, _FakePermKind


class TestGlobalService:
    def test_get_set_skill_service(self):
        from houyi_studio.server.skill.service import (
            SkillService,
            get_skill_service,
            set_skill_service,
        )

        svc = SkillService()
        set_skill_service(svc)
        assert get_skill_service() is svc

        # Reset
        set_skill_service(None)


class TestHelpers:
    """Tests for module-level helper functions."""

    def test_empty_metrics(self):
        from houyi_studio.server.skill.service import _empty_metrics

        m = _empty_metrics("test_skill")
        assert m["skill_name"] == "test_skill"
        assert m["total_calls"] == 0
        assert m["success_count"] == 0
        assert m["failure_count"] == 0
        assert m["avg_latency_ms"] == 0.0
        assert m["success_rate"] == 0.0
        assert m["last_invoked"] is None

    def test_extract_side_effects_network(self):
        from houyi_studio.server.skill.serializer import extract_side_effects

        perms = _FakePermissions(network=_FakePermKind(enabled=True))
        assert extract_side_effects(perms) == ["network"]

    def test_extract_side_effects_multiple(self):
        from houyi_studio.server.skill.serializer import extract_side_effects

        perms = _FakePermissions(
            exec_=_FakePermKind(enabled=True),
            network=_FakePermKind(enabled=True),
            filesystem=_FakePermKind(write=True),
        )
        effects = extract_side_effects(perms)
        assert "exec" in effects
        assert "network" in effects
        assert "filesystem" in effects

    def test_extract_side_effects_empty(self):
        from houyi_studio.server.skill.serializer import extract_side_effects

        perms = _FakePermissions()
        assert extract_side_effects(perms) == []

    def test_dominant_side_effect_priority(self):
        from houyi_studio.server.skill.serializer import dominant_side_effect

        perms = _FakePermissions(
            exec_=_FakePermKind(enabled=True),
            network=_FakePermKind(enabled=True),
        )
        assert dominant_side_effect(perms) == "exec"

    def test_dominant_side_effect_none(self):
        from houyi_studio.server.skill.serializer import dominant_side_effect

        perms = _FakePermissions()
        assert dominant_side_effect(perms) == "none"
