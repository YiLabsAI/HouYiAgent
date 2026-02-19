"""Tests for SkillService read operations."""

from __future__ import annotations

from unittest.mock import MagicMock

from _fakes import _FakeSkillSpec


class TestListSkills:
    def test_empty_registry(self, skill_service):
        result = skill_service.list_skills()
        assert result == []

    def test_multiple_skills(self, populated_service):
        result = populated_service.list_skills()
        assert len(result) == 2
        names = {s["name"] for s in result}
        assert names == {"web_search", "file_writer"}

    def test_summary_fields(self, populated_service):
        result = populated_service.list_skills()
        ws = next(s for s in result if s["name"] == "web_search")
        assert ws["display_name"] == "Web Search"
        assert ws["description"] == "Search the web"
        assert ws["policy_action"] == "allow"
        assert ws["side_effect"] == "network"
        assert ws["certification"] == "unverified"

    def test_policy_badge(self, populated_service):
        result = populated_service.list_skills()
        fw = next(s for s in result if s["name"] == "file_writer")
        assert fw["policy_action"] == "allow_with_consent"
        assert fw["side_effect"] == "filesystem"


class TestGetSkillDetail:
    def test_existing_skill(self, populated_service):
        detail = populated_service.get_skill_detail("web_search")
        assert detail is not None
        assert detail["name"] == "web_search"
        assert detail["version"] == "1.0.0"
        assert isinstance(detail["tools"], list)
        assert isinstance(detail["permissions"], list)
        assert isinstance(detail["policy"], dict)

    def test_nonexistent_skill(self, populated_service):
        detail = populated_service.get_skill_detail("nonexistent")
        assert detail is None

    def test_permissions_detail(self, populated_service):
        detail = populated_service.get_skill_detail("file_writer")
        assert len(detail["permissions"]) == 1
        perm = detail["permissions"][0]
        assert perm["name"] == "Filesystem: write access"
        assert perm["is_sensitive"] is True

    def test_policy_detail(self, populated_service):
        detail = populated_service.get_skill_detail("web_search")
        assert detail["policy"]["default_action"] == "allow"
        assert detail["policy"]["model_auto_invoke"] is True

    def test_version_none_defaults_to_0_0_0(self, registry):
        """When SkillSpec.version is None, get_skill_detail must return '0.0.0'."""
        from houyi_studio.server.skill.service import SkillService

        skill = _FakeSkillSpec("no_version_skill", version=None)
        registry.register(skill, overwrite=True)
        svc = SkillService(registry=registry)
        detail = svc.get_skill_detail("no_version_skill")
        assert detail is not None
        assert detail["version"] == "0.0.0", (
            f"Expected version '0.0.0' but got {detail['version']!r}"
        )

    def test_version_none_compatible_with_pydantic_model(self, registry):
        """SkillDetail pydantic model must accept the dict produced when version is None."""
        from houyi_studio.server.gateway.events import SkillDetail, SkillPermission
        from houyi_studio.server.skill.service import SkillService

        skill = _FakeSkillSpec("pydantic_compat", version=None, description="test")
        registry.register(skill, overwrite=True)
        svc = SkillService(registry=registry)
        detail_data = svc.get_skill_detail("pydantic_compat")

        skill_detail = SkillDetail(
            name=detail_data["name"],
            display_name=detail_data.get("display_name", detail_data["name"]),
            description=detail_data.get("description"),
            version=detail_data.get("version") or "0.0.0",
            author=detail_data.get("author"),
            tools=detail_data.get("tools", []),
            permissions=[
                SkillPermission(
                    name=p["name"],
                    description=p.get("description"),
                    is_sensitive=p.get("is_sensitive", False),
                )
                for p in detail_data.get("permissions", [])
            ],
            policy=detail_data.get("policy", {}),
            hooks=detail_data.get("hooks", []),
            certification=detail_data.get("certification", "unverified"),
            side_effect=detail_data.get("side_effect", "none"),
        )
        assert skill_detail.version == "0.0.0"
        assert skill_detail.name == "pydantic_compat"


class TestGetSkillMetrics:
    def test_no_metrics_store(self, populated_service):
        result = populated_service.get_skill_metrics("web_search")
        assert result["skill_name"] == "web_search"
        assert result["total_calls"] == 0
        assert result["success_rate"] == 0.0

    def test_with_metrics_store(self, populated_registry):
        from houyi_studio.server.skill.service import SkillService

        mock_store = MagicMock()
        agg = MagicMock()
        agg.total_calls = 42
        agg.success_count = 40
        agg.failure_count = 2
        agg.avg_latency_ms = 120.5
        agg.p50_latency_ms = 100.0
        agg.p99_latency_ms = 500.0
        agg.success_rate = 0.952
        agg.last_invoked = None
        mock_store.aggregate.return_value = agg

        svc = SkillService(registry=populated_registry, metrics_store=mock_store)
        result = svc.get_skill_metrics("web_search")
        assert result["total_calls"] == 42
        assert result["success_rate"] == 0.952
        assert result["avg_latency_ms"] == 120.5
        mock_store.aggregate.assert_called_once_with("web_search")


class TestIsSkillLoaded:
    """Tests for duplicate detection via is_skill_loaded."""

    def test_loaded_skill_found(self, populated_service):
        assert populated_service.is_skill_loaded("web_search") is True

    def test_unloaded_skill_not_found(self, populated_service):
        assert populated_service.is_skill_loaded("nonexistent") is False

    def test_after_unload(self, populated_service):
        populated_service.unload_skill("web_search")
        assert populated_service.is_skill_loaded("web_search") is False
