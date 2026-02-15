"""Unit tests for SkillService."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from houyi.core.skill_registry import SkillRegistry


class _FakeSkillSpec:
    """Minimal SkillSpec stand-in for testing."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.display_name = kwargs.get("display_name", name)
        self.description = kwargs.get("description", f"Skill {name}")
        self.version = kwargs.get("version", "1.0.0")
        self.author = kwargs.get("author", None)
        self.tools = kwargs.get("tools", [])
        self.permissions = kwargs.get("permissions", [])
        self.invocation_policy = kwargs.get("invocation_policy", None)
        self.hooks = kwargs.get("hooks", [])
        self.certification = kwargs.get("certification", "unverified")
        self.input_schema = kwargs.get("input_schema", None)


class _FakePermission:
    """Legacy fake — kept for backward compat if referenced."""

    def __init__(self, name, description=None, is_sensitive=False, side_effect=None):
        self.name = name
        self.description = description
        self.is_sensitive = is_sensitive
        self.side_effect = side_effect


class _FakePermKind:
    """Fake for individual permission kind (filesystem/network/exec)."""

    def __init__(self, enabled=False, write=False, delete=False, **kw):
        self.enabled = enabled
        self.write = write
        self.delete = delete
        for k, v in kw.items():
            setattr(self, k, v)


class _FakePermissions:
    """Fake Permissions dataclass (matches houyi.core.skill.policy.Permissions)."""

    def __init__(self, filesystem=None, network=None, exec_=None, descriptions=None):
        self.filesystem = filesystem or _FakePermKind()
        self.network = network or _FakePermKind()
        self.exec = exec_ or _FakePermKind()
        self._descriptions = descriptions or []

    def describe(self):
        return self._descriptions


class _FakePolicy:
    def __init__(self, default_action="allow", model_auto_invoke=True):
        self.default_action = default_action
        self.model_auto_invoke = model_auto_invoke
        self.require_consent_for = []


class _FakePolicyResult:
    def __init__(self, action_value="allow"):
        self.action = MagicMock(value=action_value)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry():
    return SkillRegistry()


@pytest.fixture
def skill_service(registry):
    from houyi_studio.server.skill_service import SkillService

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
    from houyi_studio.server.skill_service import SkillService

    return SkillService(registry=populated_registry)


# ===========================================================================
# Read Operations
# ===========================================================================


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
        # Now generated from Permissions.describe() — name is the description string
        assert perm["name"] == "Filesystem: write access"
        assert perm["is_sensitive"] is True

    def test_policy_detail(self, populated_service):
        detail = populated_service.get_skill_detail("web_search")
        assert detail["policy"]["default_action"] == "allow"
        assert detail["policy"]["model_auto_invoke"] is True

    def test_version_none_defaults_to_0_0_0(self, registry):
        """When SkillSpec.version is None, get_skill_detail must return '0.0.0'."""
        from houyi_studio.server.skill_service import SkillService

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
        from houyi_studio.server.events import SkillDetail, SkillPermission
        from houyi_studio.server.skill_service import SkillService

        skill = _FakeSkillSpec("pydantic_compat", version=None, description="test")
        registry.register(skill, overwrite=True)
        svc = SkillService(registry=registry)
        detail_data = svc.get_skill_detail("pydantic_compat")

        # This is exactly what app.py does — should NOT raise ValidationError
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
        from houyi_studio.server.skill_service import SkillService

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


# ===========================================================================
# Write Operations
# ===========================================================================


class TestLoadSkill:
    def test_file_not_found(self, skill_service):
        success, code, msg = skill_service.load_skill("/nonexistent/path.md")
        assert success is False
        assert code == "file_not_found"
        assert "not found" in msg

    def test_load_skill_md(self, skill_service, tmp_path):
        skill_file = tmp_path / "test_skill.md"
        skill_file.write_text("---\nname: test_skill\ndescription: test\n---\n")
        with patch.object(
            skill_service._registry,
            "register_from_skill_file",
            return_value="test_skill",
        ):
            success, name, msg = skill_service.load_skill(str(skill_file))
            assert success is True
            assert name == "test_skill"
            assert msg is None

    def test_load_json_manifest(self, skill_service, tmp_path):
        manifest = tmp_path / "simpleskill.json"
        manifest.write_text("{}")
        with patch.object(
            skill_service._registry,
            "register_from_manifest",
            return_value=["my_skill"],
        ):
            success, name, msg = skill_service.load_skill(str(manifest))
            assert success is True
            assert name == "my_skill"

    def test_load_exception(self, skill_service, tmp_path):
        skill_file = tmp_path / "bad.md"
        skill_file.write_text("bad content")
        with patch.object(
            skill_service._registry,
            "register_from_skill_file",
            side_effect=ValueError("parse error"),
        ):
            success, code, msg = skill_service.load_skill(str(skill_file))
            assert success is False
            assert code == "load_failed"
            assert "parse error" in msg

    def test_load_from_directory(self, skill_service, tmp_path):
        """load_skill() accepts a directory path and calls register_from_directory."""
        subdir = tmp_path / "my_skills"
        subdir.mkdir()
        (subdir / "SKILL.md").write_text("---\nname: dir_skill\n---\n")
        with patch.object(
            skill_service._registry,
            "register_from_directory",
            return_value=["dir_skill"],
        ):
            success, names, msg = skill_service.load_skill(str(subdir))
            assert success is True
            assert "dir_skill" in names

    def test_load_from_directory_empty(self, skill_service, tmp_path):
        """Directory with no SKILL.md files returns failure."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch.object(
            skill_service._registry,
            "register_from_directory",
            return_value=[],
        ):
            success, code, msg = skill_service.load_skill(str(empty_dir))
            assert success is False
            assert code == "no_skills"

    def test_load_from_url(self, skill_service):
        """load_skill() accepts an http:// or https:// URL."""
        fake_skill = MagicMock()
        fake_skill.name = "url_skill"
        with patch(
            "houyi.core.skill.spec.SkillSpec.from_url",
            return_value=fake_skill,
        ):
            success, name, msg = skill_service.load_skill("https://example.com/skill/SKILL.md")
            assert success is True
            assert name == "url_skill"

    def test_load_from_url_failure(self, skill_service):
        """URL load that raises an exception returns failure."""
        with patch(
            "houyi.core.skill.spec.SkillSpec.from_url",
            side_effect=ValueError("download failed"),
        ):
            success, code, msg = skill_service.load_skill("https://example.com/bad.md")
            assert success is False
            assert code == "url_load_failed"


class TestUnloadSkill:
    def test_unload_existing(self, populated_service):
        success, msg = populated_service.unload_skill("web_search")
        assert success is True
        assert msg is None

    def test_unload_nonexistent(self, populated_service):
        success, msg = populated_service.unload_skill("nonexistent")
        assert success is False
        assert "not found" in msg.lower()


class TestConfigureSkill:
    def test_configure_nonexistent(self, populated_service):
        success, msg = populated_service.configure_skill("nonexistent", policy_action="deny")
        assert success is False
        assert "not found" in msg.lower()

    def test_configure_invalid_policy(self, populated_service):
        success, msg = populated_service.configure_skill(
            "web_search", policy_action="invalid_action"
        )
        assert success is False
        assert "Invalid policy_action" in msg

    def test_configure_no_changes(self, populated_service):
        success, msg = populated_service.configure_skill("web_search")
        assert success is False
        assert "No configuration changes" in msg

    def test_configure_policy_action(self, populated_service):
        success, msg = populated_service.configure_skill("web_search", policy_action="deny")
        assert success is True
        assert msg is None

    def test_configure_auto_invoke(self, populated_service):
        success, msg = populated_service.configure_skill("web_search", auto_invoke=False)
        assert success is True
        assert msg is None

    def test_configure_both(self, populated_service):
        success, msg = populated_service.configure_skill(
            "web_search", policy_action="allow_with_consent", auto_invoke=True
        )
        assert success is True
        assert msg is None


# ===========================================================================
# Dry-run Validation
# ===========================================================================


class TestDryRun:
    def test_skill_not_found(self, skill_service):
        result = skill_service.dry_run("nonexistent", "tool", {})
        assert result["valid"] is False
        assert len(result["schema_errors"]) > 0

    def test_basic_dry_run(self, populated_service):
        result = populated_service.dry_run("web_search", "search", {})
        assert result["valid"] is True
        assert result["policy_result"] == "allow"

    def test_side_effects_collected(self, populated_service):
        result = populated_service.dry_run("web_search", "search", {})
        assert "network" in result["estimated_side_effects"]

    def test_policy_deny(self, populated_registry):
        from houyi_studio.server.skill_service import SkillService

        mock_enforcer = MagicMock()
        mock_enforcer.evaluate.return_value = _FakePolicyResult("deny")

        svc = SkillService(registry=populated_registry, policy_enforcer=mock_enforcer)
        result = svc.dry_run("web_search", "search", {})
        assert result["valid"] is False
        assert result["policy_result"] == "deny"


# ===========================================================================
# Consent Management
# ===========================================================================


class TestConsentManagement:
    def test_create_consent_request(self, skill_service):
        req_id = skill_service.create_consent_request(
            skill_name="web_search",
            tool_name="search",
            reason="Needs network access",
            permissions=["network"],
        )
        assert req_id.startswith("consent_")
        assert req_id in skill_service._pending_consents

    def test_respond_to_consent(self, skill_service):
        req_id = skill_service.create_consent_request(
            skill_name="web_search",
            tool_name="search",
            reason="test",
            permissions=["network"],
        )
        found = skill_service.respond_to_consent(req_id, granted=True, remember=True)
        assert found is True
        req = skill_service._pending_consents[req_id]
        assert req.granted is True
        assert req.remember is True

    def test_respond_to_nonexistent(self, skill_service):
        found = skill_service.respond_to_consent("nonexistent", granted=True)
        assert found is False

    @pytest.mark.asyncio
    async def test_wait_for_consent_granted(self, skill_service):
        req_id = skill_service.create_consent_request(
            skill_name="test",
            tool_name="tool",
            reason="test",
            permissions=[],
        )

        async def respond_later():
            await asyncio.sleep(0.05)
            skill_service.respond_to_consent(req_id, granted=True, remember=False)

        asyncio.create_task(respond_later())
        granted, remember = await skill_service.wait_for_consent(req_id, timeout=2.0)
        assert granted is True
        assert remember is False

    @pytest.mark.asyncio
    async def test_wait_for_consent_timeout(self, skill_service):
        req_id = skill_service.create_consent_request(
            skill_name="test",
            tool_name="tool",
            reason="test",
            permissions=[],
        )
        granted, remember = await skill_service.wait_for_consent(req_id, timeout=0.1)
        assert granted is False
        assert remember is False
        # Cleaned up after timeout
        assert req_id not in skill_service._pending_consents

    @pytest.mark.asyncio
    async def test_wait_for_nonexistent(self, skill_service):
        granted, remember = await skill_service.wait_for_consent("nonexistent")
        assert granted is False


# ===========================================================================
# Global Service Management
# ===========================================================================


class TestGlobalService:
    def test_get_set_skill_service(self):
        from houyi_studio.server.skill_service import (
            SkillService,
            get_skill_service,
            set_skill_service,
        )

        svc = SkillService()
        set_skill_service(svc)
        assert get_skill_service() is svc

        # Reset
        set_skill_service(None)
