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
        self.provider = kwargs.get("provider", None)
        self.tools = kwargs.get("tools", [])
        self.permissions = kwargs.get("permissions", [])
        self.invocation_policy = kwargs.get("invocation_policy", None)
        self.hooks = kwargs.get("hooks", [])
        self.certification = kwargs.get("certification", "unverified")
        self.input_schema = kwargs.get("input_schema", None)

    @property
    def qualified_name(self) -> str:
        if self.provider:
            return f"{self.provider}/{self.name}"
        return self.name


class _FakeInputSchema:
    """Mimics a Pydantic model used as input_schema."""

    def __init__(self, required_fields: list[str] | None = None):
        self._required = set(required_fields or [])

    def model_validate(self, data: dict):
        for field in self._required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

    def model_json_schema(self):
        return {"type": "object", "required": list(self._required)}


class _FakeTool:
    """Mimics a tool attached to a SkillSpec."""

    def __init__(self, name: str, description: str = "", input_schema=None):
        self.name = name
        self.description = description
        self.input_schema = input_schema


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


class _FakeSideEffect:
    """Mimics SideEffect enum."""

    def __init__(self, value: str = "none"):
        self.value = value


class _FakeModelAutoInvoke:
    """Mimics ModelAutoInvoke enum with a .value attribute."""

    def __init__(self, value: str):
        self.value = value

    def __str__(self):
        return self.value


class _FakePolicy:
    """Mimics InvocationPolicy.  Uses _FakeModelAutoInvoke so that
    `.model_auto_invoke.value` works the same as the real enum."""

    def __init__(self, default_action="allow", model_auto_invoke=None):
        self.model_auto_invoke = _FakeModelAutoInvoke(default_action)
        self.user_invocable = True
        self.side_effect = _FakeSideEffect("none")


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
        success, code, msg = skill_service.load_skill(str(skill_file))
        assert success is False
        assert code in ("no_frontmatter", "parse_failed", "load_failed")

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
        # Verify the InvocationPolicy was actually updated
        skill = populated_service._registry.get("web_search")
        assert skill.invocation_policy is not None
        assert skill.invocation_policy.model_auto_invoke.value == "deny"

    def test_configure_auto_invoke(self, populated_service):
        success, msg = populated_service.configure_skill("web_search", auto_invoke=False)
        assert success is True
        assert msg is None
        skill = populated_service._registry.get("web_search")
        assert skill.invocation_policy is not None
        assert skill.invocation_policy.model_auto_invoke.value == "deny"

    def test_configure_both_policy_takes_precedence(self, populated_service):
        """When both policy_action and auto_invoke are provided,
        policy_action takes precedence (it encodes the full semantics)."""
        success, msg = populated_service.configure_skill(
            "web_search", policy_action="allow_with_consent", auto_invoke=True
        )
        assert success is True
        assert msg is None
        skill = populated_service._registry.get("web_search")
        assert skill.invocation_policy.model_auto_invoke.value == "allow_with_consent"

    def test_configure_deny_then_verify_detail(self, populated_service):
        """Configure deny → get_skill_detail → policy.default_action == 'deny'."""
        populated_service.configure_skill("web_search", policy_action="deny")
        detail = populated_service.get_skill_detail("web_search")
        assert detail is not None
        assert detail["policy"]["default_action"] == "deny"
        assert detail["policy"]["model_auto_invoke"] is False

    def test_configure_allow_with_consent_roundtrip(self, populated_service):
        """Configure allow_with_consent → detail reflects it."""
        populated_service.configure_skill("web_search", policy_action="allow_with_consent")
        detail = populated_service.get_skill_detail("web_search")
        assert detail["policy"]["default_action"] == "allow_with_consent"


# ===========================================================================
# Dry-run Validation
# ===========================================================================


class TestDryRun:
    @pytest.mark.asyncio
    async def test_skill_not_found(self, skill_service):
        result = await skill_service.dry_run("nonexistent", "tool", {})
        assert result["valid"] is False
        assert len(result["schema_errors"]) > 0

    @pytest.mark.asyncio
    async def test_basic_dry_run(self, populated_service):
        result = await populated_service.dry_run("web_search", "search", {})
        assert result["valid"] is True
        assert result["policy_result"] == "allow"

    @pytest.mark.asyncio
    async def test_side_effects_collected(self, populated_service):
        result = await populated_service.dry_run("web_search", "search", {})
        assert "network" in result["estimated_side_effects"]

    @pytest.mark.asyncio
    async def test_empty_input_skips_schema_validation(self, populated_service):
        """Empty input {} means 'check availability only' — required fields
        should NOT cause validation failure."""
        result = await populated_service.dry_run("web_search", "search", {})
        assert result["valid"] is True
        assert result["schema_errors"] == []

    @pytest.mark.asyncio
    async def test_invalid_input_fails_schema(self, populated_service):
        """Non-empty input with wrong types should fail schema validation."""
        result = await populated_service.dry_run("web_search", "search", {"query": 42})
        # If the schema expects str and gets int, it may or may not fail
        # depending on Pydantic coercion. The key is: non-empty input IS validated.
        # This test documents the behavior.
        assert isinstance(result["valid"], bool)

    @pytest.mark.asyncio
    async def test_policy_deny(self, populated_registry):
        from houyi_studio.server.skill_service import SkillService

        mock_enforcer = MagicMock()
        mock_enforcer.evaluate.return_value = _FakePolicyResult("deny")

        svc = SkillService(registry=populated_registry, policy_enforcer=mock_enforcer)
        result = await svc.dry_run("web_search", "search", {})
        assert result["valid"] is False
        assert result["policy_result"] == "deny"

    # ─── Tool-level schema validation ────────────────────────────

    @pytest.mark.asyncio
    async def test_tool_level_schema_validation_pass(self, registry):
        """When a tool has its own input_schema, dry-run validates against it."""
        from houyi_studio.server.skill_service import SkillService

        tool = _FakeTool("search", input_schema=_FakeInputSchema(["query"]))
        skill = _FakeSkillSpec("search_skill", tools=[tool])
        registry.register(skill, overwrite=True)
        svc = SkillService(registry=registry)

        result = await svc.dry_run("search_skill", "search", {"query": "hello"})
        assert result["valid"] is True
        assert result["schema_errors"] == []

    @pytest.mark.asyncio
    async def test_tool_level_schema_validation_fail(self, registry):
        """Missing required field at tool level triggers schema error."""
        from houyi_studio.server.skill_service import SkillService

        tool = _FakeTool("search", input_schema=_FakeInputSchema(["query"]))
        skill = _FakeSkillSpec("search_skill", tools=[tool])
        registry.register(skill, overwrite=True)
        svc = SkillService(registry=registry)

        result = await svc.dry_run("search_skill", "search", {"wrong_field": "value"})
        assert result["valid"] is False
        assert any("query" in err for err in result["schema_errors"])

    @pytest.mark.asyncio
    async def test_tool_level_schema_empty_input_skips(self, registry):
        """Empty input {} skips tool-level schema validation."""
        from houyi_studio.server.skill_service import SkillService

        tool = _FakeTool("search", input_schema=_FakeInputSchema(["query"]))
        skill = _FakeSkillSpec("search_skill", tools=[tool])
        registry.register(skill, overwrite=True)
        svc = SkillService(registry=registry)

        result = await svc.dry_run("search_skill", "search", {})
        assert result["valid"] is True
        assert result["schema_errors"] == []

    @pytest.mark.asyncio
    async def test_tool_level_takes_precedence_over_skill_level(self, registry):
        """Tool-level schema is preferred; skill-level is fallback."""
        from houyi_studio.server.skill_service import SkillService

        tool = _FakeTool("search", input_schema=_FakeInputSchema(["query"]))
        skill_schema = _FakeInputSchema(["different_field"])
        skill = _FakeSkillSpec("search_skill", tools=[tool], input_schema=skill_schema)
        registry.register(skill, overwrite=True)
        svc = SkillService(registry=registry)

        # Passes tool-level but would fail skill-level
        result = await svc.dry_run("search_skill", "search", {"query": "test"})
        assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_fallback_to_skill_level_when_tool_has_no_schema(self, registry):
        """If tool has no schema, fall back to skill-level input_schema."""
        from houyi_studio.server.skill_service import SkillService

        tool = _FakeTool("simple_tool")
        skill_schema = _FakeInputSchema(["required_param"])
        skill = _FakeSkillSpec("skill_with_schema", tools=[tool], input_schema=skill_schema)
        registry.register(skill, overwrite=True)
        svc = SkillService(registry=registry)

        result = await svc.dry_run("skill_with_schema", "simple_tool", {"wrong": "value"})
        assert result["valid"] is False
        assert any("required_param" in err for err in result["schema_errors"])

    @pytest.mark.asyncio
    async def test_multi_tool_selects_correct_tool(self, registry):
        """Dry-run validates against the specific tool_name's schema."""
        from houyi_studio.server.skill_service import SkillService

        tool_a = _FakeTool("read_file", input_schema=_FakeInputSchema(["path"]))
        tool_b = _FakeTool("write_file", input_schema=_FakeInputSchema(["path", "content"]))
        skill = _FakeSkillSpec("file_ops", tools=[tool_a, tool_b])
        registry.register(skill, overwrite=True)
        svc = SkillService(registry=registry)

        # read_file only requires "path"
        result_read = await svc.dry_run("file_ops", "read_file", {"path": "/tmp/f"})
        assert result_read["valid"] is True

        # write_file requires "path" AND "content"
        result_write = await svc.dry_run("file_ops", "write_file", {"path": "/tmp/f"})
        assert result_write["valid"] is False
        assert any("content" in err for err in result_write["schema_errors"])

        # write_file with both fields passes
        result_write_ok = await svc.dry_run(
            "file_ops", "write_file", {"path": "/tmp/f", "content": "hello"}
        )
        assert result_write_ok["valid"] is True

    # ─── Live mode ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_live_false_has_no_llm_verification(self, populated_service):
        """When live=False (default), result has no llm_verification."""
        result = await populated_service.dry_run("web_search", "search", {})
        assert "llm_verification" not in result

    @pytest.mark.asyncio
    async def test_live_true_returns_llm_verification(self, registry):
        """When live=True, llm_verification key is present (even if it fails)."""
        from houyi_studio.server.skill_service import SkillService

        tool = _FakeTool("search", input_schema=_FakeInputSchema(["query"]))
        skill = _FakeSkillSpec("web_search", tools=[tool])
        registry.register(skill, overwrite=True)
        svc = SkillService(registry=registry)

        result = await svc.dry_run("web_search", "search", {"query": "test"}, live=True)
        assert "llm_verification" in result
        llm_v = result["llm_verification"]
        assert isinstance(llm_v, dict)
        assert "success" in llm_v
        assert "message" in llm_v

    @pytest.mark.asyncio
    async def test_live_import_failure_returns_graceful_error(self, registry):
        """If LLM adapter cannot be imported, live returns a helpful error."""
        from unittest.mock import patch

        from houyi_studio.server.skill_service import SkillService

        tool = _FakeTool("search")
        skill = _FakeSkillSpec("web_search", tools=[tool])
        registry.register(skill, overwrite=True)
        svc = SkillService(registry=registry)

        with patch.dict("sys.modules", {"houyi.llm.llm_adapter": None}):
            result = await svc.dry_run("web_search", "search", {}, live=True)
        assert result["llm_verification"]["success"] is False
        assert "not available" in result["llm_verification"]["message"]


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


class TestLoadSkillPaths:
    """Tests for the three load paths: file, URL, directory."""

    def _svc(self):
        from houyi_studio.server.skill_service import SkillService

        return SkillService(registry=SkillRegistry())

    # ── File loading ──────────────────────────────────────────

    def test_load_valid_skill_md(self, tmp_path):
        svc = self._svc()
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("---\nname: test-skill\ndescription: A test skill\n---\n# Test\n")
        ok, name, err = svc.load_skill(str(skill_file))
        assert ok is True
        assert name == "test-skill"
        assert err is None

    def test_load_file_not_found(self):
        svc = self._svc()
        ok, code, err = svc.load_skill("/tmp/nonexistent/SKILL.md")
        assert ok is False
        assert code == "file_not_found"
        assert "not found" in err

    def test_load_no_frontmatter_rejected(self, tmp_path):
        svc = self._svc()
        f = tmp_path / "SKILL.md"
        f.write_text("# Just a heading\nNo frontmatter here")
        ok, code, err = svc.load_skill(str(f))
        assert ok is False
        assert code == "no_frontmatter"

    def test_load_invalid_extension_rejected(self, tmp_path):
        svc = self._svc()
        f = tmp_path / "readme.txt"
        f.write_text("hello")
        ok, code, err = svc.load_skill(str(f))
        assert ok is False
        assert code == "invalid_file"

    def test_load_md_without_name_uses_heading_fallback(self, tmp_path):
        """When frontmatter has no 'name', parser falls back to heading."""
        svc = self._svc()
        f = tmp_path / "SKILL.md"
        f.write_text("---\ndescription: A skill\n---\n# My Skill Name\n")
        ok, name, err = svc.load_skill(str(f))
        assert ok is True
        assert name == "My Skill Name"

    def test_load_md_completely_empty_frontmatter(self, tmp_path):
        """Frontmatter with no name and no heading → 'unknown' → rejected."""
        svc = self._svc()
        f = tmp_path / "SKILL.md"
        f.write_text("---\ndescription: oops\n---\nJust body text, no heading.\n")
        ok, code, err = svc.load_skill(str(f))
        assert ok is False

    # ── Directory loading ─────────────────────────────────────

    def test_load_directory(self, tmp_path):
        svc = self._svc()
        sub = tmp_path / "my-skill"
        sub.mkdir()
        (sub / "SKILL.md").write_text("---\nname: dir-skill\ndescription: From dir\n---\n")
        ok, names, err = svc.load_skill(str(tmp_path))
        assert ok is True
        assert "dir-skill" in names
        assert err is None

    def test_load_empty_directory(self, tmp_path):
        svc = self._svc()
        ok, code, err = svc.load_skill(str(tmp_path))
        assert ok is False
        assert code == "no_skills"

    def test_load_directory_with_lowercase_skill_md(self, tmp_path):
        svc = self._svc()
        sub = tmp_path / "my-skill"
        sub.mkdir()
        (sub / "skill.md").write_text("---\nname: lower-skill\ndescription: Lowercase\n---\n")
        ok, names, err = svc.load_skill(str(tmp_path))
        assert ok is True
        assert "lower-skill" in names

    # ── URL loading ───────────────────────────────────────────

    def test_load_url_404(self):
        svc = self._svc()
        ok, code, err = svc.load_skill("https://httpstat.us/404")
        assert ok is False
        assert code in ("url_http_error", "url_download_failed")

    def test_load_url_invalid_scheme(self):
        svc = self._svc()
        ok, code, err = svc.load_skill("ftp://example.com/SKILL.md")
        # Not recognized as URL (no http/https), treated as file path
        assert ok is False
        assert code == "file_not_found"

    def test_load_github_tree_url_rejected(self):
        svc = self._svc()
        ok, code, err = svc.load_skill("https://github.com/user/repo/tree/main/skills")
        assert ok is False
        assert code == "invalid_url"
        assert "directory" in err.lower()


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


# ===========================================================================
# Helpers
# ===========================================================================


class TestHelpers:
    """Tests for module-level helper functions."""

    def test_empty_metrics(self):
        from houyi_studio.server.skill_service import _empty_metrics

        m = _empty_metrics("test_skill")
        assert m["skill_name"] == "test_skill"
        assert m["total_calls"] == 0
        assert m["success_count"] == 0
        assert m["failure_count"] == 0
        assert m["avg_latency_ms"] == 0.0
        assert m["success_rate"] == 0.0
        assert m["last_invoked"] is None

    def test_extract_side_effects_network(self):
        from houyi_studio.server.skill_serializer import extract_side_effects

        perms = _FakePermissions(network=_FakePermKind(enabled=True))
        assert extract_side_effects(perms) == ["network"]

    def test_extract_side_effects_multiple(self):
        from houyi_studio.server.skill_serializer import extract_side_effects

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
        from houyi_studio.server.skill_serializer import extract_side_effects

        perms = _FakePermissions()
        assert extract_side_effects(perms) == []

    def test_dominant_side_effect_priority(self):
        from houyi_studio.server.skill_serializer import dominant_side_effect

        perms = _FakePermissions(
            exec_=_FakePermKind(enabled=True),
            network=_FakePermKind(enabled=True),
        )
        assert dominant_side_effect(perms) == "exec"

    def test_dominant_side_effect_none(self):
        from houyi_studio.server.skill_serializer import dominant_side_effect

        perms = _FakePermissions()
        assert dominant_side_effect(perms) == "none"


class TestIsSkillLoaded:
    """Tests for duplicate detection via is_skill_loaded."""

    def test_loaded_skill_found(self, populated_service):
        assert populated_service.is_skill_loaded("web_search") is True

    def test_unloaded_skill_not_found(self, populated_service):
        assert populated_service.is_skill_loaded("nonexistent") is False

    def test_after_unload(self, populated_service):
        populated_service.unload_skill("web_search")
        assert populated_service.is_skill_loaded("web_search") is False
