"""Tests for SkillService dry-run validation."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from _fakes import (
    _FakeInputSchema,
    _FakeSkillSpec,
    _FakeTool,
)


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
        assert isinstance(result["valid"], bool)

    @pytest.mark.asyncio
    async def test_policy_deny(self, populated_registry):
        """When skill's InvocationPolicy is set to deny, dry-run should reflect it."""
        from _fakes import _FakePolicy
        from houyi_studio.server.skill.service import SkillService

        skill = populated_registry.get("web_search")
        skill.invocation_policy = _FakePolicy("deny")

        svc = SkillService(registry=populated_registry)
        result = await svc.dry_run("web_search", "search", {})
        assert result["valid"] is False
        assert result["policy_result"] == "deny"

    # ─── Tool-level schema validation ────────────────────────────

    @pytest.mark.asyncio
    async def test_tool_level_schema_validation_pass(self, registry):
        """When a tool has its own input_schema, dry-run validates against it."""
        from houyi_studio.server.skill.service import SkillService

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
        from houyi_studio.server.skill.service import SkillService

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
        from houyi_studio.server.skill.service import SkillService

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
        from houyi_studio.server.skill.service import SkillService

        tool = _FakeTool("search", input_schema=_FakeInputSchema(["query"]))
        skill_schema = _FakeInputSchema(["different_field"])
        skill = _FakeSkillSpec("search_skill", tools=[tool], input_schema=skill_schema)
        registry.register(skill, overwrite=True)
        svc = SkillService(registry=registry)

        result = await svc.dry_run("search_skill", "search", {"query": "test"})
        assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_fallback_to_skill_level_when_tool_has_no_schema(self, registry):
        """If tool has no schema, fall back to skill-level input_schema."""
        from houyi_studio.server.skill.service import SkillService

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
        from houyi_studio.server.skill.service import SkillService

        tool_a = _FakeTool("read_file", input_schema=_FakeInputSchema(["path"]))
        tool_b = _FakeTool("write_file", input_schema=_FakeInputSchema(["path", "content"]))
        skill = _FakeSkillSpec("file_ops", tools=[tool_a, tool_b])
        registry.register(skill, overwrite=True)
        svc = SkillService(registry=registry)

        result_read = await svc.dry_run("file_ops", "read_file", {"path": "/tmp/f"})
        assert result_read["valid"] is True

        result_write = await svc.dry_run("file_ops", "write_file", {"path": "/tmp/f"})
        assert result_write["valid"] is False
        assert any("content" in err for err in result_write["schema_errors"])

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
        from houyi_studio.server.skill.service import SkillService

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
        from houyi_studio.server.skill.service import SkillService

        tool = _FakeTool("search")
        skill = _FakeSkillSpec("web_search", tools=[tool])
        registry.register(skill, overwrite=True)
        svc = SkillService(registry=registry)

        with patch.dict("sys.modules", {"houyi.adapters.llm": None}):
            result = await svc.dry_run("web_search", "search", {}, live=True)
        assert result["llm_verification"]["success"] is False
        assert "not available" in result["llm_verification"]["message"]
