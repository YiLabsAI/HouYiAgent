"""Tests for SkillLoader loading/unloading/URL handling."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from houyi_studio.server.skill.dry_run import DryRunValidator
from houyi_studio.server.skill.loader import (
    ERR_FILE_NOT_FOUND,
    ERR_INVALID_FILE,
    ERR_NO_SKILLS,
    SKILL_MD_UPPER,
    SkillLoader,
    _validate_parsed_skill,
    normalize_github_url,
    validate_skill_content,
)
from pydantic import BaseModel

from houyi.core.skill.spec import SkillSpec
from houyi.core.skill_registry import SkillRegistry


class _In(BaseModel):
    action: str


class _Out(BaseModel):
    success: bool


class _FakeSkillSpec:
    def __init__(self, name="test", description="desc"):
        self.name = name
        self.description = description


# ── normalize_github_url ──────────────────────────────────────────────


class TestNormalizeGithubUrl:
    def test_blob_url_converted(self):
        url = "https://github.com/user/repo/blob/main/SKILL.md"
        raw = normalize_github_url(url)
        assert "raw.githubusercontent.com" in raw
        assert "/blob/" not in raw

    def test_tree_url_rejected(self):
        with pytest.raises(ValueError, match="directory"):
            normalize_github_url("https://github.com/user/repo/tree/main/skills")

    def test_passthrough(self):
        url = "https://example.com/some/SKILL.md"
        assert normalize_github_url(url) == url


# ── validate_skill_content ────────────────────────────────────────────


class TestValidateSkillContent:
    def test_rejects_html(self):
        with pytest.raises(ValueError, match="HTML"):
            validate_skill_content("<!DOCTYPE html><html>", "http://x")

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            validate_skill_content("   ", "http://x")

    def test_accepts_valid(self):
        validate_skill_content("---\nname: test\n---\n# Test", "http://x")


# ── _validate_parsed_skill ────────────────────────────────────────────


class TestValidateParsedSkill:
    def test_rejects_unknown_name(self):
        skill = _FakeSkillSpec(name="unknown")
        with pytest.raises(ValueError, match="name"):
            _validate_parsed_skill(skill)

    def test_rejects_empty_name(self):
        skill = _FakeSkillSpec(name="")
        with pytest.raises(ValueError, match="name"):
            _validate_parsed_skill(skill)

    def test_accepts_valid(self):
        _validate_parsed_skill(_FakeSkillSpec(name="good"))


# ── SkillLoader ───────────────────────────────────────────────────────


class TestSkillLoader:
    @pytest.fixture
    def loader(self):
        return SkillLoader(SkillRegistry())

    def test_is_loaded_empty(self, loader):
        assert loader.is_loaded("anything") is False

    def test_load_nonexistent_path(self, loader):
        ok, code, msg = loader.load("/no/such/path")
        assert ok is False
        assert code == ERR_FILE_NOT_FOUND

    def test_load_unsupported_extension(self, tmp_path, loader):
        f = tmp_path / "skill.yaml"
        f.write_text("name: x")
        ok, code, _ = loader.load(str(f))
        assert ok is False
        assert code == ERR_INVALID_FILE

    def test_load_empty_directory(self, tmp_path, loader):
        ok, code, _ = loader.load(str(tmp_path))
        assert ok is False
        assert code == ERR_NO_SKILLS

    def test_unload_missing(self, loader):
        ok, msg = loader.unload("nonexistent")
        assert ok is False

    def test_load_and_unload_skill_md(self, tmp_path, loader):
        md = tmp_path / SKILL_MD_UPPER
        md.write_text("---\nname: test-skill\ndescription: A test\n---\n# Test")
        ok, name, _ = loader.load(str(md))
        assert ok is True
        assert name == "test-skill"
        assert loader.is_loaded("test-skill") is True

        ok2, _ = loader.unload("test-skill")
        assert ok2 is True
        assert loader.is_loaded("test-skill") is False

    def test_load_skill_md_installs_full_package_into_managed_skills(self, tmp_path):
        registry = SkillRegistry()
        loader = SkillLoader(registry)

        project_root = tmp_path / "project"
        source_pkg = tmp_path / "source" / "planning-with-files"
        scripts_dir = source_pkg / "scripts"
        templates_dir = source_pkg / "templates"
        scripts_dir.mkdir(parents=True)
        templates_dir.mkdir(parents=True)

        (source_pkg / SKILL_MD_UPPER).write_text(
            "---\nname: planning-with-files\ndescription: external planning\n---\n# Planning\n",
            encoding="utf-8",
        )
        (scripts_dir / "check-complete.sh").write_text("echo ok\n", encoding="utf-8")
        (templates_dir / "task_plan.md").write_text("# plan\n", encoding="utf-8")

        with patch.object(SkillLoader, "_project_root", return_value=project_root):
            ok, loaded_name, err = loader.load(str(source_pkg / SKILL_MD_UPPER))

        assert ok is True
        assert err is None
        assert loaded_name == "planning-with-files"

        installed_pkg = project_root / "skills" / "planning-with-files"
        assert (installed_pkg / "SKILL.md").exists()
        assert (installed_pkg / "scripts" / "check-complete.sh").exists()
        assert (installed_pkg / "templates" / "task_plan.md").exists()

    def test_load_conflicting_external_skill_hydrates_from_core_runtime(self, tmp_path):
        registry = SkillRegistry()
        loader = SkillLoader(registry)
        project_root = tmp_path / "project"

        core = SkillSpec(
            name="planning-with-files",
            description="core planning",
            input_schema=_In,
            output_schema=_Out,
            is_core=True,
        )

        async def _core_executor(**kwargs):
            return {"success": True, "kwargs": kwargs}

        core.bind_executor(_core_executor)
        registry.register(core, overwrite=True)

        md = tmp_path / SKILL_MD_UPPER
        md.write_text(
            "---\nname: planning-with-files\ndescription: External planning\n---\n# External"
        )

        with patch.object(SkillLoader, "_project_root", return_value=project_root):
            ok, name, err = loader.load(str(md))

        assert ok is True
        assert err is None
        assert name == "ext__planning-with-files"

        external = registry.get("ext__planning-with-files")
        assert external is not None
        assert callable(external.executor)
        assert external.input_schema is _In
        assert external.output_schema is _Out

    @pytest.mark.asyncio
    async def test_loaded_ext_planning_uses_hydrated_schema_in_dry_run(self, tmp_path):
        registry = SkillRegistry()
        loader = SkillLoader(registry)
        project_root = tmp_path / "project"

        core = SkillSpec(
            name="planning-with-files",
            description="core planning",
            input_schema=_In,
            output_schema=_Out,
            is_core=True,
        )

        async def _core_executor(**kwargs):
            return {"success": True, "kwargs": kwargs}

        core.bind_executor(_core_executor)
        registry.register(core, overwrite=True)

        md = tmp_path / SKILL_MD_UPPER
        md.write_text(
            "---\nname: planning-with-files\ndescription: External planning\n---\n# External"
        )

        with patch.object(SkillLoader, "_project_root", return_value=project_root):
            ok, loaded_name, err = loader.load(str(md))
        assert ok is True
        assert err is None
        assert loaded_name == "ext__planning-with-files"

        validator = DryRunValidator(registry)
        result = await validator.validate(
            "ext__planning-with-files",
            "ext__planning-with-files",
            {},
        )

        assert result["valid"] is True

        invalid = await validator.validate(
            "ext__planning-with-files",
            "ext__planning-with-files",
            {"wrong": "x"},
        )
        assert invalid["valid"] is False
