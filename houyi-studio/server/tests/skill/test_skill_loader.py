"""Tests for SkillLoader loading/unloading/URL handling."""

from __future__ import annotations

import pytest
from houyi_studio.server.skill_loader import (
    ERR_FILE_NOT_FOUND,
    ERR_INVALID_FILE,
    ERR_NO_SKILLS,
    SKILL_MD_UPPER,
    SkillLoader,
    _validate_parsed_skill,
    normalize_github_url,
    validate_skill_content,
)

from houyi.core.skill_registry import SkillRegistry


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
