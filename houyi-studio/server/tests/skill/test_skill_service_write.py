"""Tests for SkillService write operations: load, unload, configure."""

from __future__ import annotations

from unittest.mock import patch

from houyi.core.skill_registry import SkillRegistry


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
        with patch.object(
            skill_service._loader,
            "_load_from_url",
            return_value=(True, "url_skill", None),
        ):
            success, name, msg = skill_service.load_skill("https://example.com/skill/SKILL.md")
            assert success is True
            assert name == "url_skill"

    def test_load_from_url_failure(self, skill_service):
        """URL load that raises an exception returns failure."""
        with patch.object(
            skill_service._loader,
            "_load_from_url",
            return_value=(False, "url_load_failed", "download failed"),
        ):
            success, code, msg = skill_service.load_skill("https://example.com/bad.md")
            assert success is False
            assert code == "url_load_failed"


class TestLoadSkillPaths:
    """Tests for the three load paths: file, URL, directory."""

    def _svc(self):
        from houyi_studio.server.skill.service import SkillService

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
        assert code in ("url_http_error", "url_download_failed", "url_unreachable")

    def test_load_url_invalid_scheme(self):
        svc = self._svc()
        ok, code, err = svc.load_skill("ftp://example.com/SKILL.md")
        assert ok is False
        assert code == "file_not_found"

    def test_load_github_tree_url_rejected(self):
        svc = self._svc()
        ok, code, err = svc.load_skill("https://github.com/user/repo/tree/main/skills")
        assert ok is False
        assert code == "invalid_url"
        assert "directory" in err.lower()


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
