"""Tests for SkillLoader loading/unloading/URL handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from houyi_studio.server.skill.dry_run import DryRunValidator
from houyi_studio.server.skill.loader import (
    ENV_MANAGED_SKILLS_DIR,
    ERR_FILE_NOT_FOUND,
    ERR_INVALID_FILE,
    ERR_NO_SKILLS,
    ERR_VALIDATION_FAILED,
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

    def test_managed_skills_root_uses_tmp_houyi_layout(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_MANAGED_SKILLS_DIR, str(tmp_path / ".houyi" / "skills"))
        loader = SkillLoader(SkillRegistry())
        assert loader._managed_skills_root() == (tmp_path / ".houyi" / "skills")
        assert loader._managed_skills_root() != (tmp_path / "skills")

    def test_is_loaded_empty(self, loader):
        assert loader.is_loaded("anything") is False

    def test_load_nonexistent_path(self, loader):
        ok, code, msg = loader.load("/no/such/path")
        assert ok is False
        assert code == ERR_FILE_NOT_FOUND

    def test_load_invalid_install_strategy(self, tmp_path, loader):
        skill_dir = tmp_path / "pkg"
        skill_dir.mkdir()
        (skill_dir / SKILL_MD_UPPER).write_text("---\nname: pkg\n---\n", encoding="utf-8")

        ok, code, msg = loader.load(str(skill_dir), install_strategy="hardlink")
        assert ok is False
        assert code == ERR_VALIDATION_FAILED
        assert msg is not None
        assert "install_strategy" in msg

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

        ok, loaded_name, err = loader.load(str(source_pkg / SKILL_MD_UPPER))

        assert ok is True
        assert err is None
        assert loaded_name == "planning-with-files"

        installed_pkg = tmp_path / ".houyi" / "skills" / "planning-with-files"
        assert (installed_pkg / "SKILL.md").exists()
        assert (installed_pkg / "scripts" / "check-complete.sh").exists()
        assert (installed_pkg / "templates" / "task_plan.md").exists()

    def test_load_directory_symlink_strategy_installs_linked_source(self, tmp_path):
        registry = SkillRegistry()
        loader = SkillLoader(registry)

        source_dir = tmp_path / "community-skills"
        pkg = source_dir / "docx"
        pkg.mkdir(parents=True)
        (pkg / SKILL_MD_UPPER).write_text("---\nname: docx\n---\n", encoding="utf-8")

        ok, loaded_name, err = loader.load(str(source_dir), install_strategy="symlink")

        assert ok is True
        assert err is None
        assert "docx" in loaded_name

        installed = tmp_path / ".houyi" / "skills" / "community-skills"
        assert installed.is_symlink()
        assert installed.resolve() == source_dir.resolve()

    def test_load_conflicting_external_skill_hydrates_from_core_runtime(self, tmp_path):
        registry = SkillRegistry()
        loader = SkillLoader(registry)

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
    async def test_hydrate_script_compat_runtime_binds_executor_for_instruction_commands(
        self, tmp_path
    ):
        registry = SkillRegistry()
        loader = SkillLoader(registry)

        skill_dir = tmp_path / "docx"
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir(parents=True)
        run_py = scripts_dir / "run.py"
        run_py.write_text(
            """
import json
import sys

def main():
    print(json.dumps({"argv": sys.argv[1:]}))

if __name__ == "__main__":
    main()
""".strip()
            + "\n",
            encoding="utf-8",
        )

        md = skill_dir / SKILL_MD_UPPER
        md.write_text(
            "---\nname: docx\ndescription: DOCX helper\n---\n"
            "```bash\n"
            'python scripts/run.py docx_tool.py --question "..." --doc-path "..."\n'
            "```\n",
            encoding="utf-8",
        )
        ok, loaded_name, err = loader.load(str(md))

        assert ok is True
        assert err is None
        assert loaded_name == "docx"

        loaded = registry.get("docx")
        assert loaded is not None
        assert callable(loaded.executor)

        result = await loaded.executor(
            question="Summarize architecture",
            doc_path="/tmp/demo.docx",
        )
        assert result.get("ok") is True
        assert "docx_tool.py" in result.get("command", [])
        assert "--question" in result.get("command", [])

    def test_build_script_compat_command_prefers_command_payload(self):
        args = SkillLoader._build_script_compat_command(
            {
                "command": "python scripts/run.py auth_manager.py status",
            },
            templates=[],
        )
        assert args == ["python", "scripts/run.py", "auth_manager.py", "status"]

    def test_build_script_compat_command_maps_flags_from_template(self):
        args = SkillLoader._build_script_compat_command(
            {
                "script": "ask_question.py",
                "operation": "status",
                "question": "What changed?",
                "notebook_url": "https://notebooklm.google.com/notebook/example",
            },
            templates=[
                {
                    "raw": 'python scripts/run.py ask_question.py --question "..." --notebook-url "..."',
                    "base_tokens": ["python", "scripts/run.py", "ask_question.py"],
                    "flags": ["question", "notebook_url"],
                }
            ],
        )
        assert args[:3] == ["python", "scripts/ask_question.py", "status"]
        assert "--question" in args
        assert "--notebook-url" in args

    def test_build_script_compat_command_bypasses_run_wrapper_with_script_payload(self):
        args = SkillLoader._build_script_compat_command(
            {
                "script": "auth_manager.py",
                "operation": "status",
            },
            templates=[
                {
                    "raw": "python scripts/run.py auth_manager.py status",
                    "base_tokens": ["python", "scripts/run.py", "auth_manager.py", "status"],
                    "flags": [],
                }
            ],
        )
        assert args == ["python", "scripts/auth_manager.py", "status"]

    @pytest.mark.asyncio
    async def test_loaded_ext_planning_uses_hydrated_schema_in_dry_run(self, tmp_path):
        registry = SkillRegistry()
        loader = SkillLoader(registry)

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

    def test_load_from_github_url_uses_managed_install_verify_gate(self, tmp_path):
        registry = SkillRegistry()
        loader = SkillLoader(registry)

        managed_sources = tmp_path / "sources"
        managed_skills = tmp_path / "skills"
        clone_root = managed_sources / "user" / "repo"
        target_skill = clone_root / "skills" / "pdf" / "SKILL.md"
        target_skill.parent.mkdir(parents=True)
        target_skill.write_text("---\nname: pdf\n---\n", encoding="utf-8")

        with (
            patch.object(SkillLoader, "_managed_sources_root", return_value=managed_sources),
            patch.object(SkillLoader, "_managed_global_skills_root", return_value=managed_skills),
            patch.object(
                loader, "_load_from_skill_md", return_value=(True, "pdf", None)
            ) as load_md,
        ):
            ok, name, err = loader._load_from_url(
                "https://github.com/user/repo/blob/main/skills/pdf/SKILL.md"
            )

        assert ok is True
        assert err is None
        assert name == "pdf"
        assert (managed_skills / "pdf").exists()
        load_md.assert_called_once_with(target_skill, str(target_skill))

    def test_resolve_github_link_binding_uses_repo_root_for_root_skill_md(self):
        alias, target = SkillLoader._resolve_github_link_binding("skills", "SKILL.md")
        assert alias == "skills"
        assert target == Path(".")

    def test_load_from_github_url_replaces_existing_non_symlink_alias_directory(self, tmp_path):
        registry = SkillRegistry()
        loader = SkillLoader(registry)

        managed_sources = tmp_path / "sources"
        managed_skills = tmp_path / "skills"
        clone_root = managed_sources / "anthropics" / "skills"
        target_skill = clone_root / "skills" / "docx" / "SKILL.md"
        target_skill.parent.mkdir(parents=True)
        target_skill.write_text("---\nname: docx\n---\n", encoding="utf-8")

        existing_alias = managed_skills / "docx"
        existing_alias.mkdir(parents=True, exist_ok=True)
        (existing_alias / "stale.txt").write_text("stale", encoding="utf-8")

        with (
            patch.object(SkillLoader, "_managed_sources_root", return_value=managed_sources),
            patch.object(SkillLoader, "_managed_global_skills_root", return_value=managed_skills),
            patch.object(loader, "_load_from_skill_md", return_value=(True, "docx", None)),
        ):
            ok, name, err = loader._load_from_url(
                "https://github.com/anthropics/skills/blob/main/skills/docx/SKILL.md"
            )

        assert ok is True
        assert err is None
        assert name == "docx"
        assert (managed_skills / "docx").is_symlink()
        assert (managed_skills / "docx").resolve() == (clone_root / "skills" / "docx").resolve()
        assert not (managed_skills / "docx" / "stale.txt").exists()

    def test_load_from_github_url_rebinds_existing_local_alias_symlink(self, tmp_path):
        registry = SkillRegistry()
        loader = SkillLoader(registry)

        managed_sources = tmp_path / "sources"
        managed_skills = tmp_path / "skills"

        local_docx = managed_sources / "local" / "docx"
        local_docx.mkdir(parents=True, exist_ok=True)
        (local_docx / "SKILL.md").write_text("---\nname: docx\n---\n", encoding="utf-8")

        current_alias = managed_skills / "docx"
        current_alias.parent.mkdir(parents=True, exist_ok=True)
        current_alias.symlink_to(local_docx, target_is_directory=True)

        clone_root = managed_sources / "anthropics" / "skills"
        github_docx = clone_root / "skills" / "docx"
        target_skill = github_docx / "SKILL.md"
        target_skill.parent.mkdir(parents=True, exist_ok=True)
        target_skill.write_text("---\nname: docx\n---\n", encoding="utf-8")

        with (
            patch.object(SkillLoader, "_managed_sources_root", return_value=managed_sources),
            patch.object(SkillLoader, "_managed_global_skills_root", return_value=managed_skills),
            patch.object(loader, "_load_from_skill_md", return_value=(True, "docx", None)),
        ):
            ok, name, err = loader._load_from_url(
                "https://github.com/anthropics/skills/blob/main/skills/docx/SKILL.md"
            )

        assert ok is True
        assert err is None
        assert name == "docx"
        assert (managed_skills / "docx").is_symlink()
        assert (managed_skills / "docx").resolve() == github_docx.resolve()
        assert (managed_skills / "docx").resolve() != local_docx.resolve()

    def test_load_from_github_url_root_skill_uses_skill_name_alias(self, tmp_path):
        registry = SkillRegistry()
        loader = SkillLoader(registry)

        managed_sources = tmp_path / "sources"
        managed_skills = tmp_path / "skills"
        clone_root = managed_sources / "PleasePrompto" / "notebooklm-skill"
        target_skill = clone_root / "SKILL.md"
        target_skill.parent.mkdir(parents=True, exist_ok=True)
        target_skill.write_text("---\nname: notebooklm\n---\n", encoding="utf-8")

        stale_repo_alias = managed_skills / "notebooklm-skill"
        stale_repo_alias.parent.mkdir(parents=True, exist_ok=True)
        stale_repo_alias.symlink_to(clone_root, target_is_directory=True)

        with (
            patch.object(SkillLoader, "_managed_sources_root", return_value=managed_sources),
            patch.object(SkillLoader, "_managed_global_skills_root", return_value=managed_skills),
            patch.object(loader, "_load_from_skill_md", return_value=(True, "notebooklm", None)),
        ):
            ok, name, err = loader._load_from_url(
                "https://github.com/PleasePrompto/notebooklm-skill/blob/master/SKILL.md"
            )

        assert ok is True
        assert err is None
        assert name == "notebooklm"
        assert (managed_skills / "notebooklm").is_symlink()
        assert (managed_skills / "notebooklm").resolve() == clone_root.resolve()
        assert not stale_repo_alias.exists()

    def test_load_from_github_url_fails_when_verify_missing_skills_dir(self, tmp_path):
        registry = SkillRegistry()
        loader = SkillLoader(registry)

        managed_sources = tmp_path / "sources"
        managed_skills = tmp_path / "skills"
        clone_root = managed_sources / "user" / "repo"
        clone_root.mkdir(parents=True)

        with (
            patch.object(SkillLoader, "_managed_sources_root", return_value=managed_sources),
            patch.object(SkillLoader, "_managed_global_skills_root", return_value=managed_skills),
        ):
            ok, code, err = loader._load_from_url(
                "https://github.com/user/repo/blob/main/skills/pdf/SKILL.md"
            )

        assert ok is False
        assert code == "url_load_failed"
        assert err is not None
        assert "verify failed" in err.lower()
