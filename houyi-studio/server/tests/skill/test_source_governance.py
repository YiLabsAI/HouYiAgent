"""Tests for source governance lifecycle helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from houyi_studio.server.skill.source_governance import (
    build_fallback_lifecycle_plan,
    discover_install_doc,
    infer_skills_subdir,
)


def test_discover_install_doc_prefers_codex_install(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".codex").mkdir(parents=True)
    (repo / ".codex" / "INSTALL.md").write_text("# install", encoding="utf-8")

    found = discover_install_doc(repo)
    assert found == repo / ".codex" / "INSTALL.md"


def test_infer_skills_subdir_prefers_skills_dir(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "skills" / "example").mkdir(parents=True)
    (repo / "skills" / "example" / "SKILL.md").write_text("---\nname: a\n---", encoding="utf-8")

    assert infer_skills_subdir(repo) == "skills"


def test_infer_skills_subdir_raises_without_skill_content(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)

    with pytest.raises(ValueError, match="No skill directory found"):
        infer_skills_subdir(repo)


def test_build_fallback_lifecycle_plan_matches_codex_style() -> None:
    plan = build_fallback_lifecycle_plan(
        repo_url="https://github.com/obra/superpowers.git",
        alias="superpowers",
        skills_subdir="skills",
    )

    assert plan.strategy == "generated_clone_symlink"
    assert plan.install_commands[0] == (
        "git clone https://github.com/obra/superpowers.git "
        "~/.houyi/sources/github.com/obra/superpowers"
    )
    assert plan.install_commands[2] == (
        "ln -s ~/.houyi/sources/github.com/obra/superpowers/skills ~/.houyi/skills/superpowers"
    )
    assert plan.verify_command == "ls -la ~/.houyi/skills/superpowers"
    assert plan.update_command == "git -C ~/.houyi/sources/github.com/obra/superpowers pull"
    assert plan.uninstall_command == "rm ~/.houyi/skills/superpowers"
