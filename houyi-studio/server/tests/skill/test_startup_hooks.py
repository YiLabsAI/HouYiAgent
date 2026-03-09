from __future__ import annotations

import sys
import types

import pytest
from houyi_studio.server.skill import startup_hooks as startup_hooks_module
from houyi_studio.server.skill.service import get_skill_service
from houyi_studio.server.skill.startup_hooks import (
    _default_startup_skills_dir,
    _discover_external_skill_names,
    _group_registered_names,
    _hydrate_external_runtime,
    _init_skill_service,
    _iter_external_skill_files,
    _prune_stale_external_skills,
    _read_declared_skill_name,
    _resolve_external_skill_scan_dirs,
)
from pydantic import BaseModel

from houyi.domain.skill.registry import SkillRegistry
from houyi.domain.skill.spec import SkillSpec
from houyi.infrastructure.config.env_config import ENV_STARTUP_SKILLS_DIR, EnvConfig


@pytest.fixture(autouse=True)
def _reset_env_config_singleton():
    EnvConfig._reset()
    yield
    EnvConfig._reset()


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    r: str


class _Empty(BaseModel):
    pass


def _skill(name: str, *, is_core: bool) -> SkillSpec:
    return SkillSpec(
        name=name,
        description=f"skill {name}",
        input_schema=_In,
        output_schema=_Out,
        is_core=is_core,
    )


def test_group_registered_names_groups_core_then_external_and_dedupes() -> None:
    registry = SkillRegistry()
    registry.register(_skill("planning-with-files", is_core=True), overwrite=True)
    registry.register(_skill("ext__planning-with-files", is_core=False), overwrite=True)
    registry.register(_skill("web_search", is_core=True), overwrite=True)

    registered = [
        "planning-with-files",
        "ext__planning-with-files",
        "planning-with-files",
        "web_search",
    ]

    core_names, external_names = _group_registered_names(registered, registry)

    assert core_names == ["planning-with-files", "web_search"]
    assert external_names == ["ext__planning-with-files"]


def test_group_registered_names_treats_unknown_as_external() -> None:
    registry = SkillRegistry()
    registry.register(_skill("web_search", is_core=True), overwrite=True)

    core_names, external_names = _group_registered_names(["web_search", "missing"], registry)

    assert core_names == ["web_search"]
    assert external_names == ["missing"]


def test_hydrate_external_runtime_copies_core_executor_and_schema() -> None:
    registry = SkillRegistry()

    core = SkillSpec(
        name="planning-with-files",
        description="core planning",
        input_schema=_In,
        output_schema=_Out,
        is_core=True,
    )

    async def _executor(**kwargs):
        return {"ok": True, "kwargs": kwargs}

    core.bind_executor(_executor)
    registry.register(core, overwrite=True)

    external = SkillSpec(
        name="ext__planning-with-files",
        description="external planning",
        input_schema=_Empty,
        output_schema=_Empty,
        is_core=False,
    )
    registry.register(external, overwrite=True)

    hydrated = _hydrate_external_runtime(["ext__planning-with-files"], registry)
    hydrated_skill = registry.get("ext__planning-with-files")

    assert hydrated == ["ext__planning-with-files"]
    assert hydrated_skill is not None
    assert callable(hydrated_skill.executor)
    assert hydrated_skill.input_schema is _In
    assert hydrated_skill.output_schema is _Out


def test_hydrate_external_runtime_replaces_stale_external_hooks_with_core_runtime() -> None:
    registry = SkillRegistry()

    core = SkillSpec(
        name="planning-with-files",
        description="core planning",
        input_schema=_In,
        output_schema=_Out,
        is_core=True,
        hooks=[{"type": "handler", "handler": "houyi.skills.planning.hooks:stop_hook"}],
        skill_dir="/repo/houyi/skills/planning",
        skill_md_path="/repo/houyi/skills/planning/SKILL.md",
    )

    async def _executor(**kwargs):
        return {"ok": True, "kwargs": kwargs}

    core.bind_executor(_executor)
    registry.register(core, overwrite=True)

    external = SkillSpec(
        name="ext__planning-with-files",
        description="external planning",
        input_schema=_Empty,
        output_schema=_Empty,
        is_core=False,
        hooks=[
            {
                "type": "command",
                "command": "sh /Users/von/.claude/plugins/planning-with-files/scripts/check-complete.sh",
            }
        ],
        skill_dir="/external/skills/planning-with-files",
        skill_md_path="/external/skills/planning-with-files/SKILL.md",
    )
    registry.register(external, overwrite=True)

    hydrated = _hydrate_external_runtime(["ext__planning-with-files"], registry)
    hydrated_skill = registry.get("ext__planning-with-files")

    assert hydrated == ["ext__planning-with-files"]
    assert hydrated_skill is not None
    assert hydrated_skill.hooks == core.hooks
    assert hydrated_skill.skill_dir == core.skill_dir
    assert hydrated_skill.skill_md_path == core.skill_md_path


def test_hydrate_external_runtime_keeps_existing_external_executor() -> None:
    registry = SkillRegistry()

    core = SkillSpec(
        name="planning-with-files",
        description="core planning",
        input_schema=_In,
        output_schema=_Out,
        is_core=True,
    )

    async def _core_executor(**kwargs):
        return {"from": "core", "kwargs": kwargs}

    core.bind_executor(_core_executor)
    registry.register(core, overwrite=True)

    external = SkillSpec(
        name="ext__planning-with-files",
        description="external planning",
        input_schema=_In,
        output_schema=_Out,
        is_core=False,
    )

    async def _external_executor(**kwargs):
        return {"from": "external", "kwargs": kwargs}

    external.bind_executor(_external_executor)
    registry.register(external, overwrite=True)

    hydrated = _hydrate_external_runtime(["ext__planning-with-files"], registry)
    hydrated_skill = registry.get("ext__planning-with-files")

    assert hydrated == []
    assert hydrated_skill is not None
    assert hydrated_skill.executor is _external_executor


def test_discover_external_skill_names_reads_skill_md(tmp_path) -> None:
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "rag-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: rag-skill
description: test
---
""",
        encoding="utf-8",
    )

    discovered = _discover_external_skill_names(skills_dir)
    assert discovered == {"rag-skill"}


def test_discover_external_skill_names_from_symlinked_package(tmp_path) -> None:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(parents=True)
    source_pkg = tmp_path / "sources" / "local" / "frontend-design"
    source_pkg.mkdir(parents=True)
    (source_pkg / "SKILL.md").write_text(
        """---
name: frontend-design
description: test
---
""",
        encoding="utf-8",
    )

    (skills_dir / "frontend-design").symlink_to(source_pkg, target_is_directory=True)

    discovered = _discover_external_skill_names(skills_dir)
    assert "frontend-design" in discovered


def test_iter_external_skill_files_prefers_uppercase_skill_md(tmp_path) -> None:
    skills_dir = tmp_path / "skills"
    pkg = skills_dir / "duplicate-entry"
    pkg.mkdir(parents=True)
    (pkg / "SKILL.md").write_text("---\nname: duplicate-entry\n---\n", encoding="utf-8")
    (pkg / "skill.md").write_text("---\nname: duplicate-entry\n---\n", encoding="utf-8")

    files = _iter_external_skill_files(skills_dir)

    assert len(files) == 1
    assert files[0].name == "SKILL.md"


def test_read_declared_skill_name_returns_frontmatter_name(tmp_path) -> None:
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("---\nname: notebooklm\n---\n", encoding="utf-8")

    assert _read_declared_skill_name(skill_file) == "notebooklm"


def test_load_external_skills_skips_duplicate_declared_name(monkeypatch, tmp_path) -> None:
    skills_dir = tmp_path / "skills"
    src_a = skills_dir / "src-a"
    src_b = skills_dir / "src-b"
    src_a.mkdir(parents=True)
    src_b.mkdir(parents=True)
    (src_a / "SKILL.md").write_text("---\nname: notebooklm\n---\n", encoding="utf-8")
    (src_b / "SKILL.md").write_text("---\nname: notebooklm\n---\n", encoding="utf-8")

    class _FakeRegistry:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def register_from_skill_file(self, source: str, overwrite: bool = False) -> str:
            self.calls.append(source)
            return "notebooklm"

    fake_registry = _FakeRegistry()
    monkeypatch.setattr(startup_hooks_module, "DEFAULT_SKILL_REGISTRY", fake_registry)
    monkeypatch.setattr(
        startup_hooks_module, "_resolve_external_skill_scan_dirs", lambda: [skills_dir]
    )
    monkeypatch.setattr(
        startup_hooks_module, "_prune_stale_external_skills", lambda *args, **kwargs: []
    )

    registered: list[str] = []
    startup_hooks_module._load_external_skills(registered)

    assert len(fake_registry.calls) == 1
    assert registered == ["notebooklm"]


def test_prune_stale_external_skills_removes_missing_skill(tmp_path) -> None:
    registry = SkillRegistry()
    skills_dir = tmp_path / "skills"
    stale_dir = skills_dir / "legacy"
    stale_dir.mkdir(parents=True)

    stale = SkillSpec(
        name="kb-retriever",
        description="legacy",
        input_schema=_In,
        output_schema=_Out,
        skill_dir=stale_dir,
        is_core=False,
    )
    registry.register(stale, overwrite=True)

    remaining = {"rag-skill"}
    pruned = _prune_stale_external_skills(skills_dir, remaining, registry)

    assert pruned == ["kb-retriever"]
    assert registry.get("kb-retriever") is None


def test_prune_keeps_ext_alias_when_canonical_name_is_discovered(tmp_path) -> None:
    registry = SkillRegistry()
    skills_dir = tmp_path / "skills"
    planning_dir = skills_dir / "planning-with-files"
    planning_dir.mkdir(parents=True)

    ext_alias = SkillSpec(
        name="ext__planning-with-files",
        description="external alias",
        input_schema=_In,
        output_schema=_Out,
        skill_dir=planning_dir,
        is_core=False,
    )
    registry.register(ext_alias, overwrite=True)

    pruned = _prune_stale_external_skills(skills_dir, {"planning-with-files"}, registry)

    assert pruned == []
    assert registry.get("ext__planning-with-files") is not None


def test_resolve_external_skill_scan_dirs_defaults_to_user_houyi() -> None:
    dirs = _resolve_external_skill_scan_dirs()
    assert dirs == [_default_startup_skills_dir()]


def test_resolve_external_skill_scan_dirs_honors_configured_dir(monkeypatch, tmp_path) -> None:
    configured = tmp_path / "custom-skills"
    monkeypatch.setenv(ENV_STARTUP_SKILLS_DIR, str(configured))
    dirs = _resolve_external_skill_scan_dirs()
    assert dirs == [configured]


def test_init_skill_service_initializes_global_service() -> None:
    _init_skill_service()
    assert get_skill_service() is not None


def test_register_console_skills_registers_builtin_local_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    monkeypatch.setattr(startup_hooks_module, "_init_skill_service", lambda: None)
    monkeypatch.setattr(startup_hooks_module, "_load_external_skills", lambda registered: None)
    monkeypatch.setattr(startup_hooks_module, "_hydrate_external_runtime", lambda registered: [])
    monkeypatch.setattr(startup_hooks_module, "_group_registered_names", lambda names: (names, []))

    def _capture_core(skill, registered_skills):
        registered_skills.append(skill.name)

    monkeypatch.setattr(startup_hooks_module, "_register_builtin_core", _capture_core)

    def _fake_register_builtin_local_tools(_registry):
        tool_names = [
            "houyi_read_file",
            "houyi_write_file",
            "houyi_find_files",
            "houyi_list_dir",
            "houyi_grep",
            "houyi_shell_exec",
        ]
        captured.extend(tool_names)
        return tool_names

    fake_module = types.SimpleNamespace(
        register_builtin_local_tools=_fake_register_builtin_local_tools
    )
    monkeypatch.setitem(sys.modules, "houyi.skills.builtin.local_tools", fake_module)

    startup_hooks_module.register_console_skills()

    assert captured == [
        "houyi_read_file",
        "houyi_write_file",
        "houyi_find_files",
        "houyi_list_dir",
        "houyi_grep",
        "houyi_shell_exec",
    ]
