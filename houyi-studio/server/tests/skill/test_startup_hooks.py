from __future__ import annotations

from houyi_studio.server.skill.startup_hooks import (
    _discover_external_skill_names,
    _group_registered_names,
    _hydrate_external_runtime,
    _prune_stale_external_skills,
)
from pydantic import BaseModel

from houyi.core.skill.spec import SkillSpec
from houyi.core.skill_registry import SkillRegistry


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
