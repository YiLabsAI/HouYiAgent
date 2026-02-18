"""Tests for skill registration, listing, and lifecycle management.

Covers:
  - Each skill category can register into SkillRegistry
  - Registered skills have valid name and description
  - SKILL.md files parse correctly
  - Unregister/re-register lifecycle
  - Full 13-skill batch registration
"""

import glob
import os

import pytest

from houyi.core.skill import SkillSpec
from houyi.core.skill_registry import SkillRegistry

SKILLS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "skills")


@pytest.fixture
def registry() -> SkillRegistry:
    return SkillRegistry()


# ── Weather tools ────────────────────────────────────────────────


class TestWeatherRegistration:
    def test_get_weather(self, registry: SkillRegistry):
        from houyi.skills.weather import get_weather

        registry.register(get_weather, overwrite=True)
        assert "get_weather" in [s.name for s in registry.list()]

    def test_get_weather_has_hooks(self, registry: SkillRegistry):
        from houyi.skills.weather import get_weather

        registry.register(get_weather, overwrite=True)
        assert len(get_weather.hooks) == 2, "get_weather should have PreToolUse + PostToolUse hooks"

    def test_get_date(self, registry: SkillRegistry):
        from houyi.skills.weather import get_date

        registry.register(get_date, overwrite=True)
        assert "get_date" in [s.name for s in registry.list()]

    def test_spec_completeness(self):
        from houyi.skills.weather import get_weather

        assert get_weather.name == "get_weather"
        assert get_weather.description


# ── Location tool ────────────────────────────────────────────────


class TestLocationRegistration:
    def test_register(self, registry: SkillRegistry):
        from houyi.skills.location import get_location

        registry.register(get_location, overwrite=True)
        assert "get_location" in [s.name for s in registry.list()]

    def test_spec_completeness(self):
        from houyi.skills.location import get_location

        assert get_location.name == "get_location"
        assert get_location.description


# ── Web search ───────────────────────────────────────────────────


class TestWebSearchRegistration:
    def test_register(self, registry: SkillRegistry):
        from houyi.web_search.skill import build_web_search_skill

        skill = build_web_search_skill()
        registry.register(skill, overwrite=True)
        assert skill.name in [s.name for s in registry.list()]

    def test_spec_completeness(self):
        from houyi.web_search.skill import build_web_search_skill

        skill = build_web_search_skill()
        assert skill.description


# ── Planning ─────────────────────────────────────────────────────


class TestPlanningRegistration:
    def test_register(self, registry: SkillRegistry):
        from houyi.skills.planning import PlanningSkill

        spec = PlanningSkill().to_spec()
        registry.register(spec, overwrite=True)
        assert spec.name in [s.name for s in registry.list()]

    def test_spec_completeness(self):
        from houyi.skills.planning import PlanningSkill

        spec = PlanningSkill().to_spec()
        assert spec.name
        assert spec.description


# ── RAG skills ───────────────────────────────────────────────────


class TestRagRegistration:
    def test_kb_search(self, registry: SkillRegistry):
        from houyi.rag.skills.kb_search import kb_search_skill

        registry.register(kb_search_skill, overwrite=True)
        assert kb_search_skill.name in [s.name for s in registry.list()]

    def test_kb_ingest(self, registry: SkillRegistry):
        from houyi.rag.skills.kb_ingest import kb_ingest_skill

        registry.register(kb_ingest_skill, overwrite=True)
        assert kb_ingest_skill.name in [s.name for s in registry.list()]

    def test_kb_graph(self, registry: SkillRegistry):
        from houyi.rag.skills.kb_graph import kb_graph_skill

        registry.register(kb_graph_skill, overwrite=True)
        assert kb_graph_skill.name in [s.name for s in registry.list()]

    def test_kb_analyze(self, registry: SkillRegistry):
        from houyi.rag.skills.kb_analyze import kb_analyze_skill

        registry.register(kb_analyze_skill, overwrite=True)
        assert kb_analyze_skill.name in [s.name for s in registry.list()]


# ── SKILL.md file loading ────────────────────────────────────────


class TestSkillMdParsing:
    def test_parse_single_md(self, registry: SkillRegistry):
        skill_md_path = os.path.join(SKILLS_DIR, "planning-with-files", "SKILL.md")
        if not os.path.exists(skill_md_path):
            pytest.skip("planning-with-files/SKILL.md not found")
        spec = SkillSpec.from_file(skill_md_path)
        assert spec.name
        assert spec.description
        registry.register(spec, overwrite=True)
        assert spec.name in [s.name for s in registry.list()]

    def test_parse_all_md_files(self, registry: SkillRegistry):
        if not os.path.isdir(SKILLS_DIR):
            pytest.skip("skills/ directory not found")
        md_files = glob.glob(os.path.join(SKILLS_DIR, "**", "SKILL.md"), recursive=True)
        assert len(md_files) > 0
        for md_path in md_files:
            spec = SkillSpec.from_file(md_path)
            assert spec.name, f"{md_path} has no name"
            registry.register(spec, overwrite=True)
        assert len(registry.list()) >= len(md_files)


# ── Lifecycle: unregister / re-register ──────────────────────────


class TestSkillLifecycle:
    def test_unregister_removes_skill(self, registry: SkillRegistry):
        from houyi.skills.weather import get_weather

        registry.register(get_weather, overwrite=True)
        assert "get_weather" in [s.name for s in registry.list()]
        registry.unregister("get_weather")
        assert "get_weather" not in [s.name for s in registry.list()]

    def test_reregister_after_unregister(self, registry: SkillRegistry):
        from houyi.skills.weather import get_weather

        registry.register(get_weather, overwrite=True)
        registry.unregister("get_weather")
        registry.register(get_weather, overwrite=True)
        assert "get_weather" in [s.name for s in registry.list()]

    def test_get_unknown_returns_none(self, registry: SkillRegistry):
        assert registry.get("nonexistent_xyz") is None


# ── Batch: all 13 skills ─────────────────────────────────────────


class TestFullSkillInventory:
    def test_13_skills_register_and_validate(self, registry: SkillRegistry):
        from houyi.rag.skills.kb_analyze import kb_analyze_skill
        from houyi.rag.skills.kb_graph import kb_graph_skill
        from houyi.rag.skills.kb_ingest import kb_ingest_skill
        from houyi.rag.skills.kb_search import kb_search_skill
        from houyi.skills.location import get_location
        from houyi.skills.planning import PlanningSkill
        from houyi.skills.weather import get_date, get_weather
        from houyi.web_search.skill import build_web_search_skill

        # @tool built-in (4)
        for s in [get_weather, get_date, get_location, build_web_search_skill()]:
            registry.register(s, overwrite=True)

        # Planning (1)
        registry.register(PlanningSkill().to_spec(), overwrite=True)

        # RAG (4)
        for s in [kb_search_skill, kb_ingest_skill, kb_graph_skill, kb_analyze_skill]:
            registry.register(s, overwrite=True)

        # SKILL.md — flat *.md files (calculator, file_reader, etc.)
        if os.path.isdir(SKILLS_DIR):
            for md_path in glob.glob(os.path.join(SKILLS_DIR, "*.md")):
                registry.register(SkillSpec.from_file(md_path), overwrite=True)

        # SKILL.md — subdirectory external skills (planning-with-files, superpowers, etc.)
        if os.path.isdir(SKILLS_DIR):
            loaded = registry.register_from_directory(
                SKILLS_DIR,
                pattern="SKILL.md",
                recursive=True,
                overwrite=False,
            )

        all_skills = registry.list()
        names = sorted(s.name for s in all_skills)
        assert len(all_skills) >= 13, f"Expected >= 13, got {len(all_skills)}: {names}"

        for skill in all_skills:
            assert skill.name, "Skill without name"
            assert skill.description, f"Skill {skill.name} has no description"

    def test_external_skills_from_directory(self, registry: SkillRegistry):
        """Verify that external SKILL.md files from skills/ subdirs load correctly."""
        if not os.path.isdir(SKILLS_DIR):
            pytest.skip("skills/ directory not found")

        loaded = registry.register_from_directory(
            SKILLS_DIR,
            pattern="SKILL.md",
            recursive=True,
            overwrite=True,
        )
        assert len(loaded) >= 6, f"Expected >= 6 external skills, got {len(loaded)}: {loaded}"

        names = set(loaded)
        # Acceptance skills #9-#13 (real names from community repos)
        # planning-with-files, using-superpowers, skill-creator, frontend-design, notebooklm
        expected_external = {
            "planning-with-files",
            "using-superpowers",
            "skill-creator",
            "frontend-design",
            "notebooklm",
            "kb-retriever",
        }
        missing = expected_external - names
        assert not missing, f"Missing external skills: {missing}. Loaded: {names}"

        for name in loaded:
            skill = registry.get(name)
            assert skill is not None, f"Skill {name} not found in registry after loading"
            assert skill.description, f"Skill {name} has no description"
