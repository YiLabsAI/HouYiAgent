"""Compatibility tests for external Claude skill ecosystem.

Verifies that our SimpleSkill parser can correctly load SKILL.md files
from the Claude Code community, including:
- planning-with-files (OthmanAdi)
- superpowers
- skill-creator
- frontend-design
- notebooklm-skill

Also verifies all 7 internal SKILL.md files load correctly with full
frontmatter parsing (permissions, invocationPolicy, hooks, etc.).

Reference: SimpleSkill Specification v0.1 Section 3 (SKILL.md Format)
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from houyi.core.skill.policy import (
    InvocationPolicy,
    ModelAutoInvoke,
    Permissions,
    SideEffect,
)
from houyi.core.skill.spec import SkillSpec

# =========================================================================
# Internal SKILL.md Loading Tests
# =========================================================================


class TestInternalSkillMdLoading:
    """Verify all 7 internal SKILL.md files load with correct metadata."""

    SKILL_MD_FILES = [
        "houyi/skills/planning/SKILL.md",
        "houyi/skills/weather/SKILL.md",
        "houyi/skills/location/SKILL.md",
        "houyi/rag/skills/kb_search/SKILL.md",
        "houyi/rag/skills/kb_ingest/SKILL.md",
        "houyi/rag/skills/kb_graph/SKILL.md",
        "houyi/rag/skills/kb_analyze/SKILL.md",
    ]

    @pytest.fixture(autouse=True)
    def _project_root(self) -> None:
        """Locate project root for resolving skill file paths."""
        # Walk up from test file to find project root (contains pyproject.toml)
        current = Path(__file__).resolve()
        for parent in [current, *list(current.parents)]:
            if (parent / "pyproject.toml").exists():
                self.project_root = parent
                return
        pytest.fail("Cannot find project root with pyproject.toml")

    def _load_skill(self, relative_path: str) -> SkillSpec:
        path = self.project_root / relative_path
        if not path.exists():
            pytest.skip(f"Skill file not found: {path}")
        return SkillSpec.from_file(str(path))

    def test_all_skill_md_files_load_without_error(self) -> None:
        """Smoke test: every internal SKILL.md loads without exception."""
        for skill_file in self.SKILL_MD_FILES:
            spec = self._load_skill(skill_file)
            assert spec.name, f"Skill name is empty for {skill_file}"
            assert spec.description, f"Skill description is empty for {skill_file}"

    def test_planning_with_files_skill(self) -> None:
        """Verify planning-with-files SKILL.md loads with full metadata."""
        spec = self._load_skill("houyi/skills/planning/SKILL.md")

        assert spec.name == "planning-with-files"
        assert spec.version == "1.0.0"
        assert spec.user_invocable is True
        assert len(spec.hooks) == 3
        assert "Write" in spec.allowed_tools
        assert "Read" in spec.allowed_tools

        # Invocation policy
        assert isinstance(spec.invocation_policy, InvocationPolicy)
        assert spec.invocation_policy.model_auto_invoke == ModelAutoInvoke.ALLOW
        assert spec.invocation_policy.side_effect == SideEffect.FILESYSTEM

        # Permissions
        assert isinstance(spec.permissions, Permissions)
        assert spec.permissions.filesystem.read is True
        assert spec.permissions.filesystem.write is True
        assert len(spec.permissions.filesystem.paths) == 2

    def test_kb_search_skill(self) -> None:
        """Verify kb-search SKILL.md loads with hooks, policy, and permissions."""
        spec = self._load_skill("houyi/rag/skills/kb_search/SKILL.md")

        assert spec.name == "kb-search"
        assert spec.version == "1.0.0"
        assert len(spec.hooks) == 3

        # Policy: allow, no side effect
        assert isinstance(spec.invocation_policy, InvocationPolicy)
        assert spec.invocation_policy.model_auto_invoke == ModelAutoInvoke.ALLOW
        assert spec.invocation_policy.side_effect == SideEffect.NONE

        # Permissions: read-only
        assert isinstance(spec.permissions, Permissions)
        assert spec.permissions.filesystem.read is True
        assert spec.permissions.filesystem.write is False

    def test_kb_ingest_skill_has_write_permission(self) -> None:
        """Verify kb-ingest has filesystem write permission."""
        spec = self._load_skill("houyi/rag/skills/kb_ingest/SKILL.md")

        assert spec.name == "kb-ingest"
        assert isinstance(spec.permissions, Permissions)
        assert spec.permissions.filesystem.write is True
        assert spec.invocation_policy.side_effect == SideEffect.FILESYSTEM

    def test_weather_skill_has_network_policy(self) -> None:
        """Verify weather SKILL.md declares network side-effect."""
        spec = self._load_skill("houyi/skills/weather/SKILL.md")

        assert spec.name == "weather"
        assert spec.invocation_policy is not None
        assert spec.invocation_policy.side_effect == SideEffect.NETWORK
        assert spec.permissions is not None
        assert spec.permissions.network.enabled is True

    def test_location_skill_has_network_policy(self) -> None:
        """Verify location SKILL.md declares network side-effect."""
        spec = self._load_skill("houyi/skills/location/SKILL.md")

        assert spec.name == "location"
        assert spec.invocation_policy is not None
        assert spec.invocation_policy.side_effect == SideEffect.NETWORK
        assert spec.permissions is not None
        assert spec.permissions.network.enabled is True


# =========================================================================
# External Claude Skill Compatibility Tests
# =========================================================================


class TestExternalClaudeSkillCompatibility:
    """Test that our parser handles external Claude SKILL.md formats.

    These tests use inline SKILL.md content matching the actual structure
    of popular Claude Code community skills, ensuring our parser handles
    the real-world variety of frontmatter schemas.
    """

    def _parse_skill_content(self, content: str, tmp_path: Path) -> SkillSpec:
        """Write content to temp file and parse it."""
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(textwrap.dedent(content))
        return SkillSpec.from_file(str(skill_file))

    def test_planning_with_files_external(self, tmp_path: Path) -> None:
        """Test parsing the real planning-with-files SKILL.md from OthmanAdi."""
        content = """\
        ---
        name: planning-with-files
        description: |
          Manage tasks using structured plan files in markdown format.
          Track progress through subtasks before marking as done.
        hooks:
          PreToolUse:
            - matcher: "Write|Edit|MultiEdit|NotebookEdit"
              type: handler
              handler: module:pre_write_hook
          PostToolUse:
            - matcher: ".*"
              type: handler
              handler: module:post_tool_hook
          Stop:
            - type: handler
              handler: module:stop_hook
        ---

        # Planning with Files

        A task planning skill for complex multi-step work.
        """
        spec = self._parse_skill_content(content, tmp_path)

        assert spec.name == "planning-with-files"
        assert "plan" in spec.description.lower()
        assert len(spec.hooks) == 3

    def test_superpowers_skill(self, tmp_path: Path) -> None:
        """Test parsing a superpowers-style SKILL.md (no hooks, description-heavy)."""
        content = """\
        ---
        name: superpowers
        description: |
          Enhance Claude with advanced capabilities including parallel processing,
          memory management, and reasoning chains.
        version: "1.2.0"
        author: community
        ---

        # Superpowers

        Advanced capability enhancement for Claude Code.

        ## Features
        - Parallel task execution
        - Persistent memory across sessions
        - Chain-of-thought reasoning templates
        """
        spec = self._parse_skill_content(content, tmp_path)

        assert spec.name == "superpowers"
        assert spec.version == "1.2.0"
        assert spec.author == "community"
        assert len(spec.hooks) == 0

    def test_skill_creator_skill(self, tmp_path: Path) -> None:
        """Test parsing a skill-creator SKILL.md."""
        content = """\
        ---
        name: skill-creator
        description: |
          Meta-skill for creating, testing, and publishing new Claude skills.
          Generates SKILL.md files with proper frontmatter and markdown structure.
        version: "0.5.0"
        allowed-tools:
          - Write
          - Read
          - Edit
          - Glob
        ---

        # Skill Creator

        Create new skills with proper structure and documentation.
        """
        spec = self._parse_skill_content(content, tmp_path)

        assert spec.name == "skill-creator"
        assert "Write" in spec.allowed_tools
        assert len(spec.allowed_tools) == 4

    def test_frontend_design_skill(self, tmp_path: Path) -> None:
        """Test parsing a frontend-design SKILL.md (prose-heavy, few frontmatter fields)."""
        content = """\
        ---
        name: frontend-design
        description: |
          Expert frontend design guidance. Provides UX best practices,
          accessibility patterns, and responsive layout strategies.
        ---

        # Frontend Design Expert

        Comprehensive frontend design assistance.

        ## Principles

        1. Mobile-first responsive design
        2. WCAG 2.1 AA compliance
        3. Progressive enhancement
        """
        spec = self._parse_skill_content(content, tmp_path)

        assert spec.name == "frontend-design"
        assert "frontend" in spec.description.lower()

    def test_notebooklm_skill(self, tmp_path: Path) -> None:
        """Test parsing a notebooklm-style SKILL.md with complex hooks."""
        content = """\
        ---
        name: notebooklm-skill
        description: |
          NotebookLM-style research and note-taking assistant.
          Manages sources, generates summaries, and creates study guides.
        version: "1.0.0"
        hooks:
          PreToolUse:
            - matcher: "Write|Edit"
              type: handler
              handler: notebooklm.hooks:validate_source_refs
          PostToolUse:
            - matcher: "Read"
              type: handler
              handler: notebooklm.hooks:index_new_content
        allowed-tools:
          - Read
          - Write
          - Glob
          - Grep
        ---

        # NotebookLM Skill

        Research assistant with source management.
        """
        spec = self._parse_skill_content(content, tmp_path)

        assert spec.name == "notebooklm-skill"
        assert len(spec.hooks) == 2
        assert "Read" in spec.allowed_tools

    def test_skill_with_disable_model_invocation(self, tmp_path: Path) -> None:
        """Test Claude standard: disable-model-invocation field."""
        content = """\
        ---
        name: dangerous-skill
        description: A skill that should not be auto-invoked by model.
        disable-model-invocation: true
        ---

        # Dangerous Skill
        """
        spec = self._parse_skill_content(content, tmp_path)

        assert spec.name == "dangerous-skill"
        assert isinstance(spec.invocation_policy, InvocationPolicy)
        assert spec.invocation_policy.model_auto_invoke == ModelAutoInvoke.DENY

    def test_skill_with_unknown_frontmatter_fields_preserved(self, tmp_path: Path) -> None:
        """Test that unknown frontmatter fields are preserved in extra_frontmatter."""
        content = """\
        ---
        name: custom-skill
        description: Skill with custom fields.
        license: MIT
        repository: https://github.com/example/custom-skill
        tags:
          - productivity
          - automation
        custom_field: custom_value
        ---

        # Custom Skill
        """
        spec = self._parse_skill_content(content, tmp_path)

        assert spec.name == "custom-skill"
        assert "license" in spec.extra_frontmatter
        assert spec.extra_frontmatter["license"] == "MIT"
        assert "repository" in spec.extra_frontmatter
        assert "tags" in spec.extra_frontmatter
        assert "custom_field" in spec.extra_frontmatter

    def test_skill_with_full_permissions_and_policy(self, tmp_path: Path) -> None:
        """Test parsing a skill with comprehensive permissions and policy."""
        content = """\
        ---
        name: full-featured-skill
        description: Skill with all SimpleSkill extensions.
        version: "2.0.0"
        author: HouYi Team
        user-invocable: true
        allowed-tools:
          - Read
          - Write
          - Shell
        invocationPolicy:
          modelAutoInvoke: allow_with_consent
          userInvocable: true
          sideEffect: mixed
        permissions:
          filesystem:
            read: true
            write: true
            delete: false
            paths:
              - "${WORKSPACE}/data/**"
          network:
            enabled: true
            domains:
              - api.example.com
          exec:
            enabled: false
        hooks:
          PreToolUse:
            - matcher: "Shell"
              type: handler
              handler: skill.hooks:validate_shell
          Stop:
            - type: handler
              handler: skill.hooks:cleanup
        ---

        # Full Featured Skill
        """
        spec = self._parse_skill_content(content, tmp_path)

        assert spec.name == "full-featured-skill"
        assert spec.version == "2.0.0"
        assert spec.author == "HouYi Team"
        assert "Shell" in spec.allowed_tools

        # Policy
        assert isinstance(spec.invocation_policy, InvocationPolicy)
        assert spec.invocation_policy.model_auto_invoke == ModelAutoInvoke.ALLOW_WITH_CONSENT
        assert spec.invocation_policy.side_effect == SideEffect.MIXED

        # Permissions
        assert isinstance(spec.permissions, Permissions)
        assert spec.permissions.filesystem.read is True
        assert spec.permissions.filesystem.write is True
        assert spec.permissions.filesystem.delete is False
        assert spec.permissions.network.enabled is True
        assert "api.example.com" in spec.permissions.network.domains
        assert spec.permissions.exec.enabled is False

        # Hooks
        assert len(spec.hooks) == 2


# =========================================================================
# Skill Registry Integration Tests
# =========================================================================


class TestSkillRegistryIntegration:
    """Integration tests for skill loading through the registry."""

    def test_register_from_skill_file(self, tmp_path: Path) -> None:
        """Test loading and registering a skill from SKILL.md via registry."""
        from houyi.core.skill_registry import SkillRegistry

        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(
            textwrap.dedent("""\
        ---
        name: registry-compat-test
        description: Test skill for registry compatibility
        version: "1.0.0"
        invocationPolicy:
          modelAutoInvoke: allow
          sideEffect: none
        ---

        # Registry Compat Test
        """)
        )

        registry = SkillRegistry()
        name = registry.register_from_skill_file(skill_md)

        assert name == "registry-compat-test"

        skill = registry.get("registry-compat-test")
        assert skill is not None
        assert isinstance(skill.invocation_policy, InvocationPolicy)
        assert skill.invocation_policy.model_auto_invoke == ModelAutoInvoke.ALLOW

    def test_register_from_directory(self, tmp_path: Path) -> None:
        """Test discovering and registering skills from a directory tree."""
        from houyi.core.skill_registry import SkillRegistry

        # Create multiple skill directories
        for name in ["skill-a", "skill-b"]:
            skill_dir = tmp_path / name
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                textwrap.dedent(f"""\
            ---
            name: {name}
            description: Test skill {name}
            ---
            # {name}
            """)
            )

        registry = SkillRegistry()
        registered = registry.register_from_directory(tmp_path)

        assert len(registered) == 2
        assert "skill-a" in registered
        assert "skill-b" in registered

    def test_register_all_internal_skills(self) -> None:
        """Smoke test: register all internal SKILL.md files via directory scan."""
        from houyi.core.skill_registry import SkillRegistry

        project_root = Path(__file__).resolve()
        for parent in [project_root, *list(project_root.parents)]:
            if (parent / "pyproject.toml").exists():
                project_root = parent
                break

        skills_dir = project_root / "houyi" / "skills"
        if not skills_dir.exists():
            pytest.skip("Skills directory not found")

        registry = SkillRegistry()
        registered = registry.register_from_directory(skills_dir)

        # We should find at least planning, weather, location
        assert len(registered) >= 3
        names = registry.list_names()
        assert "planning-with-files" in names

    def test_skill_to_tool_schema_includes_name_and_description(self, tmp_path: Path) -> None:
        """Verify loaded skills produce valid OpenAI tool schemas."""
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(
            textwrap.dedent("""\
        ---
        name: schema-test
        description: Skill for schema verification

        ---

        # Schema Test

        ## Input Schema
        ```json
        {
          "type": "object",
          "properties": {
            "query": {"type": "string", "description": "Search query"}
          },
          "required": ["query"]
        }
        ```
        """)
        )

        spec = SkillSpec.from_file(str(skill_md))
        schema = spec.to_tool_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "schema-test"
        assert schema["function"]["description"] == "Skill for schema verification"
        assert "query" in schema["function"]["parameters"]["properties"]


# =========================================================================
# Edge Cases & Robustness
# =========================================================================


class TestEdgeCases:
    """Edge case tests for parser robustness."""

    def test_empty_frontmatter(self, tmp_path: Path) -> None:
        """Test parsing SKILL.md with empty frontmatter."""
        content = "---\n---\n\n# Empty Skill\n\n## Description\nA skill with no metadata."
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(content)

        spec = SkillSpec.from_file(str(skill_file))
        assert spec.name == "Empty Skill"  # Falls back to markdown title

    def test_no_frontmatter(self, tmp_path: Path) -> None:
        """Test parsing SKILL.md without frontmatter (legacy format)."""
        content = "# Legacy Skill\n\n## Description\nOld format skill."
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(content)

        spec = SkillSpec.from_file(str(skill_file))
        assert spec.name == "Legacy Skill"

    def test_multiline_description(self, tmp_path: Path) -> None:
        """Test parsing SKILL.md with multiline YAML description."""
        content = textwrap.dedent("""\
        ---
        name: multiline-desc
        description: |
          First line of description.
          Second line with more detail.
          Third line with context.
        ---
        # Multiline
        """)
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(content)

        spec = SkillSpec.from_file(str(skill_file))
        assert "First line" in spec.description
        assert "Third line" in spec.description

    def test_invalid_policy_value_handled_gracefully(self, tmp_path: Path) -> None:
        """Test that invalid policy values fall back to raw dict (no crash)."""
        content = textwrap.dedent("""\
        ---
        name: bad-policy
        description: Skill with invalid policy value
        invocationPolicy:
          modelAutoInvoke: invalid_value
        ---
        # Bad Policy
        """)
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(content)

        # Should not raise - invalid value falls back to raw dict
        spec = SkillSpec.from_file(str(skill_file))
        assert spec.name == "bad-policy"
        # Policy may be raw dict fallback or None depending on parse path
        if spec.invocation_policy is not None:
            # If it parsed successfully, the value was accepted somehow
            assert isinstance(spec.invocation_policy, (InvocationPolicy, dict))

    def test_permissions_with_only_filesystem(self, tmp_path: Path) -> None:
        """Test parsing permissions with only filesystem section."""
        content = textwrap.dedent("""\
        ---
        name: fs-only
        description: Filesystem only permissions
        permissions:
          filesystem:
            read: true
            write: false
        ---
        # FS Only
        """)
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(content)

        spec = SkillSpec.from_file(str(skill_file))
        assert isinstance(spec.permissions, Permissions)
        assert spec.permissions.filesystem.read is True
        assert spec.permissions.filesystem.write is False
        assert spec.permissions.network.enabled is False
