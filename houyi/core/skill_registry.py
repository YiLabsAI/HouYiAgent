"""Skill registry with automatic hooks and policy integration.

This module provides the central registry for skills, automatically handling:
- Hooks registration with SkillHooksManager
- Policy enforcement setup
- Loading skills from manifest files (simpleskill.json)
- Loading skills from SKILL.md files
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from houyi.core.skill.spec import SkillSpec

if TYPE_CHECKING:
    from houyi.core.skill.hooks import SkillHooksManager
    from houyi.core.skill.manifest import SkillManifest
    from houyi.core.skill.policy import PolicyEnforcer

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Central registry for skills with hooks, policy, and namespace support.

    Skills are indexed by plain name **and** by qualified name
    (``provider/name``) when a provider is set.  Lookups via :meth:`get` check
    the qualified form first, then fall back to plain name.
    """

    def __init__(
        self,
        hooks_manager: SkillHooksManager | None = None,
        policy_enforcer: PolicyEnforcer | None = None,
    ) -> None:
        self._skills: dict[str, SkillSpec] = {}
        self._qualified: dict[str, SkillSpec] = {}
        self._hooks_manager = hooks_manager
        self._policy_enforcer = policy_enforcer

    def set_hooks_manager(self, manager: SkillHooksManager) -> None:
        """Set the hooks manager for automatic hooks registration."""
        self._hooks_manager = manager

    def set_policy_enforcer(self, enforcer: PolicyEnforcer) -> None:
        """Set the policy enforcer for automatic policy setup."""
        self._policy_enforcer = enforcer

    def register(self, skill: SkillSpec, *, overwrite: bool = False) -> None:
        """Register a skill with automatic hooks and policy integration.

        When a skill has a ``provider``, the qualified name
        (``provider/name``) is also indexed.  Two skills with the same
        ``name`` but different ``provider`` values can coexist; plain-name
        lookups will return whichever was registered first (or the one that
        overwrote it).

        Raises:
            ValueError: If skill name is empty or already registered
                (when not overwriting and same provider).
        """
        name = str(getattr(skill, "name", "") or "").strip()
        if not name:
            raise ValueError("Skill name is required")

        existing = self._skills.get(name)
        if not overwrite and existing is not None:
            existing_provider = getattr(existing, "provider", None) or ""
            new_provider = getattr(skill, "provider", None) or ""
            if existing_provider == new_provider:
                raise ValueError(f"Skill already registered: {name}")
            # Different provider — store under qualified name only;
            # the plain-name slot keeps the earlier registration.
            qname = skill.qualified_name
            self._qualified[qname] = skill
            self._register_hooks(skill, qname)
            logger.debug(
                "Registered skill %s (provider=%s) as namespaced; "
                "plain-name slot kept for provider=%s",
                qname,
                new_provider,
                existing_provider,
            )
            return

        if overwrite and name in self._skills and self._hooks_manager:
            self._hooks_manager.unregister_hooks(name)

        self._skills[name] = skill
        if skill.provider:
            self._qualified[skill.qualified_name] = skill
        self._register_hooks(skill, name)
        logger.debug("Registered skill: %s", skill.qualified_name)

    def _register_hooks(self, skill: SkillSpec, key: str) -> None:
        if self._hooks_manager and hasattr(skill, "hooks") and skill.hooks:
            self._hooks_manager.register_hooks(skill)
            logger.debug("Registered %d hooks for skill: %s", len(skill.hooks), key)

    def unregister(self, name: str) -> bool:
        """Unregister a skill by plain or qualified name."""
        skill = self._qualified.get(name) or self._skills.get(name)
        if skill is None:
            return False
        self._qualified.pop(name, None)
        if skill.provider:
            self._qualified.pop(skill.qualified_name, None)
        if self._skills.get(skill.name) is skill:
            del self._skills[skill.name]
        if self._hooks_manager:
            self._hooks_manager.unregister_hooks(skill.name)
        logger.info("Unregistered skill: %s", skill.qualified_name)
        return True

    def get(self, name: str) -> SkillSpec | None:
        """Get a skill by plain or qualified (``provider/name``) key."""
        return self._qualified.get(name) or self._skills.get(name)

    def list(self) -> list[SkillSpec]:
        """List all registered skills (deduplicated)."""
        seen: set[int] = set()
        result: list[SkillSpec] = []
        for skill in list(self._skills.values()) + list(self._qualified.values()):
            sid = id(skill)
            if sid not in seen:
                seen.add(sid)
                result.append(skill)
        return result

    def list_names(self) -> list[str]:
        """List all registered skill names (plain names)."""
        return list(self._skills.keys())

    def list_qualified_names(self) -> list[str]:
        """List all qualified skill names (``provider/name``)."""
        return [s.qualified_name for s in self.list()]

    def as_tool_schemas(self) -> list[dict[str, Any]]:
        """Convert all skills to OpenAI function calling schemas."""
        return [skill.to_tool_schema() for skill in self.list()]

    def clear(self) -> None:
        """Clear all registered skills and their hooks."""
        if self._hooks_manager:
            self._hooks_manager.clear()
        self._skills.clear()
        self._qualified.clear()
        logger.info("Cleared all skills")

    def register_from_manifest(
        self,
        manifest_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> list[str]:
        """Load and register all skills from a simpleskill.json manifest.

        This method implements Layer A (Manifest) integration by:
        1. Parsing the manifest file
        2. Loading each skill definition from the contributions
        3. Registering skills with their hooks and policies

        Args:
            manifest_path: Path to simpleskill.json manifest file
            overwrite: Whether to overwrite existing skills

        Returns:
            List of registered skill names

        Raises:
            FileNotFoundError: If manifest file doesn't exist
            ValueError: If manifest parsing fails
        """
        from houyi.core.skill.manifest import SkillManifest

        manifest_path = Path(manifest_path)
        manifest = SkillManifest.from_file(manifest_path)
        registered: list[str] = []

        # Get base directory for resolving relative paths
        base_dir = manifest_path.parent

        # Register skills from contributions
        if manifest.contributions and manifest.contributions.skills:
            for skill_contrib in manifest.contributions.skills:
                try:
                    if skill_contrib.path:
                        # Path-based: load SKILL.md from relative path
                        skill_path = base_dir / skill_contrib.path
                        skill = SkillSpec.from_file(str(skill_path))
                    else:
                        # Inline: create SkillSpec from contribution metadata
                        from pydantic import BaseModel as _BaseModel

                        _empty = type(
                            f"Empty_{skill_contrib.id.replace('-', '_')}",
                            (_BaseModel,),
                            {},
                        )
                        skill = SkillSpec(
                            name=skill_contrib.id,
                            description=skill_contrib.description,
                            input_schema=_empty,
                            output_schema=_empty,
                            invocation_policy=skill_contrib.invocation_policy,
                        )
                    self.register(skill, overwrite=overwrite)
                    registered.append(skill.name)
                except Exception as e:
                    logger.warning(
                        "Failed to load skill '%s' from manifest %s: %s",
                        skill_contrib.id,
                        manifest_path,
                        e,
                    )

        logger.info(
            "Loaded %d skills from manifest: %s",
            len(registered),
            manifest_path,
        )
        return registered

    def register_from_skill_file(
        self,
        skill_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> str:
        """Load and register a skill from a SKILL.md file.

        Args:
            skill_path: Path to SKILL.md file
            overwrite: Whether to overwrite existing skill

        Returns:
            Name of registered skill

        Raises:
            FileNotFoundError: If skill file doesn't exist
            ValueError: If skill parsing fails
        """
        skill = SkillSpec.from_file(str(skill_path))
        self.register(skill, overwrite=overwrite)
        return skill.name

    def register_from_directory(
        self,
        directory: str | Path,
        *,
        pattern: str = "SKILL.md",
        recursive: bool = True,
        overwrite: bool = False,
    ) -> list[str]:
        """Discover and register skills from a directory.

        Searches for SKILL.md files and registers each skill found.

        Args:
            directory: Directory to search
            pattern: Filename pattern to match (default: SKILL.md)
            recursive: Whether to search subdirectories
            overwrite: Whether to overwrite existing skills

        Returns:
            List of registered skill names
        """
        directory = Path(directory)
        registered: list[str] = []

        glob_pattern = f"**/{pattern}" if recursive else pattern
        for skill_path in directory.glob(glob_pattern):
            try:
                skill_name = self.register_from_skill_file(
                    skill_path,
                    overwrite=overwrite,
                )
                registered.append(skill_name)
            except ValueError as e:
                if "already registered" in str(e):
                    # Parse the duplicate skill name from the error message
                    dup_name = str(e).replace("Skill already registered: ", "")
                    logger.warning(
                        "Skill '%s' from %s skipped: already registered "
                        "(built-in takes priority over external SKILL.md). "
                        "To override, use overwrite=True or remove the "
                        "duplicate from the skills/ directory.",
                        dup_name,
                        skill_path,
                    )
                else:
                    logger.warning(
                        "Failed to load skill from %s: %s",
                        skill_path,
                        e,
                    )
            except Exception as e:
                logger.warning(
                    "Failed to load skill from %s: %s",
                    skill_path,
                    e,
                )

        logger.debug(
            "Discovered and registered %d skills from: %s",
            len(registered),
            directory,
        )
        return registered

    def get_manifest(self, manifest_path: str | Path) -> SkillManifest:
        """Load a manifest without registering its skills.

        Useful for inspecting manifest contents before registration.

        Args:
            manifest_path: Path to simpleskill.json

        Returns:
            Parsed SkillManifest
        """
        from houyi.core.skill.manifest import SkillManifest

        return SkillManifest.from_file(manifest_path)


def create_default_registry() -> SkillRegistry:
    """Create a default registry with global hooks manager."""
    from houyi.core.skill.hooks import DEFAULT_HOOKS_MANAGER

    return SkillRegistry(hooks_manager=DEFAULT_HOOKS_MANAGER)


# Default global registry instance
DEFAULT_SKILL_REGISTRY = create_default_registry()

__all__ = [
    "DEFAULT_SKILL_REGISTRY",
    "SkillRegistry",
    "create_default_registry",
]
