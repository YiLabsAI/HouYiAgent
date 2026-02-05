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
    """Central registry for skills with hooks and policy integration."""

    def __init__(
        self,
        hooks_manager: SkillHooksManager | None = None,
        policy_enforcer: PolicyEnforcer | None = None,
    ) -> None:
        """Initialize skill registry.

        Args:
            hooks_manager: Optional hooks manager for automatic hooks registration
            policy_enforcer: Optional policy enforcer for automatic policy setup
        """
        self._skills: dict[str, SkillSpec] = {}
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

        Args:
            skill: The skill specification to register
            overwrite: Whether to overwrite existing skill with same name

        Raises:
            ValueError: If skill name is empty or already registered (when not overwriting)
        """
        name = str(getattr(skill, "name", "") or "").strip()
        if not name:
            raise ValueError("Skill name is required")
        if not overwrite and name in self._skills:
            raise ValueError(f"Skill already registered: {name}")

        # Unregister old hooks if overwriting
        if overwrite and name in self._skills and self._hooks_manager:
            self._hooks_manager.unregister_hooks(name)

        # Register skill
        self._skills[name] = skill

        # Auto-register hooks if available
        if self._hooks_manager and hasattr(skill, "hooks") and skill.hooks:
            self._hooks_manager.register_hooks(skill)
            logger.debug("Registered %d hooks for skill: %s", len(skill.hooks), name)

        logger.info("Registered skill: %s", name)

    def unregister(self, name: str) -> bool:
        """Unregister a skill and its hooks.

        Args:
            name: Name of the skill to unregister

        Returns:
            True if skill was unregistered, False if not found
        """
        if name not in self._skills:
            return False

        # Unregister hooks
        if self._hooks_manager:
            self._hooks_manager.unregister_hooks(name)

        del self._skills[name]
        logger.info("Unregistered skill: %s", name)
        return True

    def get(self, name: str) -> SkillSpec | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def list(self) -> list[SkillSpec]:
        """List all registered skills."""
        return list(self._skills.values())

    def list_names(self) -> list[str]:
        """List all registered skill names."""
        return list(self._skills.keys())

    def as_tool_schemas(self) -> list[dict[str, Any]]:
        """Convert all skills to OpenAI function calling schemas."""
        return [skill.to_tool_schema() for skill in self._skills.values()]

    def clear(self) -> None:
        """Clear all registered skills and their hooks."""
        if self._hooks_manager:
            self._hooks_manager.clear()
        self._skills.clear()
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
                skill_path = base_dir / skill_contrib.path
                try:
                    skill = SkillSpec.from_file(str(skill_path))
                    self.register(skill, overwrite=overwrite)
                    registered.append(skill.name)
                except Exception as e:
                    logger.warning(
                        "Failed to load skill from %s: %s",
                        skill_path,
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
            except Exception as e:
                logger.warning(
                    "Failed to load skill from %s: %s",
                    skill_path,
                    e,
                )

        logger.info(
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
