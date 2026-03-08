"""Skill registry with automatic hooks and policy integration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from houyi.domain.skill.spec import SkillSpec

if TYPE_CHECKING:
    from houyi.domain.skill.hooks import SkillHooksManager
    from houyi.domain.skill.manifest import SkillManifest
    from houyi.domain.skill.policy import PolicyEnforcer

logger = logging.getLogger(__name__)


class CoreToolProtectionError(ValueError):
    """Raised when an operation attempts to overwrite or hijack a core built-in tool."""


class SkillRegistry:
    """Central registry for skills with hooks, policy, and namespace support."""

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
        self._hooks_manager = manager

    def set_policy_enforcer(self, enforcer: PolicyEnforcer) -> None:
        self._policy_enforcer = enforcer

    def register(self, skill: SkillSpec, *, overwrite: bool = False) -> str:
        name = str(getattr(skill, "name", "") or "").strip()
        if not name:
            raise ValueError("Skill name is required")

        existing = self._skills.get(name)
        if existing is not None and getattr(existing, "is_core", False):
            if overwrite:
                raise CoreToolProtectionError(
                    f"Cannot overwrite core tool '{name}': core tools are protected "
                    "from external override. Remove is_core=True from the existing "
                    "registration or use a different tool name."
                )
            if getattr(skill, "is_core", False):
                raise ValueError(
                    f"Duplicate core tool registration: '{name}' is already registered "
                    "as a core tool. Each core tool name must be unique."
                )
            prefixed_name = f"ext__{name}"
            skill = skill.model_copy(update={"name": prefixed_name})
            logger.warning(
                "External tool '%s' conflicts with core tool; renamed to '%s'",
                name,
                prefixed_name,
            )
            self._skills[prefixed_name] = skill
            if skill.provider:
                self._qualified[skill.qualified_name] = skill
            self._register_hooks(skill, prefixed_name)
            logger.debug("Registered external skill as: %s", prefixed_name)
            return prefixed_name

        if not overwrite and existing is not None:
            existing_provider = getattr(existing, "provider", None) or ""
            new_provider = getattr(skill, "provider", None) or ""
            if existing_provider == new_provider:
                raise ValueError(f"Skill already registered: {name}")
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
            return qname

        if overwrite and name in self._skills and self._hooks_manager:
            self._hooks_manager.unregister_hooks(name)

        self._skills[name] = skill
        if skill.provider:
            self._qualified[skill.qualified_name] = skill
        self._register_hooks(skill, name)
        logger.debug("Registered skill: %s", skill.qualified_name)
        return name

    def _register_hooks(self, skill: SkillSpec, key: str) -> None:
        if self._hooks_manager and hasattr(skill, "hooks") and skill.hooks:
            self._hooks_manager.register_hooks(skill)
            logger.debug("Registered %d hooks for skill: %s", len(skill.hooks), key)

    def unregister(self, name: str) -> bool:
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
        return self._qualified.get(name) or self._skills.get(name)

    def list(self) -> list[SkillSpec]:
        seen: set[int] = set()
        result: list[SkillSpec] = []
        for skill in list(self._skills.values()) + list(self._qualified.values()):
            sid = id(skill)
            if sid not in seen:
                seen.add(sid)
                result.append(skill)
        return result

    def list_names(self) -> list[str]:
        return list(self._skills.keys())

    def list_qualified_names(self) -> list[str]:
        return [s.qualified_name for s in self.list()]

    def as_tool_schemas(self) -> list[dict[str, Any]]:
        skills = sorted(self.list(), key=lambda s: (0 if getattr(s, "is_core", False) else 1))
        return [skill.to_tool_schema() for skill in skills]

    def clear(self) -> None:
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
        from houyi.domain.skill.manifest import SkillManifest

        manifest_path = Path(manifest_path)
        manifest = SkillManifest.from_file(manifest_path)
        registered: list[str] = []
        base_dir = manifest_path.parent

        if manifest.contributions and manifest.contributions.skills:
            for skill_contrib in manifest.contributions.skills:
                try:
                    if skill_contrib.path:
                        skill_path = base_dir / skill_contrib.path
                        skill = SkillSpec.from_file(str(skill_path))
                    else:
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
                    registered_name = self.register(skill, overwrite=overwrite)
                    registered.append(registered_name)
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
        skill = SkillSpec.from_file(str(skill_path))
        registered_name = self.register(skill, overwrite=overwrite)
        return registered_name

    def register_from_directory(
        self,
        directory: str | Path,
        *,
        pattern: str = "SKILL.md",
        recursive: bool = True,
        overwrite: bool = False,
    ) -> list[str]:
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
                    dup_name = str(e).replace("Skill already registered: ", "")
                    logger.warning(
                        "Skill '%s' from %s skipped: already registered "
                        "for the same provider/name namespace. "
                        "Use overwrite=True to replace or rename/remove the "
                        "duplicate SKILL.md.",
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
        from houyi.domain.skill.manifest import SkillManifest

        return SkillManifest.from_file(manifest_path)


def create_default_registry() -> SkillRegistry:
    from houyi.domain.skill.hooks import DEFAULT_HOOKS_MANAGER

    return SkillRegistry(hooks_manager=DEFAULT_HOOKS_MANAGER)


DEFAULT_SKILL_REGISTRY = create_default_registry()

__all__ = [
    "DEFAULT_SKILL_REGISTRY",
    "CoreToolProtectionError",
    "SkillRegistry",
    "create_default_registry",
]
