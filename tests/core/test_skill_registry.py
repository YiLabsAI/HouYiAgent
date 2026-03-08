"""Tests for SkillRegistry with hooks and policy integration."""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from houyi.domain.skill.hooks import HookEvent, SkillHook, SkillHooksManager
from houyi.domain.skill.registry import (
    DEFAULT_SKILL_REGISTRY,
    SkillRegistry,
    create_default_registry,
)
from houyi.domain.skill.spec import SkillSpec


class InputModel(BaseModel):
    query: str


class OutputModel(BaseModel):
    result: str


def create_test_skill(name: str, hooks: list | None = None) -> SkillSpec:
    """Create a test skill with optional hooks."""
    return SkillSpec(
        name=name,
        description=f"Test skill: {name}",
        input_schema=InputModel,
        output_schema=OutputModel,
        hooks=hooks or [],
    )


class TestSkillRegistry:
    """Test SkillRegistry class."""

    def test_register_and_get(self) -> None:
        """Test basic registration and retrieval."""
        registry = SkillRegistry()
        skill = create_test_skill("test-skill")

        registry.register(skill)

        retrieved = registry.get("test-skill")
        assert retrieved is not None
        assert retrieved.name == "test-skill"

    def test_register_empty_name_raises(self) -> None:
        """Test that registering skill with empty name raises error."""
        registry = SkillRegistry()
        skill = SkillSpec(
            name="",
            description="Empty name skill",
            input_schema=InputModel,
            output_schema=OutputModel,
        )

        with pytest.raises(ValueError, match="Skill name is required"):
            registry.register(skill)

    def test_register_duplicate_raises(self) -> None:
        """Test that registering duplicate skill raises error."""
        registry = SkillRegistry()
        skill = create_test_skill("duplicate-skill")

        registry.register(skill)

        with pytest.raises(ValueError, match="Skill already registered"):
            registry.register(skill)

    def test_register_with_overwrite(self) -> None:
        """Test that overwrite=True allows re-registration."""
        registry = SkillRegistry()
        skill1 = create_test_skill("overwrite-skill")
        skill2 = SkillSpec(
            name="overwrite-skill",
            description="Updated description",
            input_schema=InputModel,
            output_schema=OutputModel,
        )

        registry.register(skill1)
        registry.register(skill2, overwrite=True)

        retrieved = registry.get("overwrite-skill")
        assert retrieved is not None
        assert retrieved.description == "Updated description"

    def test_get_nonexistent_returns_none(self) -> None:
        """Test that getting nonexistent skill returns None."""
        registry = SkillRegistry()

        result = registry.get("nonexistent")
        assert result is None

    def test_list_skills(self) -> None:
        """Test listing all registered skills."""
        registry = SkillRegistry()
        registry.register(create_test_skill("skill-1"))
        registry.register(create_test_skill("skill-2"))
        registry.register(create_test_skill("skill-3"))

        skills = registry.list()
        assert len(skills) == 3
        names = {s.name for s in skills}
        assert names == {"skill-1", "skill-2", "skill-3"}

    def test_list_names(self) -> None:
        """Test listing skill names."""
        registry = SkillRegistry()
        registry.register(create_test_skill("skill-a"))
        registry.register(create_test_skill("skill-b"))

        names = registry.list_names()
        assert set(names) == {"skill-a", "skill-b"}

    def test_as_tool_schemas(self) -> None:
        """Test converting skills to tool schemas."""
        registry = SkillRegistry()
        registry.register(create_test_skill("tool-skill"))

        schemas = registry.as_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "tool-skill"

    def test_unregister(self) -> None:
        """Test unregistering a skill."""
        registry = SkillRegistry()
        skill = create_test_skill("unregister-skill")
        registry.register(skill)

        result = registry.unregister("unregister-skill")
        assert result is True
        assert registry.get("unregister-skill") is None

    def test_unregister_nonexistent(self) -> None:
        """Test unregistering nonexistent skill returns False."""
        registry = SkillRegistry()

        result = registry.unregister("nonexistent")
        assert result is False

    def test_clear(self) -> None:
        """Test clearing all skills."""
        registry = SkillRegistry()
        registry.register(create_test_skill("skill-1"))
        registry.register(create_test_skill("skill-2"))

        registry.clear()

        assert len(registry.list()) == 0


class TestSkillRegistryHooksIntegration:
    """Test SkillRegistry integration with SkillHooksManager."""

    def test_register_with_hooks_manager(self) -> None:
        """Test that hooks are registered with hooks manager."""
        hooks_manager = SkillHooksManager()
        registry = SkillRegistry(hooks_manager=hooks_manager)

        hooks = [
            SkillHook(event=HookEvent.PRE_TOOL_USE, matcher="Write"),
            SkillHook(event=HookEvent.POST_TOOL_USE),
        ]
        skill = create_test_skill("hooked-skill", hooks=hooks)

        registry.register(skill)

        # Verify hooks were registered
        pre_hooks = hooks_manager.get_registered_hooks(HookEvent.PRE_TOOL_USE)
        post_hooks = hooks_manager.get_registered_hooks(HookEvent.POST_TOOL_USE)
        assert len(pre_hooks) == 1
        assert len(post_hooks) == 1

    def test_unregister_removes_hooks(self) -> None:
        """Test that unregistering removes hooks from manager."""
        hooks_manager = SkillHooksManager()
        registry = SkillRegistry(hooks_manager=hooks_manager)

        hooks = [SkillHook(event=HookEvent.PRE_TOOL_USE)]
        skill = create_test_skill("hooked-skill", hooks=hooks)

        registry.register(skill)
        assert len(hooks_manager.get_registered_hooks(HookEvent.PRE_TOOL_USE)) == 1

        registry.unregister("hooked-skill")
        assert len(hooks_manager.get_registered_hooks(HookEvent.PRE_TOOL_USE)) == 0

    def test_overwrite_updates_hooks(self) -> None:
        """Test that overwriting updates hooks correctly."""
        hooks_manager = SkillHooksManager()
        registry = SkillRegistry(hooks_manager=hooks_manager)

        # First registration with PRE_TOOL_USE hook
        hooks1 = [SkillHook(event=HookEvent.PRE_TOOL_USE)]
        skill1 = create_test_skill("update-skill", hooks=hooks1)
        registry.register(skill1)

        # Overwrite with POST_TOOL_USE hook
        hooks2 = [SkillHook(event=HookEvent.POST_TOOL_USE)]
        skill2 = create_test_skill("update-skill", hooks=hooks2)
        registry.register(skill2, overwrite=True)

        # Old hooks should be gone, new hooks should be present
        pre_hooks = hooks_manager.get_registered_hooks(HookEvent.PRE_TOOL_USE)
        post_hooks = hooks_manager.get_registered_hooks(HookEvent.POST_TOOL_USE)
        assert len(pre_hooks) == 0
        assert len(post_hooks) == 1

    def test_clear_clears_hooks(self) -> None:
        """Test that clear() also clears hooks."""
        hooks_manager = SkillHooksManager()
        registry = SkillRegistry(hooks_manager=hooks_manager)

        hooks = [SkillHook(event=HookEvent.PRE_TOOL_USE)]
        skill = create_test_skill("clear-skill", hooks=hooks)
        registry.register(skill)

        registry.clear()

        assert len(hooks_manager.get_registered_hooks()) == 0

    def test_set_hooks_manager(self) -> None:
        """Test setting hooks manager after creation."""
        registry = SkillRegistry()
        hooks_manager = SkillHooksManager()

        registry.set_hooks_manager(hooks_manager)

        hooks = [SkillHook(event=HookEvent.PRE_TOOL_USE)]
        skill = create_test_skill("late-hooks-skill", hooks=hooks)
        registry.register(skill)

        assert len(hooks_manager.get_registered_hooks(HookEvent.PRE_TOOL_USE)) == 1

    def test_register_skill_without_hooks(self) -> None:
        """Test registering skill without hooks doesn't cause errors."""
        hooks_manager = SkillHooksManager()
        registry = SkillRegistry(hooks_manager=hooks_manager)

        skill = create_test_skill("no-hooks-skill", hooks=[])
        registry.register(skill)

        assert registry.get("no-hooks-skill") is not None
        assert len(hooks_manager.get_registered_hooks()) == 0


class TestSkillRegistryPolicyIntegration:
    """Test SkillRegistry integration with PolicyEnforcer."""

    def test_set_policy_enforcer(self) -> None:
        """Test setting policy enforcer."""
        registry = SkillRegistry()
        enforcer = MagicMock()

        registry.set_policy_enforcer(enforcer)

        assert registry._policy_enforcer is enforcer


class TestCreateDefaultRegistry:
    """Test create_default_registry function."""

    def test_creates_registry_with_default_hooks_manager(self) -> None:
        """Test that default registry has hooks manager."""
        registry = create_default_registry()

        assert registry._hooks_manager is not None


class TestDefaultSkillRegistry:
    """Test DEFAULT_SKILL_REGISTRY global instance."""

    def test_default_registry_exists(self) -> None:
        """Test that default registry is available."""
        assert DEFAULT_SKILL_REGISTRY is not None
        assert isinstance(DEFAULT_SKILL_REGISTRY, SkillRegistry)

    def test_default_registry_has_hooks_manager(self) -> None:
        """Test that default registry has hooks manager."""
        assert DEFAULT_SKILL_REGISTRY._hooks_manager is not None


class TestProviderNamespace:
    """Test provider namespace support in SkillRegistry."""

    def _make_skill(self, name: str, provider: str | None = None) -> SkillSpec:
        return SkillSpec(
            name=name,
            provider=provider,
            description=f"{name} from {provider}",
            input_schema=InputModel,
            output_schema=OutputModel,
        )

    def test_qualified_name_property(self) -> None:
        s1 = self._make_skill("weather", "houyi")
        assert s1.qualified_name == "houyi/weather"

        s2 = self._make_skill("weather")
        assert s2.qualified_name == "weather"

    def test_same_name_different_provider_coexist(self) -> None:
        registry = SkillRegistry()
        s1 = self._make_skill("weather", "houyi")
        s2 = self._make_skill("weather", "openclaw")

        registry.register(s1)
        registry.register(s2)

        assert registry.get("houyi/weather") is s1
        assert registry.get("openclaw/weather") is s2
        # Plain name returns the first-registered
        assert registry.get("weather") is s1

    def test_same_name_same_provider_raises(self) -> None:
        registry = SkillRegistry()
        s1 = self._make_skill("weather", "houyi")
        s2 = self._make_skill("weather", "houyi")

        registry.register(s1)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(s2)

    def test_list_deduplicates(self) -> None:
        registry = SkillRegistry()
        s1 = self._make_skill("weather", "houyi")
        s2 = self._make_skill("weather", "openclaw")
        registry.register(s1)
        registry.register(s2)

        skills = registry.list()
        assert len(skills) == 2
        names = {s.qualified_name for s in skills}
        assert names == {"houyi/weather", "openclaw/weather"}

    def test_list_qualified_names(self) -> None:
        registry = SkillRegistry()
        registry.register(self._make_skill("weather", "houyi"))
        registry.register(self._make_skill("location"))

        qnames = registry.list_qualified_names()
        assert "houyi/weather" in qnames
        assert "location" in qnames

    def test_unregister_by_qualified_name(self) -> None:
        registry = SkillRegistry()
        s1 = self._make_skill("weather", "houyi")
        s2 = self._make_skill("weather", "openclaw")
        registry.register(s1)
        registry.register(s2)

        assert registry.unregister("openclaw/weather") is True
        assert registry.get("openclaw/weather") is None
        assert registry.get("weather") is s1

    def test_overwrite_with_same_name(self) -> None:
        registry = SkillRegistry()
        s1 = self._make_skill("weather", "houyi")
        s2 = self._make_skill("weather", "houyi")
        registry.register(s1)
        registry.register(s2, overwrite=True)
        assert registry.get("weather") is s2
