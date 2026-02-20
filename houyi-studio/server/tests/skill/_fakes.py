"""Reusable fake/stub classes for skill-service tests."""

from __future__ import annotations

from unittest.mock import MagicMock


class _FakeSkillSpec:
    """Minimal SkillSpec stand-in for testing."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.display_name = kwargs.get("display_name", name)
        self.description = kwargs.get("description", f"Skill {name}")
        self.version = kwargs.get("version", "1.0.0")
        self.author = kwargs.get("author")
        self.provider = kwargs.get("provider")
        self.tools = kwargs.get("tools", [])
        self.permissions = kwargs.get("permissions", [])
        self.invocation_policy = kwargs.get("invocation_policy")
        self.hooks = kwargs.get("hooks", [])
        self.certification = kwargs.get("certification", "unverified")
        self.input_schema = kwargs.get("input_schema")

    @property
    def qualified_name(self) -> str:
        if self.provider:
            return f"{self.provider}/{self.name}"
        return self.name


class _FakeInputSchema:
    """Mimics a Pydantic model used as input_schema."""

    def __init__(self, required_fields: list[str] | None = None):
        self._required = set(required_fields or [])

    def model_validate(self, data: dict):
        for field in self._required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

    def model_json_schema(self):
        return {"type": "object", "required": list(self._required)}


class _FakeTool:
    """Mimics a tool attached to a SkillSpec."""

    def __init__(self, name: str, description: str = "", input_schema=None):
        self.name = name
        self.description = description
        self.input_schema = input_schema


class _FakePermission:
    """Legacy fake — kept for backward compat if referenced."""

    def __init__(self, name, description=None, is_sensitive=False, side_effect=None):
        self.name = name
        self.description = description
        self.is_sensitive = is_sensitive
        self.side_effect = side_effect


class _FakePermKind:
    """Fake for individual permission kind (filesystem/network/exec)."""

    def __init__(self, enabled=False, write=False, delete=False, **kw):
        self.enabled = enabled
        self.write = write
        self.delete = delete
        for k, v in kw.items():
            setattr(self, k, v)


class _FakePermissions:
    """Fake Permissions dataclass (matches houyi.core.skill.policy.Permissions)."""

    def __init__(self, filesystem=None, network=None, exec_=None, descriptions=None):
        self.filesystem = filesystem or _FakePermKind()
        self.network = network or _FakePermKind()
        self.exec = exec_ or _FakePermKind()
        self._descriptions = descriptions or []

    def describe(self):
        return self._descriptions


class _FakeSideEffect:
    """Mimics SideEffect enum."""

    def __init__(self, value: str = "none"):
        self.value = value


class _FakeModelAutoInvoke:
    """Mimics ModelAutoInvoke enum with a .value attribute."""

    def __init__(self, value: str):
        self.value = value

    def __str__(self):
        return self.value


class _FakePolicy:
    """Mimics InvocationPolicy.  Uses _FakeModelAutoInvoke so that
    `.model_auto_invoke.value` works the same as the real enum."""

    def __init__(self, default_action="allow", model_auto_invoke=None):
        self.model_auto_invoke = _FakeModelAutoInvoke(default_action)
        self.user_invocable = True
        self.side_effect = _FakeSideEffect("none")


class _FakePolicyResult:
    def __init__(self, action_value="allow"):
        self.action = MagicMock(value=action_value)
