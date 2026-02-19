"""SkillSpec → dict serialization for the Console UI.

Converts ``SkillSpec`` objects into plain dicts that can be serialized to
JSON and sent to the frontend via WebSocket.  No side effects, no I/O —
pure data transformation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from houyi.core.skill.spec import SkillSpec

# ── Constants ─────────────────────────────────────────────────────────

POLICY_ALLOW = "allow"
POLICY_ALLOW_WITH_CONSENT = "allow_with_consent"
POLICY_DENY = "deny"
VALID_POLICY_ACTIONS = frozenset({POLICY_ALLOW, POLICY_ALLOW_WITH_CONSENT, POLICY_DENY})

SIDE_EFFECT_NONE = "none"
SIDE_EFFECT_EXEC = "exec"
SIDE_EFFECT_NETWORK = "network"
SIDE_EFFECT_FILESYSTEM = "filesystem"

DEFAULT_VERSION = "0.0.0"

# ── Side-effect helpers (pure functions) ──────────────────────────────


def extract_side_effects(perms: object) -> list[str]:
    """Return all side-effect tags present in *perms*."""
    effects: list[str] = []
    if hasattr(perms, "exec") and getattr(perms.exec, "enabled", False):
        effects.append(SIDE_EFFECT_EXEC)
    if hasattr(perms, "network") and getattr(perms.network, "enabled", False):
        effects.append(SIDE_EFFECT_NETWORK)
    if hasattr(perms, "filesystem") and (
        getattr(perms.filesystem, "write", False) or getattr(perms.filesystem, "delete", False)
    ):
        effects.append(SIDE_EFFECT_FILESYSTEM)
    return effects


def dominant_side_effect(perms: object) -> str:
    """Return the single most important side-effect label for UI badges."""
    for tag in (SIDE_EFFECT_EXEC, SIDE_EFFECT_NETWORK, SIDE_EFFECT_FILESYSTEM):
        if tag in extract_side_effects(perms):
            return tag
    return SIDE_EFFECT_NONE


# ── Serializer class ─────────────────────────────────────────────────


class SkillSerializer:
    """Stateless converter: ``SkillSpec`` → dict.

    Separated from ``SkillService`` so the service class does not need to
    know anything about JSON shape or UI presentation.
    """

    # ── Summary (list view) ───────────────────────────────────────

    def to_summary(self, skill: SkillSpec) -> dict[str, Any]:
        tools = self._tool_names(skill)
        side = SIDE_EFFECT_NONE
        if hasattr(skill, "permissions") and skill.permissions:
            side = dominant_side_effect(skill.permissions)

        return {
            "name": skill.name,
            "display_name": getattr(skill, "display_name", skill.name),
            "description": getattr(skill, "description", None),
            "tools": tools,
            "policy_action": self._policy_action(skill),
            "side_effect": side,
            "certification": getattr(skill, "certification", "unverified"),
        }

    # ── Full detail (detail panel) ────────────────────────────────

    def to_detail(self, skill: SkillSpec) -> dict[str, Any]:
        summary = self.to_summary(skill)
        return {
            **summary,
            "version": getattr(skill, "version", None) or DEFAULT_VERSION,
            "author": getattr(skill, "author", None),
            "tools": self._serialize_tools(skill),
            "permissions": self._serialize_permissions(skill),
            "policy": self._serialize_policy(skill),
            "hooks": self._serialize_hooks(skill),
        }

    # ── Private helpers ───────────────────────────────────────────

    @staticmethod
    def _tool_names(skill: SkillSpec) -> list[str]:
        if hasattr(skill, "tools") and skill.tools:
            return [t.name if hasattr(t, "name") else str(t) for t in skill.tools]
        return [skill.name] if hasattr(skill, "name") else []

    @staticmethod
    def _policy_action(skill: SkillSpec) -> str:
        if hasattr(skill, "invocation_policy") and skill.invocation_policy:
            mai = getattr(skill.invocation_policy, "model_auto_invoke", None)
            if mai is not None:
                return mai.value if hasattr(mai, "value") else str(mai)
        return POLICY_ALLOW

    @staticmethod
    def _serialize_tools(skill: SkillSpec) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        if hasattr(skill, "tools") and skill.tools:
            for t in skill.tools:
                info: dict[str, Any] = {
                    "name": getattr(t, "name", str(t)),
                    "description": getattr(t, "description", None),
                }
                if hasattr(t, "input_schema"):
                    info["input_schema"] = (
                        t.input_schema.model_json_schema()
                        if hasattr(t.input_schema, "model_json_schema")
                        else {}
                    )
                tools.append(info)
        else:
            tools.append(
                {
                    "name": skill.name,
                    "description": getattr(skill, "description", None),
                    "input_schema": (
                        skill.input_schema.model_json_schema()
                        if hasattr(skill, "input_schema")
                        and hasattr(skill.input_schema, "model_json_schema")
                        else {}
                    ),
                }
            )
        return tools

    @staticmethod
    def _serialize_permissions(skill: SkillSpec) -> list[dict[str, Any]]:
        perms_list: list[dict[str, Any]] = []
        if not (hasattr(skill, "permissions") and skill.permissions):
            return perms_list
        perms = skill.permissions
        if hasattr(perms, "describe"):
            for desc in perms.describe():
                perms_list.append(
                    {
                        "name": desc,
                        "description": desc,
                        "is_sensitive": True,
                    }
                )
        elif isinstance(perms, dict):
            for k, v in perms.items():
                perms_list.append(
                    {
                        "name": k,
                        "description": str(v),
                        "is_sensitive": False,
                    }
                )
        return perms_list

    def _serialize_policy(self, skill: SkillSpec) -> dict[str, Any]:
        if not (hasattr(skill, "invocation_policy") and skill.invocation_policy):
            return {}
        ip = skill.invocation_policy
        mai_val = self._policy_action(skill)
        policy: dict[str, Any] = {
            "default_action": mai_val,
            "model_auto_invoke": mai_val != POLICY_DENY,
            "user_invocable": getattr(ip, "user_invocable", True),
            "side_effect": getattr(ip, "side_effect", SIDE_EFFECT_NONE),
        }
        if hasattr(policy["side_effect"], "value"):
            policy["side_effect"] = policy["side_effect"].value
        return policy

    @staticmethod
    def _serialize_hooks(skill: SkillSpec) -> list[str]:
        if hasattr(skill, "hooks") and skill.hooks:
            return [getattr(h, "hook_type", str(h)) for h in skill.hooks]
        return []
