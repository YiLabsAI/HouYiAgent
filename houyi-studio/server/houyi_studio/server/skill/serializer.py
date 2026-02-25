"""SkillSpec → dict serialization for the Console UI.

Converts ``SkillSpec`` objects into plain dicts that can be serialized to
JSON and sent to the frontend via WebSocket.  No side effects, no I/O —
pure data transformation.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
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
_SKILL_MD_META_CACHE: dict[str, dict[str, Any]] = {}

logger = logging.getLogger(__name__)

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
        meta = self._resolve_frontmatter_meta(skill)
        version = getattr(skill, "version", None) or meta.get("version") or DEFAULT_VERSION
        author = getattr(skill, "author", None) or meta.get("author")
        return {
            **summary,
            "version": version,
            "author": author,
            "tools": self._serialize_tools(skill),
            "permissions": self._serialize_permissions(skill),
            "policy": self._serialize_policy(skill),
            "hooks": self._serialize_hooks(skill),
        }

    @staticmethod
    def _resolve_frontmatter_meta(skill: SkillSpec) -> dict[str, Any]:
        """Best-effort metadata hydration from adjacent SKILL.md for code skills."""
        executor = getattr(skill, "executor", None)
        if not callable(executor):
            return {}

        try:
            source_file = inspect.getsourcefile(executor) or inspect.getfile(executor)
        except (TypeError, OSError):
            return {}
        if not source_file:
            return {}

        skill_md = (Path(source_file).resolve().parent / "SKILL.md").resolve()
        cache_key = str(skill_md)
        if cache_key in _SKILL_MD_META_CACHE:
            return _SKILL_MD_META_CACHE[cache_key]

        if not skill_md.exists():
            _SKILL_MD_META_CACHE[cache_key] = {}
            return {}

        try:
            from houyi.core.skill.schema import parse_skill_md

            parsed = parse_skill_md(skill_md.read_text(encoding="utf-8"))
            meta = {
                "version": parsed.get("version"),
                "author": parsed.get("author"),
            }
            _SKILL_MD_META_CACHE[cache_key] = meta
            return meta
        except Exception as exc:  # pragma: no cover - defensive path
            logger.debug("Failed to parse SKILL.md metadata at %s: %s", skill_md, exc)
            _SKILL_MD_META_CACHE[cache_key] = {}
            return {}

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
            labels: list[str] = []
            for hook in skill.hooks:
                event = getattr(hook, "event", None)
                hook_type = getattr(hook, "hook_type", None)
                matcher = getattr(hook, "matcher", None)

                event_str = event.value if hasattr(event, "value") else str(event or "hook")
                type_str = (
                    hook_type.value if hasattr(hook_type, "value") else str(hook_type or "handler")
                )
                matcher_str = str(matcher) if matcher else "*"
                labels.append(f"{event_str}:{matcher_str} ({type_str})")
            return labels
        return []
