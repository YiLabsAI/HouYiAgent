"""SkillSpec → dict serialization for the Console UI.

Converts ``SkillSpec`` objects into plain dicts that can be serialized to
JSON and sent to the frontend via WebSocket.  No side effects, no I/O —
pure data transformation.
"""

from __future__ import annotations

import inspect
import logging
import re
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

SOURCE_BUILTIN = "builtin"
SOURCE_COMMUNITY = "community"
SOURCE_THIRD_PARTY = "third_party"
SOURCE_LOCAL = "local"
VALID_SOURCES = frozenset({SOURCE_BUILTIN, SOURCE_COMMUNITY, SOURCE_THIRD_PARTY, SOURCE_LOCAL})

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

        name = str(getattr(skill, "name", "") or "")
        is_external_alias = name.startswith("ext__")
        alias_target = name[len("ext__") :] if is_external_alias else None
        instructions = getattr(skill, "instructions", None)
        instructions_length = len(instructions.strip()) if isinstance(instructions, str) else 0

        return {
            "name": skill.name,
            "display_name": getattr(skill, "display_name", skill.name),
            "description": getattr(skill, "description", None),
            "tools": tools,
            "policy_action": self._policy_action(skill),
            "side_effect": side,
            "certification": getattr(skill, "certification", "unverified"),
            "is_core": bool(getattr(skill, "is_core", False)),
            "source": self._source(skill),
            "capability_tier": self._capability_tier(skill),
            "runtime_status": self._runtime_status(skill),
            "is_external_alias": is_external_alias,
            "alias_target": alias_target,
            "instructions_length": instructions_length,
            "runtime_binding": self._runtime_binding(skill),
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
            "instructions": getattr(skill, "instructions", None),
            "hook_specs": self._serialize_hook_specs(skill),
            "package_examples": self._load_package_examples(skill),
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

    @staticmethod
    def _load_package_examples(skill: SkillSpec) -> list[dict[str, Any]]:
        skill_dir = getattr(skill, "skill_dir", None)
        if skill_dir is None:
            return []

        try:
            base_dir = Path(skill_dir).resolve()
        except Exception:
            return []

        candidates = [
            base_dir / "examples.md",
            base_dir / "EXAMPLES.md",
            base_dir / "README.md",
            base_dir / "readme.md",
        ]
        for doc in candidates:
            if not doc.exists() or not doc.is_file():
                continue
            parsed = SkillSerializer._parse_markdown_examples(doc)
            if parsed:
                return parsed
        return []

    @staticmethod
    def _parse_markdown_examples(path: Path) -> list[dict[str, Any]]:
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            return []

        section_re = re.compile(r"^##\s+Example\s+(\d+)\s*:\s*(.+?)\s*$", re.MULTILINE)
        matches = list(section_re.finditer(content))
        if not matches:
            return []

        def _slug(text: str) -> str:
            return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")

        def _infer_action(title_text: str, block_text: str) -> str:
            signal = f"{title_text}\n{block_text}".lower()
            if "bug fix" in signal or "fix" in signal:
                return "update"
            if "feature" in signal or "error recovery" in signal or "status" in signal:
                return "status"
            return "create"

        def _fallback_task(title_text: str, block_text: str) -> str:
            lines = [line.strip() for line in block_text.splitlines() if line.strip()]
            for line in lines:
                cleaned = re.sub(r"^[#>*\-`\s]+", "", line).strip()
                if not cleaned:
                    continue
                if cleaned.lower().startswith("example"):
                    continue
                if cleaned.lower().startswith("action:"):
                    continue
                return cleaned.rstrip(":.")
            return f"Run workflow for {title_text}".strip()

        examples: list[dict[str, Any]] = []
        for idx, match in enumerate(matches):
            number = match.group(1).strip()
            title = match.group(2).strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
            block = content[start:end]

            request_match = re.search(r'\*\*User Request:\*\*\s*"([^"]+)"', block)
            task = request_match.group(1).strip() if request_match else _fallback_task(title, block)
            phase_lines = [
                line.strip()
                for line in re.findall(
                    r"^-\s*\[[ xX]\]\s*(Phase\s*\d+:[^\n]+)", block, re.MULTILINE
                )
            ]

            input_data: dict[str, Any] = {"action": _infer_action(title, block)}
            if task:
                input_data["task"] = task
            if phase_lines:
                input_data["subtasks"] = phase_lines

            examples.append(
                {
                    "id": f"example-{number}-{_slug(title) or number}",
                    "label": f"Example {number} · {title}",
                    "description": task,
                    "input": input_data,
                    "expectedFocus": [f"{path.name.lower()} #example-{number}"],
                    "objective": f"Validate package-native workflow for Example {number}: {title}.",
                }
            )

        return examples

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

    @staticmethod
    def _serialize_hook_specs(skill: SkillSpec) -> list[dict[str, Any]]:
        if not (hasattr(skill, "hooks") and skill.hooks):
            return []

        specs: list[dict[str, Any]] = []
        for hook in skill.hooks:
            event = getattr(hook, "event", None)
            hook_type = getattr(hook, "hook_type", None)
            matcher = getattr(hook, "matcher", None)
            specs.append(
                {
                    "event": event.value if hasattr(event, "value") else str(event or "hook"),
                    "type": hook_type.value
                    if hasattr(hook_type, "value")
                    else str(hook_type or "handler"),
                    "matcher": str(matcher or "*"),
                    "command": getattr(hook, "command", None),
                    "handler": getattr(hook, "handler_path", None),
                }
            )
        return specs

    @staticmethod
    def _source(skill: SkillSpec) -> str:
        if bool(getattr(skill, "is_core", False)):
            return SOURCE_BUILTIN

        explicit_source = SkillSerializer._explicit_trust_source(skill)
        if explicit_source in VALID_SOURCES:
            return explicit_source

        name = str(getattr(skill, "name", "") or "")
        if name.startswith("ext__"):
            return SOURCE_THIRD_PARTY

        skill_md_path = str(getattr(skill, "skill_md_path", "") or "")
        if skill_md_path.startswith(("http://", "https://")):
            return SOURCE_THIRD_PARTY

        normalized_path = skill_md_path.replace("\\", "/")
        if "/skills/" in normalized_path:
            return SOURCE_COMMUNITY

        if skill_md_path:
            return SOURCE_LOCAL

        return SOURCE_BUILTIN

    @staticmethod
    def _capability_tier(skill: SkillSpec) -> str:
        """Return integration level string from the SkillSpec computed property."""
        try:
            level = skill.capability_tier
            # CapabilityTier is an int enum; use name for the string label
            return level.name.lower() if hasattr(level, "name") else str(level)
        except Exception:
            return "metadata"

    @staticmethod
    def _runtime_status(skill: SkillSpec) -> str:
        """Return runtime status string from the SkillSpec computed property."""
        try:
            status = skill.runtime_status
            return status.value if hasattr(status, "value") else str(status)
        except Exception:
            return "unavailable"

    @staticmethod
    def _runtime_binding(skill: SkillSpec) -> str:
        """Classify how this skill executes at runtime for UI explainability."""
        if callable(getattr(skill, "executor", None)):
            return "python_executor"
        instructions = getattr(skill, "instructions", None)
        if isinstance(instructions, str) and instructions.strip():
            return "prompt_instructions"
        return "none"

    @staticmethod
    def _explicit_trust_source(skill: SkillSpec) -> str | None:
        meta = getattr(skill, "extra_frontmatter", None)
        if not isinstance(meta, dict):
            return None

        trust = meta.get("trust")
        if isinstance(trust, dict):
            source = trust.get("source")
            if isinstance(source, str):
                normalized = source.strip().lower()
                return normalized if normalized in VALID_SOURCES else None
        return None
