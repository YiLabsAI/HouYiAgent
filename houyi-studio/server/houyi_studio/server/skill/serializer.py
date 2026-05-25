"""SkillSpec → dict serialization for the Console UI.

Converts SkillSpec objects into plain dicts that can be serialized to
JSON and sent to the frontend via WebSocket.  No side effects, no I/O —
pure data transformation.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from houyi.domain.skill.spec import SkillSpec

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
    """Stateless converter: SkillSpec → dict.

    Separated from SkillService so the service class does not need to
    know anything about JSON shape or UI presentation.
    """

    # ── Summary (list view) ───────────────────────────────────────

    def to_summary(self, skill: SkillSpec) -> dict[str, Any]:
        tools = self._tool_names(skill)
        side = SIDE_EFFECT_NONE
        if hasattr(skill, "permissions") and skill.permissions:
            side = dominant_side_effect(skill.permissions)
        source = self._source(skill)

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
            "source": source,
            "source_group": self._source_group(skill, source),
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
        permissions = self._serialize_permissions(skill)
        policy = self._serialize_policy(skill)
        instructions = getattr(skill, "instructions", None)
        package_examples = self._load_package_examples(skill)
        try:
            from .dry_run import _collect_available_workflows

            available_workflows = _collect_available_workflows(skill)
        except Exception:
            available_workflows = []
        return {
            **summary,
            "version": version,
            "author": author,
            "tools": self._serialize_tools(skill),
            "permissions": permissions,
            "policy": policy,
            "hooks": self._serialize_hooks(skill),
            "instructions": instructions,
            "hook_specs": self._serialize_hook_specs(skill),
            "package_examples": package_examples,
            "available_workflows": available_workflows,
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
            from houyi.domain.skill.schema import parse_skill_md

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
        skill_doc = base_dir / "SKILL.md"
        if skill_doc.exists() and skill_doc.is_file():
            return SkillSerializer._parse_skill_usage_examples(skill_doc)
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
            input_data = SkillSerializer._derive_example_input_from_text(block, fallback_task=task)

            examples.append(
                {
                    "id": f"example-{number}-{_slug(title) or number}",
                    "label": f"Example {number} · {title}",
                    "description": task,
                    "input": input_data,
                    "expectedFocus": [f"{path.name.lower()} #example-{number}"],
                    "objective": f"Validate package-native workflow for Example {number}: {title}.",
                    "source": f"{path.name}#example-{number}",
                    "confidence": "high",
                    "confidence_reason": "Explicit Example section with structured payload.",
                    "confidence_breakdown": {
                        "title_match": 1.0,
                        "step_structure": 1.0,
                        "command_parse": 1.0,
                        "score": 1.0,
                    },
                }
            )

        return examples

    @staticmethod
    def _parse_skill_usage_examples(path: Path) -> list[dict[str, Any]]:
        """Best-effort fallback parser for SKILL.md usage snippets.

        When a package does not ship explicit Example sections, derive a single
        dry-run preset from the Quick Start code block.
        """
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            return []

        examples: list[dict[str, Any]] = []
        examples.extend(
            SkillSerializer._parse_flow_semantic_examples(content, path.name.lower(), path.name)
        )

        quick_start_match = re.search(
            r"^##\s+Quick\s+Start\b([\s\S]*?)(?=^##\s+|\Z)",
            content,
            re.MULTILINE,
        )
        scan_text = quick_start_match.group(1) if quick_start_match else content

        command_text = SkillSerializer._extract_shell_command(scan_text)
        if command_text:
            task = "Run the package quick start flow"
            request_match = re.search(r'"input"\s*:\s*"([^"]+)"', command_text)
            if request_match:
                task = request_match.group(1).strip()

            input_data = SkillSerializer._derive_example_input_from_text(
                scan_text,
                fallback_task=task,
            )

            examples.append(
                {
                    "id": "quick-start-from-skill-md",
                    "label": "Quick Start · SKILL.md",
                    "description": task,
                    "input": input_data,
                    "expectedFocus": [f"{path.name.lower()} #quick-start"],
                    "objective": f"Validate quick-start flow derived from {path.name}.",
                    "source": f"{path.name}#quick-start",
                    "confidence": "medium",
                    "confidence_reason": "Quick Start command inferred from fallback parsing.",
                    "confidence_breakdown": {
                        "title_match": 0.8,
                        "step_structure": 0.6,
                        "command_parse": 0.7,
                        "score": 0.7,
                    },
                }
            )

        return examples

    @staticmethod
    def _parse_flow_semantic_examples(
        content: str, path_name_lower: str, path_name: str
    ) -> list[dict[str, Any]]:
        sections = SkillSerializer._extract_semantic_flow_sections(content)
        if not sections:
            return []

        examples: list[dict[str, Any]] = []
        for section_title, anchor, section_confidence, section_body in sections:
            step_lines = SkillSerializer._extract_flow_step_lines(section_body)
            if not step_lines:
                continue

            id_prefix = SkillSerializer._flow_id_prefix(section_title)
            section_label = section_title.strip() or "Flow"

            for idx, line in enumerate(step_lines, start=1):
                step, command = SkillSerializer._split_flow_step_and_command(line)
                if not command:
                    continue

                input_data = SkillSerializer._input_from_shell_command(command)
                if "task" not in input_data:
                    input_data["task"] = step

                confidence, confidence_reason, confidence_breakdown = (
                    SkillSerializer._score_flow_example_confidence(
                        section_title=section_title,
                        step_line=line,
                        parsed_input=input_data,
                    )
                )

                slug = SkillSerializer._slugify(step) or str(idx)
                examples.append(
                    {
                        "id": f"{id_prefix}-{idx}-{slug}",
                        "label": f"{section_label} {idx} · {step}",
                        "description": f"{step} via command workflow.",
                        "input": input_data,
                        "expectedFocus": [f"{path_name_lower} #{anchor}"],
                        "objective": f"Validate {section_label.lower()} step: {step}.",
                        "source": f"{path_name}#{anchor}",
                        "confidence": confidence or section_confidence,
                        "confidence_reason": confidence_reason,
                        "confidence_breakdown": confidence_breakdown,
                    }
                )

        return examples

    @staticmethod
    def _extract_semantic_flow_sections(content: str) -> list[tuple[str, str, str, str]]:
        heading_re = re.compile(r"^(#{2,4})\s+(.+?)\s*$", re.MULTILINE)
        matches = list(heading_re.finditer(content))
        if not matches:
            return []

        sections: list[tuple[str, str, str, str]] = []
        for idx, match in enumerate(matches):
            title = match.group(2).strip()
            title_lower = title.lower()

            if not any(
                keyword in title_lower
                for keyword in ("flow", "workflow", "steps", "process", "how it works")
            ):
                continue

            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
            body = content[start:end]
            if not body.strip():
                continue

            if (
                ("decision" in title_lower and "flow" in title_lower)
                or "workflow" in title_lower
                or "flow" in title_lower
            ):
                confidence = "high"
            else:
                confidence = "medium"

            sections.append((title, SkillSerializer._slugify(title) or "flow", confidence, body))

        return sections

    @staticmethod
    def _extract_flow_step_lines(section_body: str) -> list[str]:
        lines: list[str] = []
        for raw_line in section_body.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if any(token in line for token in ("→", "->", "=>")):
                lines.append(line)
                continue

            normalized = line
            bullet_match = re.match(r"^(?:[-*]\s+|\d+[.)]\s+)(.+)$", line)
            if bullet_match:
                normalized = bullet_match.group(1).strip()

            if SkillSerializer._extract_shell_commands(normalized):
                lines.append(normalized)

        return lines

    @staticmethod
    def _flow_id_prefix(section_title: str) -> str:
        title_lower = section_title.lower()
        if "decision" in title_lower and "flow" in title_lower:
            return "decision-flow"
        if "workflow" in title_lower:
            return "workflow"
        if "steps" in title_lower:
            return "steps"
        if "process" in title_lower:
            return "process"
        return "flow"

    @staticmethod
    def _split_flow_step_and_command(line: str) -> tuple[str, str]:
        for separator in ("→", "->", "=>"):
            if separator in line:
                left, _, right = line.partition(separator)
                step = left.strip("` ") or "Flow step"
                commands = SkillSerializer._extract_shell_commands(right)
                if commands:
                    return step, commands[0]

        if ":" in line:
            left, _, right = line.partition(":")
            step = left.strip("` ") or "Flow step"
            commands = SkillSerializer._extract_shell_commands(right)
            if commands:
                return step, commands[0]

        commands = SkillSerializer._extract_shell_commands(line)
        step = line.strip("` ") or "Flow step"
        return step, (commands[0] if commands else "")

    @staticmethod
    def _score_flow_example_confidence(
        section_title: str,
        step_line: str,
        parsed_input: dict[str, Any],
    ) -> tuple[str, str, dict[str, float]]:
        title_lower = section_title.lower()
        line = step_line.strip()

        title_score = 1.0 if ("workflow" in title_lower or "flow" in title_lower) else 0.75
        step_score = 1.0 if any(token in line for token in ("→", "->", "=>", ":")) else 0.65

        command_score = 0.45
        if parsed_input:
            if "command" not in parsed_input or len(parsed_input.keys()) > 1:
                command_score = 1.0
            else:
                command_score = 0.7

        score = round((title_score * 0.4) + (step_score * 0.25) + (command_score * 0.35), 2)
        if score >= 0.82:
            confidence = "high"
        elif score >= 0.58:
            confidence = "medium"
        else:
            confidence = "low"

        reason_parts: list[str] = []
        if title_score >= 1.0:
            reason_parts.append("title matched flow/workflow semantics")
        else:
            reason_parts.append("title weakly matched process semantics")

        if step_score >= 1.0:
            reason_parts.append("step formatting matched workflow pattern")
        else:
            reason_parts.append("step formatting inferred from loose command line")

        if command_score >= 1.0:
            reason_parts.append("command parsed into structured arguments")
        elif command_score >= 0.7:
            reason_parts.append("command detected but parsed as raw shell command")
        else:
            reason_parts.append("command evidence is weak")

        breakdown = {
            "title_match": round(title_score, 2),
            "step_structure": round(step_score, 2),
            "command_parse": round(command_score, 2),
            "score": score,
        }

        return confidence, "; ".join(reason_parts), breakdown

    @staticmethod
    def _derive_example_input_from_text(
        text: str, fallback_task: str | None = None
    ) -> dict[str, Any]:
        # 1) explicit JSON payload inside markdown
        json_blocks = re.findall(r"```json\n([\s\S]*?)```", text)
        for block in json_blocks:
            try:
                payload = json.loads(block.strip())
                if isinstance(payload, dict) and SkillSerializer._looks_like_invocation_payload(
                    payload
                ):
                    return payload
            except (ValueError, TypeError):
                continue

        # 2) command-line example payload
        command = SkillSerializer._extract_shell_command(text)
        input_data = SkillSerializer._input_from_shell_command(command) if command else {}

        if fallback_task and "task" not in input_data:
            input_data["task"] = fallback_task

        return input_data

    @staticmethod
    def _looks_like_invocation_payload(payload: dict[str, Any]) -> bool:
        keys = {str(k) for k in payload}
        if not keys:
            return False

        schema_only_keys = {
            "$id",
            "$schema",
            "title",
            "description",
            "type",
            "properties",
            "required",
            "items",
            "definitions",
            "$defs",
            "additionalProperties",
            "oneOf",
            "anyOf",
            "allOf",
        }

        if keys <= schema_only_keys:
            return False

        return not ({"type", "properties"}.issubset(keys) and len(keys - schema_only_keys) == 0)

    @staticmethod
    def _extract_shell_command(text: str) -> str:
        commands = SkillSerializer._extract_shell_commands(text)
        return commands[0] if commands else ""

    @staticmethod
    def _extract_shell_commands(text: str) -> list[str]:
        commands: list[str] = []
        code_blocks = re.findall(r"```(?:bash|sh|zsh)?\n([\s\S]*?)```", text)
        for block in code_blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            for line in lines:
                if line.startswith("python ") or line.startswith("./") or line.startswith("sh "):
                    commands.append(line)
        if code_blocks:
            first_line = code_blocks[0].strip().splitlines()
            if first_line:
                commands.append(first_line[0].strip())

        inline_candidates = re.findall(r"(python\s+[^\n`]+|\./[^\n`]+|sh\s+[^\n`]+)", text)
        for candidate in inline_candidates:
            cmd = candidate.strip().rstrip(",.;)")
            if cmd:
                commands.append(cmd)

        deduped: list[str] = []
        seen: set[str] = set()
        for cmd in commands:
            if cmd in seen:
                continue
            seen.add(cmd)
            deduped.append(cmd)

        return deduped

    @staticmethod
    def _slugify(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")

    @staticmethod
    def _input_from_shell_command(command: str) -> dict[str, Any]:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return {"command": command} if command else {}
        if not tokens:
            return {}

        run_idx = -1
        for idx, tok in enumerate(tokens):
            if tok.endswith("scripts/run.py"):
                run_idx = idx
                break

        if run_idx >= 0 and run_idx + 1 < len(tokens):
            script = tokens[run_idx + 1]
            args = tokens[run_idx + 2 :]
            payload: dict[str, Any] = {"script": script}
            positional: list[str] = []
            i = 0
            while i < len(args):
                token = args[i]
                if token.startswith("--"):
                    key = token[2:].replace("-", "_")
                    if i + 1 < len(args) and not args[i + 1].startswith("--"):
                        payload[key] = args[i + 1]
                        i += 2
                    else:
                        payload[key] = True
                        i += 1
                    continue
                positional.append(token)
                i += 1

            if positional:
                payload["operation"] = positional[0]
                if len(positional) > 1:
                    payload["args"] = positional[1:]
            return payload

        return {"command": command}

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
        extra = getattr(skill, "extra_frontmatter", None)
        if isinstance(extra, dict):
            compat = extra.get("runtime_binding")
            if isinstance(compat, str) and compat.strip():
                return compat.strip()
        if callable(getattr(skill, "executor", None)):
            return "python_executor"
        instructions = getattr(skill, "instructions", None)
        if isinstance(instructions, str) and instructions.strip():
            return "prompt_instructions"
        return "none"

    @staticmethod
    def _source_group(skill: SkillSpec, source: str) -> str | None:
        """Best-effort package grouping key for external/community skills.

        Example: ~/.houyi/skills/superpowers/skills/using-git-worktrees/SKILL.md
        => source_group = "superpowers"
        """
        if source == SOURCE_BUILTIN or bool(getattr(skill, "is_core", False)):
            return None

        skill_md_path = str(getattr(skill, "skill_md_path", "") or "")
        if not skill_md_path or skill_md_path.startswith(("http://", "https://")):
            return None

        try:
            resolved = Path(skill_md_path).resolve()
        except Exception:
            return None

        parts = list(resolved.parts)
        for idx in range(len(parts) - 2):
            if parts[idx] == ".houyi" and parts[idx + 1] == "skills":
                candidate = parts[idx + 2].strip()
                return candidate or None

        for idx in range(len(parts) - 3):
            if (
                parts[idx] == ".houyi"
                and parts[idx + 1] == "sources"
                and parts[idx + 2]
                in {
                    "local",
                    "remote",
                }
            ):
                candidate = parts[idx + 3].strip()
                return candidate or None

        # Generic fallback for ecosystem layouts:
        #   <package>/skills/<skill-name>/SKILL.md
        # Detect the package segment right before "skills".
        for idx, part in enumerate(parts):
            if part != "skills" or idx == 0:
                continue
            candidate = str(parts[idx - 1]).strip()
            if candidate and candidate not in {".houyi", "sources", "local", "remote", "skills"}:
                return candidate

        return None

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
