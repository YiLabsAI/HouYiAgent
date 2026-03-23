"""Tool schema bridge for shared skill-to-tool conversion logic.

Centralizes tool schema collection so server-side consumers do not duplicate
schema extraction, filtering, relevance trimming, or ordering logic.
"""

from __future__ import annotations

import contextlib
import copy
import re
from dataclasses import dataclass
from typing import Any

from houyi.domain.skill.registry import SkillRegistry

_RELEVANCE_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]{3,}")
_SKILL_NAME_ALIASES = {
    "houyi_web_search": "web_search",
}


@dataclass(slots=True)
class _SchemaCacheEntry:
    fingerprint: str
    schemas: list[dict[str, Any]]


def _simplify_property_schema(prop: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in prop.items():
        if key == "title":
            continue
        if key == "anyOf" and isinstance(value, list):
            non_null = [item for item in value if item != {"type": "null"}]
            if len(non_null) == 1:
                result.update(non_null[0])
                continue
        result[key] = value
    return result


def _simplify_schema(schema: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(schema)
    cleaned.pop("title", None)
    properties = cleaned.get("properties")
    if isinstance(properties, dict):
        cleaned["properties"] = {
            name: _simplify_property_schema(spec) if isinstance(spec, dict) else spec
            for name, spec in properties.items()
        }
    return cleaned


def _to_minimal_parameter_schema(schema: dict[str, Any]) -> dict[str, Any]:
    cleaned = _simplify_schema(schema)
    properties = cleaned.get("properties")
    if not isinstance(properties, dict) or not properties:
        return cleaned

    required_fields = [str(item) for item in cleaned.get("required", []) if isinstance(item, str)]
    selected_names = required_fields[:]
    for name in properties:
        if len(selected_names) >= 3:
            break
        if name not in selected_names:
            selected_names.append(name)

    minimal_properties = {name: properties[name] for name in selected_names if name in properties}
    cleaned["properties"] = minimal_properties
    if required_fields:
        cleaned["required"] = [name for name in required_fields if name in minimal_properties]
    return cleaned


def _schema_from_input_model(input_model: Any) -> dict[str, Any]:
    if not input_model:
        return {}
    with contextlib.suppress(Exception):
        model_schema = input_model.model_json_schema()
        if isinstance(model_schema, dict):
            return _simplify_schema(model_schema)
    return {}


def build_tool_definitions_for_skill(skill: Any) -> list[dict[str, Any]]:
    tool_objects = getattr(skill, "tools", None)
    if isinstance(tool_objects, list) and tool_objects:
        definitions: list[dict[str, Any]] = []
        for tool in tool_objects:
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": str(getattr(tool, "name", "") or ""),
                        "description": str(getattr(tool, "description", "") or ""),
                        "parameters": _schema_from_input_model(getattr(tool, "input_schema", None)),
                    },
                }
            )
        return definitions

    return [
        {
            "type": "function",
            "function": {
                "name": str(getattr(skill, "name", "unknown") or "unknown"),
                "description": str(getattr(skill, "description", "") or ""),
                "parameters": _schema_from_input_model(getattr(skill, "input_schema", None)),
            },
        }
    ]


def _apply_schema_exposure(
    schema: dict[str, Any],
    *,
    schema_exposure: str,
) -> dict[str, Any]:
    if schema_exposure == "minimal":
        function_payload = dict(schema.get("function", {}))
        parameters = function_payload.get("parameters")
        if isinstance(parameters, dict):
            function_payload["parameters"] = _to_minimal_parameter_schema(parameters)
        exposed = dict(schema)
        exposed["function"] = function_payload
        return exposed
    return schema


class ToolBridge:
    """Shared bridge for collecting tool schemas from the skill registry."""

    def __init__(self, registry: SkillRegistry):
        self._registry = registry
        self._schema_cache: dict[str, _SchemaCacheEntry] = {}

    def collect_skills(
        self,
        skill_filter: list[str] | None = None,
        include_core: bool = True,
    ) -> list[Any]:
        if skill_filter:
            seen: set[int] = set()
            selected: list[Any] = []
            for skill_name in skill_filter:
                skill = self._resolve_skill(skill_name)
                if skill is None:
                    continue
                marker = id(skill)
                if marker in seen:
                    continue
                seen.add(marker)
                selected.append(skill)
            return selected

        all_skills = self._registry.list()
        if include_core:
            return list(all_skills)
        return [skill for skill in all_skills if not getattr(skill, "is_core", False)]

    def _resolve_skill(self, skill_name: str) -> Any | None:
        skill = self._registry.get(skill_name)
        if skill is not None:
            return skill
        alias = _SKILL_NAME_ALIASES.get(skill_name)
        if not alias:
            return None
        return self._registry.get(alias)

    def collect_tool_schemas(
        self,
        skill_filter: list[str] | None = None,
        include_core: bool = True,
        relevance_hint: str | None = None,
        usage_counts: dict[str, int] | None = None,
        schema_exposure: str = "full",
    ) -> list[dict[str, Any]]:
        selected_skills = self.collect_skills(skill_filter=skill_filter, include_core=include_core)
        if not selected_skills:
            return []

        relevant_skills = self._select_relevant_skills(selected_skills, relevance_hint)

        schemas: list[dict[str, Any]] = []
        for skill in relevant_skills:
            schemas.extend(
                _apply_schema_exposure(schema, schema_exposure=schema_exposure)
                for schema in self._schemas_for_skill(skill)
            )

        if usage_counts:
            schemas.sort(
                key=lambda schema: (
                    -int(usage_counts.get(str(schema.get("function", {}).get("name", "")), 0)),
                    self._core_rank(schema, selected_skills),
                    str(schema.get("function", {}).get("name", "")),
                )
            )
        return schemas

    def _schemas_for_skill(self, skill: Any) -> list[dict[str, Any]]:
        cache_key = str(getattr(skill, "qualified_name", None) or getattr(skill, "name", ""))
        fingerprint = self._schema_fingerprint(skill)
        cached = self._schema_cache.get(cache_key)
        if cached is not None and cached.fingerprint == fingerprint:
            return copy.deepcopy(cached.schemas)

        schemas = build_tool_definitions_for_skill(skill)
        self._schema_cache[cache_key] = _SchemaCacheEntry(
            fingerprint=fingerprint,
            schemas=copy.deepcopy(schemas),
        )
        return schemas

    @staticmethod
    def _schema_fingerprint(skill: Any) -> str:
        name = str(getattr(skill, "name", ""))
        version = str(getattr(skill, "version", "") or "")
        return f"{name}:{version}"

    def _select_relevant_skills(self, skills: list[Any], relevance_hint: str | None) -> list[Any]:
        if not relevance_hint:
            return skills

        tokens = {token.lower() for token in _RELEVANCE_TOKEN_PATTERN.findall(relevance_hint)}
        if not tokens:
            return skills

        matched = [skill for skill in skills if self._matches_relevance(skill, tokens)]
        return matched if matched else skills

    @staticmethod
    def _matches_relevance(skill: Any, tokens: set[str]) -> bool:
        tags = getattr(skill, "tags", None)
        metadata = getattr(skill, "metadata", None)
        metadata_tags: list[str] = []
        if isinstance(metadata, dict):
            raw_tags = metadata.get("tags")
            if isinstance(raw_tags, list):
                metadata_tags = [str(item) for item in raw_tags]

        text = " ".join(
            [
                str(getattr(skill, "name", "")),
                str(getattr(skill, "description", "")),
                " ".join(str(item) for item in (tags or [])),
                " ".join(metadata_tags),
            ]
        ).lower()
        return any(token in text for token in tokens)

    @staticmethod
    def _core_rank(schema: dict[str, Any], skills: list[Any]) -> int:
        name = str(schema.get("function", {}).get("name", ""))
        for skill in skills:
            if str(getattr(skill, "name", "")) == name and getattr(skill, "is_core", False):
                return 0
        return 1
