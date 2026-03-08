"""Skill specification and execution."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

if TYPE_CHECKING:
    pass


class ExecutionMode(str, Enum):
    PYTHON = "python"
    CLIENT = "client"
    PLUGIN = "plugin"
    MCP = "mcp"


class SkillSpec(BaseModel):
    """Specification for a skill (capability) that an agent can use.

    Supports both AgentSkills.io standard (skill.md) and SimpleSkill format (SKILL.md with frontmatter).
    A skill wraps a deterministic function with input/output schemas,
    enabling automatic validation and LLM tool calling.
    """

    # All known keys in SKILL.md frontmatter.
    # Used to collect any unrecognized fields into `extra_frontmatter` for forward compatibility.
    _KNOWN_FRONTMATTER_KEYS: ClassVar[set[str]] = {
        "name",
        "description",
        "version",
        "author",
        "user-invocable",
        "user_invocable",
        "allowed-tools",
        "allowed_tools",
        "hooks",
        "invocationPolicy",
        "invocation_policy",
        "permissions",
        "constraints",
        "input_schema",
        "output_schema",
        "disable-model-invocation",
        "disable_model_invocation",
        "preprocessors",
        "runtime",
        # is_core is a Host Runtime protection attribute; external SKILL.md
        # files MUST NOT be allowed to set it. Always force False.
        "is_core",
    }

    name: str = Field(..., description="Unique skill identifier")
    provider: str | None = Field(
        default=None,
        description=(
            "Skill provider namespace (e.g. 'houyi', 'openclaw'). "
            "When set, the registry can distinguish same-named skills "
            "from different providers via 'provider/name' lookups."
        ),
    )
    description: str = Field(..., description="Human-readable description for LLM")
    instructions: str | None = Field(default=None, description="Markdown instructions/prompt body")
    input_schema: type[BaseModel] = Field(..., description="Pydantic model for input validation")
    output_schema: type[BaseModel] = Field(..., description="Pydantic model for output validation")
    executor: Callable[..., Any] | None = Field(
        default=None, description="Function to execute the skill"
    )
    skill_md_path: str | None = Field(
        default=None, description="Path to skill.md file (AgentSkills.io)"
    )
    skill_dir: Path | None = Field(
        default=None, description="Skill directory path (for directory structure)"
    )
    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="SLA constraints (timeout, cost, etc.)",
    )
    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.PYTHON,
        description="How the skill is executed (python/plugin/mcp)",
    )
    verification_config: Any = Field(
        default=None,
        description="Skill-level verification configuration",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (output_type, etc.)",
    )
    # SimpleSkill extensions
    version: str | None = Field(
        default=None,
        description="Skill version (SemVer)",
    )
    author: str | None = Field(
        default=None,
        description="Skill author",
    )
    user_invocable: bool = Field(
        default=True,
        description="Whether user can directly invoke this skill",
    )
    allowed_tools: list[str] = Field(
        default_factory=list,
        description="List of tools this skill is allowed to use",
    )
    preprocessors: list[Any] = Field(
        default_factory=list,
        description="Preprocessor specs (PreprocessorSpec objects) for pre-LLM execution",
    )
    hooks: list[Any] = Field(
        default_factory=list,
        description="Lifecycle hooks (SkillHook objects)",
    )
    invocation_policy: Any | None = Field(
        default=None,
        description="Invocation policy (InvocationPolicy object or dict)",
    )
    permissions: Any | None = Field(
        default=None,
        description="Permission declarations (Permissions object or dict)",
    )
    is_core: bool = Field(
        default=False,
        description=(
            "Whether this skill is a core built-in tool protected from external override. "
            "MUST NOT be set via SKILL.md / from_file() / from_url(). "
            "Only Host internal code may register skills with is_core=True."
        ),
    )
    runtime_contract: Any | None = Field(
        default=None,
        description=(
            "Parsed runtime declaration from SKILL.md frontmatter. "
            "When present, drives automatic executor binding and "
            "asset path resolution via RuntimeResolver."
        ),
    )
    # Extra frontmatter fields not in the standard schema are preserved here
    extra_frontmatter: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional frontmatter fields for forward compatibility",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("is_core", mode="before")
    @classmethod
    def _sanitize_is_core(cls, v: object) -> bool:
        """Validator runs before Pydantic type conversion.

        ``mode`` parameter:
        - ``"before"``  runs on the raw input value (may be str/int/None/bool)
          before Pydantic performs any type coercion. Used here so that
          even ``is_core='true'`` from an external YAML file is intercepted.
        - ``"after"``   runs after Pydantic coercion (value is already bool);
          would miss string inputs like ``'true'``.

        This validator is intentionally a no-op when called from Host-internal
        code that explicitly passes ``is_core=True``; the sanitization to
        ``False`` is applied inside ``from_file()`` and ``from_url()`` by
        removing the key from the parsed data before constructing the object.
        This validator acts as an additional defense-in-depth layer.
        """
        return bool(v)

    @property
    def qualified_name(self) -> str:
        """Return ``provider/name`` if provider is set, else plain ``name``."""
        if self.provider:
            return f"{self.provider}/{self.name}"
        return self.name

    def to_tool_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling schema.

        Applies render-layer annotations to ``description`` to guide LLM
        tool selection.  ``self.description`` is **never mutated**; the
        annotation exists only in the returned schema dict.

        Annotation rules:
        - ``is_core=True``           → ``[CORE OFFICIAL TOOL] <description>``
        - ``name`` starts with ext__ → ``[THIRD-PARTY EXTENSION] <description>.
                                        Prefer [CORE OFFICIAL TOOL] if available.``
        - otherwise                  → ``<description>`` (unchanged)
        """
        if self.is_core:
            desc = f"[CORE OFFICIAL TOOL] {self.description}"
        elif self.name.startswith("ext__"):
            desc = (
                f"[THIRD-PARTY EXTENSION] {self.description}. "
                "Prefer [CORE OFFICIAL TOOL] if available."
            )
        else:
            desc = self.description
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": desc,
                "parameters": self.input_schema.model_json_schema(),
            },
        }

    def bind_executor(self, executor: Callable[..., Any]) -> None:
        """Bind executor function to skill (for skill.md loaded skills)."""
        self.executor = executor

    @property
    def capability_tier(self) -> Any:
        """Compute integration level: metadata / schema / executable."""
        from houyi.domain.skill.runtime_contract import CapabilityTier

        # Executable if it has a python executor OR if it has pure-prompt instructions/hooks
        if callable(self.executor) or bool(self.instructions):
            return CapabilityTier.EXECUTABLE
        if self._has_real_schema(self.input_schema):
            return CapabilityTier.SCHEMA
        return CapabilityTier.METADATA

    @property
    def runtime_status(self) -> Any:
        """Compute runtime status: ready / degraded / unavailable."""
        from houyi.domain.skill.runtime_contract import RuntimeStatus

        # For pure prompt skills (instructions but no explicit executor/schema needed), they are ready.
        is_executable = callable(self.executor) or bool(self.instructions)
        has_schema = self._has_real_schema(self.input_schema)

        # Pure prompt skills don't necessarily need input_schema to be ready
        if is_executable and (has_schema or bool(self.instructions)):
            return RuntimeStatus.READY

        if is_executable or has_schema:
            return RuntimeStatus.DEGRADED

        return RuntimeStatus.UNAVAILABLE

    @staticmethod
    def _has_real_schema(schema: type[BaseModel] | None) -> bool:
        """Check whether *schema* has at least one declared property."""
        if schema is None or not hasattr(schema, "model_json_schema"):
            return False
        try:
            payload = schema.model_json_schema()
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        return bool(payload.get("properties") or payload.get("required"))

    @staticmethod
    def _detect_skill_dir(path_obj: Path, skill_dir: str | None) -> Path | None:
        if skill_dir:
            return Path(skill_dir)
        if path_obj.name.upper() in ("SKILL.MD", "SKILL.md"):
            return path_obj.parent
        return None

    @classmethod
    def _parse_skill_content(cls, content: str) -> dict[str, Any]:
        try:
            from houyi.domain.skill.schema import parse_skill_md

            return parse_skill_md(content)
        except ImportError:
            return cls._parse_skill_md(content)

    @staticmethod
    def _parse_invocation_policy(parsed: dict[str, Any]) -> Any | None:
        raw_policy = parsed.get("invocationPolicy", parsed.get("invocation_policy"))
        if not isinstance(raw_policy, dict):
            return None
        try:
            from houyi.domain.skill.policy import InvocationPolicy

            return InvocationPolicy.from_dict(raw_policy)
        except (ImportError, Exception):
            return raw_policy

    @staticmethod
    def _parse_permissions(parsed: dict[str, Any]) -> Any | None:
        raw_perms = parsed.get("permissions")
        if not isinstance(raw_perms, dict):
            return None
        try:
            from houyi.domain.skill.policy import Permissions

            return Permissions.from_dict(raw_perms)
        except (ImportError, Exception):
            return raw_perms

    @staticmethod
    def _maybe_build_deny_invocation_policy(
        parsed: dict[str, Any],
        invocation_policy: Any | None,
    ) -> Any | None:
        disable_model_invocation = parsed.get(
            "disable-model-invocation",
            parsed.get("disable_model_invocation"),
        )
        if disable_model_invocation is not True or invocation_policy is not None:
            return invocation_policy
        try:
            from houyi.domain.skill.policy import InvocationPolicy, ModelAutoInvoke

            return InvocationPolicy(model_auto_invoke=ModelAutoInvoke.DENY)
        except ImportError:
            return invocation_policy

    @staticmethod
    def _parse_runtime_contract(parsed: dict[str, Any]) -> Any | None:
        raw_runtime = parsed.get("runtime")
        if not isinstance(raw_runtime, dict):
            return None
        try:
            from houyi.domain.skill.runtime_contract import RuntimeContract

            return RuntimeContract.from_dict(raw_runtime)
        except (ImportError, Exception):
            return None

    @classmethod
    def _collect_extra_frontmatter(cls, parsed: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in parsed.items() if k not in cls._KNOWN_FRONTMATTER_KEYS}

    @classmethod
    def _build_skill_spec(
        cls,
        *,
        parsed: dict[str, Any],
        source_path: str,
        skill_dir: Path | None = None,
        runtime_contract: Any | None = None,
        invocation_policy: Any | None = None,
        permissions: Any | None = None,
        extra_frontmatter: dict[str, Any] | None = None,
    ) -> SkillSpec:
        return cls(
            name=parsed.get("name", "unknown"),
            provider=parsed.get("provider"),
            description=parsed.get("description", ""),
            instructions=parsed.get("instructions"),
            input_schema=cls._json_to_pydantic(parsed.get("input_schema", {}), "Input"),
            output_schema=cls._json_to_pydantic(parsed.get("output_schema", {}), "Output"),
            executor=None,
            skill_md_path=source_path,
            skill_dir=skill_dir,
            constraints=parsed.get("constraints", {}),
            version=parsed.get("version"),
            author=parsed.get("author"),
            user_invocable=parsed.get("user-invocable", parsed.get("user_invocable", True)),
            allowed_tools=parsed.get("allowed-tools", parsed.get("allowed_tools", [])),
            preprocessors=_parse_preprocessors(parsed.get("preprocessors", [])),
            hooks=parsed.get("hooks", []),
            invocation_policy=invocation_policy,
            permissions=permissions,
            runtime_contract=runtime_contract,
            extra_frontmatter=extra_frontmatter or {},
        )

    @staticmethod
    def _download_skill_content(url: str) -> str:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.read().decode("utf-8")

    @staticmethod
    def _cache_downloaded_skill(url: str, content: str, cache: bool) -> str | None:
        if not cache:
            return None
        cache_dir = Path.home() / ".houyi" / "skill_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        parsed_url = urlparse(url)
        filename = Path(parsed_url.path).name or "skill.md"
        cache_path = cache_dir / filename
        cache_path.write_text(content, encoding="utf-8")
        return str(cache_path)

    @staticmethod
    def _resolve_literal_field_type(prop: dict[str, Any], enum_values: list[Any] | None) -> Any:
        field_type: Any = SkillSpec._json_type_to_python(prop.get("type", "string"))
        if not isinstance(enum_values, list) or not enum_values:
            return field_type
        try:
            return Literal.__getitem__(tuple(enum_values))
        except TypeError:
            return SkillSpec._json_type_to_python(prop.get("type", "string"))

    @staticmethod
    def _build_field_kwargs(prop: dict[str, Any], enum_values: list[Any] | None) -> dict[str, Any]:
        field_kwargs: dict[str, Any] = {}
        description = prop.get("description")
        if isinstance(description, str) and description:
            field_kwargs["description"] = description
        minimum = prop.get("minimum")
        if isinstance(minimum, int | float):
            field_kwargs["ge"] = minimum
        maximum = prop.get("maximum")
        if isinstance(maximum, int | float):
            field_kwargs["le"] = maximum

        json_schema_extra: dict[str, Any] = {}
        if isinstance(enum_values, list) and enum_values:
            json_schema_extra["enum"] = enum_values
        fmt = prop.get("format")
        if isinstance(fmt, str) and fmt:
            json_schema_extra["format"] = fmt
        if json_schema_extra:
            field_kwargs["json_schema_extra"] = json_schema_extra
        return field_kwargs

    @staticmethod
    def _build_pydantic_field(
        name: str,
        prop: dict[str, Any],
        required: list[Any],
    ) -> tuple[Any, Field]:
        enum_values = prop.get("enum")
        field_type = SkillSpec._resolve_literal_field_type(prop, enum_values)
        default = ... if name in required else prop.get("default")
        field_kwargs = SkillSpec._build_field_kwargs(prop, enum_values)
        return field_type, Field(default, **field_kwargs)

    @classmethod
    def from_file(cls, path: str, skill_dir: str | None = None) -> SkillSpec:
        """Load skill from skill.md or SKILL.md file.

        Supports both:
        - AgentSkills.io standard (skill.md without frontmatter)
        - SimpleSkill format (SKILL.md with YAML frontmatter)

        Args:
            path: Path to skill.md or SKILL.md file
            skill_dir: Optional skill directory path (auto-detected if not provided)

        Returns:
            SkillSpec instance (executor needs to be bound separately)
        """
        path_obj = Path(path)
        content = path_obj.read_text(encoding="utf-8")
        parsed = cls._parse_skill_content(content)
        invocation_policy = cls._parse_invocation_policy(parsed)
        invocation_policy = cls._maybe_build_deny_invocation_policy(parsed, invocation_policy)
        return cls._build_skill_spec(
            parsed=parsed,
            source_path=path,
            skill_dir=cls._detect_skill_dir(path_obj, skill_dir),
            runtime_contract=cls._parse_runtime_contract(parsed),
            invocation_policy=invocation_policy,
            permissions=cls._parse_permissions(parsed),
            extra_frontmatter=cls._collect_extra_frontmatter(parsed),
        )

    @classmethod
    def from_url(cls, url: str, cache: bool = True) -> SkillSpec:
        """Load skill from URL (AgentSkills.io standard).

        Args:
            url: URL to skill.md file
            cache: Whether to cache the downloaded file (default: True)

        Returns:
            SkillSpec instance (executor needs to be bound separately)

        Raises:
            urllib.error.URLError: If URL cannot be accessed
            ValueError: If content cannot be parsed
        """
        try:
            content = cls._download_skill_content(url)
            parsed = cls._parse_skill_content(content)
            invocation_policy = cls._parse_invocation_policy(parsed)
            invocation_policy = cls._maybe_build_deny_invocation_policy(parsed, invocation_policy)
            cache_path = cls._cache_downloaded_skill(url, content, cache)
            return cls._build_skill_spec(
                parsed=parsed,
                source_path=cache_path or url,
                invocation_policy=invocation_policy,
                permissions=cls._parse_permissions(parsed),
                extra_frontmatter=cls._collect_extra_frontmatter(parsed),
            )
        except urllib.error.URLError as e:
            raise urllib.error.URLError(f"Failed to load skill from {url}: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to parse skill from {url}: {e}") from e

    @classmethod
    def from_registry(
        cls,
        skill_name: str,
        version: str | None = None,
        cache: bool = True,
        base_url: str | None = None,
    ) -> SkillSpec:
        """Load a skill from a compatibility remote registry.

        This API is kept for compatibility with remote registry loading flows
        used by earlier integrations. In HouYi's current SimpleSkill
        architecture, the primary registration model is Host-side
        ProviderRegistry + local/discovered manifests.

        Args:
            skill_name: Name of the skill (e.g., "web_search")
            version: Optional version (e.g., "v1.0.0"), defaults to latest
            cache: Whether to cache the downloaded file (default: True)
            base_url: Optional registry base URL override. If not provided,
                reads ``HOUYI_SKILL_REGISTRY_BASE_URL``.

        Returns:
            SkillSpec instance (executor needs to be bound separately)

        Example:
            >>> skill = SkillSpec.from_registry("web_search")
            >>> skill = SkillSpec.from_registry("web_search", version="v1.0.0")
        """
        resolved_base_url = (base_url or os.getenv("HOUYI_SKILL_REGISTRY_BASE_URL") or "").strip()
        if not resolved_base_url:
            raise ValueError(
                "Remote skill registry base URL is not configured. "
                "Pass base_url=... or set HOUYI_SKILL_REGISTRY_BASE_URL."
            )
        resolved_base_url = resolved_base_url.rstrip("/")

        if version:
            url = f"{resolved_base_url}/{skill_name}/{version}/skill.md"
        else:
            url = f"{resolved_base_url}/{skill_name}/skill.md"

        return cls.from_url(url, cache=cache)

    @staticmethod
    def _parse_skill_md(content: str) -> dict[str, Any]:
        """Parse skill.md format."""
        result = {}

        title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
        result["name"] = title_match.group(1).strip() if title_match else "unknown"

        desc_match = re.search(r"## Description\s+(.+?)(?=##|$)", content, re.DOTALL)
        result["description"] = desc_match.group(1).strip() if desc_match else ""

        input_match = re.search(r"## Input Schema\s+```json\s+(.+?)\s+```", content, re.DOTALL)
        if input_match:
            result["input_schema"] = json.loads(input_match.group(1))

        output_match = re.search(r"## Output Schema\s+```json\s+(.+?)\s+```", content, re.DOTALL)
        if output_match:
            result["output_schema"] = json.loads(output_match.group(1))

        return result

    @staticmethod
    def _json_to_pydantic(json_schema: dict, model_name: str = "DynamicModel") -> type[BaseModel]:
        """Convert JSON Schema to Pydantic model (simplified)."""
        if not json_schema or not json_schema.get("properties"):
            return type(f"Empty{model_name}", (BaseModel,), {})

        properties = json_schema.get("properties", {})
        required = json_schema.get("required", [])

        fields = {}
        for name, prop in properties.items():
            fields[name] = SkillSpec._build_pydantic_field(name, prop, required)

        model: type[BaseModel] = create_model(model_name, **fields)  # type: ignore[call-overload]
        return model

    @staticmethod
    def _json_type_to_python(json_type: str) -> type:
        """JSON type to Python type mapping."""
        mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        return mapping.get(json_type, str)

    def export_skill_md(
        self,
        path: str,
        metadata: dict[str, Any] | None = None,
        examples: list[dict[str, Any]] | None = None,
    ) -> None:
        """Export skill to skill.md file (AgentSkills.io standard).

        Args:
            path: Path to save skill.md file
            metadata: Optional metadata (language, runtime, timeout, cost, etc.)
            examples: Optional list of examples with 'input' and 'output' keys

        Example:
            >>> skill.export_skill_md(
            ...     "skills/my_skill.md",
            ...     metadata={"language": "Python", "runtime": "async"},
            ...     examples=[{"input": {...}, "output": {...}}]
            ... )
        """
        # Build skill.md content
        content = f"# {self.name}\n\n"

        # Description
        content += "## Description\n"
        content += f"{self.description}\n\n"

        # Input Schema
        content += "## Input Schema\n"
        content += "```json\n"
        content += json.dumps(self.input_schema.model_json_schema(), indent=2)
        content += "\n```\n\n"

        # Output Schema
        content += "## Output Schema\n"
        content += "```json\n"
        content += json.dumps(self.output_schema.model_json_schema(), indent=2)
        content += "\n```\n\n"

        # Examples
        if examples:
            content += "## Examples\n\n"
            for i, example in enumerate(examples, 1):
                content += f"### Example {i}\n"
                content += "**Input:**\n"
                content += "```json\n"
                content += json.dumps(example.get("input", {}), indent=2)
                content += "\n```\n\n"
                content += "**Output:**\n"
                content += "```json\n"
                content += json.dumps(example.get("output", {}), indent=2)
                content += "\n```\n\n"

        # Implementation metadata
        if metadata:
            content += "## Implementation\n"
            for key, value in metadata.items():
                content += f"- {key.replace('_', ' ').title()}: {value}\n"
            content += "\n"

        # Constraints
        if self.constraints:
            content += "## Constraints\n"
            for key, value in self.constraints.items():
                content += f"- {key.replace('_', ' ').title()}: {value}\n"
            content += "\n"

        # Write to file
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content)


def _parse_preprocessors(raw: list[Any]) -> list[Any]:
    """Parse preprocessor entries from manifest frontmatter.

    Each entry can be a dict (parsed to PreprocessorSpec) or already a
    PreprocessorSpec instance.  Returns a list of PreprocessorSpec objects.
    """
    if not raw:
        return []
    from houyi.domain.skill.preprocessor import PreprocessorSpec

    result: list[Any] = []
    for item in raw:
        if isinstance(item, dict):
            result.append(PreprocessorSpec.from_dict(item))
        elif isinstance(item, PreprocessorSpec):
            result.append(item)
        # else: skip unknown types
    return result
