"""Skill specification and execution."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, create_model

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

    name: str = Field(..., description="Unique skill identifier")
    description: str = Field(..., description="Human-readable description for LLM")
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
    # Extra frontmatter fields not in the standard schema are preserved here
    extra_frontmatter: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional frontmatter fields for forward compatibility",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_tool_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema.model_json_schema(),
            },
        }

    def bind_executor(self, executor: Callable[..., Any]) -> None:
        """Bind executor function to skill (for skill.md loaded skills)."""
        self.executor = executor

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
        content = path_obj.read_text()

        # Auto-detect skill directory
        detected_skill_dir: Path | None = None
        if skill_dir:
            detected_skill_dir = Path(skill_dir)
        elif path_obj.name.upper() in ("SKILL.MD", "SKILL.md"):
            detected_skill_dir = path_obj.parent

        # Try new parser with YAML frontmatter support
        try:
            from houyi.core.skill.schema import parse_skill_md

            parsed = parse_skill_md(content)
        except ImportError:
            # Fallback to legacy parser
            parsed = cls._parse_skill_md(content)

        # Parse invocation policy from frontmatter
        invocation_policy = None
        raw_policy = parsed.get("invocationPolicy", parsed.get("invocation_policy"))
        if isinstance(raw_policy, dict):
            try:
                from houyi.core.skill.policy import InvocationPolicy

                invocation_policy = InvocationPolicy.from_dict(raw_policy)
            except (ImportError, Exception):
                invocation_policy = raw_policy

        # Parse permissions from frontmatter
        permissions = None
        raw_perms = parsed.get("permissions")
        if isinstance(raw_perms, dict):
            try:
                from houyi.core.skill.policy import Permissions

                permissions = Permissions.from_dict(raw_perms)
            except (ImportError, Exception):
                permissions = raw_perms

        # Handle disable-model-invocation (Claude standard field)
        disable_model_invocation = parsed.get(
            "disable-model-invocation",
            parsed.get("disable_model_invocation"),
        )
        if disable_model_invocation is True and invocation_policy is None:
            try:
                from houyi.core.skill.policy import InvocationPolicy, ModelAutoInvoke

                invocation_policy = InvocationPolicy(
                    model_auto_invoke=ModelAutoInvoke.DENY,
                )
            except ImportError:
                pass

        # Collect extra frontmatter fields for forward compatibility
        known_keys = {
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
        }
        extra_frontmatter = {k: v for k, v in parsed.items() if k not in known_keys}

        # Build SkillSpec
        return cls(
            name=parsed.get("name", "unknown"),
            description=parsed.get("description", ""),
            input_schema=cls._json_to_pydantic(parsed.get("input_schema", {}), "Input"),
            output_schema=cls._json_to_pydantic(parsed.get("output_schema", {}), "Output"),
            executor=None,
            skill_md_path=path,
            skill_dir=detected_skill_dir,
            constraints=parsed.get("constraints", {}),
            version=parsed.get("version"),
            author=parsed.get("author"),
            user_invocable=parsed.get("user-invocable", parsed.get("user_invocable", True)),
            allowed_tools=parsed.get("allowed-tools", parsed.get("allowed_tools", [])),
            preprocessors=_parse_preprocessors(parsed.get("preprocessors", [])),
            hooks=parsed.get("hooks", []),
            invocation_policy=invocation_policy,
            permissions=permissions,
            extra_frontmatter=extra_frontmatter,
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
            # Download content
            with urllib.request.urlopen(url, timeout=10) as response:
                content = response.read().decode("utf-8")

            # Try new parser with YAML frontmatter support
            try:
                from houyi.core.skill.schema import parse_skill_md

                parsed = parse_skill_md(content)
            except ImportError:
                # Fallback to legacy parser
                parsed = cls._parse_skill_md(content)

            # Optionally cache to local file
            cache_path = None
            if cache:
                # Create cache directory
                cache_dir = Path.home() / ".houyi" / "skill_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)

                # Generate cache filename from URL
                parsed_url = urlparse(url)
                filename = Path(parsed_url.path).name or "skill.md"
                cache_path = str(cache_dir / filename)

                # Save to cache
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(content)

            # Parse invocation policy
            invocation_policy = None
            raw_policy = parsed.get("invocationPolicy", parsed.get("invocation_policy"))
            if isinstance(raw_policy, dict):
                try:
                    from houyi.core.skill.policy import InvocationPolicy

                    invocation_policy = InvocationPolicy.from_dict(raw_policy)
                except (ImportError, Exception):
                    invocation_policy = raw_policy

            # Parse permissions
            permissions = None
            raw_perms = parsed.get("permissions")
            if isinstance(raw_perms, dict):
                try:
                    from houyi.core.skill.policy import Permissions

                    permissions = Permissions.from_dict(raw_perms)
                except (ImportError, Exception):
                    permissions = raw_perms

            # Handle disable-model-invocation
            disable_model_invocation = parsed.get(
                "disable-model-invocation",
                parsed.get("disable_model_invocation"),
            )
            if disable_model_invocation is True and invocation_policy is None:
                try:
                    from houyi.core.skill.policy import InvocationPolicy, ModelAutoInvoke

                    invocation_policy = InvocationPolicy(
                        model_auto_invoke=ModelAutoInvoke.DENY,
                    )
                except ImportError:
                    pass

            # Collect extra frontmatter fields
            known_keys = {
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
            }
            extra_frontmatter = {k: v for k, v in parsed.items() if k not in known_keys}

            return cls(
                name=parsed.get("name", "unknown"),
                description=parsed.get("description", ""),
                input_schema=cls._json_to_pydantic(parsed.get("input_schema", {}), "Input"),
                output_schema=cls._json_to_pydantic(parsed.get("output_schema", {}), "Output"),
                executor=None,
                skill_md_path=cache_path or url,
                constraints=parsed.get("constraints", {}),
                version=parsed.get("version"),
                author=parsed.get("author"),
                user_invocable=parsed.get("user-invocable", parsed.get("user_invocable", True)),
                allowed_tools=parsed.get("allowed-tools", parsed.get("allowed_tools", [])),
                preprocessors=_parse_preprocessors(parsed.get("preprocessors", [])),
                hooks=parsed.get("hooks", []),
                invocation_policy=invocation_policy,
                permissions=permissions,
                extra_frontmatter=extra_frontmatter,
            )
        except urllib.error.URLError as e:
            raise urllib.error.URLError(f"Failed to load skill from {url}: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to parse skill from {url}: {e}") from e

    @classmethod
    def from_registry(
        cls, skill_name: str, version: str | None = None, cache: bool = True
    ) -> SkillSpec:
        """Load skill from AgentSkills.io registry.

        Args:
            skill_name: Name of the skill (e.g., "web_search")
            version: Optional version (e.g., "v1.0.0"), defaults to latest
            cache: Whether to cache the downloaded file (default: True)

        Returns:
            SkillSpec instance (executor needs to be bound separately)

        Example:
            >>> skill = SkillSpec.from_registry("web_search")
            >>> skill = SkillSpec.from_registry("web_search", version="v1.0.0")
        """
        # Construct URL to AgentSkills.io registry
        base_url = "https://raw.githubusercontent.com/agentskills/skills/main"

        if version:
            url = f"{base_url}/{skill_name}/{version}/skill.md"
        else:
            url = f"{base_url}/{skill_name}/skill.md"

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
            field_type = SkillSpec._json_type_to_python(prop.get("type", "string"))
            default = ... if name in required else prop.get("default")
            fields[name] = (field_type, default)

        return create_model(model_name, **fields)

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
    from houyi.core.skill.preprocessor import PreprocessorSpec

    result: list[Any] = []
    for item in raw:
        if isinstance(item, dict):
            result.append(PreprocessorSpec.from_dict(item))
        elif isinstance(item, PreprocessorSpec):
            result.append(item)
        # else: skip unknown types
    return result
