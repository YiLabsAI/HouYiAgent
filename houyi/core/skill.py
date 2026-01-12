"""Skill specification and execution."""

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, create_model


class SkillSpec(BaseModel):
    """Specification for a skill (capability) that an agent can use.

    Supports AgentSkills.io standard (skill.md format).
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
    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="SLA constraints (timeout, cost, etc.)",
    )
    verification_config: Any = Field(
        default=None,
        description="Skill-level verification configuration",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (output_type, etc.)",
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
    def from_file(cls, path: str) -> "SkillSpec":
        """Load skill from skill.md file (AgentSkills.io standard).

        Args:
            path: Path to skill.md file

        Returns:
            SkillSpec instance (executor needs to be bound separately)
        """
        content = Path(path).read_text()
        parsed = cls._parse_skill_md(content)

        return cls(
            name=parsed["name"],
            description=parsed["description"],
            input_schema=cls._json_to_pydantic(parsed.get("input_schema", {}), "Input"),
            output_schema=cls._json_to_pydantic(parsed.get("output_schema", {}), "Output"),
            executor=None,
            skill_md_path=path,
            constraints=parsed.get("constraints", {}),
        )

    @classmethod
    def from_url(cls, url: str, cache: bool = True) -> "SkillSpec":
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

            # Parse content
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

            return cls(
                name=parsed["name"],
                description=parsed["description"],
                input_schema=cls._json_to_pydantic(parsed.get("input_schema", {}), "Input"),
                output_schema=cls._json_to_pydantic(parsed.get("output_schema", {}), "Output"),
                executor=None,
                skill_md_path=cache_path or url,
                constraints=parsed.get("constraints", {}),
            )
        except urllib.error.URLError as e:
            raise urllib.error.URLError(f"Failed to load skill from {url}: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to parse skill from {url}: {e}") from e

    @classmethod
    def from_registry(
        cls, skill_name: str, version: str | None = None, cache: bool = True
    ) -> "SkillSpec":
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
