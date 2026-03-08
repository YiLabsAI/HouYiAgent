"""Observability configuration.

Provides privacy-first configuration for span data collection.
By default, sensitive content (prompts, responses) is NOT collected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PrivacyConfig:
    """Privacy configuration for observability.

    Controls what data is collected in spans. Defaults to privacy-preserving
    settings that only collect metadata, not content.

    Attributes:
        capture_prompts: Whether to capture LLM prompts (default: False)
        capture_responses: Whether to capture LLM responses (default: False)
        capture_tool_inputs: Whether to capture tool input data (default: False)
        capture_tool_outputs: Whether to capture tool output data (default: False)
        capture_retriever_docs: Whether to capture retrieved documents (default: False)
        redact_patterns: Regex patterns to redact from any captured content
        max_content_length: Max length for any captured content (truncate beyond)
    """

    capture_prompts: bool = False
    capture_responses: bool = False
    capture_tool_inputs: bool = False
    capture_tool_outputs: bool = False
    capture_retriever_docs: bool = False
    redact_patterns: list[str] = field(default_factory=list)
    max_content_length: int = 1000

    def should_capture_llm_content(self) -> bool:
        """Check if LLM content should be captured."""
        return self.capture_prompts or self.capture_responses

    def should_capture_tool_content(self) -> bool:
        """Check if tool content should be captured."""
        return self.capture_tool_inputs or self.capture_tool_outputs


@dataclass
class ObservabilityConfig:
    """Main observability configuration.

    Attributes:
        enabled: Whether observability is enabled (default: True)
        privacy: Privacy configuration
        exporters: List of exporter configurations
        sample_rate: Sampling rate for traces (0.0-1.0, default: 1.0)
    """

    enabled: bool = True
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    exporters: list[dict[str, Any]] = field(default_factory=lambda: [{"type": "console"}])
    sample_rate: float = 1.0

    @classmethod
    def default(cls) -> ObservabilityConfig:
        """Create default configuration (privacy-preserving)."""
        return cls()

    @classmethod
    def development(cls) -> ObservabilityConfig:
        """Create development configuration (captures content for debugging)."""
        return cls(
            privacy=PrivacyConfig(
                capture_prompts=True,
                capture_responses=True,
                capture_tool_inputs=True,
                capture_tool_outputs=True,
                capture_retriever_docs=True,
            )
        )


# Global configuration instance (can be overridden)
_config: ObservabilityConfig | None = None


def get_config() -> ObservabilityConfig:
    """Get current observability configuration."""
    global _config
    if _config is None:
        _config = ObservabilityConfig.default()
    return _config


def set_config(config: ObservabilityConfig) -> None:
    """Set observability configuration."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset to default configuration."""
    global _config
    _config = None
