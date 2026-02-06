"""Skill configuration management for SimpleSkill v0.1.

This module provides centralized configuration for the skill system with:
- Environment variable support
- Default values
- Runtime configuration updates

Reference: SimpleSkill Specification v0.1 Section 5 (Host Runtime API)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HookConfig:
    """Configuration for skill hooks execution."""

    timeout_seconds: float = 30.0
    """Maximum time to wait for a hook to complete."""

    max_concurrent: int = 10
    """Maximum concurrent hook executions."""

    fail_on_error: bool = False
    """Whether to fail the operation if a hook fails."""


@dataclass
class ConsentConfig:
    """Configuration for consent management."""

    cache_ttl_seconds: float = 3600.0
    """Time-to-live for cached consent decisions."""

    require_explicit_consent: bool = True
    """Whether to require explicit consent for sensitive operations."""

    auto_deny_timeout: bool = False
    """Whether to auto-deny if consent prompt times out."""

    consent_prompt_timeout_seconds: float = 60.0
    """Timeout for interactive consent prompts."""


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""

    enabled: bool = True
    """Whether metrics collection is enabled."""

    export_interval_seconds: float = 60.0
    """Interval for automatic metrics export."""

    max_samples_per_skill: int = 1000
    """Maximum latency samples to keep per skill."""

    export_path: str | None = None
    """Path for metrics export (None = no auto-export)."""


@dataclass
class PolicyConfig:
    """Configuration for policy enforcement."""

    default_model_auto_invoke: str = "allow_with_consent"
    """Default modelAutoInvoke policy: allow | deny | allow_with_consent."""

    strict_mode: bool = False
    """Whether to use strict policy enforcement."""

    allow_unknown_skills: bool = True
    """Whether to allow invocation of skills without explicit policies."""


@dataclass
class SkillConfig:
    """Central configuration for the skill system.

    Load from environment variables with from_env() or create with defaults.

    Environment variables:
        HOUYI_HOOK_TIMEOUT: Hook execution timeout in seconds (default: 30)
        HOUYI_HOOK_MAX_CONCURRENT: Max concurrent hooks (default: 10)
        HOUYI_CONSENT_CACHE_TTL: Consent cache TTL in seconds (default: 3600)
        HOUYI_CONSENT_REQUIRE_EXPLICIT: Require explicit consent (default: true)
        HOUYI_METRICS_ENABLED: Enable metrics collection (default: true)
        HOUYI_METRICS_EXPORT_PATH: Path for metrics export (optional)
        HOUYI_POLICY_DEFAULT_AUTO_INVOKE: Default auto-invoke policy (default: allow_with_consent)
        HOUYI_POLICY_STRICT_MODE: Enable strict policy mode (default: false)
    """

    hooks: HookConfig = field(default_factory=HookConfig)
    """Hook execution configuration."""

    consent: ConsentConfig = field(default_factory=ConsentConfig)
    """Consent management configuration."""

    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    """Metrics collection configuration."""

    policy: PolicyConfig = field(default_factory=PolicyConfig)
    """Policy enforcement configuration."""

    @classmethod
    def from_env(cls) -> SkillConfig:
        """Create configuration from environment variables.

        Returns:
            SkillConfig instance with values from environment or defaults.
        """
        return cls(
            hooks=HookConfig(
                timeout_seconds=_get_float_env("HOUYI_HOOK_TIMEOUT", 30.0),
                max_concurrent=_get_int_env("HOUYI_HOOK_MAX_CONCURRENT", 10),
                fail_on_error=_get_bool_env("HOUYI_HOOK_FAIL_ON_ERROR", False),
            ),
            consent=ConsentConfig(
                cache_ttl_seconds=_get_float_env("HOUYI_CONSENT_CACHE_TTL", 3600.0),
                require_explicit_consent=_get_bool_env("HOUYI_CONSENT_REQUIRE_EXPLICIT", True),
                auto_deny_timeout=_get_bool_env("HOUYI_CONSENT_AUTO_DENY_TIMEOUT", False),
                consent_prompt_timeout_seconds=_get_float_env("HOUYI_CONSENT_PROMPT_TIMEOUT", 60.0),
            ),
            metrics=MetricsConfig(
                enabled=_get_bool_env("HOUYI_METRICS_ENABLED", True),
                export_interval_seconds=_get_float_env("HOUYI_METRICS_EXPORT_INTERVAL", 60.0),
                max_samples_per_skill=_get_int_env("HOUYI_METRICS_MAX_SAMPLES", 1000),
                export_path=os.getenv("HOUYI_METRICS_EXPORT_PATH"),
            ),
            policy=PolicyConfig(
                default_model_auto_invoke=os.getenv(
                    "HOUYI_POLICY_DEFAULT_AUTO_INVOKE", "allow_with_consent"
                ),
                strict_mode=_get_bool_env("HOUYI_POLICY_STRICT_MODE", False),
                allow_unknown_skills=_get_bool_env("HOUYI_POLICY_ALLOW_UNKNOWN", True),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "hooks": {
                "timeout_seconds": self.hooks.timeout_seconds,
                "max_concurrent": self.hooks.max_concurrent,
                "fail_on_error": self.hooks.fail_on_error,
            },
            "consent": {
                "cache_ttl_seconds": self.consent.cache_ttl_seconds,
                "require_explicit_consent": self.consent.require_explicit_consent,
                "auto_deny_timeout": self.consent.auto_deny_timeout,
                "consent_prompt_timeout_seconds": self.consent.consent_prompt_timeout_seconds,
            },
            "metrics": {
                "enabled": self.metrics.enabled,
                "export_interval_seconds": self.metrics.export_interval_seconds,
                "max_samples_per_skill": self.metrics.max_samples_per_skill,
                "export_path": self.metrics.export_path,
            },
            "policy": {
                "default_model_auto_invoke": self.policy.default_model_auto_invoke,
                "strict_mode": self.policy.strict_mode,
                "allow_unknown_skills": self.policy.allow_unknown_skills,
            },
        }


def _get_float_env(key: str, default: float) -> float:
    """Get float value from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_int_env(key: str, default: int) -> int:
    """Get integer value from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_bool_env(key: str, default: bool) -> bool:
    """Get boolean value from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


# Global default configuration instance
DEFAULT_SKILL_CONFIG: SkillConfig = SkillConfig()


def get_skill_config() -> SkillConfig:
    """Get the current skill configuration.

    Returns:
        The global default configuration instance.
    """
    return DEFAULT_SKILL_CONFIG


def load_skill_config_from_env() -> SkillConfig:
    """Load skill configuration from environment and update global default.

    Returns:
        Updated configuration instance.
    """
    global DEFAULT_SKILL_CONFIG
    DEFAULT_SKILL_CONFIG = SkillConfig.from_env()
    return DEFAULT_SKILL_CONFIG
