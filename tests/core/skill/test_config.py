"""Tests for skill configuration management."""

from __future__ import annotations

import os
from unittest.mock import patch

from houyi.core.skill.config import (
    DEFAULT_SKILL_CONFIG,
    ConsentConfig,
    HookConfig,
    MetricsConfig,
    PolicyConfig,
    SkillConfig,
    _get_bool_env,
    _get_float_env,
    _get_int_env,
    get_skill_config,
    load_skill_config_from_env,
)


class TestHookConfig:
    """Tests for HookConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = HookConfig()
        assert config.timeout_seconds == 30.0
        assert config.max_concurrent == 10
        assert config.fail_on_error is False

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = HookConfig(
            timeout_seconds=60.0,
            max_concurrent=5,
            fail_on_error=True,
        )
        assert config.timeout_seconds == 60.0
        assert config.max_concurrent == 5
        assert config.fail_on_error is True


class TestConsentConfig:
    """Tests for ConsentConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ConsentConfig()
        assert config.cache_ttl_seconds == 3600.0
        assert config.require_explicit_consent is True
        assert config.auto_deny_timeout is False
        assert config.consent_prompt_timeout_seconds == 60.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ConsentConfig(
            cache_ttl_seconds=1800.0,
            require_explicit_consent=False,
            auto_deny_timeout=True,
            consent_prompt_timeout_seconds=30.0,
        )
        assert config.cache_ttl_seconds == 1800.0
        assert config.require_explicit_consent is False
        assert config.auto_deny_timeout is True
        assert config.consent_prompt_timeout_seconds == 30.0


class TestMetricsConfig:
    """Tests for MetricsConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = MetricsConfig()
        assert config.enabled is True
        assert config.export_interval_seconds == 60.0
        assert config.max_samples_per_skill == 1000
        assert config.export_path is None

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = MetricsConfig(
            enabled=False,
            export_interval_seconds=120.0,
            max_samples_per_skill=500,
            export_path="/tmp/metrics.json",
        )
        assert config.enabled is False
        assert config.export_interval_seconds == 120.0
        assert config.max_samples_per_skill == 500
        assert config.export_path == "/tmp/metrics.json"


class TestPolicyConfig:
    """Tests for PolicyConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = PolicyConfig()
        assert config.default_model_auto_invoke == "allow_with_consent"
        assert config.strict_mode is False
        assert config.allow_unknown_skills is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = PolicyConfig(
            default_model_auto_invoke="deny",
            strict_mode=True,
            allow_unknown_skills=False,
        )
        assert config.default_model_auto_invoke == "deny"
        assert config.strict_mode is True
        assert config.allow_unknown_skills is False


class TestSkillConfig:
    """Tests for SkillConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SkillConfig()
        assert isinstance(config.hooks, HookConfig)
        assert isinstance(config.consent, ConsentConfig)
        assert isinstance(config.metrics, MetricsConfig)
        assert isinstance(config.policy, PolicyConfig)

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        config = SkillConfig()
        d = config.to_dict()

        assert "hooks" in d
        assert d["hooks"]["timeout_seconds"] == 30.0
        assert "consent" in d
        assert d["consent"]["cache_ttl_seconds"] == 3600.0
        assert "metrics" in d
        assert d["metrics"]["enabled"] is True
        assert "policy" in d
        assert d["policy"]["default_model_auto_invoke"] == "allow_with_consent"

    def test_from_env_with_defaults(self) -> None:
        """Test loading from environment with defaults (no env vars set)."""
        with patch.dict(os.environ, {}, clear=True):
            config = SkillConfig.from_env()
            assert config.hooks.timeout_seconds == 30.0
            assert config.consent.cache_ttl_seconds == 3600.0
            assert config.metrics.enabled is True

    def test_from_env_with_custom_values(self) -> None:
        """Test loading from environment with custom values."""
        env_vars = {
            "HOUYI_HOOK_TIMEOUT": "60",
            "HOUYI_HOOK_MAX_CONCURRENT": "20",
            "HOUYI_HOOK_FAIL_ON_ERROR": "true",
            "HOUYI_CONSENT_CACHE_TTL": "1800",
            "HOUYI_CONSENT_REQUIRE_EXPLICIT": "false",
            "HOUYI_CONSENT_AUTO_DENY_TIMEOUT": "yes",
            "HOUYI_CONSENT_PROMPT_TIMEOUT": "45",
            "HOUYI_METRICS_ENABLED": "false",
            "HOUYI_METRICS_EXPORT_INTERVAL": "120",
            "HOUYI_METRICS_MAX_SAMPLES": "500",
            "HOUYI_METRICS_EXPORT_PATH": "/tmp/metrics.json",
            "HOUYI_POLICY_DEFAULT_AUTO_INVOKE": "deny",
            "HOUYI_POLICY_STRICT_MODE": "1",
            "HOUYI_POLICY_ALLOW_UNKNOWN": "false",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = SkillConfig.from_env()

            # Check hooks
            assert config.hooks.timeout_seconds == 60.0
            assert config.hooks.max_concurrent == 20
            assert config.hooks.fail_on_error is True

            # Check consent
            assert config.consent.cache_ttl_seconds == 1800.0
            assert config.consent.require_explicit_consent is False
            assert config.consent.auto_deny_timeout is True
            assert config.consent.consent_prompt_timeout_seconds == 45.0

            # Check metrics
            assert config.metrics.enabled is False
            assert config.metrics.export_interval_seconds == 120.0
            assert config.metrics.max_samples_per_skill == 500
            assert config.metrics.export_path == "/tmp/metrics.json"

            # Check policy
            assert config.policy.default_model_auto_invoke == "deny"
            assert config.policy.strict_mode is True
            assert config.policy.allow_unknown_skills is False


class TestEnvHelpers:
    """Tests for environment variable helper functions."""

    def test_get_float_env_with_value(self) -> None:
        """Test getting float from environment."""
        with patch.dict(os.environ, {"TEST_FLOAT": "42.5"}):
            assert _get_float_env("TEST_FLOAT", 0.0) == 42.5

    def test_get_float_env_with_default(self) -> None:
        """Test getting float with default value."""
        with patch.dict(os.environ, {}, clear=True):
            assert _get_float_env("TEST_FLOAT", 10.5) == 10.5

    def test_get_float_env_with_invalid(self) -> None:
        """Test getting float with invalid value returns default."""
        with patch.dict(os.environ, {"TEST_FLOAT": "not_a_number"}):
            assert _get_float_env("TEST_FLOAT", 5.0) == 5.0

    def test_get_int_env_with_value(self) -> None:
        """Test getting int from environment."""
        with patch.dict(os.environ, {"TEST_INT": "42"}):
            assert _get_int_env("TEST_INT", 0) == 42

    def test_get_int_env_with_default(self) -> None:
        """Test getting int with default value."""
        with patch.dict(os.environ, {}, clear=True):
            assert _get_int_env("TEST_INT", 10) == 10

    def test_get_int_env_with_invalid(self) -> None:
        """Test getting int with invalid value returns default."""
        with patch.dict(os.environ, {"TEST_INT": "not_a_number"}):
            assert _get_int_env("TEST_INT", 5) == 5

    def test_get_bool_env_true_values(self) -> None:
        """Test getting bool with various true values."""
        for value in ["1", "true", "TRUE", "yes", "YES", "on", "ON"]:
            with patch.dict(os.environ, {"TEST_BOOL": value}):
                assert _get_bool_env("TEST_BOOL", False) is True

    def test_get_bool_env_false_values(self) -> None:
        """Test getting bool with various false values."""
        for value in ["0", "false", "FALSE", "no", "NO", "off", "OFF", ""]:
            with patch.dict(os.environ, {"TEST_BOOL": value}):
                assert _get_bool_env("TEST_BOOL", True) is False

    def test_get_bool_env_with_default(self) -> None:
        """Test getting bool with default value."""
        with patch.dict(os.environ, {}, clear=True):
            assert _get_bool_env("TEST_BOOL", True) is True
            assert _get_bool_env("TEST_BOOL", False) is False


class TestGlobalConfig:
    """Tests for global configuration functions."""

    def test_get_skill_config_returns_default(self) -> None:
        """Test that get_skill_config returns the default instance."""
        config = get_skill_config()
        assert isinstance(config, SkillConfig)
        assert config is DEFAULT_SKILL_CONFIG

    def test_load_skill_config_from_env_updates_global(self) -> None:
        """Test that load_skill_config_from_env updates the global default."""
        with patch.dict(os.environ, {"HOUYI_HOOK_TIMEOUT": "99"}, clear=True):
            config = load_skill_config_from_env()
            assert config.hooks.timeout_seconds == 99.0
            # Note: This modifies the global, so subsequent tests may be affected
