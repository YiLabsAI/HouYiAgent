"""Settings Store: file-based persistence for global chat settings.

Stores global configuration (providers, defaults, display) in a single
settings.json file with atomic writes (write .tmp then rename).

"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_DEFAULT_SETTINGS_PATH = "data/settings.json"


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""

    id: str
    name: str
    api_key: str = ""
    base_url: str = ""
    models: list[str] = Field(default_factory=list)
    enabled: bool = True


class DefaultsConfig(BaseModel):
    """Default values for new conversations."""

    model: str = ""
    system_instructions: str = "You are a helpful assistant."
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = True


class DisplayConfig(BaseModel):
    """Display configuration for user and assistant."""

    user_name: str = "You"
    user_avatar: str | None = None
    assistant_name: str = "Assistant"
    assistant_avatar: str | None = None


class GlobalSettings(BaseModel):
    """Top-level global settings model.

    Persisted as settings.json. Provides defaults that conversations
    inherit unless overridden at the conversation level.
    """

    version: int = 1
    providers: list[ProviderConfig] = Field(default_factory=list)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    updated_at: float = Field(default_factory=time.time)


class SettingsStore:
    """File-based settings store with atomic writes.

    Thread-safe for single-writer scenarios (FastAPI async).
    Uses write-to-tmp-then-rename for crash safety.
    """

    def __init__(self, settings_path: str | Path | None = None):
        """Initialize settings store.

        Args:
            settings_path: Path to settings.json file.
                          Defaults to 'data/settings.json' relative to CWD.
        """
        self._path = Path(settings_path) if settings_path else Path(_DEFAULT_SETTINGS_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._settings: GlobalSettings | None = None

    def get(self) -> GlobalSettings:
        """Get current global settings.

        Loads from disk on first access, then caches in memory.

        Returns:
            Current GlobalSettings.
        """
        if self._settings is None:
            self._settings = self._load()
        return self._settings

    def update(self, settings: GlobalSettings) -> GlobalSettings:
        """Update global settings.

        Persists to disk with atomic write.

        Args:
            settings: Updated settings.

        Returns:
            The persisted GlobalSettings.
        """
        settings.updated_at = time.time()
        self._write(settings)
        self._settings = settings
        return settings

    def update_partial(self, updates: dict[str, Any]) -> GlobalSettings:
        """Partially update settings from a dict.

        Merges updates into the current settings. Supports nested keys
        like 'defaults.model' or 'display.user_name'.

        Args:
            updates: Dict of field updates.

        Returns:
            The updated GlobalSettings.
        """
        current = self.get()
        data = current.model_dump()

        for key, value in updates.items():
            parts = key.split(".")
            target = data
            for part in parts[:-1]:
                if part in target and isinstance(target[part], dict):
                    target = target[part]
                else:
                    break
            else:
                target[parts[-1]] = value

        updated = GlobalSettings(**data)
        return self.update(updated)

    def get_available_models(self) -> list[dict[str, str]]:
        """Get list of available models from all enabled providers.

        Also includes models from the default env-configured provider
        (SiliconFlow) if it has an API key but no explicit settings entry,
        so users always see their env-configured models in the dropdown.

        Returns:
            List of dicts with 'model' and 'provider' keys.
        """
        import os

        settings = self.get()
        models = []
        has_siliconflow = False
        for provider in settings.providers:
            if not provider.enabled:
                continue
            if "siliconflow" in provider.id.lower() or "siliconflow" in provider.name.lower():
                has_siliconflow = True
            for model_id in provider.models:
                models.append(
                    {
                        "model": model_id,
                        "provider": provider.name,
                        "provider_id": provider.id,
                    }
                )

        # Auto-include env-configured SiliconFlow models if not in settings
        if not has_siliconflow and os.getenv("SILICONFLOW_API_KEY"):
            from houyi.llm.models import DEFAULT_MODEL

            env_model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)
            env_models = [env_model]
            # Add common DeepSeek models if the default is one of them
            common = [
                "deepseek-ai/DeepSeek-V3",
                "deepseek-ai/DeepSeek-V3.1",
                "deepseek-ai/DeepSeek-R1",
            ]
            for m in common:
                if m not in env_models:
                    env_models.append(m)
            for model_id in env_models:
                models.append(
                    {
                        "model": model_id,
                        "provider": "SiliconFlow (env)",
                        "provider_id": "_env_siliconflow",
                    }
                )

        return models

    def _load(self) -> GlobalSettings:
        """Load settings from disk, or create defaults if not found."""
        if not self._path.exists():
            logger.info("No settings file found, creating defaults at %s", self._path)
            defaults = GlobalSettings()
            self._write(defaults)
            return defaults

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return GlobalSettings(**data)
        except Exception as e:
            logger.error("Failed to read settings from %s: %s", self._path, e)
            return GlobalSettings()

    def _write(self, settings: GlobalSettings) -> None:
        """Atomic write: write to .tmp then rename."""
        tmp_path = self._path.with_suffix(".json.tmp")
        data = settings.model_dump(mode="json")
        tmp_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_path.rename(self._path)
        logger.debug("Settings written to %s", self._path)
