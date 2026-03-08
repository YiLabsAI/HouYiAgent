"""Tests for SettingsStore: file-based global settings persistence."""

import json
from pathlib import Path

import pytest
from houyi_studio.server.chat.settings_store import (
    DefaultsConfig,
    ProviderConfig,
    SettingsStore,
)

from houyi.adapters.llm.models import (
    DEEPSEEK_R1,
    DEEPSEEK_V3,
    GPT_4O,
    PROVIDER_DISPLAY_NAMES,
    PROVIDER_OPENAI,
    PROVIDER_SILICONFLOW,
)


@pytest.fixture
def store(tmp_path: Path) -> SettingsStore:
    return SettingsStore(settings_path=tmp_path / "settings.json")


class TestSettingsStoreBasic:
    def test_get_creates_defaults(self, store: SettingsStore):
        settings = store.get()
        assert settings.version == 1
        assert settings.defaults.temperature == 0.7
        assert settings.defaults.stream is True
        assert settings.display.user_name == "You"

    def test_get_returns_cached(self, store: SettingsStore):
        s1 = store.get()
        s2 = store.get()
        assert s1 is s2

    def test_update_persists(self, store: SettingsStore, tmp_path: Path):
        settings = store.get()
        settings.defaults.model = GPT_4O
        settings.display.user_name = "Alice"
        store.update(settings)

        # Re-read from disk
        store2 = SettingsStore(settings_path=tmp_path / "settings.json")
        reloaded = store2.get()
        assert reloaded.defaults.model == GPT_4O
        assert reloaded.display.user_name == "Alice"

    def test_update_sets_updated_at(self, store: SettingsStore):
        settings = store.get()
        old_ts = settings.updated_at
        settings.defaults.model = "test"
        updated = store.update(settings)
        assert updated.updated_at >= old_ts


class TestSettingsStoreProviders:
    def test_get_available_models_empty(self, store: SettingsStore, monkeypatch):
        # Isolate from host env — SILICONFLOW_API_KEY auto-includes env models
        monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
        models = store.get_available_models()
        assert models == []

    def test_get_available_models(self, store: SettingsStore):
        settings = store.get()
        sf_name = PROVIDER_DISPLAY_NAMES[PROVIDER_SILICONFLOW]
        openai_name = PROVIDER_DISPLAY_NAMES[PROVIDER_OPENAI]
        settings.providers = [
            ProviderConfig(
                id=PROVIDER_SILICONFLOW,
                name=sf_name,
                models=[DEEPSEEK_V3, DEEPSEEK_R1],
                enabled=True,
            ),
            ProviderConfig(
                id=PROVIDER_OPENAI,
                name=openai_name,
                models=[GPT_4O],
                enabled=False,
            ),
        ]
        store.update(settings)

        models = store.get_available_models()
        assert len(models) == 2
        assert models[0]["model"] == DEEPSEEK_V3
        assert models[0]["provider"] == sf_name
        # Disabled provider should be excluded
        assert all(m["provider"] != openai_name for m in models)


class TestSettingsStorePartialUpdate:
    def test_update_partial_nested(self, store: SettingsStore):
        store.get()  # Initialize
        updated = store.update_partial({"defaults.model": GPT_4O})
        assert updated.defaults.model == GPT_4O
        # Other fields unchanged
        assert updated.defaults.temperature == 0.7

    def test_update_partial_display(self, store: SettingsStore):
        store.get()
        updated = store.update_partial({"display.user_name": "Bob"})
        assert updated.display.user_name == "Bob"


class TestSettingsStoreStreamField:
    """Test stream field in DefaultsConfig."""

    def test_defaults_stream_true(self):
        defaults = DefaultsConfig()
        assert defaults.stream is True

    def test_defaults_stream_false(self):
        defaults = DefaultsConfig(stream=False)
        assert defaults.stream is False

    def test_stream_persists(self, store: SettingsStore, tmp_path):
        settings = store.get()
        settings.defaults.stream = False
        store.update(settings)

        store2 = SettingsStore(settings_path=tmp_path / "settings.json")
        reloaded = store2.get()
        assert reloaded.defaults.stream is False

    def test_stream_backward_compat_missing_field(self, tmp_path):
        """Old settings.json without stream field should default to True."""
        path = tmp_path / "settings.json"
        data = {
            "version": 1,
            "providers": [],
            "defaults": {
                "model": "",
                "system_instructions": "",
                "temperature": 0.7,
                "max_tokens": 4096,
            },
            "display": {},
            "updated_at": 0,
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        store = SettingsStore(settings_path=path)
        settings = store.get()
        assert settings.defaults.stream is True


class TestSettingsStoreAtomicWrite:
    def test_no_tmp_file_left(self, store: SettingsStore, tmp_path: Path):
        settings = store.get()
        store.update(settings)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_corrupt_file_returns_defaults(self, tmp_path: Path):
        path = tmp_path / "settings.json"
        path.write_text("not valid json", encoding="utf-8")
        store = SettingsStore(settings_path=path)
        settings = store.get()
        assert settings.version == 1
