from __future__ import annotations

import logging
import time

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from houyi.infrastructure.config.env_config import (
    ENV_CHAT_DATA_DIR,
    ENV_CHAT_SETTINGS_PATH,
    ENV_CHAT_SYSTEM_PROMPT,
    ENV_DEEPSEEK_MODEL,
    ENV_EMBEDDING_MODEL,
    ENV_EMBEDDING_PROVIDER,
    ENV_GOOGLE_API_KEY,
    ENV_GOOGLE_APPLICATION_CREDENTIALS,
    ENV_GOOGLE_CLOUD_PROJECT,
    ENV_OPENAI_API_KEY,
    EnvConfig,
)


def _stub_startup_dependencies(app_module, monkeypatch, tmp_path) -> None:
    EnvConfig._reset()
    monkeypatch.setattr(app_module, "get_execution_engine", lambda: object())
    monkeypatch.setattr(app_module, "register_console_skills", lambda: None)

    captured: dict[str, object] = {}

    def _json_store(data_dir):
        captured["chat_data_dir"] = data_dir
        return object()

    monkeypatch.setattr(app_module, "JsonStore", _json_store)
    monkeypatch.setattr(app_module, "SettingsStore", lambda settings_path: object())
    monkeypatch.setattr(app_module, "ChatService", lambda **kwargs: object())
    monkeypatch.setattr(app_module, "register_chat_routes", lambda *args, **kwargs: APIRouter())

    # Prevent the lifespan's Research subsystem init from polluting the
    # module-level _research_service_ref global (side effect not reverted
    # by monkeypatch, leaks into subsequent tests).
    from houyi.skills.builtin import deep_research as _dr_mod

    monkeypatch.setattr(_dr_mod, "_research_service_ref", None)

    monkeypatch.setenv(ENV_CHAT_DATA_DIR, str(tmp_path / "chat-data"))
    monkeypatch.setenv(ENV_CHAT_SETTINGS_PATH, str(tmp_path / "settings.json"))
    return captured


def test_lifespan_local_failfast(monkeypatch, tmp_path) -> None:
    from houyi_studio.server.gateway import app as app_module

    _stub_startup_dependencies(app_module, monkeypatch, tmp_path)
    monkeypatch.setenv(ENV_EMBEDDING_PROVIDER, "local")

    called: dict[str, bool] = {"strict": False}

    def _fail_if_strict(*, strict_explicit: bool = False, **kwargs):
        called["strict"] = strict_explicit
        raise RuntimeError("Embedding provider 'local' from env is unavailable at runtime")

    monkeypatch.setattr(app_module, "resolve_embedding_config", _fail_if_strict)

    app = FastAPI(lifespan=app_module.lifespan)
    try:
        with TestClient(app):
            raise AssertionError(
                "startup should fail when explicit local embedding runtime is unavailable"
            )
    except RuntimeError as exc:
        assert "Embedding provider 'local'" in str(exc)

    assert called["strict"] is True


def test_lifespan_auto_embedding(monkeypatch, tmp_path) -> None:
    from houyi_studio.server.gateway import app as app_module

    _stub_startup_dependencies(app_module, monkeypatch, tmp_path)
    monkeypatch.delenv(ENV_EMBEDDING_PROVIDER, raising=False)

    called: dict[str, bool] = {"strict": True}

    def _no_provider(*, strict_explicit: bool = False, **kwargs):
        called["strict"] = strict_explicit
        return None, "no_provider"

    monkeypatch.setattr(app_module, "resolve_embedding_config", _no_provider)

    app = FastAPI(lifespan=app_module.lifespan)
    with TestClient(app):
        pass

    assert called["strict"] is False


def test_lifespan_resolver_failfast(monkeypatch, tmp_path, caplog) -> None:
    from houyi_studio.server.gateway import app as app_module

    _stub_startup_dependencies(app_module, monkeypatch, tmp_path)
    monkeypatch.setenv(ENV_EMBEDDING_PROVIDER, "local")
    monkeypatch.delenv(ENV_EMBEDDING_MODEL, raising=False)

    from houyi_studio.server.rag import embedding_config as embedding_config_module

    monkeypatch.setattr(
        app_module, "resolve_embedding_config", embedding_config_module.resolve_embedding_config
    )
    monkeypatch.setattr(
        embedding_config_module,
        "_is_provider_runtime_available",
        lambda provider: provider != "local",
    )

    app = FastAPI(lifespan=app_module.lifespan)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(
            RuntimeError, match="Embedding provider 'local' from env is unavailable at runtime"
        ):
            with TestClient(app):
                pass

    assert "Embedding config validation failed" in caplog.text
    assert "Embedding env snapshot:" in caplog.text
    assert "Explicit EMBEDDING_PROVIDER=local requires fastembed" in caplog.text
    assert (
        "Quick fallback (auto-detect): unset EMBEDDING_PROVIDER && unset EMBEDDING_MODEL"
        in caplog.text
    )


def test_lifespan_warns_provider(monkeypatch, tmp_path, caplog) -> None:
    from houyi_studio.server.gateway import app as app_module

    _stub_startup_dependencies(app_module, monkeypatch, tmp_path)

    # Ensure true auto mode and no provider hints from host env.
    for env_key in (
        ENV_EMBEDDING_PROVIDER,
        ENV_EMBEDDING_MODEL,
        ENV_GOOGLE_API_KEY,
        ENV_GOOGLE_CLOUD_PROJECT,
        ENV_GOOGLE_APPLICATION_CREDENTIALS,
        ENV_OPENAI_API_KEY,
    ):
        monkeypatch.delenv(env_key, raising=False)

    monkeypatch.setenv(ENV_DEEPSEEK_MODEL, "deepseek-chat")
    monkeypatch.setenv(ENV_CHAT_SYSTEM_PROMPT, "")

    from houyi_studio.server.rag import embedding_config as embedding_config_module

    monkeypatch.setattr(
        app_module, "resolve_embedding_config", embedding_config_module.resolve_embedding_config
    )
    monkeypatch.setattr(
        embedding_config_module, "_auto_detect_embedding", lambda: (None, "no_provider")
    )

    app = FastAPI(lifespan=app_module.lifespan)
    with caplog.at_level(logging.WARNING):
        with TestClient(app):
            pass

    assert "No embedding provider detected at startup" in caplog.text


def test_lifespan_timeout_degrades(monkeypatch, tmp_path, caplog) -> None:
    from houyi_studio.server.gateway import app as app_module

    _stub_startup_dependencies(app_module, monkeypatch, tmp_path)
    monkeypatch.delenv(ENV_EMBEDDING_PROVIDER, raising=False)
    monkeypatch.setenv("HOUYI_EMBEDDING_STARTUP_TIMEOUT_SECONDS", "0.01")

    def _slow_resolve(*, strict_explicit: bool = False, **kwargs):
        time.sleep(0.2)
        return None, "no_provider"

    monkeypatch.setattr(app_module, "resolve_embedding_config", _slow_resolve)

    app = FastAPI(lifespan=app_module.lifespan)
    with caplog.at_level(logging.WARNING):
        with TestClient(app):
            pass

    assert "Embedding config resolution timed out" in caplog.text
    assert "No embedding provider detected at startup" in caplog.text


def test_lifespan_default_chat_dir(monkeypatch, tmp_path) -> None:
    from houyi_studio.server.gateway import app as app_module

    captured = _stub_startup_dependencies(app_module, monkeypatch, tmp_path)
    monkeypatch.delenv(ENV_CHAT_DATA_DIR, raising=False)
    expected = tmp_path / "project-root" / "data/conversations"
    monkeypatch.setattr(app_module, "resolve_chat_data_dir", lambda value: expected)

    app = FastAPI(lifespan=app_module.lifespan)
    with TestClient(app):
        pass

    assert captured["chat_data_dir"] == expected


def test_lifespan_relative_chat_dir(monkeypatch, tmp_path) -> None:
    from houyi_studio.server.gateway import app as app_module

    captured = _stub_startup_dependencies(app_module, monkeypatch, tmp_path)
    monkeypatch.setenv(ENV_CHAT_DATA_DIR, "custom/chat-data")
    expected = tmp_path / "project-root" / "custom/chat-data"
    monkeypatch.setattr(app_module, "resolve_chat_data_dir", lambda value: expected)

    app = FastAPI(lifespan=app_module.lifespan)
    with TestClient(app):
        pass

    assert captured["chat_data_dir"] == expected


def test_lifespan_absolute_chat_dir(monkeypatch, tmp_path) -> None:
    from houyi_studio.server.gateway import app as app_module

    captured = _stub_startup_dependencies(app_module, monkeypatch, tmp_path)
    absolute = tmp_path / "absolute-chat-data"
    monkeypatch.setenv(ENV_CHAT_DATA_DIR, str(absolute))

    app = FastAPI(lifespan=app_module.lifespan)
    with TestClient(app):
        pass

    assert captured["chat_data_dir"] == absolute
