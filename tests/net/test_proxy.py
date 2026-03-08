"""Proxy detection scenarios for infrastructure network helpers."""

from __future__ import annotations

from unittest.mock import patch

from houyi.infrastructure.config.env_config import ENV_PROXY_URL
from houyi.infrastructure.net import proxy as proxy_module
from houyi.infrastructure.net.proxy import detect_proxy


class TestDetectProxy:
    """Cross-platform proxy detection via urllib.request.getproxies."""

    def test_explicit_url_takes_precedence(self, monkeypatch) -> None:
        monkeypatch.setenv(ENV_PROXY_URL, "http://explicit:1234")
        with patch.object(proxy_module, "getproxies", return_value={"https": "http://system:5678"}):
            assert detect_proxy() == "http://explicit:1234"

    def test_returns_https_proxy(self, monkeypatch) -> None:
        monkeypatch.delenv(ENV_PROXY_URL, raising=False)
        with patch.object(
            proxy_module,
            "getproxies",
            return_value={"https": "http://127.0.0.1:7890", "http": "http://127.0.0.1:7890"},
        ):
            assert detect_proxy() == "http://127.0.0.1:7890"

    def test_falls_back_to_http(self, monkeypatch) -> None:
        monkeypatch.delenv(ENV_PROXY_URL, raising=False)
        with patch.object(
            proxy_module, "getproxies", return_value={"http": "http://127.0.0.1:1087"}
        ):
            assert detect_proxy() == "http://127.0.0.1:1087"

    def test_returns_none_when_no_proxy(self, monkeypatch) -> None:
        monkeypatch.delenv(ENV_PROXY_URL, raising=False)
        with patch.object(proxy_module, "getproxies", return_value={}):
            assert detect_proxy() is None

    def test_ignores_socks_only(self, monkeypatch) -> None:
        monkeypatch.delenv(ENV_PROXY_URL, raising=False)
        with patch.object(
            proxy_module,
            "getproxies",
            return_value={"socks": "socks5://127.0.0.1:1080"},
        ):
            assert detect_proxy() is None

    def test_empty_explicit_url_falls_through(self, monkeypatch) -> None:
        monkeypatch.setenv(ENV_PROXY_URL, "  ")
        with patch.object(proxy_module, "getproxies", return_value={"https": "http://sys:9999"}):
            assert detect_proxy() == "http://sys:9999"
