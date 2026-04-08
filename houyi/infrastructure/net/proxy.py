"""Cross-platform proxy detection shared by adapters and providers."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from urllib.request import getproxies

from houyi.infrastructure.config.env_config import (
    ENV_PROXY_URL,
    ENV_WEB_SEARCH_PROXY_POLICY,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProxyResolution:
    policy: str
    proxy_url: str | None
    proxy_source: str


def _detect_proxy_source() -> tuple[str | None, str]:
    explicit = os.getenv(ENV_PROXY_URL, "").strip()
    if explicit:
        logger.debug("Using explicit proxy (%s): %s", ENV_PROXY_URL, explicit)
        return explicit, "explicit"

    proxies = getproxies()
    proxy_url = proxies.get("https") or proxies.get("http")
    if proxy_url:
        logger.debug("System proxy detected: %s (all: %s)", proxy_url, proxies)
        return proxy_url, "system"

    logger.debug("No system proxy detected (getproxies=%s)", proxies)
    return None, "direct"


def resolve_web_search_proxy() -> ProxyResolution:
    policy_raw = (os.getenv(ENV_WEB_SEARCH_PROXY_POLICY) or "").strip().lower()
    if policy_raw:
        if policy_raw == "off":
            return ProxyResolution(policy="off", proxy_url=None, proxy_source="direct")
        if policy_raw != "auto":
            logger.warning(
                "Unsupported %s=%r; falling back to auto",
                ENV_WEB_SEARCH_PROXY_POLICY,
                policy_raw,
            )
        proxy_url, proxy_source = _detect_proxy_source()
        return ProxyResolution(policy="auto", proxy_url=proxy_url, proxy_source=proxy_source)

    proxy_url, proxy_source = _detect_proxy_source()
    return ProxyResolution(policy="auto", proxy_url=proxy_url, proxy_source=proxy_source)


def detect_proxy() -> str | None:
    """Return the best available HTTPS/HTTP proxy URL, or ``None``.

    Resolution order:
    1. ``HOUYI_PROXY_URL`` env var (explicit override)
    2. System HTTPS proxy via ``urllib.request.getproxies()``
    3. System HTTP proxy (fallback)
    """
    proxy_url, _ = _detect_proxy_source()
    return proxy_url
