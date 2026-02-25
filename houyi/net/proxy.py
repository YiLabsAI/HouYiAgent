"""Cross-platform proxy detection shared by LLM adapters and web search providers.

Uses ``urllib.request.getproxies()`` for detection:
* **macOS** — SystemConfiguration framework (``scutil --proxy`` equivalent)
* **Windows** — Internet Settings registry
* **Linux** — ``HTTPS_PROXY`` / ``HTTP_PROXY`` environment variables

Explicit override via ``HOUYI_PROXY_URL`` always takes precedence.
"""

from __future__ import annotations

import logging
import os
from urllib.request import getproxies

from houyi.config.env_config import ENV_PROXY_URL

logger = logging.getLogger(__name__)


def detect_proxy() -> str | None:
    """Return the best available HTTPS/HTTP proxy URL, or ``None``.

    Resolution order:
    1. ``HOUYI_PROXY_URL`` env var (explicit override)
    2. System HTTPS proxy via ``urllib.request.getproxies()``
    3. System HTTP proxy (fallback)
    """
    explicit = os.getenv(ENV_PROXY_URL, "").strip()
    if explicit:
        logger.debug("Using explicit proxy (%s): %s", ENV_PROXY_URL, explicit)
        return explicit

    proxies = getproxies()
    proxy_url = proxies.get("https") or proxies.get("http")
    if proxy_url:
        logger.debug("System proxy detected: %s (all: %s)", proxy_url, proxies)
    else:
        logger.debug("No system proxy detected (getproxies=%s)", proxies)
    return proxy_url
