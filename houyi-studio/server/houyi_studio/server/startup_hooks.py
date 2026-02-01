"""Startup hooks for console server."""

from __future__ import annotations

import logging
import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from houyi.core.skill_registry import DEFAULT_SKILL_REGISTRY
from houyi.web_search.skill import build_web_search_skill

logger = logging.getLogger(__name__)


def register_console_skills() -> None:
    """Register default skills for console runtime."""
    DEFAULT_SKILL_REGISTRY.register(build_web_search_skill(), overwrite=True)
    if _should_load_e2e_tools():
        _load_e2e_tools()


def _should_load_e2e_tools() -> bool:
    disable_e2e = (os.getenv("HOUYI_DISABLE_E2E_TOOLS") or "").strip().lower()
    return disable_e2e not in {"1", "true", "yes", "on"}


def _load_e2e_tools() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "integration"
        / "fixtures"
        / "console_e2e_tools.py"
    )
    if not script_path.exists():
        logger.warning("E2E tool script not found: %s", script_path)
        return
    spec = spec_from_file_location("console_e2e_tools", script_path)
    if spec and spec.loader:
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        logger.info("Loaded console E2E tools from %s", script_path)
