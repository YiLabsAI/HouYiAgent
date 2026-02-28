"""Path resolution helpers for managed skill storage.

Goals:
- Keep managed skill state under a user-writable directory.
- Prefer explicit env overrides when provided.
- Fall back safely if home directory is unavailable/unwritable.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

ENV_HOUYI_HOME_DIR = "HOUYI_HOME_DIR"
ENV_MANAGED_SKILLS_DIR = "HOUYI_MANAGED_SKILLS_DIR"

_DEFAULT_HOUYI_DIRNAME = ".houyi"
_SKILLS_SUBDIR = "skills"
_SOURCES_HOME_SUBDIR = "sources"
_GITHUB_SOURCES_SUBDIR = "github.com"
_LOCAL_SOURCES_SUBDIR = "local"
_SKILL_CACHE_SUBDIR = "skill_cache"


def _ensure_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        return False
    return os.access(path, os.W_OK | os.X_OK)


def resolve_houyi_home_dir() -> Path:
    configured = os.getenv(ENV_HOUYI_HOME_DIR, "").strip()
    if configured:
        return Path(configured).expanduser()

    default_home = Path.home() / _DEFAULT_HOUYI_DIRNAME
    if _ensure_writable_dir(default_home):
        return default_home

    fallback = Path(tempfile.gettempdir()) / _DEFAULT_HOUYI_DIRNAME
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def resolve_managed_skills_dir() -> Path:
    configured = os.getenv(ENV_MANAGED_SKILLS_DIR, "").strip()
    if configured:
        return Path(configured).expanduser()
    return resolve_houyi_home_dir() / _SKILLS_SUBDIR


def resolve_managed_sources_home() -> Path:
    configured_skills = os.getenv(ENV_MANAGED_SKILLS_DIR, "").strip()
    if configured_skills and not os.getenv(ENV_HOUYI_HOME_DIR, "").strip():
        return Path(configured_skills).expanduser().parent / _SOURCES_HOME_SUBDIR
    return resolve_houyi_home_dir() / _SOURCES_HOME_SUBDIR


def resolve_managed_sources_root() -> Path:
    return resolve_managed_sources_home() / _GITHUB_SOURCES_SUBDIR


def resolve_managed_local_sources_root() -> Path:
    return resolve_managed_sources_home() / _LOCAL_SOURCES_SUBDIR


def resolve_skill_cache_dir() -> Path:
    return resolve_houyi_home_dir() / _SKILL_CACHE_SUBDIR
