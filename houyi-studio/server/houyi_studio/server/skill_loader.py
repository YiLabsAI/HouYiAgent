"""Skill loading, unloading, and source validation.

This module owns everything related to *getting skills into and out of the
registry*: file loading, URL fetching/caching, directory scanning, content
validation, and GitHub URL normalisation.

All other concerns (serialisation, dry-run, consent, metrics) live elsewhere.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from houyi.core.skill_registry import SkillRegistry

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

# Error codes (second element of the return tuple)
ERR_FILE_NOT_FOUND = "file_not_found"
ERR_INVALID_FILE = "invalid_file"
ERR_INVALID_URL = "invalid_url"
ERR_INVALID_CONTENT = "invalid_content"
ERR_NO_SKILLS = "no_skills"
ERR_NO_FRONTMATTER = "no_frontmatter"
ERR_PARSE_FAILED = "parse_failed"
ERR_READ_FAILED = "read_failed"
ERR_LOAD_FAILED = "load_failed"
ERR_VALIDATION_FAILED = "validation_failed"
ERR_URL_HTTP_ERROR = "url_http_error"
ERR_URL_UNREACHABLE = "url_unreachable"
ERR_URL_DOWNLOAD_FAILED = "url_download_failed"
ERR_URL_LOAD_FAILED = "url_load_failed"
ERR_DUPLICATE_SKILL = "duplicate_skill"

# Skill file conventions
SKILL_MD_UPPER = "SKILL.md"
SKILL_MD_LOWER = "skill.md"
SKILL_NAME_UNKNOWN = "unknown"
FRONTMATTER_DELIMITER = "---"

# GitHub URL handling
_GITHUB_BLOB_RE = re.compile(r"https?://github\.com/([^/]+)/([^/]+)/blob/(.+)")
_GITHUB_RAW_HOST = "raw.githubusercontent.com"
_GITHUB_RAW_TEMPLATE = "https://{host}/{owner}/{repo}/{rest}"

# Cache directory
SKILL_CACHE_DIR = Path.home() / ".houyi" / "skill_cache"

# Return type alias
LoadResult = tuple[bool, str, str | None]


class SkillLoader:
    """Loads / unloads skills from various sources into a ``SkillRegistry``.

    Responsibilities (SRP):
    - Local file loading (.md, .json)
    - URL loading (with GitHub normalisation + caching)
    - Directory scanning
    - Content & parse validation
    - Unloading

    Open/Closed: new source types (e.g. S3, Git clone) can be added as new
    private methods without modifying existing ones.
    """

    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry

    # ── Public API ────────────────────────────────────────────────

    def is_loaded(self, skill_name: str) -> bool:
        """Check whether *skill_name* is currently in the registry."""
        return self._registry.get(skill_name) is not None

    def load(self, source: str) -> LoadResult:
        """Load skill(s) from *source* (file path, URL, or directory).

        Returns ``(success, skill_name_or_error_code, error_message_or_None)``.
        """
        if source.startswith("http://") or source.startswith("https://"):
            return self._load_from_url(source)

        path = Path(source)
        if not path.exists():
            return False, ERR_FILE_NOT_FOUND, f"Skill source not found: {source}"
        if path.is_dir():
            return self._load_from_directory(source)

        name_lower = path.name.lower()
        if name_lower.endswith(".json"):
            return self._load_from_manifest(source)
        if not name_lower.endswith(".md"):
            return (
                False,
                ERR_INVALID_FILE,
                (
                    f"Unsupported file type: '{path.suffix}'. "
                    f"Expected {SKILL_MD_UPPER} or simpleskill.json."
                ),
            )
        if name_lower != SKILL_MD_LOWER:
            logger.warning(
                "File '%s' does not follow the %s naming convention. Attempting to parse anyway.",
                path.name,
                SKILL_MD_UPPER,
            )
        return self._load_from_skill_md(path, source)

    def unload(self, skill_name: str) -> tuple[bool, str | None]:
        """Remove *skill_name* from the registry."""
        if not self._registry.get(skill_name):
            return False, f"Skill not found: {skill_name}"
        self._registry.unregister(skill_name)
        return True, None

    # ── Private: individual source strategies ─────────────────────

    def _load_from_manifest(self, source: str) -> LoadResult:
        try:
            names = self._registry.register_from_manifest(source, overwrite=True)
            if names:
                return True, names[0], None
            return False, ERR_NO_SKILLS, "Manifest contains no skills"
        except Exception as e:
            logger.exception("Failed to load skill from manifest: %s", source)
            return False, ERR_LOAD_FAILED, str(e)

    def _load_from_skill_md(self, path: Path, source: str) -> LoadResult:
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            return False, ERR_READ_FAILED, f"Cannot read file: {e}"

        if not content.strip().startswith(FRONTMATTER_DELIMITER):
            return (
                False,
                ERR_NO_FRONTMATTER,
                (
                    f"File '{path.name}' does not have YAML frontmatter. "
                    f"A valid {SKILL_MD_UPPER} must start with "
                    f"'{FRONTMATTER_DELIMITER}' followed by YAML metadata "
                    f"including at least a 'name' field."
                ),
            )

        try:
            skill_name = self._registry.register_from_skill_file(source, overwrite=True)
            if not skill_name or skill_name == SKILL_NAME_UNKNOWN:
                return (
                    False,
                    ERR_PARSE_FAILED,
                    (
                        f"Failed to extract skill name from {source}. "
                        f"Ensure the YAML frontmatter has a 'name' field."
                    ),
                )
            skill = self._registry.get(skill_name)
            if skill:
                _validate_parsed_skill(skill)
            return True, skill_name, None
        except ValueError as e:
            return False, ERR_VALIDATION_FAILED, str(e)
        except Exception as e:
            logger.exception("Failed to load skill from %s", source)
            return False, ERR_LOAD_FAILED, str(e)

    def _load_from_url(self, url: str) -> LoadResult:
        import urllib.error
        import urllib.request

        try:
            raw_url = normalize_github_url(url)
        except ValueError as e:
            return False, ERR_INVALID_URL, str(e)

        try:
            with urllib.request.urlopen(raw_url, timeout=15) as resp:
                content = resp.read().decode("utf-8")
                content_type = resp.headers.get("content-type", "")
        except urllib.error.HTTPError as e:
            return False, ERR_URL_HTTP_ERROR, (f"HTTP {e.code}: {e.reason} — URL: {raw_url}")
        except urllib.error.URLError as e:
            return False, ERR_URL_UNREACHABLE, (f"Cannot reach URL: {e.reason} — URL: {raw_url}")
        except Exception as e:
            return False, ERR_URL_DOWNLOAD_FAILED, f"Download failed: {e}"

        try:
            validate_skill_content(content, raw_url)
        except ValueError as e:
            return False, ERR_INVALID_CONTENT, str(e)

        if "text/html" in content_type:
            return (
                False,
                ERR_INVALID_CONTENT,
                (
                    f"Server returned HTML (content-type: {content_type}). "
                    f"Expected text/plain or text/markdown. "
                    f"Please use a raw content URL."
                ),
            )

        try:
            from houyi.core.skill.schema import parse_skill_md

            parse_skill_md(content)
        except Exception as e:
            return False, ERR_PARSE_FAILED, (f"Failed to parse {SKILL_MD_UPPER} content: {e}")

        try:
            from houyi.core.skill.spec import SkillSpec

            cache_path = _cache_url_content(raw_url, content)
            skill = SkillSpec.from_file(cache_path)
            _validate_parsed_skill(skill)
            self._registry.register(skill, overwrite=True)
            logger.info(
                "Loaded skill '%s' from URL: %s (cached: %s)",
                skill.name,
                url,
                cache_path,
            )
            return True, skill.name, None
        except ValueError as e:
            return False, ERR_VALIDATION_FAILED, str(e)
        except Exception as e:
            logger.exception("Failed to load skill from URL: %s", url)
            return False, ERR_URL_LOAD_FAILED, str(e)

    def _load_from_directory(self, directory: str) -> LoadResult:
        dir_path = Path(directory)
        all_names: list[str] = []

        for pattern in (SKILL_MD_UPPER, SKILL_MD_LOWER):
            try:
                names = self._registry.register_from_directory(
                    directory,
                    pattern=pattern,
                    recursive=True,
                    overwrite=True,
                )
                all_names.extend(n for n in names if n and n != SKILL_NAME_UNKNOWN)
            except Exception as e:
                logger.warning("Error scanning for %s in %s: %s", pattern, directory, e)

        if all_names:
            unique = list(dict.fromkeys(all_names))
            logger.info(
                "Loaded %d skills from directory %s: %s",
                len(unique),
                directory,
                ", ".join(unique),
            )
            return True, ", ".join(unique), None

        md_files = list(dir_path.rglob("*.md"))
        if md_files:
            names_found = [f.name for f in md_files[:5]]
            return (
                False,
                ERR_NO_SKILLS,
                (
                    f"No {SKILL_MD_UPPER} files found in {directory}. "
                    f"Found: {', '.join(names_found)}. "
                    f"Rename to {SKILL_MD_UPPER} with YAML frontmatter to load."
                ),
            )
        return (
            False,
            ERR_NO_SKILLS,
            (
                f"No {SKILL_MD_UPPER} files found in {directory}. "
                f"Ensure the directory contains {SKILL_MD_UPPER} files "
                f"with YAML frontmatter."
            ),
        )


# ── Free functions (stateless, importable for testing) ────────────────


def normalize_github_url(url: str) -> str:
    """Convert GitHub blob URLs to raw content URLs; others pass through."""
    m = _GITHUB_BLOB_RE.match(url)
    if m:
        owner, repo, rest = m.group(1), m.group(2), m.group(3)
        raw = _GITHUB_RAW_TEMPLATE.format(
            host=_GITHUB_RAW_HOST,
            owner=owner,
            repo=repo,
            rest=rest,
        )
        logger.info("Converted GitHub blob URL → raw: %s", raw)
        return raw
    if "/tree/" in url and "github.com" in url:
        raise ValueError(
            f"Cannot load a directory URL. Please provide a direct URL "
            f"to a {SKILL_MD_UPPER} file. Got: {url}"
        )
    return url


def validate_skill_content(content: str, url: str) -> None:
    """Raise ``ValueError`` if *content* looks like HTML or is empty."""
    stripped = content.strip()
    if stripped.startswith("<!DOCTYPE") or stripped.startswith("<html"):
        raise ValueError(
            f"URL returned an HTML page instead of a {SKILL_MD_UPPER} file. "
            f"If this is a GitHub URL, use the 'Raw' button to get the "
            f"direct link, or we will auto-convert blob URLs. URL: {url}"
        )
    if not stripped:
        raise ValueError(f"URL returned empty content: {url}")


def _validate_parsed_skill(skill: object) -> None:
    """Raise ``ValueError`` if the skill has no usable name."""
    name = getattr(skill, "name", None)
    if not name or name == SKILL_NAME_UNKNOWN:
        raise ValueError(
            f"Failed to parse skill: 'name' field not found in "
            f"{SKILL_MD_UPPER}. Ensure the file has YAML frontmatter "
            f"with a 'name' field (e.g., '---\\nname: my-skill\\n---')."
        )
    if not getattr(skill, "description", ""):
        logger.warning("Skill '%s' has no description", name)


def _cache_url_content(raw_url: str, content: str) -> str:
    """Write *content* to the skill cache dir and return the path."""
    from urllib.parse import urlparse

    SKILL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url_path = Path(urlparse(raw_url).path)
    filename = url_path.name or SKILL_MD_UPPER
    parent = url_path.parent.name
    if parent:
        filename = f"{parent}_{filename}"
    cache_path = str(SKILL_CACHE_DIR / filename)
    with open(cache_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return cache_path
