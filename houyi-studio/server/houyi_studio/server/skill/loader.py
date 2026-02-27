"""Skill loading, unloading, and source validation.

This module owns everything related to *getting skills into and out of the
registry*: file loading, URL fetching/caching, directory scanning, content
validation, and GitHub URL normalisation.

All other concerns (serialisation, dry-run, consent, metrics) live elsewhere.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from houyi.core.skill_registry import CoreToolProtectionError, SkillRegistry

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
            self._hydrate_external_runtime(names)
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

        effective_source = source
        installed_skill_path = self._install_local_skill_package(path)
        if installed_skill_path is not None:
            effective_source = str(installed_skill_path)

        try:
            try:
                skill_name = self._registry.register_from_skill_file(
                    effective_source, overwrite=True
                )
            except CoreToolProtectionError:
                skill_name = self._registry.register_from_skill_file(
                    effective_source, overwrite=False
                )
            if not skill_name or skill_name == SKILL_NAME_UNKNOWN:
                return (
                    False,
                    ERR_PARSE_FAILED,
                    (
                        f"Failed to extract skill name from {effective_source}. "
                        f"Ensure the YAML frontmatter has a 'name' field."
                    ),
                )
            skill = self._registry.get(skill_name)
            if skill:
                _validate_parsed_skill(skill)
            self._hydrate_external_runtime([skill_name])
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
            try:
                registered_name = self._registry.register(skill, overwrite=True)
            except CoreToolProtectionError:
                registered_name = self._registry.register(skill, overwrite=False)
            self._hydrate_external_runtime([registered_name])
            logger.info(
                "Loaded skill '%s' from URL: %s (cached: %s)",
                registered_name,
                url,
                cache_path,
            )
            return True, registered_name, None
        except ValueError as e:
            return False, ERR_VALIDATION_FAILED, str(e)
        except Exception as e:
            logger.exception("Failed to load skill from URL: %s", url)
            return False, ERR_URL_LOAD_FAILED, str(e)

    def _load_from_directory(self, directory: str) -> LoadResult:
        dir_path = Path(directory)

        direct_skill = dir_path / SKILL_MD_UPPER
        if direct_skill.exists() and direct_skill.is_file():
            return self._load_from_skill_md(direct_skill, str(direct_skill))

        all_names: list[str] = []

        for pattern in (SKILL_MD_UPPER, SKILL_MD_LOWER):
            try:
                names = self._registry.register_from_directory(
                    directory,
                    pattern=pattern,
                    recursive=True,
                    overwrite=False,
                )
                all_names.extend(n for n in names if n and n != SKILL_NAME_UNKNOWN)
            except Exception as e:
                logger.warning("Error scanning for %s in %s: %s", pattern, directory, e)

        if all_names:
            unique = list(dict.fromkeys(all_names))
            self._hydrate_external_runtime(unique)
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

    @staticmethod
    def _project_root() -> Path:
        this_file = Path(__file__).resolve()
        return this_file.parents[5]

    def _managed_skills_root(self) -> Path:
        return self._project_root() / "skills"

    def _install_local_skill_package(self, skill_md_path: Path) -> Path | None:
        """Ensure local skill loads use a full installed package under project skills/."""
        try:
            source_skill_md = skill_md_path.resolve()
        except Exception:
            return None

        package_dir = source_skill_md.parent
        managed_root = self._managed_skills_root().resolve()

        if managed_root == package_dir or managed_root in package_dir.parents:
            return source_skill_md

        target_package_dir = managed_root / package_dir.name
        target_skill_md = target_package_dir / source_skill_md.name

        try:
            target_package_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(package_dir, target_package_dir, dirs_exist_ok=True)
        except Exception as exc:
            logger.warning(
                "Failed to install full skill package '%s' into managed skills dir: %s",
                package_dir,
                exc,
            )
            return source_skill_md

        if target_skill_md.exists():
            logger.info(
                "Installed skill package '%s' to %s",
                package_dir.name,
                target_package_dir,
            )
            return target_skill_md

        return source_skill_md

    def _hydrate_external_runtime(self, loaded_names: list[str]) -> None:
        """Hydrate ext__ aliases with core runtime when external spec is metadata-only.

        Two-phase resolution:
        1. RuntimeResolver: if a skill declares a ``runtime`` contract with an
           ``adapter``, dynamically import and bind it as executor.
        2. Core fallback: if the skill still lacks an executor, inherit from the
           matching core skill (ext__X -> X).
        """
        self._resolve_runtime_contracts(loaded_names)
        from houyi.core.skill.runtime_contract import CapabilityTier

        for name in loaded_names:
            if not name or not name.startswith("ext__"):
                continue

            external = self._registry.get(name)
            if external is None:
                continue

            if getattr(external, "capability_tier", None) == CapabilityTier.EXECUTABLE and callable(
                getattr(external, "executor", None)
            ):
                continue

            has_executor = callable(getattr(external, "executor", None))
            has_schema = self._has_schema(getattr(external, "input_schema", None))
            if has_executor and has_schema:
                continue

            core_name = name[len("ext__") :]
            core = self._registry.get(core_name)
            if core is None:
                continue
            core_executor = getattr(core, "executor", None)
            if not callable(core_executor):
                continue

            updates: dict[str, object] = {"executor": core_executor}
            if not self._has_schema(getattr(external, "input_schema", None)):
                updates["input_schema"] = core.input_schema
            if not self._has_schema(getattr(external, "output_schema", None)):
                updates["output_schema"] = core.output_schema

            self._registry.register(external.model_copy(update=updates), overwrite=True)
            logger.info(
                "Hydrated external skill runtime '%s' from core '%s'",
                name,
                core_name,
            )

    def _resolve_runtime_contracts(self, loaded_names: list[str]) -> None:
        """Phase 1: resolve runtime contracts via adapter import."""
        try:
            from houyi.core.skill.runtime_resolver import RuntimeResolver
        except ImportError:
            return

        resolver = RuntimeResolver()
        for name in loaded_names:
            if not name:
                continue
            skill = self._registry.get(name)
            if skill is None:
                continue
            rc = getattr(skill, "runtime_contract", None)
            if rc is None:
                continue
            resolved = resolver.resolve(skill)
            if resolved is not skill:
                self._registry.register(resolved, overwrite=True)

    @staticmethod
    def _has_schema(schema: object | None) -> bool:
        if schema is None or not hasattr(schema, "model_json_schema"):
            return False
        try:
            payload = schema.model_json_schema()
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        return bool(payload.get("properties") or payload.get("required"))


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
