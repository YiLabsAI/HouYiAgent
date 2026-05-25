"""Skill loading, unloading, and source validation.

This module owns everything related to *getting skills into and out of the
registry*: file loading, URL fetching/caching, directory scanning, content
validation, and GitHub URL normalisation.

All other concerns (serialisation, dry-run, consent, metrics) live elsewhere.
"""

from __future__ import annotations

import contextlib
import logging
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from houyi.domain.skill.registry import CoreToolProtectionError, SkillRegistry

from .paths import (
    ENV_MANAGED_SKILLS_DIR as PATHS_ENV_MANAGED_SKILLS_DIR,
)
from .paths import (
    resolve_managed_local_sources_root,
    resolve_managed_skills_dir,
    resolve_managed_sources_root,
    resolve_skill_cache_dir,
)

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

INSTALL_STRATEGY_COPY = "copy"
INSTALL_STRATEGY_SYMLINK = "symlink"
VALID_INSTALL_STRATEGIES = {INSTALL_STRATEGY_COPY, INSTALL_STRATEGY_SYMLINK}

# Skill file conventions
SKILL_MD_UPPER = "SKILL.md"
SKILL_MD_LOWER = "skill.md"
SKILL_NAME_UNKNOWN = "unknown"
FRONTMATTER_DELIMITER = "---"

# GitHub URL handling
_GITHUB_BLOB_RE = re.compile(r"https?://github\.com/([^/]+)/([^/]+)/blob/(.+)")
_GITHUB_RAW_RE = re.compile(r"https?://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.+)")
_GITHUB_RAW_HOST = "raw.githubusercontent.com"
_GITHUB_RAW_TEMPLATE = "https://{host}/{owner}/{repo}/{rest}"

# Cache directory
SKILL_CACHE_DIR = resolve_skill_cache_dir()
ENV_MANAGED_SKILLS_DIR = PATHS_ENV_MANAGED_SKILLS_DIR

# Return type alias
LoadResult = tuple[bool, str, str | None]


class SkillLoader:
    """Loads / unloads skills from various sources into a SkillRegistry.

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

    def load(self, source: str, install_strategy: str | None = None) -> LoadResult:
        """Load skill(s) from *source* (file path, URL, or directory).

        Returns (success, skill_name_or_error_code, error_message_or_None).
        """
        if source.startswith("http://") or source.startswith("https://"):
            return self._load_from_url(source)

        strategy = install_strategy or INSTALL_STRATEGY_COPY
        if strategy not in VALID_INSTALL_STRATEGIES:
            return (
                False,
                ERR_VALIDATION_FAILED,
                f"Invalid install_strategy '{strategy}'. Must be one of: copy, symlink",
            )

        path = Path(source)
        if not path.exists():
            return False, ERR_FILE_NOT_FOUND, f"Skill source not found: {source}"
        if path.is_dir():
            return self._load_from_directory(source, install_strategy=strategy)

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
        return self._load_from_skill_md(path, source, install_strategy=strategy)

    def unload(self, skill_name: str) -> tuple[bool, str | None]:
        """Remove *skill_name* from the registry."""
        if not self._registry.get(skill_name):
            return False, f"Skill not found: {skill_name}"
        self._registry.unregister(skill_name)
        return True, None

    def refresh_managed_external_skills(self) -> None:
        """Rescan managed skill directories and prune stale registry entries.

        This keeps list_skills aligned with disk changes (manual delete/move),
        especially for managed ~/.houyi/skills + ~/.houyi/sources/local.
        """
        managed_skills_root = self._managed_global_skills_root()
        managed_local_sources_root = self._managed_local_sources_root()

        if managed_skills_root.exists() and managed_skills_root.is_dir():
            for pattern in (SKILL_MD_UPPER, SKILL_MD_LOWER):
                try:
                    self._registry.register_from_directory(
                        str(managed_skills_root),
                        pattern=pattern,
                        recursive=True,
                        overwrite=False,
                    )
                except Exception as exc:
                    logger.debug(
                        "Managed skill refresh scan failed for pattern %s in %s: %s",
                        pattern,
                        managed_skills_root,
                        exc,
                    )

        managed_roots = tuple(
            root.resolve()
            for root in (managed_skills_root, managed_local_sources_root)
            if root.exists()
        )
        if not managed_roots:
            return

        for skill in list(self._registry.list()):
            if bool(getattr(skill, "is_core", False)):
                continue
            raw_skill_md_path = str(getattr(skill, "skill_md_path", "") or "")
            if not raw_skill_md_path or raw_skill_md_path.startswith(("http://", "https://")):
                continue
            try:
                resolved = Path(raw_skill_md_path).resolve()
            except Exception:
                continue

            under_managed_root = any(
                resolved == root or root in resolved.parents for root in managed_roots
            )
            if not under_managed_root:
                continue
            if resolved.exists():
                continue

            skill_name = str(getattr(skill, "name", "") or "")
            if skill_name:
                self._registry.unregister(skill_name)

    def remove_from_disk(self, skill_name: str) -> tuple[bool, str | None]:
        """Delete managed on-disk package links/data for *skill_name* and unload it.

        Removal scope is intentionally restricted to managed skill roots:
        - ~/.houyi/skills/<package>
        - ~/.houyi/sources/local/<package>
        """
        skill = self._registry.get(skill_name)
        if not skill:
            return False, f"Skill not found: {skill_name}"
        if bool(getattr(skill, "is_core", False)):
            return False, "Core skills cannot be removed from disk"

        managed_skills_root = self._managed_global_skills_root().resolve()
        managed_local_sources_root = self._managed_local_sources_root().resolve()

        package_names: set[str] = set()
        skill_md_path = str(getattr(skill, "skill_md_path", "") or "")
        if skill_md_path and not skill_md_path.startswith(("http://", "https://")):
            raw_path = Path(skill_md_path).expanduser()
            try:
                resolved = Path(skill_md_path).resolve()
            except Exception:
                resolved = None

            for root in (managed_skills_root, managed_local_sources_root):
                for candidate in (raw_path, resolved):
                    if candidate is None:
                        continue
                    try:
                        absolute_candidate = (
                            candidate if candidate.is_absolute() else candidate.resolve()
                        )
                    except Exception:
                        continue
                    if absolute_candidate == root or root in absolute_candidate.parents:
                        rel_parts = absolute_candidate.relative_to(root).parts
                        if rel_parts:
                            package_names.add(rel_parts[0].strip())

        if not package_names:
            source_group = str(getattr(skill, "source_group", "") or "").strip()
            if source_group:
                package_names.add(source_group)

        package_names = {name for name in package_names if name and name not in {".", ".."}}
        if not package_names:
            return False, "Skill is not managed on local disk; nothing to remove"

        delete_targets: list[Path] = []
        for package in sorted(package_names):
            delete_targets.append(managed_skills_root / package)
            delete_targets.append(managed_local_sources_root / package)

        removed_any = False
        for target in delete_targets:
            if not target.exists() and not target.is_symlink():
                continue
            try:
                if target.is_symlink() or target.is_file():
                    target.unlink()
                else:
                    shutil.rmtree(target)
                removed_any = True
            except Exception as exc:
                return False, f"Failed to remove managed path '{target}': {exc}"

        if not removed_any:
            return False, "No managed skill files found to remove"

        self._registry.unregister(skill_name)
        self.refresh_managed_external_skills()
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

    def _load_from_skill_md(
        self,
        path: Path,
        source: str,
        install_strategy: str = INSTALL_STRATEGY_COPY,
    ) -> LoadResult:
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
        installed_skill_path = self._install_local_skill_package(
            path, install_strategy=install_strategy
        )
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

        github_load = self._load_from_github_managed_install(url, raw_url)
        if github_load is not None:
            return github_load

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
            from houyi.domain.skill.schema import parse_skill_md

            parse_skill_md(content)
        except Exception as e:
            return False, ERR_PARSE_FAILED, (f"Failed to parse {SKILL_MD_UPPER} content: {e}")

        try:
            from houyi.domain.skill.spec import SkillSpec

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

    def _load_from_github_managed_install(self, source_url: str, raw_url: str) -> LoadResult | None:
        """Load GitHub skills via managed install gate (clone + symlink + verify)."""
        parsed = _GITHUB_RAW_RE.match(raw_url)
        if not parsed:
            return None

        owner, repo, ref, repo_relative_path = (
            parsed.group(1),
            parsed.group(2),
            parsed.group(3),
            parsed.group(4),
        )
        clone_root = self._managed_sources_root() / owner / repo
        clone_root.parent.mkdir(parents=True, exist_ok=True)

        if not clone_root.exists():
            clone_url = f"https://github.com/{owner}/{repo}.git"
            cmd = ["git", "clone", "--depth", "1", "--branch", ref, clone_url, str(clone_root)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "git clone failed").strip()
                return False, ERR_URL_LOAD_FAILED, f"Managed install failed during clone: {err}"

        link_alias, link_target_rel = self._resolve_github_link_binding(repo, repo_relative_path)
        managed_skill_file = clone_root / repo_relative_path
        if not managed_skill_file.exists() or not managed_skill_file.is_file():
            return (
                False,
                ERR_URL_LOAD_FAILED,
                f"Managed install verify failed: skill file not found at {managed_skill_file}",
            )

        if link_alias == repo and link_target_rel == Path("."):
            skill_name_alias = self._extract_alias_from_skill_file(managed_skill_file)
            if skill_name_alias:
                logger.info(
                    "Resolved root GitHub skill alias from %s to skill name '%s'",
                    repo,
                    skill_name_alias,
                )
                link_alias = skill_name_alias

        if link_alias != repo:
            self._prune_stale_repo_alias(
                managed_skills_root=self._managed_global_skills_root(),
                repo_alias=repo,
                clone_root=clone_root,
            )

        managed_link = self._managed_global_skills_root() / link_alias
        link_target = clone_root / link_target_rel
        managed_link.parent.mkdir(parents=True, exist_ok=True)

        if (
            managed_link.exists()
            and managed_link.is_symlink()
            and managed_link.resolve() != link_target.resolve()
        ):
            managed_link.unlink()
        if managed_link.exists() and not managed_link.is_symlink():
            shutil.rmtree(managed_link)
        if not managed_link.exists():
            managed_link.symlink_to(link_target, target_is_directory=True)

        if not managed_link.exists() or not link_target.exists():
            return (
                False,
                ERR_URL_LOAD_FAILED,
                f"Managed install verify failed: {managed_link} does not point to an existing skills directory",
            )

        result = self._load_from_skill_md(managed_skill_file, str(managed_skill_file))
        if result[0]:
            logger.info(
                "Loaded skill '%s' via managed GitHub install: %s (source: %s)",
                result[1],
                managed_skill_file,
                source_url,
            )
        return result

    @staticmethod
    def _resolve_github_link_binding(repo: str, repo_relative_path: str) -> tuple[str, Path]:
        """Derive symlink alias and target dir from a GitHub raw skill path.

        Examples:
        - skills/docx/SKILL.md -> ("docx", Path("skills/docx"))
        - SKILL.md             -> (repo, Path("."))
        """
        rel = Path(repo_relative_path)
        if rel.name.lower() == SKILL_MD_LOWER and rel.parent.name:
            return rel.parent.name, rel.parent
        if rel.name.lower() == "simpleskill.json" and rel.parent.name:
            return rel.parent.name, rel.parent
        return repo, Path(".")

    @staticmethod
    def _extract_alias_from_skill_file(skill_file: Path) -> str | None:
        """Extract alias candidate from SKILL.md frontmatter name."""
        try:
            from houyi.domain.skill.schema import parse_skill_md

            parsed = parse_skill_md(skill_file.read_text(encoding="utf-8"))
        except Exception:
            return None

        name = parsed.get("name") if isinstance(parsed, dict) else None
        if not isinstance(name, str):
            return None
        alias = name.strip()
        if not alias or "/" in alias or alias in {".", ".."}:
            return None
        return alias

    @staticmethod
    def _prune_stale_repo_alias(
        managed_skills_root: Path, repo_alias: str, clone_root: Path
    ) -> None:
        """Remove stale repo-named alias if it points to this clone root."""
        stale_alias = managed_skills_root / repo_alias
        if not stale_alias.exists() or not stale_alias.is_symlink():
            return
        try:
            if stale_alias.resolve() == clone_root.resolve():
                stale_alias.unlink(missing_ok=True)
        except Exception:
            return

    def _load_from_directory(
        self,
        directory: str,
        install_strategy: str = INSTALL_STRATEGY_COPY,
    ) -> LoadResult:
        dir_path = Path(directory)
        effective_dir = self._install_local_directory_package(dir_path, install_strategy)
        if effective_dir is None:
            return (
                False,
                ERR_LOAD_FAILED,
                f"Failed to install directory source using strategy '{install_strategy}': {directory}",
            )

        direct_skill = effective_dir / SKILL_MD_UPPER
        if direct_skill.exists() and direct_skill.is_file():
            return self._load_from_skill_md(
                direct_skill,
                str(direct_skill),
                install_strategy=install_strategy,
            )

        all_names: list[str] = []

        for pattern in (SKILL_MD_UPPER, SKILL_MD_LOWER):
            try:
                names = self._registry.register_from_directory(
                    str(effective_dir),
                    pattern=pattern,
                    recursive=True,
                    overwrite=False,
                )
                all_names.extend(n for n in names if n and n != SKILL_NAME_UNKNOWN)
            except Exception as e:
                logger.warning("Error scanning for %s in %s: %s", pattern, effective_dir, e)

        if all_names:
            unique = list(dict.fromkeys(all_names))
            self._hydrate_external_runtime(unique)
            logger.info(
                "Loaded %d skills from directory %s: %s",
                len(unique),
                effective_dir,
                ", ".join(unique),
            )
            return True, ", ".join(unique), None

        md_files = list(effective_dir.rglob("*.md"))
        if md_files:
            names_found = [f.name for f in md_files[:5]]
            return (
                False,
                ERR_NO_SKILLS,
                (
                    f"No {SKILL_MD_UPPER} files found in {effective_dir}. "
                    f"Found: {', '.join(names_found)}. "
                    f"Rename to {SKILL_MD_UPPER} with YAML frontmatter to load."
                ),
            )
        return (
            False,
            ERR_NO_SKILLS,
            (
                f"No {SKILL_MD_UPPER} files found in {effective_dir}. "
                f"Ensure the directory contains {SKILL_MD_UPPER} files "
                f"with YAML frontmatter."
            ),
        )

    @staticmethod
    def _project_root() -> Path:
        this_file = Path(__file__).resolve()
        return this_file.parents[5]

    def _managed_skills_root(self) -> Path:
        return resolve_managed_skills_dir()

    @staticmethod
    def _managed_sources_root() -> Path:
        return resolve_managed_sources_root()

    @staticmethod
    def _managed_local_sources_root() -> Path:
        return resolve_managed_local_sources_root()

    @staticmethod
    def _managed_global_skills_root() -> Path:
        return resolve_managed_skills_dir()

    def _install_local_skill_package(
        self,
        skill_md_path: Path,
        install_strategy: str = INSTALL_STRATEGY_COPY,
    ) -> Path | None:
        """Materialize local package under managed sources and expose via skills symlink."""
        try:
            source_skill_md = skill_md_path.resolve()
        except Exception:
            return None

        package_dir = source_skill_md.parent
        managed_root = self._managed_skills_root().resolve()
        managed_local_sources_root = self._managed_local_sources_root().resolve()
        managed_sources_root = self._managed_sources_root().resolve()
        managed_global_root = self._managed_global_skills_root().resolve()

        if managed_root == package_dir or managed_root in package_dir.parents:
            return source_skill_md
        if package_dir in managed_root.parents:
            return source_skill_md
        if (
            managed_local_sources_root == package_dir
            or managed_local_sources_root in package_dir.parents
        ):
            return source_skill_md
        if managed_sources_root == package_dir or managed_sources_root in package_dir.parents:
            return source_skill_md
        if managed_global_root == package_dir or managed_global_root in package_dir.parents:
            return source_skill_md

        source_package_dir = managed_local_sources_root / package_dir.name
        managed_link_dir = managed_root / package_dir.name
        target_skill_md = managed_link_dir / source_skill_md.name

        try:
            managed_local_sources_root.mkdir(parents=True, exist_ok=True)
            managed_root.mkdir(parents=True, exist_ok=True)

            if install_strategy == INSTALL_STRATEGY_SYMLINK:
                if (
                    source_package_dir.exists()
                    and source_package_dir.is_symlink()
                    and source_package_dir.resolve() != package_dir
                ):
                    source_package_dir.unlink()
                if source_package_dir.exists() and not source_package_dir.is_symlink():
                    shutil.rmtree(source_package_dir)
                if not source_package_dir.exists():
                    source_package_dir.symlink_to(package_dir, target_is_directory=True)
            else:
                if source_package_dir.exists() and source_package_dir.is_symlink():
                    source_package_dir.unlink()
                source_package_dir.mkdir(parents=True, exist_ok=True)
                shutil.copytree(package_dir, source_package_dir, dirs_exist_ok=True)

            if (
                managed_link_dir.exists()
                and managed_link_dir.is_symlink()
                and managed_link_dir.resolve() != source_package_dir.resolve()
            ):
                managed_link_dir.unlink()
            if managed_link_dir.exists() and not managed_link_dir.is_symlink():
                shutil.rmtree(managed_link_dir)
            if not managed_link_dir.exists():
                managed_link_dir.symlink_to(source_package_dir, target_is_directory=True)
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
                managed_link_dir,
            )
            return target_skill_md

        return source_skill_md

    def _install_local_directory_package(
        self,
        directory_path: Path,
        install_strategy: str = INSTALL_STRATEGY_COPY,
    ) -> Path | None:
        try:
            source_dir = directory_path.resolve()
        except Exception:
            return None

        managed_root = self._managed_skills_root().resolve()
        managed_local_sources_root = self._managed_local_sources_root().resolve()
        if source_dir == managed_root or managed_root in source_dir.parents:
            return source_dir
        if source_dir in managed_root.parents:
            return source_dir
        if (
            source_dir == managed_local_sources_root
            or managed_local_sources_root in source_dir.parents
        ):
            return source_dir

        source_target_dir = managed_local_sources_root / source_dir.name
        managed_link_dir = managed_root / source_dir.name
        try:
            managed_local_sources_root.mkdir(parents=True, exist_ok=True)
            managed_root.mkdir(parents=True, exist_ok=True)
            if install_strategy == INSTALL_STRATEGY_SYMLINK:
                if (
                    source_target_dir.exists()
                    and source_target_dir.is_symlink()
                    and source_target_dir.resolve() != source_dir
                ):
                    source_target_dir.unlink()
                if source_target_dir.exists() and not source_target_dir.is_symlink():
                    shutil.rmtree(source_target_dir)
                if not source_target_dir.exists():
                    source_target_dir.symlink_to(source_dir, target_is_directory=True)
            else:
                if source_target_dir.exists() and source_target_dir.is_symlink():
                    source_target_dir.unlink()
                source_target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source_dir, source_target_dir, dirs_exist_ok=True)

            if (
                managed_link_dir.exists()
                and managed_link_dir.is_symlink()
                and managed_link_dir.resolve() != source_target_dir.resolve()
            ):
                managed_link_dir.unlink()
            if managed_link_dir.exists() and not managed_link_dir.is_symlink():
                shutil.rmtree(managed_link_dir)
            if not managed_link_dir.exists():
                managed_link_dir.symlink_to(source_target_dir, target_is_directory=True)
        except Exception as exc:
            logger.warning(
                "Failed to install directory source '%s' using strategy '%s': %s",
                source_dir,
                install_strategy,
                exc,
            )
            return None

        return managed_link_dir

    def _hydrate_external_runtime(self, loaded_names: list[str]) -> None:
        """Hydrate ext__ aliases with core runtime when external spec is metadata-only.

        Two-phase resolution:
        1. RuntimeResolver: if a skill declares a runtime contract with an
           adapter, dynamically import and bind it as executor.
        2. Core fallback: if the skill still lacks an executor, inherit from the
           matching core skill (ext__X -> X).
        """
        self._resolve_runtime_contracts(loaded_names)
        from houyi.domain.skill.runtime_contract import CapabilityTier

        for name in loaded_names:
            if not name:
                continue

            self._hydrate_script_compat_runtime(name)

            if not name.startswith("ext__"):
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

    def _hydrate_script_compat_runtime(self, loaded_name: str) -> None:
        """Bind a script-compatible executor for instruction-driven script skills.

        Many community SKILL.md files do not provide a HouYi-specific
        runtime.adapter block, but still define executable workflows via
        command examples (e.g. python scripts/run.py ...). This method scans
        instructions for command templates and binds a generic executor.
        """
        skill = self._registry.get(loaded_name)
        if skill is None or callable(getattr(skill, "executor", None)):
            return

        raw_skill_dir = getattr(skill, "skill_dir", None)
        instructions = getattr(skill, "instructions", None)
        if not raw_skill_dir or not isinstance(instructions, str) or not instructions.strip():
            return

        try:
            skill_dir = Path(raw_skill_dir).resolve()
        except Exception:
            return

        templates = self._extract_script_command_templates(instructions)
        if not templates:
            return

        async def _script_compat_executor(**kwargs):
            import asyncio
            import json
            import sys

            def _normalize_command(command: list[str]) -> list[str]:
                normalized: list[str] = []
                for idx, token in enumerate(command):
                    if idx == 0 and token == "python":
                        normalized.append(sys.executable)
                        continue
                    if token.startswith("-"):
                        normalized.append(token)
                        continue

                    path_token = Path(token)
                    if not path_token.is_absolute():
                        candidate = (skill_dir / path_token).resolve()
                        if candidate.exists():
                            normalized.append(str(candidate))
                            continue
                    normalized.append(token)
                return normalized

            def _dependency_state(command: list[str]) -> tuple[list[str], list[str]]:
                required = self._infer_required_binaries(command)
                return required, self._missing_binaries(required)

            explicit_command = isinstance(kwargs.get("command"), str) and bool(
                str(kwargs.get("command", "")).strip()
            )
            explicit_workflow = isinstance(kwargs.get("workflow_id"), str) and bool(
                str(kwargs.get("workflow_id", "")).strip()
            )

            cmd = self._build_script_compat_command(kwargs, templates)
            if not cmd:
                raise ValueError("No executable script command could be derived from payload")

            normalized_cmd = _normalize_command(cmd)
            _, missing_bins = _dependency_state(normalized_cmd)
            if missing_bins and not explicit_command and not explicit_workflow:
                for idx in range(len(templates)):
                    candidate_payload = dict(kwargs)
                    candidate_payload["workflow_id"] = f"template_{idx + 1}"
                    candidate_cmd = self._build_script_compat_command(candidate_payload, templates)
                    if not candidate_cmd:
                        continue
                    candidate_normalized = _normalize_command(candidate_cmd)
                    _, candidate_missing = _dependency_state(candidate_normalized)
                    if candidate_missing:
                        continue
                    normalized_cmd = candidate_normalized
                    _, missing_bins = _dependency_state(normalized_cmd)
                    break

            if missing_bins:
                missing_msg = (
                    "Missing required runtime dependency: "
                    + ", ".join(missing_bins)
                    + ". Please install it (e.g. LibreOffice provides 'soffice')."
                )
                return {
                    "ok": False,
                    "exit_code": 127,
                    "error_code": "missing_dependency",
                    "missing_dependencies": missing_bins,
                    "command": normalized_cmd,
                    "output": "",
                    "stderr": missing_msg,
                }

            async def _run_command_once(command: list[str]) -> tuple[int, str, str]:
                proc = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=str(skill_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180.0)
                except TimeoutError:
                    proc.kill()
                    await proc.wait()
                    raise
                return (
                    proc.returncode,
                    stdout.decode("utf-8", errors="replace").strip(),
                    stderr.decode("utf-8", errors="replace").strip(),
                )

            def _is_broken_skill_venv(returncode: int, stderr_text: str) -> bool:
                if returncode == 250 and "Library not loaded: @rpath/libpython" in stderr_text:
                    return True
                lowered = stderr_text.lower()
                return "libpython" in lowered and "no such file" in lowered

            def _is_missing_venv_pip(stderr_text: str) -> bool:
                lowered = stderr_text.lower()
                return ".venv/bin/pip" in lowered and "no such file or directory" in lowered

            async def _repair_skill_venv_if_possible() -> bool:
                setup_script = skill_dir / "scripts" / "setup_environment.py"
                if not setup_script.exists():
                    return False
                repair_cmd = [sys.executable, str(setup_script)]
                repair_code, repair_stdout, repair_stderr = await _run_command_once(repair_cmd)
                if repair_code != 0:
                    if _is_missing_venv_pip(repair_stderr):
                        broken_venv = skill_dir / ".venv"
                        with contextlib.suppress(Exception):
                            if broken_venv.exists():
                                shutil.rmtree(broken_venv)
                        repair_code, repair_stdout, repair_stderr = await _run_command_once(
                            repair_cmd
                        )
                        if repair_code == 0:
                            logger.info(
                                "Script compatibility runtime rebuilt broken skill .venv and repaired environment"
                            )
                            return True
                    logger.warning(
                        "Script compatibility runtime failed to repair skill env: cmd=%s code=%d stdout=%s stderr=%s",
                        repair_cmd,
                        repair_code,
                        repair_stdout[:500],
                        repair_stderr[:500],
                    )
                    return False
                logger.info(
                    "Script compatibility runtime repaired skill env via setup_environment.py"
                )
                return True

            try:
                code, stdout_text, stderr_text = await _run_command_once(normalized_cmd)
            except TimeoutError:
                return {
                    "ok": False,
                    "error": "Script compatibility execution timed out",
                    "command": normalized_cmd,
                }

            if _is_broken_skill_venv(code, stderr_text):
                repaired = False
                with contextlib.suppress(TimeoutError):
                    repaired = await _repair_skill_venv_if_possible()
                if repaired:
                    try:
                        code, stdout_text, stderr_text = await _run_command_once(normalized_cmd)
                    except TimeoutError:
                        return {
                            "ok": False,
                            "error": "Script compatibility execution timed out",
                            "command": normalized_cmd,
                        }

            parsed_output: object = stdout_text
            if stdout_text:
                with contextlib.suppress(Exception):
                    parsed_output = json.loads(stdout_text)

            return {
                "ok": code == 0,
                "exit_code": code,
                "command": normalized_cmd,
                "output": parsed_output,
                "stderr": stderr_text,
            }

        extra_frontmatter = dict(getattr(skill, "extra_frontmatter", {}) or {})
        extra_frontmatter["runtime_binding"] = "script_executor_compat"

        updated = skill.model_copy(
            update={
                "executor": _script_compat_executor,
                "extra_frontmatter": extra_frontmatter,
            }
        )
        self._registry.register(updated, overwrite=True)
        logger.info(
            "Hydrated script compatibility runtime for '%s' (%d command templates)",
            loaded_name,
            len(templates),
        )

    @staticmethod
    def _extract_script_command_templates(instructions: str) -> list[dict[str, object]]:
        templates: list[dict[str, object]] = []

        code_blocks = re.findall(r"```(?:bash|sh|zsh)?\n([\s\S]*?)```", instructions)
        lines: list[str] = []
        for block in code_blocks:
            lines.extend(line.strip() for line in block.splitlines() if line.strip())

        if not lines:
            lines = [
                line.strip()
                for line in instructions.splitlines()
                if line.strip().startswith(("python ", "./", "sh "))
            ]

        for line in lines:
            if not line.startswith(("python ", "./", "sh ")):
                continue
            with contextlib.suppress(ValueError):
                tokens = shlex.split(line)
                if not tokens:
                    continue
                base_tokens: list[str] = []
                flags: list[str] = []
                saw_flag = False
                for token in tokens:
                    if token.startswith("--"):
                        saw_flag = True
                        flags.append(token[2:].replace("-", "_"))
                        continue
                    if saw_flag:
                        continue
                    if token.startswith("["):
                        continue
                    base_tokens.append(token)
                templates.append(
                    {
                        "raw": line,
                        "base_tokens": base_tokens,
                        "flags": flags,
                    }
                )

        unique: list[dict[str, object]] = []
        seen = set()
        for item in templates:
            key = tuple(item.get("base_tokens", []))
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    @staticmethod
    def _build_script_compat_command(
        payload: dict[str, object],
        templates: list[dict[str, object]],
    ) -> list[str]:
        command = payload.get("command")
        if isinstance(command, str) and command.strip():
            with contextlib.suppress(ValueError):
                return shlex.split(command)
            return [command.strip()]

        if not templates:
            return []

        workflow_id = str(payload.get("workflow_id", "")).strip().lower()
        if workflow_id.startswith("template_"):
            suffix = workflow_id.removeprefix("template_")
            with contextlib.suppress(ValueError):
                selected_index = int(suffix) - 1
                if 0 <= selected_index < len(templates):
                    selected_template = templates[selected_index]
                    selected_tokens = [str(t) for t in (selected_template.get("base_tokens") or [])]
                    if selected_tokens:
                        return selected_tokens

        operation = str(payload.get("operation", "")).strip().lower()
        script = str(payload.get("script", "")).strip().lower()

        def _score(template: dict[str, object]) -> int:
            base_tokens = [str(t).lower() for t in (template.get("base_tokens") or [])]
            flags = [str(f) for f in (template.get("flags") or [])]
            score = 0
            if operation and operation in base_tokens:
                score += 6
            if script and any(script in tok for tok in base_tokens):
                score += 6
            for flag in flags:
                if flag in payload or flag.replace("_", "-") in payload:
                    score += 2
            return score

        scored_templates = [(template, _score(template)) for template in templates]
        selected, best_score = max(scored_templates, key=lambda item: item[1])

        # If we have multiple executable examples but payload does not map to any
        # of them, avoid defaulting to the first template unexpectedly.
        if len(templates) > 1 and best_score <= 0:
            return []

        base_tokens = [str(t) for t in (selected.get("base_tokens") or [])]
        flags = [str(f) for f in (selected.get("flags") or [])]
        if not base_tokens:
            return []

        cmd = [*base_tokens]
        script_value = str(payload.get("script", "")).strip()
        operation_value = str(payload.get("operation", "")).strip()
        if len(base_tokens) >= 2 and base_tokens[1].endswith("scripts/run.py") and script_value:
            runner = base_tokens[0]
            script_token = Path(script_value).name
            cmd = [runner, base_tokens[1], script_token]
            if operation_value:
                cmd.append(operation_value)

        used: set[str] = set()

        for flag in flags:
            key_variants = (flag, flag.replace("_", "-"))
            value = None
            key_used = None
            for key in key_variants:
                if key in payload:
                    value = payload[key]
                    key_used = key
                    break
            if value is None:
                continue
            used.add(key_used or flag)

            option = f"--{flag.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(option)
                continue
            if value in (None, ""):
                continue
            cmd.extend([option, str(value)])

        meta_keys = {
            "task",
            "objective",
            "expected_focus",
            "subtasks",
            "description",
            "script",
            "operation",
            "command",
            "workflow_id",
        }

        for key, value in payload.items():
            if key in used or key in meta_keys:
                continue
            if value in (None, ""):
                continue
            option = f"--{str(key).replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(option)
                continue
            cmd.extend([option, str(value)])

        return cmd

    def _resolve_runtime_contracts(self, loaded_names: list[str]) -> None:
        """Phase 1: resolve runtime contracts via adapter import."""
        try:
            from houyi.domain.skill.runtime_resolver import RuntimeResolver
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
    def _infer_required_binaries(command: list[str]) -> list[str]:
        required: list[str] = []
        if not command:
            return required

        head = Path(command[0]).name.lower()
        if head in {"soffice", "pandoc", "pdftoppm"}:
            required.append(head)

        lowered_tokens = [str(token).lower() for token in command]
        if any(token.endswith("scripts/office/soffice.py") for token in lowered_tokens):
            required.append("soffice")

        unique: list[str] = []
        seen: set[str] = set()
        for item in required:
            if item in seen:
                continue
            seen.add(item)
            unique.append(item)
        return unique

    @staticmethod
    def _missing_binaries(required: list[str]) -> list[str]:
        return [binary for binary in required if shutil.which(binary) is None]

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
    """Raise ValueError if *content* looks like HTML or is empty."""
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
    """Raise ValueError if the skill has no usable name."""
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
