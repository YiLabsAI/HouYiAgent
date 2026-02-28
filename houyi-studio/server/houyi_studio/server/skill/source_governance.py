"""Source governance helpers for community skill repositories.

This module provides a minimal compatibility layer for repositories that do not
ship an explicit installation guide. It follows a Codex-like flow:
clone -> symlink -> verify/update/uninstall.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

INSTALL_DOC_CANDIDATES: tuple[str, ...] = (".codex/INSTALL.md",)


@dataclass(frozen=True)
class SourceLifecyclePlan:
    strategy: str
    install_commands: tuple[str, ...]
    verify_command: str
    update_command: str
    uninstall_command: str
    install_doc: str | None = None


def discover_install_doc(repo_root: Path) -> Path | None:
    """Return the first install document found in a repository."""
    for candidate in INSTALL_DOC_CANDIDATES:
        path = repo_root / candidate
        if path.exists() and path.is_file():
            return path
    return None


def infer_skills_subdir(repo_root: Path, preferred_subdir: str = "skills") -> str:
    """Infer the directory that should be symlinked into managed skills.

    Priority:
    1. Explicit preferred subdir if present (default: ``skills``)
    2. Repo root if it directly contains SKILL.md
    3. First direct child directory that contains one or more SKILL.md files
    """
    preferred = repo_root / preferred_subdir
    if preferred.exists() and preferred.is_dir():
        return preferred_subdir

    if (repo_root / "SKILL.md").exists():
        return "."

    for child in sorted(repo_root.iterdir()):
        if not child.is_dir():
            continue
        if any(p.is_file() for p in child.rglob("SKILL.md")):
            return child.name

    raise ValueError(f"No skill directory found under repository: {repo_root}")


def build_fallback_lifecycle_plan(
    *,
    repo_url: str,
    alias: str,
    source_home: str = "~/.houyi/sources",
    managed_home: str = "~/.houyi/skills",
    skills_subdir: str = "skills",
) -> SourceLifecyclePlan:
    """Generate a Codex-style lifecycle plan for repositories without INSTALL.md."""
    repo_slug = _repo_slug(repo_url)
    source_repo_dir = f"{source_home}/{repo_slug}"
    link_target = (
        f"{source_repo_dir}" if skills_subdir == "." else f"{source_repo_dir}/{skills_subdir}"
    )
    link_path = f"{managed_home}/{alias}"

    install_commands = (
        f"git clone {repo_url} {source_repo_dir}",
        f"mkdir -p {managed_home}",
        f"ln -s {link_target} {link_path}",
    )

    return SourceLifecyclePlan(
        strategy="generated_clone_symlink",
        install_commands=install_commands,
        verify_command=f"ls -la {link_path}",
        update_command=f"git -C {source_repo_dir} pull",
        uninstall_command=f"rm {link_path}",
    )


def _repo_slug(repo_url: str) -> str:
    normalized = repo_url.strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]

    if normalized.startswith("https://"):
        normalized = normalized[len("https://") :]
    elif normalized.startswith("http://"):
        normalized = normalized[len("http://") :]
    elif normalized.startswith("git@"):
        normalized = normalized.split("@", 1)[1].replace(":", "/", 1)

    return normalized
