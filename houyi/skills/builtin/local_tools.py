"""Built-in local tools for workspace development workflows.

This module defines six core tools:
- houyi_read_file
- houyi_write_file
- houyi_find_files
- houyi_list_dir
- houyi_grep
- houyi_shell_exec
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from houyi.domain.skill.policy import (
    ExecPerm,
    FilesystemPerm,
    InvocationPolicy,
    ModelAutoInvoke,
    Permissions,
    SideEffect,
)
from houyi.domain.skill.registry import SkillRegistry
from houyi.domain.skill.spec import SkillSpec

DEFAULT_ENCODING = "utf-8"
DEFAULT_MAX_READ_CHARS = 200_000
DEFAULT_MAX_OUTPUT_CHARS = 50_000
DEFAULT_MAX_RESULTS = 200
DEFAULT_SHELL_TIMEOUT_SECONDS = 30
MAX_SHELL_TIMEOUT_SECONDS = 120

DANGEROUS_COMMAND_PATTERNS = (
    r"\brm\s+-rf\s+/$",
    r"\bsudo\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r":\(\)\s*\{\s*:\|:&\s*;\s*\}",
)


class ToolResponse(BaseModel):
    success: bool
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class ReadFileInput(BaseModel):
    path: str
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)


class WriteFileInput(BaseModel):
    path: str
    content: str
    create_parents: bool = True


class FindFilesInput(BaseModel):
    root_path: str = "."
    pattern: str = "*"
    search_mode: Literal["glob", "contains", "exact"] = "contains"
    iterative_subdirs: bool = False
    max_depth: int = Field(default=16, ge=0, le=64)
    max_results: int = Field(default=DEFAULT_MAX_RESULTS, ge=1, le=1000)


class ListDirInput(BaseModel):
    path: str = "."
    recursive: bool = False
    max_entries: int = Field(default=DEFAULT_MAX_RESULTS, ge=1, le=1000)


class GrepInput(BaseModel):
    path: str = "."
    query: str
    case_sensitive: bool = False
    max_results: int = Field(default=DEFAULT_MAX_RESULTS, ge=1, le=1000)


class ShellExecInput(BaseModel):
    command: str
    cwd: str | None = None
    timeout_seconds: int = Field(
        default=DEFAULT_SHELL_TIMEOUT_SECONDS, ge=1, le=MAX_SHELL_TIMEOUT_SECONDS
    )


def _workspace_root() -> Path:
    configured = (os.getenv("HOUYI_WORKSPACE_ROOT") or "").strip()
    return Path(configured).expanduser().resolve() if configured else Path.cwd().resolve()


def _resolve_workspace_path(path_value: str, *, for_creation: bool = False) -> Path:
    base = _workspace_root()
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    else:
        candidate = candidate.resolve()

    target = candidate if not for_creation else candidate.parent
    if target != base and base not in target.parents:
        raise ValueError(f"Path must stay within workspace root: {base}")
    return candidate


def _trim_text(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars]


def _iter_bfs_entries(root: Path, max_depth: int) -> list[Path]:
    discovered: list[Path] = []
    queue: deque[tuple[Path, int]] = deque([(root, 0)])
    while queue:
        current, depth = queue.popleft()
        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for entry in entries:
            discovered.append(entry)
            if entry.is_dir() and depth < max_depth:
                queue.append((entry, depth + 1))
    return discovered


def _iter_files_under(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for item in root.rglob("*"):
        if item.is_file():
            yield item


def _iter_file_lines(file_path: Path) -> Iterator[tuple[int, str]]:
    with file_path.open("r", encoding=DEFAULT_ENCODING, errors="ignore") as handle:
        for index, line in enumerate(handle, start=1):
            yield index, line.rstrip("\r\n")


async def _read_file_executor(
    *,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict[str, Any]:
    try:
        file_path = _resolve_workspace_path(path)
    except ValueError as exc:
        return ToolResponse(success=False, message=str(exc)).model_dump()
    if not file_path.is_file():
        return ToolResponse(success=False, message=f"File not found: {file_path}").model_dump()

    text = file_path.read_text(encoding=DEFAULT_ENCODING)
    lines = text.splitlines()
    if start_line is not None or end_line is not None:
        start = max((start_line or 1) - 1, 0)
        end = end_line or len(lines)
        if end < start + 1:
            return ToolResponse(
                success=False, message="end_line must be greater than or equal to start_line"
            ).model_dump()
        selected = lines[start:end]
    else:
        selected = lines

    content = _trim_text("\n".join(selected), DEFAULT_MAX_READ_CHARS)
    return ToolResponse(
        success=True,
        data={
            "path": str(file_path),
            "content": content,
            "line_count": len(selected),
            "truncated": len(content) >= DEFAULT_MAX_READ_CHARS,
        },
    ).model_dump()


async def _write_file_executor(
    *,
    path: str,
    content: str,
    create_parents: bool = True,
) -> dict[str, Any]:
    try:
        file_path = _resolve_workspace_path(path, for_creation=True)
    except ValueError as exc:
        return ToolResponse(success=False, message=str(exc)).model_dump()
    if create_parents:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(content, encoding=DEFAULT_ENCODING)
    return ToolResponse(
        success=True,
        data={
            "path": str(file_path),
            "bytes_written": len(content.encode(DEFAULT_ENCODING)),
        },
    ).model_dump()


def _resolve_search_mode(*, pattern: str, search_mode: str) -> str:
    has_glob_meta = any(ch in pattern for ch in "*?[")
    return "glob" if (search_mode == "contains" and has_glob_meta) else search_mode


def _collect_search_matches(
    *,
    iterator: Any,
    matcher: Any,
    root: Path,
    max_results: int,
) -> tuple[list[str], set[str]]:
    matches: list[str] = []
    searched_dirs: set[str] = {str(root)}
    for candidate in iterator:
        if candidate.is_dir():
            searched_dirs.add(str(candidate))
        if matcher(candidate.name):
            matches.append(str(candidate))
            if len(matches) >= max_results:
                break
    return matches, searched_dirs


async def _find_files_executor(
    *,
    root_path: str = ".",
    pattern: str = "*",
    search_mode: Literal["glob", "contains", "exact"] = "contains",
    iterative_subdirs: bool = False,
    max_depth: int = 16,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> dict[str, Any]:
    effective_mode = _resolve_search_mode(pattern=pattern, search_mode=search_mode)

    def _matches(candidate_name: str) -> bool:
        if effective_mode == "exact":
            return candidate_name.lower() == pattern.lower()
        if effective_mode == "contains":
            return pattern.lower() in candidate_name.lower()
        return fnmatch.fnmatch(candidate_name, pattern)

    def _iter_candidates() -> Any:
        if not iterative_subdirs:
            return root.rglob("*")
        return _iter_bfs_entries(root, max_depth)

    try:
        root = _resolve_workspace_path(root_path)
    except ValueError as exc:
        return ToolResponse(
            success=False,
            message=f"{exc}. Use a relative path under workspace root (e.g. '.' or 'subdir').",
        ).model_dump()
    if not root.exists() or not root.is_dir():
        return ToolResponse(success=False, message=f"Directory not found: {root}").model_dump()

    matches, searched_dirs = _collect_search_matches(
        iterator=_iter_candidates(),
        matcher=_matches,
        root=root,
        max_results=max_results,
    )

    return ToolResponse(
        success=True,
        data={
            "root_path": str(root),
            "pattern": pattern,
            "search_mode": search_mode,
            "effective_mode": effective_mode,
            "iterative_subdirs": iterative_subdirs,
            "max_depth": max_depth,
            "searched_dirs": sorted(searched_dirs),
            "matches": matches,
            "truncated": len(matches) >= max_results,
        },
    ).model_dump()


async def _list_dir_executor(
    *,
    path: str = ".",
    recursive: bool = False,
    max_entries: int = DEFAULT_MAX_RESULTS,
) -> dict[str, Any]:
    try:
        directory = _resolve_workspace_path(path)
    except ValueError as exc:
        return ToolResponse(success=False, message=str(exc)).model_dump()
    if not directory.exists() or not directory.is_dir():
        return ToolResponse(success=False, message=f"Directory not found: {directory}").model_dump()

    iterator = directory.rglob("*") if recursive else directory.iterdir()
    entries: list[dict[str, Any]] = []
    for entry in iterator:
        entries.append({"path": str(entry), "is_dir": entry.is_dir()})
        if len(entries) >= max_entries:
            break

    return ToolResponse(
        success=True,
        data={
            "path": str(directory),
            "entries": entries,
            "recursive": recursive,
            "truncated": len(entries) >= max_entries,
        },
    ).model_dump()


async def _grep_executor(
    *,
    path: str,
    query: str,
    case_sensitive: bool = False,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> dict[str, Any]:
    try:
        root = _resolve_workspace_path(path)
    except ValueError as exc:
        return ToolResponse(success=False, message=str(exc)).model_dump()
    if not root.exists():
        return ToolResponse(success=False, message=f"Path not found: {root}").model_dump()

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        pattern = re.compile(query, flags)
    except re.error as exc:
        return ToolResponse(success=False, message=f"Invalid regex: {exc}").model_dump()

    files = _iter_files_under(root)
    results: list[dict[str, Any]] = []
    for file_path in files:
        for index, line in _iter_file_lines(file_path):
            if pattern.search(line):
                results.append({"path": str(file_path), "line": index, "text": line})
                if len(results) >= max_results:
                    return ToolResponse(
                        success=True,
                        data={"matches": results, "truncated": True},
                    ).model_dump()

    return ToolResponse(success=True, data={"matches": results, "truncated": False}).model_dump()


def _validate_command(command: str) -> None:
    normalized = command.strip()
    for pattern in DANGEROUS_COMMAND_PATTERNS:
        if re.search(pattern, normalized):
            raise ValueError(f"Blocked dangerous command pattern: {pattern}")


async def _shell_exec_executor(
    *,
    command: str,
    cwd: str | None = None,
    timeout_seconds: int = DEFAULT_SHELL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    effective_timeout_seconds = int(min(timeout_seconds, MAX_SHELL_TIMEOUT_SECONDS))
    try:
        _validate_command(command)
        working_dir = _resolve_workspace_path(cwd or ".")
    except ValueError as exc:
        return ToolResponse(success=False, message=str(exc)).model_dump()

    started_at = asyncio.get_running_loop().time()
    process = await asyncio.create_subprocess_shell(
        command,
        cwd=str(working_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=float(effective_timeout_seconds),
        )
    except TimeoutError:
        process.kill()
        await process.communicate()
        duration_ms = round((asyncio.get_running_loop().time() - started_at) * 1000, 2)
        return ToolResponse(
            success=False,
            message=f"Command timed out after {effective_timeout_seconds}s",
            data={
                "command": command,
                "cwd": str(working_dir),
                "timed_out": True,
                "timeout_seconds": effective_timeout_seconds,
                "duration_ms": duration_ms,
                "retry_count": 0,
            },
        ).model_dump()

    duration_ms = round((asyncio.get_running_loop().time() - started_at) * 1000, 2)
    stdout_text = _trim_text(
        (stdout_bytes or b"").decode(DEFAULT_ENCODING, errors="ignore"), DEFAULT_MAX_OUTPUT_CHARS
    )
    stderr_text = _trim_text(
        (stderr_bytes or b"").decode(DEFAULT_ENCODING, errors="ignore"), DEFAULT_MAX_OUTPUT_CHARS
    )
    success = process.returncode == 0
    return ToolResponse(
        success=success,
        message="" if success else f"Command exited with code {process.returncode}",
        data={
            "command": command,
            "cwd": str(working_dir),
            "returncode": process.returncode,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "stdout_truncated": len(stdout_text) >= DEFAULT_MAX_OUTPUT_CHARS,
            "stderr_truncated": len(stderr_text) >= DEFAULT_MAX_OUTPUT_CHARS,
            "timed_out": False,
            "timeout_seconds": effective_timeout_seconds,
            "duration_ms": duration_ms,
            "retry_count": 0,
        },
    ).model_dump()


def _policy_allow_filesystem_read() -> InvocationPolicy:
    return InvocationPolicy(
        model_auto_invoke=ModelAutoInvoke.ALLOW, side_effect=SideEffect.FILESYSTEM
    )


def _policy_allow_with_consent(side_effect: SideEffect) -> InvocationPolicy:
    return InvocationPolicy(
        model_auto_invoke=ModelAutoInvoke.ALLOW_WITH_CONSENT,
        side_effect=side_effect,
    )


def build_builtin_local_tools() -> list[SkillSpec]:
    """Build six built-in local tool specs."""
    return [
        SkillSpec(
            name="houyi_read_file",
            description="Read local file content with optional line range.",
            input_schema=ReadFileInput,
            output_schema=ToolResponse,
            executor=_read_file_executor,
            is_core=True,
            invocation_policy=_policy_allow_filesystem_read(),
            permissions=Permissions(filesystem=FilesystemPerm(read=True)),
            metadata={"tags": ["file", "read", "workspace"]},
        ),
        SkillSpec(
            name="houyi_write_file",
            description="Write content to a local file under workspace.",
            input_schema=WriteFileInput,
            output_schema=ToolResponse,
            executor=_write_file_executor,
            is_core=True,
            invocation_policy=_policy_allow_with_consent(SideEffect.FILESYSTEM),
            permissions=Permissions(filesystem=FilesystemPerm(write=True)),
            metadata={"tags": ["file", "write", "workspace"]},
        ),
        SkillSpec(
            name="houyi_find_files",
            description="Find files by glob pattern under a workspace directory.",
            input_schema=FindFilesInput,
            output_schema=ToolResponse,
            executor=_find_files_executor,
            is_core=True,
            invocation_policy=_policy_allow_filesystem_read(),
            permissions=Permissions(filesystem=FilesystemPerm(read=True)),
            metadata={"tags": ["file", "search", "glob"]},
        ),
        SkillSpec(
            name="houyi_list_dir",
            description="List directory entries under workspace.",
            input_schema=ListDirInput,
            output_schema=ToolResponse,
            executor=_list_dir_executor,
            is_core=True,
            invocation_policy=_policy_allow_filesystem_read(),
            permissions=Permissions(filesystem=FilesystemPerm(read=True)),
            metadata={"tags": ["directory", "list", "workspace"]},
        ),
        SkillSpec(
            name="houyi_grep",
            description="Search text patterns in files under workspace.",
            input_schema=GrepInput,
            output_schema=ToolResponse,
            executor=_grep_executor,
            is_core=True,
            invocation_policy=_policy_allow_filesystem_read(),
            permissions=Permissions(filesystem=FilesystemPerm(read=True)),
            metadata={"tags": ["search", "grep", "regex"]},
        ),
        SkillSpec(
            name="houyi_shell_exec",
            description="Execute a shell command in workspace with timeout and output limits.",
            input_schema=ShellExecInput,
            output_schema=ToolResponse,
            executor=_shell_exec_executor,
            is_core=True,
            invocation_policy=_policy_allow_with_consent(SideEffect.EXEC),
            permissions=Permissions(exec=ExecPerm(enabled=True)),
            metadata={"tags": ["shell", "command", "execution"]},
        ),
    ]


def register_builtin_local_tools(registry: SkillRegistry) -> list[str]:
    """Register built-in local tools as core skills."""
    registered: list[str] = []
    for skill in build_builtin_local_tools():
        registry.register(skill, overwrite=True)
        registered.append(skill.name)
    return registered


__all__ = [
    "build_builtin_local_tools",
    "register_builtin_local_tools",
]
