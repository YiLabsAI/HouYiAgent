from __future__ import annotations

from pathlib import Path

import pytest

from houyi.domain.skill.policy import ModelAutoInvoke
from houyi.domain.skill.registry import SkillRegistry
from houyi.skills.builtin import local_tools


@pytest.fixture
def workspace_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOUYI_WORKSPACE_ROOT", str(tmp_path))
    return tmp_path


def test_build_builtin_local_tools_contains_six_expected_names() -> None:
    tools = local_tools.build_builtin_local_tools()
    names = [tool.name for tool in tools]
    assert names == [
        "houyi_read_file",
        "houyi_write_file",
        "houyi_find_files",
        "houyi_list_dir",
        "houyi_grep",
        "houyi_shell_exec",
    ]


def test_build_builtin_local_tools_sets_policy_by_risk() -> None:
    tools = {tool.name: tool for tool in local_tools.build_builtin_local_tools()}

    assert tools["houyi_read_file"].invocation_policy.model_auto_invoke == ModelAutoInvoke.ALLOW
    assert tools["houyi_write_file"].invocation_policy.model_auto_invoke == (
        ModelAutoInvoke.ALLOW_WITH_CONSENT
    )
    assert tools["houyi_shell_exec"].invocation_policy.model_auto_invoke == (
        ModelAutoInvoke.ALLOW_WITH_CONSENT
    )


def test_register_builtin_local_tools_registers_all() -> None:
    registry = SkillRegistry()

    registered = local_tools.register_builtin_local_tools(registry)

    assert len(registered) == 6
    assert all(registry.get(name) is not None for name in registered)


@pytest.mark.asyncio
async def test_read_file_executor_reads_with_line_range(workspace_root: Path) -> None:
    target = workspace_root / "a.txt"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")

    result = await local_tools._read_file_executor(path="a.txt", start_line=2, end_line=3)

    assert result["success"] is True
    assert result["data"]["content"] == "two\nthree"


@pytest.mark.asyncio
async def test_write_file_executor_creates_parents(workspace_root: Path) -> None:
    result = await local_tools._write_file_executor(
        path="nested/target.txt",
        content="hello",
        create_parents=True,
    )

    assert result["success"] is True
    assert (workspace_root / "nested" / "target.txt").read_text(encoding="utf-8") == "hello"


@pytest.mark.asyncio
async def test_find_files_executor_filters_by_pattern(workspace_root: Path) -> None:
    (workspace_root / "keep.py").write_text("print('x')", encoding="utf-8")
    (workspace_root / "skip.txt").write_text("x", encoding="utf-8")

    result = await local_tools._find_files_executor(root_path=".", pattern="*.py")

    assert result["success"] is True
    assert any(item.endswith("keep.py") for item in result["data"]["matches"])
    assert all(not item.endswith("skip.txt") for item in result["data"]["matches"])


@pytest.mark.asyncio
async def test_find_files_executor_iterative_subdirs_exact_match(workspace_root: Path) -> None:
    nested = workspace_root / "src" / "skills"
    nested.mkdir(parents=True)
    (nested / "skill.md").write_text("# skill", encoding="utf-8")

    result = await local_tools._find_files_executor(
        root_path=".",
        pattern="skill.md",
        search_mode="exact",
        iterative_subdirs=True,
        max_depth=4,
    )

    assert result["success"] is True
    assert any(item.endswith("src/skills/skill.md") for item in result["data"]["matches"])
    assert result["data"]["iterative_subdirs"] is True
    assert any(path.endswith("src") for path in result["data"]["searched_dirs"])


@pytest.mark.asyncio
async def test_find_files_executor_iterative_subdirs_respects_max_depth(
    workspace_root: Path,
) -> None:
    deep = workspace_root / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "target.md").write_text("x", encoding="utf-8")

    result = await local_tools._find_files_executor(
        root_path=".",
        pattern="target.md",
        search_mode="exact",
        iterative_subdirs=True,
        max_depth=1,
    )

    assert result["success"] is True
    assert all(not item.endswith("target.md") for item in result["data"]["matches"])


@pytest.mark.asyncio
async def test_find_files_executor_default_contains_mode(workspace_root: Path) -> None:
    (workspace_root / "skill.md").write_text("x", encoding="utf-8")

    result = await local_tools._find_files_executor(root_path=".", pattern="skill")

    assert result["success"] is True
    assert result["data"]["search_mode"] == "contains"
    assert result["data"]["effective_mode"] == "contains"
    assert any(item.endswith("skill.md") for item in result["data"]["matches"])


@pytest.mark.asyncio
async def test_find_files_executor_wildcard_pattern_auto_uses_glob(workspace_root: Path) -> None:
    (workspace_root / "main.py").write_text("print('x')", encoding="utf-8")

    result = await local_tools._find_files_executor(root_path=".", pattern="*.py")

    assert result["success"] is True
    assert result["data"]["search_mode"] == "contains"
    assert result["data"]["effective_mode"] == "glob"
    assert any(item.endswith("main.py") for item in result["data"]["matches"])


@pytest.mark.asyncio
async def test_list_dir_executor_non_recursive(workspace_root: Path) -> None:
    (workspace_root / "a.txt").write_text("a", encoding="utf-8")
    nested = workspace_root / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("b", encoding="utf-8")

    result = await local_tools._list_dir_executor(path=".", recursive=False)

    paths = [entry["path"] for entry in result["data"]["entries"]]
    assert any(path.endswith("a.txt") for path in paths)
    assert any(path.endswith("nested") for path in paths)
    assert all(not path.endswith("b.txt") for path in paths)


@pytest.mark.asyncio
async def test_grep_executor_matches_lines(workspace_root: Path) -> None:
    (workspace_root / "x.txt").write_text("alpha\nneedle here\nomega\n", encoding="utf-8")

    result = await local_tools._grep_executor(path=".", query="needle")

    assert result["success"] is True
    assert len(result["data"]["matches"]) == 1
    assert result["data"]["matches"][0]["line"] == 2


@pytest.mark.asyncio
async def test_shell_exec_executor_blocks_dangerous_command(workspace_root: Path) -> None:
    result = await local_tools._shell_exec_executor(command="sudo ls", cwd=".")

    assert result["success"] is False
    assert "Blocked dangerous command pattern" in result["message"]


@pytest.mark.asyncio
async def test_shell_exec_executor_runs_safe_command(workspace_root: Path) -> None:
    result = await local_tools._shell_exec_executor(command="echo hello", cwd=".")

    assert result["success"] is True
    assert "hello" in result["data"]["stdout"]


@pytest.mark.asyncio
async def test_executors_reject_paths_outside_workspace(workspace_root: Path) -> None:
    outside_path = str(workspace_root.parent / "outside.txt")

    read_result = await local_tools._read_file_executor(path=outside_path)
    write_result = await local_tools._write_file_executor(path=outside_path, content="x")

    assert read_result["success"] is False
    assert write_result["success"] is False
    assert "workspace root" in read_result["message"]
    assert "workspace root" in write_result["message"]
