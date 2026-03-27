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


def test_build_tools_names() -> None:
    tools = local_tools.build_builtin_local_tools()
    names = [tool.name for tool in tools]
    assert names == [
        "houyi_read_file",
        "houyi_write_file",
        "houyi_find_files",
        "houyi_list_dir",
        "houyi_grep",
        "houyi_shell_exec",
        "houyi_local_cli",
        "houyi_local_cli_chain",
    ]


def test_build_tools_policies() -> None:
    tools = {tool.name: tool for tool in local_tools.build_builtin_local_tools()}

    assert tools["houyi_read_file"].invocation_policy.model_auto_invoke == ModelAutoInvoke.ALLOW
    assert tools["houyi_write_file"].invocation_policy.model_auto_invoke == (
        ModelAutoInvoke.ALLOW_WITH_CONSENT
    )
    assert tools["houyi_shell_exec"].invocation_policy.model_auto_invoke == (
        ModelAutoInvoke.ALLOW_WITH_CONSENT
    )
    assert tools["houyi_local_cli"].invocation_policy.model_auto_invoke == ModelAutoInvoke.ALLOW
    assert (
        tools["houyi_local_cli_chain"].invocation_policy.model_auto_invoke == ModelAutoInvoke.ALLOW
    )


def test_build_tools_descriptions() -> None:
    tools = {tool.name: tool for tool in local_tools.build_builtin_local_tools()}

    assert "target path is already verified" in tools["houyi_read_file"].description
    assert "narrow with find, list, or grep before read" in tools["houyi_read_file"].description
    assert (
        "path, skill location, or file choice is still unclear"
        in tools["houyi_find_files"].description
    )
    assert "verify workspace structure or candidate folders" in tools["houyi_list_dir"].description
    assert (
        "narrow multiple candidates before choosing which file to read"
        in tools["houyi_grep"].description
    )
    assert "prefer find, list, or grep before read" in tools["houyi_local_cli"].description
    assert "keep narrowing when multiple candidates remain" in tools["houyi_local_cli"].description


def test_register_tools() -> None:
    registry = SkillRegistry()

    registered = local_tools.register_builtin_local_tools(registry)

    assert len(registered) == 8
    assert all(registry.get(name) is not None for name in registered)


@pytest.mark.asyncio
async def test_read_file_line(workspace_root: Path) -> None:
    target = workspace_root / "a.txt"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")

    result = await local_tools._read_file_executor(path="a.txt", start_line=2, end_line=3)

    assert result["success"] is True
    assert result["data"]["content"] == "two\nthree"


@pytest.mark.asyncio
async def test_write_file(workspace_root: Path) -> None:
    result = await local_tools._write_file_executor(
        path="nested/target.txt",
        content="hello",
        create_parents=True,
    )

    assert result["success"] is True
    assert (workspace_root / "nested" / "target.txt").read_text(encoding="utf-8") == "hello"


@pytest.mark.asyncio
async def test_find_files_pattern(workspace_root: Path) -> None:
    (workspace_root / "keep.py").write_text("print('x')", encoding="utf-8")
    (workspace_root / "skip.txt").write_text("x", encoding="utf-8")

    result = await local_tools._find_files_executor(root_path=".", pattern="*.py")

    assert result["success"] is True
    assert any(item.endswith("keep.py") for item in result["data"]["matches"])
    assert all(not item.endswith("skip.txt") for item in result["data"]["matches"])


@pytest.mark.asyncio
async def test_find_files_iterative(workspace_root: Path) -> None:
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
    assert any(
        Path(item).parts[-3:] == ("src", "skills", "skill.md") for item in result["data"]["matches"]
    )
    assert result["data"]["iterative_subdirs"] is True
    assert any(Path(path).name == "src" for path in result["data"]["searched_dirs"])


@pytest.mark.asyncio
async def test_find_files_max_depth(
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
async def test_find_files_contains_mode(workspace_root: Path) -> None:
    (workspace_root / "skill.md").write_text("x", encoding="utf-8")

    result = await local_tools._find_files_executor(root_path=".", pattern="skill")

    assert result["success"] is True
    assert result["data"]["search_mode"] == "contains"
    assert result["data"]["effective_mode"] == "contains"
    assert any(item.endswith("skill.md") for item in result["data"]["matches"])


@pytest.mark.asyncio
async def test_find_files_glob_mode(workspace_root: Path) -> None:
    (workspace_root / "main.py").write_text("print('x')", encoding="utf-8")

    result = await local_tools._find_files_executor(root_path=".", pattern="*.py")

    assert result["success"] is True
    assert result["data"]["search_mode"] == "contains"
    assert result["data"]["effective_mode"] == "glob"
    assert any(item.endswith("main.py") for item in result["data"]["matches"])


@pytest.mark.asyncio
async def test_list_dir_non_recursive(workspace_root: Path) -> None:
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
async def test_grep_matches_lines(workspace_root: Path) -> None:
    (workspace_root / "x.txt").write_text("alpha\nneedle here\nomega\n", encoding="utf-8")

    result = await local_tools._grep_executor(path=".", query="needle")

    assert result["success"] is True
    assert len(result["data"]["matches"]) == 1
    assert result["data"]["matches"][0]["line"] == 2


@pytest.mark.asyncio
async def test_blocks_dangerous_command(workspace_root: Path) -> None:
    result = await local_tools._shell_exec_executor(command="sudo ls", cwd=".")

    assert result["success"] is False
    assert "Blocked dangerous command pattern" in result["message"]


@pytest.mark.asyncio
async def test_runs_safe_command(workspace_root: Path) -> None:
    result = await local_tools._shell_exec_executor(command="echo hello", cwd=".")

    assert result["success"] is True
    assert "hello" in result["data"]["stdout"]


@pytest.mark.asyncio
async def test_reject_outside_workspace(workspace_root: Path) -> None:
    outside_path = str(workspace_root.parent / "outside.txt")

    read_result = await local_tools._read_file_executor(path=outside_path)
    write_result = await local_tools._write_file_executor(path=outside_path, content="x")

    assert read_result["success"] is False
    assert write_result["success"] is False
    assert "workspace root" in read_result["message"]
    assert "workspace root" in write_result["message"]


@pytest.mark.asyncio
async def test_read_dispatch(workspace_root: Path) -> None:
    target = workspace_root / "a.txt"
    target.write_text("one\ntwo\n", encoding="utf-8")

    result = await local_tools._local_cli_executor(command="read", path="a.txt", start_line=2)

    assert result["success"] is True
    assert result["data"]["content"] == "two"


@pytest.mark.asyncio
async def test_grep_requires_query(workspace_root: Path) -> None:
    _ = workspace_root

    result = await local_tools._local_cli_executor(command="grep", path=".")

    assert result["success"] is False
    assert "query is required" in result["message"]


@pytest.mark.asyncio
async def test_chain_runs_pipe(workspace_root: Path) -> None:
    skills_dir = workspace_root / "houyi" / "skills" / "web_search"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("# Web Search\nweb search target\n", encoding="utf-8")

    result = await local_tools._local_cli_chain_executor(
        workflow=(
            "find path=houyi/skills pattern=SKILL.md search_mode=exact "
            "| read start_line=1 end_line=2"
        )
    )

    assert result["success"] is True
    steps = result["data"]["steps"]
    assert len(steps) == 2
    assert steps[0]["command"] == "find"
    assert steps[1]["command"] == "read"
    assert "Web Search" in steps[1]["result"]["data"]["content"]
    assert result["data"]["mode"] == "plan"
    assert result["data"]["workflow_id"] == "local_cli_chain"
    assert result["data"]["continuation_token"].startswith("local_cli_chain:v1:")
    assert result["data"]["replan_required"] is False
    assert result["data"]["repair_scope"] == "retry_failed_step_only"
    assert result["data"]["reused_step_count"] == 0
    assert len(result["data"]["frozen_success_steps"]) == 2


@pytest.mark.asyncio
async def test_chain_runs_structured_steps(workspace_root: Path) -> None:
    skills_dir = workspace_root / "houyi" / "skills" / "web_search"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("# Web Search\nweb search target\n", encoding="utf-8")

    result = await local_tools._local_cli_chain_executor(
        steps=[
            local_tools.LocalCliChainStepInput(
                command="find",
                path="houyi/skills",
                pattern="SKILL.md",
                search_mode="exact",
            ),
            local_tools.LocalCliChainStepInput(command="read", start_line=1, end_line=2),
        ]
    )

    assert result["success"] is True
    assert result["data"]["input_mode"] == "steps"
    steps = result["data"]["steps"]
    assert len(steps) == 2
    assert steps[0]["command"] == "find"
    assert steps[1]["command"] == "read"
    assert "Web Search" in steps[1]["result"]["data"]["content"]


@pytest.mark.asyncio
async def test_chain_normalizes_structured_steps(workspace_root: Path) -> None:
    skills_dir = workspace_root / "houyi" / "skills" / "web_search"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("# Web Search\nweb search target\n", encoding="utf-8")

    result = await local_tools._local_cli_chain_executor(
        steps=[
            {
                "command": "find",
                "path": "houyi/skills",
                "pattern": "SKILL.md",
                "search_mode": "exact",
            },
            {"command": "read", "start_line": 1, "end_line": 2},
        ]
    )

    assert result["success"] is True
    assert result["data"]["input_mode"] == "steps"
    steps = result["data"]["steps"]
    assert len(steps) == 2
    assert steps[0]["command"] == "find"
    assert steps[1]["command"] == "read"
    assert "Web Search" in steps[1]["result"]["data"]["content"]


@pytest.mark.asyncio
async def test_chain_runs(workspace_root: Path) -> None:
    skills_dir = workspace_root / "houyi" / "skills" / "web_search"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("# Web Search\nweb search target\n", encoding="utf-8")

    result = await local_tools._local_cli_chain_executor(
        workflow=(
            "find(path=houyi/skills, pattern=SKILL.md, search_mode=exact) "
            "| read(start_line=1, end_line=2)"
        )
    )

    assert result["success"] is True
    steps = result["data"]["steps"]
    assert len(steps) == 2
    assert steps[0]["command"] == "find"
    assert steps[1]["command"] == "read"
    assert "Web Search" in steps[1]["result"]["data"]["content"]


@pytest.mark.asyncio
async def test_chain_runs_positional_path(workspace_root: Path) -> None:
    skills_dir = workspace_root / "houyi" / "skills" / "web_search"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("# Web Search\nweb search target\n", encoding="utf-8")

    result = await local_tools._local_cli_chain_executor(
        workflow="read houyi/skills/web_search/SKILL.md start_line=1 end_line=1"
    )

    assert result["success"] is True
    steps = result["data"]["steps"]
    assert len(steps) == 1
    assert steps[0]["command"] == "read"
    assert "Web Search" in steps[0]["result"]["data"]["content"]


@pytest.mark.asyncio
async def test_chain_runs_fallback(workspace_root: Path) -> None:
    skills_dir = workspace_root / "houyi" / "skills" / "web_search"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("# Web Search\nweb search target\n", encoding="utf-8")

    result = await local_tools._local_cli_chain_executor(
        workflow=(
            "find path=houyi/skills pattern=missing.md search_mode=exact "
            "|| find path=houyi/skills pattern=SKILL.md search_mode=exact"
        )
    )

    assert result["success"] is True
    steps = result["data"]["steps"]
    assert len(steps) == 2
    assert steps[0]["success"] is False
    assert steps[1]["success"] is True
    assert steps[1]["result"]["data"]["matches"]


@pytest.mark.asyncio
async def test_chain_rejects_ambiguous(workspace_root: Path) -> None:
    web_search_dir = workspace_root / "houyi" / "skills" / "web_search"
    web_search_dir.mkdir(parents=True)
    (web_search_dir / "SKILL.md").write_text("# Web Search\nweb search target\n", encoding="utf-8")
    local_search_dir = workspace_root / "houyi" / "skills" / "local_search"
    local_search_dir.mkdir(parents=True)
    (local_search_dir / "SKILL.md").write_text(
        "# Local Search\nlocal search target\n", encoding="utf-8"
    )

    result = await local_tools._local_cli_chain_executor(
        workflow=(
            "find path=houyi/skills pattern=SKILL.md search_mode=exact "
            "| read start_line=1 end_line=1"
        )
    )

    assert result["success"] is False
    steps = result["data"]["steps"]
    assert len(steps) == 2
    assert steps[0]["success"] is True
    assert steps[1]["success"] is False
    assert steps[1]["result"]["data"]["failure_kind"] == "projection_failed"
    assert "unique path candidate" in steps[1]["result"]["message"]


@pytest.mark.asyncio
async def test_chain_narrowing_ambiguous_find(workspace_root: Path) -> None:
    web_search_dir = workspace_root / "houyi" / "skills" / "web_search"
    web_search_dir.mkdir(parents=True)
    (web_search_dir / "SKILL.md").write_text("# Web Search\nweb search target\n", encoding="utf-8")
    local_search_dir = workspace_root / "houyi" / "skills" / "local_search"
    local_search_dir.mkdir(parents=True)
    (local_search_dir / "SKILL.md").write_text(
        "# Local Search\nlocal search target\n", encoding="utf-8"
    )

    result = await local_tools._local_cli_chain_executor(
        workflow=(
            "find path=houyi/skills pattern=SKILL.md search_mode=exact "
            "| grep query='web search target' "
            "| read start_line=1 end_line=1"
        )
    )

    assert result["success"] is True
    steps = result["data"]["steps"]
    assert len(steps) == 3
    assert steps[0]["success"] is True
    assert steps[1]["success"] is True
    assert steps[2]["success"] is True
    projected = steps[1]["projection"]["path"].replace("\\", "/")
    assert projected.endswith("houyi/skills/web_search/SKILL.md")
    assert "Web Search" in steps[2]["result"]["data"]["content"]


@pytest.mark.asyncio
async def test_chain_fallback(workspace_root: Path) -> None:
    web_search_dir = workspace_root / "houyi" / "skills" / "web_search"
    web_search_dir.mkdir(parents=True)
    (web_search_dir / "SKILL.md").write_text("# Web Search\nweb search target\n", encoding="utf-8")
    local_search_dir = workspace_root / "houyi" / "skills" / "local_search"
    local_search_dir.mkdir(parents=True)
    (local_search_dir / "SKILL.md").write_text(
        "# Local Search\nlocal search target\n", encoding="utf-8"
    )

    result = await local_tools._local_cli_chain_executor(
        workflow=(
            "find path=houyi/skills pattern=SKILL.md search_mode=exact "
            "| read start_line=1 end_line=1 "
            "|| read path=houyi/skills/web_search/SKILL.md start_line=1 end_line=1"
        )
    )

    assert result["success"] is True
    steps = result["data"]["steps"]
    assert len(steps) == 3
    assert steps[1]["result"]["data"]["failure_kind"] == "projection_failed"
    assert steps[2]["success"] is True
    assert "Web Search" in steps[2]["result"]["data"]["content"]


@pytest.mark.asyncio
async def test_chain_missing_read_hint(workspace_root: Path) -> None:
    skills_dir = workspace_root / "houyi" / "skills" / "web_search"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("# Web Search\nweb search target\n", encoding="utf-8")

    result = await local_tools._local_cli_chain_executor(
        steps=[
            local_tools.LocalCliChainStepInput(
                command="read",
                path="houyi/skills/websearch/SKILL.md",
                start_line=1,
                end_line=1,
            )
        ]
    )

    assert result["success"] is False
    assert result["data"]["failure_kind"] == "step_execution_failed"
    assert "Do not guess another path directly" in result["data"]["recovery_hint"]
    assert "Prefer the provided recovery_step_template" in result["data"]["recovery_hint"]
    assert (
        "Only add extra narrowing if that first find still returns multiple candidates"
        in result["data"]["recovery_hint"]
    )
    template = result["data"]["recovery_step_template"]
    assert template["reason"] == "read_file_not_found"
    assert template["steps"][0]["command"] == "find"
    assert template["steps"][0]["path"] == "houyi/skills"
    assert template["steps"][0]["pattern"] == "SKILL.md"
    assert template["steps"][0]["search_mode"] == "exact"
    assert template["steps"][1]["command"] == "read"
    assert template["steps"][1]["path_from_previous_find"] is True
    assert template["steps"][1]["start_line"] == 1
    assert template["steps"][1]["end_line"] == 1


def test_chain_continue_needs_token() -> None:
    with pytest.raises(ValueError, match="continuation_token is required"):
        local_tools.LocalCliChainInput(
            mode="continue",
            steps=[local_tools.LocalCliChainStepInput(command="read", path="a.txt")],
            resume_from_step_index=0,
        )


def test_chain_repair_needs_fields() -> None:
    with pytest.raises(ValueError, match="failed_step_index is required"):
        local_tools.LocalCliChainInput(
            mode="repair",
            continuation_token="wf:repair",
            steps=[local_tools.LocalCliChainStepInput(command="read", path="a.txt")],
            repair_action="replace_failed_step",
        )

    with pytest.raises(ValueError, match="repair_action is required"):
        local_tools.LocalCliChainInput(
            mode="repair",
            continuation_token="wf:repair",
            failed_step_index=0,
            steps=[local_tools.LocalCliChainStepInput(command="read", path="a.txt")],
        )


@pytest.mark.asyncio
async def test_chain_reports_metadata(workspace_root: Path) -> None:
    target = workspace_root / "a.txt"
    target.write_text("one\ntwo\n", encoding="utf-8")

    result = await local_tools._local_cli_chain_executor(
        mode="continue",
        workflow_id="read_preview",
        continuation_token="read_preview:plan",
        resume_from_step_index=0,
        steps=[
            local_tools.LocalCliChainStepInput(
                command="read", path="a.txt", start_line=1, end_line=1
            )
        ],
    )

    assert result["success"] is True
    assert result["data"]["mode"] == "continue"
    assert result["data"]["workflow_id"] == "read_preview"
    assert result["data"]["input_continuation_token"] == "read_preview:plan"
    assert result["data"]["continuation_token"].startswith("local_cli_chain:v1:")
    assert result["data"]["resume_from_step_index"] == 0


@pytest.mark.asyncio
async def test_chain_reports_repair_metadata(workspace_root: Path) -> None:
    target = workspace_root / "a.txt"
    target.write_text("one\ntwo\n", encoding="utf-8")

    result = await local_tools._local_cli_chain_executor(
        mode="repair",
        workflow_id="read_preview",
        continuation_token="read_preview:plan",
        failed_step_index=0,
        repair_action="replace_failed_step",
        steps=[
            local_tools.LocalCliChainStepInput(
                command="read", path="a.txt", start_line=1, end_line=1
            )
        ],
    )

    assert result["success"] is True
    assert result["data"]["mode"] == "repair"
    assert result["data"]["workflow_id"] == "read_preview"
    assert result["data"]["input_continuation_token"] == "read_preview:plan"
    assert result["data"]["continuation_token"].startswith("local_cli_chain:v1:")
    assert result["data"]["failed_step_index"] == 0
    assert result["data"]["repair_action"] == "replace_failed_step"


@pytest.mark.asyncio
async def test_chain_continue_frozen_steps(workspace_root: Path) -> None:
    target = workspace_root / "a.txt"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")

    planned = await local_tools._local_cli_chain_executor(
        workflow_id="read_preview",
        steps=[
            local_tools.LocalCliChainStepInput(command="list", path="."),
            local_tools.LocalCliChainStepInput(
                command="read", path="a.txt", start_line=1, end_line=1
            ),
        ],
    )

    continued = await local_tools._local_cli_chain_executor(
        mode="continue",
        workflow_id="read_preview",
        continuation_token=planned["data"]["continuation_token"],
        resume_from_step_index=2,
        steps=[
            local_tools.LocalCliChainStepInput(
                command="read", path="a.txt", start_line=2, end_line=2
            )
        ],
    )

    assert continued["success"] is True
    assert continued["data"]["reused_step_count"] == 2
    assert continued["data"]["steps"][0]["reused"] is True
    assert continued["data"]["steps"][1]["reused"] is True
    assert continued["data"]["steps"][2]["step_index"] == 2
    assert continued["data"]["final"]["data"]["content"] == "two"


@pytest.mark.asyncio
async def test_chain_repair_rejects_step(workspace_root: Path) -> None:
    target = workspace_root / "a.txt"
    target.write_text("one\ntwo\n", encoding="utf-8")

    planned = await local_tools._local_cli_chain_executor(
        workflow_id="read_preview",
        steps=[
            local_tools.LocalCliChainStepInput(command="list", path="."),
            local_tools.LocalCliChainStepInput(
                command="read", path="a.txt", start_line=1, end_line=1
            ),
        ],
    )

    repaired = await local_tools._local_cli_chain_executor(
        mode="repair",
        workflow_id="read_preview",
        continuation_token=planned["data"]["continuation_token"],
        failed_step_index=1,
        repair_action="replace_failed_step",
        steps=[
            local_tools.LocalCliChainStepInput(
                command="read", path="a.txt", start_line=2, end_line=2
            )
        ],
    )

    assert repaired["success"] is False
    assert repaired["data"]["failure_kind"] == "replan_required"
    assert repaired["data"]["replan_required"] is True
    assert repaired["data"]["repair_scope"] == "retry_failed_step_only"


@pytest.mark.asyncio
async def test_chain_continue_rejects(workspace_root: Path) -> None:
    target = workspace_root / "a.txt"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")

    planned = await local_tools._local_cli_chain_executor(
        workflow_id="read_preview",
        steps=[
            local_tools.LocalCliChainStepInput(command="list", path="."),
            local_tools.LocalCliChainStepInput(
                command="read", path="a.txt", start_line=1, end_line=1
            ),
        ],
    )

    continued = await local_tools._local_cli_chain_executor(
        mode="continue",
        workflow_id="read_preview",
        continuation_token=planned["data"]["continuation_token"],
        resume_from_step_index=1,
        steps=[
            local_tools.LocalCliChainStepInput(
                command="read", path="a.txt", start_line=2, end_line=2
            )
        ],
    )

    assert continued["success"] is False
    assert continued["data"]["failure_kind"] == "replan_required"
    assert continued["data"]["replan_required"] is True
    assert continued["data"]["repair_scope"] == "retry_failed_step_only"


@pytest.mark.asyncio
async def test_chain_rejects_badstep(workspace_root: Path) -> None:
    _ = workspace_root

    result = await local_tools._local_cli_chain_executor(workflow="write path=a.txt")

    assert result["success"] is False
    assert "unsupported chain command" in result["message"]
    assert result["data"]["failure_kind"] == "unsupported_chain_command"
    assert "read, list, find, or grep" in result["data"]["recovery_hint"]


@pytest.mark.asyncio
async def test_chain_rejects_invalid_argument(workspace_root: Path) -> None:
    _ = workspace_root

    result = await local_tools._local_cli_chain_executor(workflow="read path=a.txt extra")

    assert result["success"] is False
    assert "invalid chain argument" in result["message"]
    assert result["data"]["failure_kind"] == "invalid_chain_argument"
    assert "function-style steps" in result["data"]["recovery_hint"]
