"""Tests for planning-with-files skill.

Tests the planning skill's core functionality:
- Plan creation and parsing
- Progress tracking
- Hook execution
- Subtask updates
"""

import tempfile
from pathlib import Path

import pytest

from houyi.domain.skill.policy import ModelAutoInvoke
from houyi.skills.planning import (
    PlanningSkill,
    create_plan,
    find_plan_file,
    get_progress,
    is_plan_complete,
    parse_plan,
    update_plan_subtask,
)


class TestParsePlan:
    """Test plan parsing functionality."""

    def test_parse_empty_plan(self):
        content = """# Task: Test Task

## Status: in_progress

## Subtasks

## Notes
"""
        plan = parse_plan(content)
        assert plan["task"] == "Test Task"
        assert plan["status"] == "in_progress"
        assert plan["subtasks"] == []

    def test_parse_plan_with_subtasks(self):
        content = """# Task: Implement Feature

## Status: in_progress

## Subtasks

- [ ] Design the API
- [x] Write unit tests
- [ ] Implement core logic
- [x] Add documentation

## Notes

Some notes here.
"""
        plan = parse_plan(content)
        assert plan["task"] == "Implement Feature"
        assert plan["status"] == "in_progress"
        assert len(plan["subtasks"]) == 4
        assert plan["subtasks"][0]["description"] == "Design the API"
        assert plan["subtasks"][0]["completed"] is False
        assert plan["subtasks"][1]["description"] == "Write unit tests"
        assert plan["subtasks"][1]["completed"] is True
        assert "Some notes here" in plan["notes"]

    def test_parse_completed_plan(self):
        content = """# Task: Done Task

## Status: completed

## Subtasks

- [x] Task 1
- [x] Task 2
"""
        plan = parse_plan(content)
        assert plan["status"] == "completed"
        assert all(s["completed"] for s in plan["subtasks"])


class TestGetProgress:
    """Test progress calculation."""

    def test_empty_progress(self):
        plan = {"subtasks": []}
        progress = get_progress(plan)
        assert progress["total"] == 0
        assert progress["completed"] == 0
        assert progress["remaining"] == 0
        assert progress["percentage"] == 0

    def test_partial_progress(self):
        plan = {
            "subtasks": [
                {"description": "A", "completed": True},
                {"description": "B", "completed": False},
                {"description": "C", "completed": True},
                {"description": "D", "completed": False},
            ]
        }
        progress = get_progress(plan)
        assert progress["total"] == 4
        assert progress["completed"] == 2
        assert progress["remaining"] == 2
        assert progress["percentage"] == 50.0

    def test_complete_progress(self):
        plan = {
            "subtasks": [
                {"description": "A", "completed": True},
                {"description": "B", "completed": True},
            ]
        }
        progress = get_progress(plan)
        assert progress["total"] == 2
        assert progress["completed"] == 2
        assert progress["remaining"] == 0
        assert progress["percentage"] == 100.0


class TestIsComplete:
    """Test completion checking."""

    def test_empty_is_complete(self):
        plan = {"subtasks": []}
        assert is_plan_complete(plan)

    def test_incomplete(self):
        plan = {
            "subtasks": [
                {"completed": True},
                {"completed": False},
            ]
        }
        assert not is_plan_complete(plan)

    def test_complete(self):
        plan = {
            "subtasks": [
                {"completed": True},
                {"completed": True},
            ]
        }
        assert is_plan_complete(plan)


class TestCreatePlan:
    """Test plan creation."""

    def test_create_basic_plan(self):
        content = create_plan("My Task")
        assert "# Task: My Task" in content
        assert "## Status: in_progress" in content
        assert "- [ ]" in content

    def test_create_plan_with_subtasks(self):
        subtasks = ["Step 1", "Step 2", "Step 3"]
        content = create_plan("My Task", subtasks)
        assert "# Task: My Task" in content
        assert "- [ ] Step 1" in content
        assert "- [ ] Step 2" in content
        assert "- [ ] Step 3" in content


class TestUpdatePlanSubtask:
    """Test subtask update functionality."""

    def test_complete_subtask(self):
        content = """# Task: Test

## Subtasks

- [ ] Task 1
- [ ] Task 2
- [ ] Task 3
"""
        updated = update_plan_subtask(content, 1, completed=True)
        assert "- [ ] Task 1" in updated
        assert "- [x] Task 2" in updated
        assert "- [ ] Task 3" in updated

    def test_uncomplete_subtask(self):
        content = """# Task: Test

## Subtasks

- [x] Task 1
- [x] Task 2
"""
        updated = update_plan_subtask(content, 0, completed=False)
        assert "- [ ] Task 1" in updated
        assert "- [x] Task 2" in updated


class TestFindPlanFile:
    """Test plan file discovery."""

    def test_find_in_current_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            plan_path = tmpdir / "PLAN.md"
            plan_path.write_text("# Task: Test")

            found = find_plan_file(tmpdir)
            assert found == plan_path

    def test_find_in_plan_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            plan_dir = tmpdir / ".plan"
            plan_dir.mkdir()
            plan_path = plan_dir / "PLAN.md"
            plan_path.write_text("# Task: Test")

            found = find_plan_file(tmpdir)
            assert found == plan_path

    def test_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            found = find_plan_file(tmpdir)
            assert found is None


class TestPlanningSkill:
    """Test PlanningSkill class."""

    @pytest.mark.asyncio
    async def test_create_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            skill = PlanningSkill(workspace=tmpdir)

            result = await skill.execute(
                "create",
                task="Test Task",
                subtasks=["Step 1", "Step 2"],
            )

            assert result["success"]
            assert "plan_path" in result

            plan_path = Path(result["plan_path"])
            assert plan_path.exists()

            content = plan_path.read_text()
            assert "Test Task" in content
            assert "Step 1" in content

    @pytest.mark.asyncio
    async def test_get_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            skill = PlanningSkill(workspace=tmpdir)

            # Create plan first
            await skill.execute(
                "create",
                task="Test Task",
                subtasks=["Step 1", "Step 2"],
            )

            # Get status
            result = await skill.execute("status")

            assert result["success"]
            assert result["task"] == "Test Task"
            assert result["status"] == "in_progress"
            assert result["progress"]["total"] == 2
            assert result["progress"]["completed"] == 0

    @pytest.mark.asyncio
    async def test_update_subtask(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            skill = PlanningSkill(workspace=tmpdir)

            # Create plan
            await skill.execute(
                "create",
                task="Test Task",
                subtasks=["Step 1", "Step 2"],
            )

            # Update subtask
            result = await skill.execute(
                "update",
                subtask_index=0,
                completed=True,
            )

            assert result["success"]
            assert result["progress"]["completed"] == 1

    @pytest.mark.asyncio
    async def test_create_requires_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            skill = PlanningSkill(workspace=tmpdir)

            result = await skill.execute("create")
            assert not result["success"]
            assert "Missing required field for create: task" in result["message"]

    @pytest.mark.asyncio
    async def test_update_requires_subtask_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            skill = PlanningSkill(workspace=tmpdir)

            await skill.execute(
                "create",
                task="Test Task",
                subtasks=["Step 1"],
            )

            result = await skill.execute("update", completed=True)
            assert not result["success"]
            assert "Missing required field for update: subtask_index" in result["message"]

    @pytest.mark.asyncio
    async def test_update_rejects_invalid_subtask_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            skill = PlanningSkill(workspace=tmpdir)

            await skill.execute(
                "create",
                task="Test Task",
                subtasks=["Step 1"],
            )

            bad_type = await skill.execute("update", subtask_index="abc")
            assert not bad_type["success"]
            assert "must be an integer" in bad_type["message"]

            negative = await skill.execute("update", subtask_index=-1)
            assert not negative["success"]
            assert "must be >= 0" in negative["message"]

    @pytest.mark.asyncio
    async def test_complete_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            skill = PlanningSkill(workspace=tmpdir)

            # Create plan
            await skill.execute(
                "create",
                task="Test Task",
                subtasks=["Step 1"],
            )

            # Try to complete (should fail - subtasks incomplete)
            result = await skill.execute("complete")
            assert not result["success"]

            # Complete subtask
            await skill.execute("update", subtask_index=0, completed=True)

            # Now complete should work
            result = await skill.execute("complete")
            assert result["success"]
            assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_status_no_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            skill = PlanningSkill(workspace=tmpdir)

            result = await skill.execute("status")
            assert not result["success"]
            assert "No plan file" in result["message"]

    def test_to_spec(self):
        skill = PlanningSkill()
        spec = skill.to_spec()
        schema = spec.input_schema.model_json_schema()
        props = schema.get("properties", {})

        assert spec.name == "planning-with-files"
        assert spec.version == "1.0.0"
        assert spec.user_invocable is True
        assert len(spec.hooks) > 0
        assert callable(spec.executor)
        assert spec.invocation_policy.model_auto_invoke == ModelAutoInvoke.ALLOW_WITH_CONSENT
        assert props.get("action", {}).get("enum") == ["create", "update", "complete", "status"]
        assert props.get("subtask_index", {}).get("minimum") == 0
        assert props.get("completed", {}).get("type") == "boolean"
        assert "status" not in props


class TestPreWriteHook:
    """Test pre_write_hook function."""

    @pytest.mark.asyncio
    async def test_no_plan_file(self):
        from houyi.domain.skill.hooks import HookContext
        from houyi.skills.planning.hooks import pre_write_hook

        with tempfile.TemporaryDirectory() as tmpdir:
            context = HookContext(cwd=Path(tmpdir), tool_name="Write")
            result = await pre_write_hook(context)

            assert result.success
            assert result.output is None

    @pytest.mark.asyncio
    async def test_with_plan_file(self):
        from houyi.domain.skill.hooks import HookContext
        from houyi.skills.planning.hooks import pre_write_hook

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            plan_path = tmpdir / "PLAN.md"
            plan_path.write_text("""# Task: Test Task

## Status: in_progress

## Subtasks

- [ ] Task 1
- [x] Task 2
- [ ] Task 3
""")
            context = HookContext(cwd=tmpdir, tool_name="Write")
            result = await pre_write_hook(context)

            assert result.success
            assert result.output is not None
            assert "Test Task" in result.output
            assert "Progress:" in result.output

    @pytest.mark.asyncio
    async def test_with_many_remaining_tasks(self):
        from houyi.domain.skill.hooks import HookContext
        from houyi.skills.planning.hooks import pre_write_hook

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            plan_path = tmpdir / "PLAN.md"
            plan_path.write_text("""# Task: Big Task

## Status: in_progress

## Subtasks

- [ ] Task 1
- [ ] Task 2
- [ ] Task 3
- [ ] Task 4
- [ ] Task 5
""")
            context = HookContext(cwd=tmpdir, tool_name="Write")
            result = await pre_write_hook(context)

            assert result.success
            assert "... and" in (result.output or "")

    @pytest.mark.asyncio
    async def test_handles_exception(self):
        from houyi.domain.skill.hooks import HookContext
        from houyi.skills.planning.hooks import pre_write_hook

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            plan_path = tmpdir / "PLAN.md"
            plan_path.write_text("invalid plan content")

            context = HookContext(cwd=tmpdir, tool_name="Write")

            # Even if parsing fails, should not crash
            result = await pre_write_hook(context)
            assert result.success


class TestPostToolHook:
    """Test post_tool_hook function."""

    @pytest.mark.asyncio
    async def test_no_plan_file(self):
        from houyi.domain.skill.hooks import HookContext
        from houyi.skills.planning.hooks import post_tool_hook

        with tempfile.TemporaryDirectory() as tmpdir:
            context = HookContext(cwd=Path(tmpdir), tool_name="Write")
            result = await post_tool_hook(context)

            assert result.success

    @pytest.mark.asyncio
    async def test_with_plan_file(self):
        from houyi.domain.skill.hooks import HookContext
        from houyi.skills.planning.hooks import post_tool_hook

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            plan_path = tmpdir / "PLAN.md"
            plan_path.write_text("""# Task: Test Task

## Status: in_progress

## Subtasks

- [ ] Task 1
- [x] Task 2
""")
            context = HookContext(cwd=tmpdir, tool_name="Write", tool_result={"success": True})
            result = await post_tool_hook(context)

            assert result.success
            assert "Progress:" in (result.output or "")

    @pytest.mark.asyncio
    async def test_handles_exception(self):
        from unittest.mock import patch

        from houyi.domain.skill.hooks import HookContext
        from houyi.skills.planning.hooks import post_tool_hook

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            plan_path = tmpdir / "PLAN.md"
            plan_path.write_text("# Task: Test")

            context = HookContext(cwd=tmpdir)

            with patch(
                "houyi.skills.planning.hooks.parse_plan", side_effect=Exception("Parse error")
            ):
                result = await post_tool_hook(context)

            assert result.success


class TestStopHook:
    """Test stop_hook function."""

    @pytest.mark.asyncio
    async def test_no_plan_file(self):
        from houyi.domain.skill.hooks import HookContext
        from houyi.skills.planning.hooks import stop_hook

        with tempfile.TemporaryDirectory() as tmpdir:
            context = HookContext(cwd=Path(tmpdir))
            result = await stop_hook(context)

            assert result.success

    @pytest.mark.asyncio
    async def test_complete_plan(self):
        from houyi.domain.skill.hooks import HookContext
        from houyi.skills.planning.hooks import stop_hook

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            plan_path = tmpdir / "PLAN.md"
            plan_path.write_text("""# Task: Done Task

## Status: completed

## Subtasks

- [x] Task 1
- [x] Task 2
""")
            context = HookContext(cwd=tmpdir)
            result = await stop_hook(context)

            assert result.success
            assert "complete" in (result.output or "").lower()

    @pytest.mark.asyncio
    async def test_incomplete_plan(self):
        from houyi.domain.skill.hooks import HookContext
        from houyi.skills.planning.hooks import stop_hook

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            plan_path = tmpdir / "PLAN.md"
            plan_path.write_text("""# Task: Incomplete Task

## Status: in_progress

## Subtasks

- [ ] Task 1
- [x] Task 2
- [ ] Task 3
""")
            context = HookContext(cwd=tmpdir)
            result = await stop_hook(context)

            assert result.success  # Doesn't hard block
            assert "WARNING" in (result.output or "")
            assert "incomplete" in (result.output or "").lower()

    @pytest.mark.asyncio
    async def test_many_incomplete_tasks(self):
        from houyi.domain.skill.hooks import HookContext
        from houyi.skills.planning.hooks import stop_hook

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            plan_path = tmpdir / "PLAN.md"
            plan_path.write_text("""# Task: Big Incomplete Task

## Status: in_progress

## Subtasks

- [ ] Task 1
- [ ] Task 2
- [ ] Task 3
- [ ] Task 4
- [ ] Task 5
- [ ] Task 6
- [ ] Task 7
""")
            context = HookContext(cwd=tmpdir)
            result = await stop_hook(context)

            assert "... and" in (result.output or "")

    @pytest.mark.asyncio
    async def test_handles_exception(self):
        from unittest.mock import patch

        from houyi.domain.skill.hooks import HookContext
        from houyi.skills.planning.hooks import stop_hook

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            plan_path = tmpdir / "PLAN.md"
            plan_path.write_text("# Task: Test")

            context = HookContext(cwd=tmpdir)

            with patch(
                "houyi.skills.planning.hooks.parse_plan", side_effect=Exception("Parse error")
            ):
                result = await stop_hook(context)

            assert result.success


class TestFindPlanFileExtended:
    """Extended tests for find_plan_file."""

    def test_find_in_parent_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            plan_path = tmpdir / "PLAN.md"
            plan_path.write_text("# Task: Test")

            subdir = tmpdir / "sub" / "dir"
            subdir.mkdir(parents=True)

            found = find_plan_file(subdir)
            assert found == plan_path

    def test_default_cwd(self):
        # Test that it uses Path.cwd() by default
        result = find_plan_file(None)
        # Should not crash
        assert result is None or isinstance(result, Path)
