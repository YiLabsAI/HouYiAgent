from __future__ import annotations

from dataclasses import dataclass

from houyi.application.tool_calling.tool_call_messages import (
    apply_fast_path_tool_choice,
    build_fast_path_prompt,
    build_tool_call_messages,
)


@dataclass
class _Skill:
    name: str


class TestToolCallMessages:
    def test_apply_fast_path_tool_choice_disabled_keeps(self) -> None:
        assert apply_fast_path_tool_choice(fast_path_enabled=False, tool_choice=None) is None
        assert apply_fast_path_tool_choice(fast_path_enabled=False, tool_choice="auto") == "auto"
        assert (
            apply_fast_path_tool_choice(fast_path_enabled=False, tool_choice="required")
            == "required"
        )

    def test_apply_fast_path_tool_choice_enabled_sets_required_for_none_or_auto(self) -> None:
        assert apply_fast_path_tool_choice(fast_path_enabled=True, tool_choice=None) == "required"
        assert apply_fast_path_tool_choice(fast_path_enabled=True, tool_choice="auto") == "required"

    def test_apply_fast_path_tool_choice_enabled_keeps_explicit_choice(self) -> None:
        assert (
            apply_fast_path_tool_choice(fast_path_enabled=True, tool_choice="required")
            == "required"
        )
        assert apply_fast_path_tool_choice(
            fast_path_enabled=True, tool_choice={"type": "function"}
        ) == {"type": "function"}

    def test_build_fast_path_prompt_prefers_tool_names(self) -> None:
        prompt = build_fast_path_prompt(tool_names=["a", "b"], skills=[_Skill("x")])  # type: ignore[arg-type]
        assert "Call each tool exactly once" in prompt
        assert "a, b" in prompt

    def test_build_fast_path_prompt_falls_back_to_skills(self) -> None:
        prompt = build_fast_path_prompt(tool_names=[], skills=[_Skill("x"), _Skill("y")])  # type: ignore[arg-type]
        assert "x, y" in prompt

    def test_build_tool_call_messages_basic(self) -> None:
        result = build_tool_call_messages(
            system_prompt="sys",
            user_prompt=None,
            prompt="p",
            fast_path_enabled=False,
            fast_path_prompt="fp",
            tool_choice=None,
        )
        assert result.user_content == "p"
        assert result.tool_choice is None
        assert result.messages == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "p"},
        ]

    def test_build_tool_call_messages_fast_path_adds_prompt_and_sets_choice(self) -> None:
        result = build_tool_call_messages(
            system_prompt=None,
            user_prompt="u",
            prompt="p",
            fast_path_enabled=True,
            fast_path_prompt="fp",
            tool_choice="auto",
        )
        assert result.user_content == "u"
        assert result.tool_choice == "required"
        assert result.messages == [
            {"role": "system", "content": "fp"},
            {"role": "user", "content": "u"},
        ]
