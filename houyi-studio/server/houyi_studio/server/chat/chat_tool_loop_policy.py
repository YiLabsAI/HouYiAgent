from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import SendMessageRequest


@dataclass
class ToolLoopGateDecision:
    enabled_skills: list[str]
    mode: str
    reason: str


class ChatToolLoopPolicy:
    def __init__(
        self,
        *,
        builtin_tool_names: set[str] | frozenset[str],
        explicit_tool_names: set[str] | frozenset[str],
        web_search_skill_name: str,
        repo_tool_skills: set[str] | frozenset[str],
        web_tool_skills: set[str] | frozenset[str],
        conservative_strategy: str,
        balanced_strategy: str,
        aggressive_strategy: str,
        looks_like_repo_intent: Any,
        looks_like_web_intent: Any,
        looks_like_tool_intent: Any,
    ) -> None:
        self._builtin_tool_names = builtin_tool_names
        self._explicit_tool_names = explicit_tool_names
        self._web_search_skill_name = web_search_skill_name
        self._repo_tool_skills = repo_tool_skills
        self._web_tool_skills = web_tool_skills
        self._conservative_strategy = conservative_strategy
        self._balanced_strategy = balanced_strategy
        self._aggressive_strategy = aggressive_strategy
        self._looks_like_repo_intent = looks_like_repo_intent
        self._looks_like_web_intent = looks_like_web_intent
        self._looks_like_tool_intent = looks_like_tool_intent

    def resolve_enabled_chat_skills(self, request: SendMessageRequest) -> list[str]:
        if request.enable_tool_calls is False:
            return []
        resolved: set[str] = set(self._builtin_tool_names)
        user_content = str(request.content or "").strip()
        if self._looks_like_tool_intent(user_content):
            resolved.update(self._explicit_tool_names)
        auto_web_search = self._looks_like_web_intent(user_content)
        if request.enable_web_search or auto_web_search:
            resolved.add(self._web_search_skill_name)
        if request.enable_deep_research:
            resolved.add("deep_research")
        if request.enable_skills:
            for skill_name in request.enable_skills:
                if isinstance(skill_name, str) and skill_name.strip():
                    resolved.add(skill_name.strip())
        return sorted(resolved)

    def gate_tool_loop(
        self,
        *,
        request: SendMessageRequest,
        resolved_skills: list[str],
    ) -> ToolLoopGateDecision:
        if request.enable_tool_calls is False:
            return ToolLoopGateDecision([], "disabled_by_request", "request_disable")
        if not resolved_skills:
            return ToolLoopGateDecision([], "disabled_no_skills", "no_resolved_skills")
        explicit_skills = [
            name.strip()
            for name in (request.enable_skills or [])
            if isinstance(name, str) and name.strip()
        ]
        strategy = (request.tool_call_strategy or self._balanced_strategy).strip().lower()
        if strategy not in {
            self._conservative_strategy,
            self._balanced_strategy,
            self._aggressive_strategy,
        }:
            strategy = self._balanced_strategy
        user_content = str(request.content or "").strip()
        if request.enable_web_search:
            web_skills = [
                skill_name for skill_name in resolved_skills if skill_name in self._web_tool_skills
            ]
            if web_skills:
                return ToolLoopGateDecision(web_skills, "enabled", "explicit_web_search_request")
        if explicit_skills:
            explicit_requested = set(explicit_skills)
            filtered_skills = [
                skill_name
                for skill_name in resolved_skills
                if skill_name in explicit_requested or skill_name == self._web_search_skill_name
            ]
            if self._looks_like_web_intent(user_content):
                filtered_skills = [
                    skill_name
                    for skill_name in filtered_skills
                    if skill_name in self._web_tool_skills
                ]
            return ToolLoopGateDecision(filtered_skills, "enabled", "explicit_skill_request")
        if strategy == self._aggressive_strategy:
            return ToolLoopGateDecision(
                resolved_skills,
                "enabled",
                "strategy_aggressive_default_on",
            )
        if strategy == self._conservative_strategy:
            return ToolLoopGateDecision(
                [],
                "disabled_by_gating",
                "strategy_conservative_requires_explicit",
            )
        if self._looks_like_repo_intent(user_content):
            repo_skills = [
                skill_name for skill_name in resolved_skills if skill_name in self._repo_tool_skills
            ]
            if repo_skills:
                return ToolLoopGateDecision(repo_skills, "enabled", "heuristic_repo_intent")
        if self._looks_like_web_intent(user_content):
            web_skills = [
                skill_name for skill_name in resolved_skills if skill_name in self._web_tool_skills
            ]
            if web_skills:
                return ToolLoopGateDecision(web_skills, "enabled", "heuristic_web_intent")
        if self._looks_like_tool_intent(user_content):
            return ToolLoopGateDecision(resolved_skills, "enabled", "heuristic_tool_intent")
        return ToolLoopGateDecision([], "disabled_by_gating", "heuristic_no_tool_intent")
