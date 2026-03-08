"""Tests for ToolCallService governance component wiring.

Verifies that ToolCallService._get_runner() correctly wires governance
components (policy_enforcer, consent_manager, metrics_store) from
SkillService into ToolCallRunner, and handles edge cases like missing
SkillService.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from houyi.application.tool_calling.runner import ToolCallRunner


class TestToolCallServiceGetRunner:
    """Tests for _get_runner() lazy initialization and governance wiring."""

    def _make_service(self, **overrides: Any) -> Any:
        """Create a ToolCallService with mocked dependencies."""
        # Import here to avoid module-level side effects
        from houyi_studio.server.tooling.service import ToolCallService

        return ToolCallService(
            connection_manager=overrides.get("connection_manager", MagicMock()),
            record_llm_call=overrides.get("record_llm_call", MagicMock()),
            tool_call_cache=overrides.get("tool_call_cache", {}),
            llm_tool_call_cache=overrides.get("llm_tool_call_cache", {}),
            skill_registry=overrides.get("skill_registry"),
        )

    def _patch_get_skill_service(self, monkeypatch: Any, *, side_effect: Any = None) -> MagicMock:
        from houyi_studio.server.tooling import service as tooling_service

        mock_get_svc = MagicMock(side_effect=side_effect)
        monkeypatch.setattr(tooling_service, "get_skill_service", mock_get_svc)
        return mock_get_svc

    def test_runner_is_tool_call_runner(self, monkeypatch) -> None:
        """_get_runner returns a ToolCallRunner instance."""
        mock_get_svc = self._patch_get_skill_service(monkeypatch)
        mock_svc = MagicMock()
        mock_svc.policy_enforcer = None
        mock_svc.consent_manager = None
        mock_svc.metrics_store = None
        mock_get_svc.return_value = mock_svc

        svc = self._make_service()
        runner = svc._get_runner()

        assert isinstance(runner, ToolCallRunner)

    def test_governance_components_wired_from_skill_service(self, monkeypatch) -> None:
        """policy_enforcer, consent_manager, metrics_store from SkillService are passed to ToolCallRunner."""
        mock_get_svc = self._patch_get_skill_service(monkeypatch)
        mock_policy = MagicMock()
        mock_consent = MagicMock()
        mock_metrics = MagicMock()

        mock_svc = MagicMock()
        mock_svc.policy_enforcer = mock_policy
        mock_svc.consent_manager = mock_consent
        mock_svc.metrics_store = mock_metrics
        mock_get_svc.return_value = mock_svc

        svc = self._make_service()
        runner = svc._get_runner()

        assert runner.policy_enforcer is mock_policy
        assert runner.consent_manager is mock_consent
        assert runner.metrics_store is mock_metrics

    def test_runner_has_hooks_manager(self, monkeypatch) -> None:
        """Runner should have the DEFAULT_HOOKS_MANAGER regardless of SkillService."""
        from houyi.domain.skill.hooks import DEFAULT_HOOKS_MANAGER

        mock_get_svc = self._patch_get_skill_service(monkeypatch)
        mock_svc = MagicMock()
        mock_svc.policy_enforcer = None
        mock_svc.consent_manager = None
        mock_svc.metrics_store = None
        mock_get_svc.return_value = mock_svc

        svc = self._make_service()
        runner = svc._get_runner()

        assert runner.skill_hooks_manager is DEFAULT_HOOKS_MANAGER

    def test_lazy_initialization_caches_runner(self, monkeypatch) -> None:
        """Second call to _get_runner() returns the cached instance."""
        mock_get_svc = self._patch_get_skill_service(monkeypatch)
        mock_svc = MagicMock()
        mock_svc.policy_enforcer = None
        mock_svc.consent_manager = None
        mock_svc.metrics_store = None
        mock_get_svc.return_value = mock_svc

        svc = self._make_service()
        runner1 = svc._get_runner()
        runner2 = svc._get_runner()

        assert runner1 is runner2
        # get_skill_service should only be called once (lazy init)
        assert mock_get_svc.call_count == 1

    def test_skill_service_unavailable_falls_back_gracefully(self, monkeypatch) -> None:
        """When SkillService is not available, runner is still created without governance."""
        self._patch_get_skill_service(
            monkeypatch,
            side_effect=RuntimeError("SkillService not initialized"),
        )

        svc = self._make_service()
        runner = svc._get_runner()

        assert isinstance(runner, ToolCallRunner)
        assert runner.policy_enforcer is None
        assert runner.consent_manager is None
        assert runner.metrics_store is None

    def test_skill_service_returns_none_properties(self, monkeypatch) -> None:
        """When SkillService properties are None, runner still works."""
        mock_get_svc = self._patch_get_skill_service(monkeypatch)
        mock_svc = MagicMock()
        mock_svc.policy_enforcer = None
        mock_svc.consent_manager = None
        mock_svc.metrics_store = None
        mock_get_svc.return_value = mock_svc

        svc = self._make_service()
        runner = svc._get_runner()

        assert runner.policy_enforcer is None
        assert runner.consent_manager is None
        assert runner.metrics_store is None


class TestToolCallServiceSelectSkills:
    """Tests for _select_skills() registry lookup."""

    def _make_service(self, registry: Any = None) -> Any:
        from houyi_studio.server.tooling.service import ToolCallService

        return ToolCallService(
            connection_manager=MagicMock(),
            record_llm_call=MagicMock(),
            tool_call_cache={},
            llm_tool_call_cache={},
            skill_registry=registry,
        )

    def test_returns_matching_skills(self) -> None:
        mock_registry = MagicMock()
        skill_a = MagicMock()
        skill_b = MagicMock()
        mock_registry.get.side_effect = lambda name: {"a": skill_a, "b": skill_b}.get(name)

        svc = self._make_service(registry=mock_registry)
        result = svc._select_skills(["a", "b"])

        assert result == [skill_a, skill_b]

    def test_skips_unknown_skills(self) -> None:
        mock_registry = MagicMock()
        skill_a = MagicMock()
        mock_registry.get.side_effect = lambda name: {"a": skill_a}.get(name)

        svc = self._make_service(registry=mock_registry)
        result = svc._select_skills(["a", "unknown"])

        assert result == [skill_a]

    def test_empty_tool_names(self) -> None:
        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        svc = self._make_service(registry=mock_registry)
        result = svc._select_skills([])

        assert result == []


class TestToolCallServiceExecute:
    def _make_service(self, registry: Any = None) -> Any:
        from houyi_studio.server.tooling.service import ToolCallService

        return ToolCallService(
            connection_manager=MagicMock(),
            record_llm_call=MagicMock(),
            tool_call_cache={},
            llm_tool_call_cache={},
            skill_registry=registry,
        )

    async def test_execute_tool_calls_returns_adapter_init_error_when_openai_missing(
        self,
        monkeypatch,
    ) -> None:
        from houyi_studio.server.tooling import service as tooling_service

        from houyi.adapters import llm as llm_adapters

        svc = self._make_service(registry=MagicMock())

        skill = SimpleNamespace(name="demo", preprocessors=[])
        svc._select_skills = lambda tool_names: [skill]
        svc._tool_bridge = SimpleNamespace(
            collect_tool_schemas=lambda skill_filter, include_core: [
                {"type": "function", "function": {"name": "demo"}}
            ]
        )

        with (
            patch.object(tooling_service, "ConsoleToolCallResponseAssembler"),
            patch.object(
                tooling_service,
                "wrap_tool_choice",
                side_effect=lambda adapter, tool_choice: adapter,
            ),
            patch.object(
                tooling_service.ToolCallAdapterRegistry, "resolve"
            ) as mock_registry_resolve,
            patch.object(
                llm_adapters,
                "OpenAIAdapter",
                side_effect=ImportError("openai missing"),
            ),
        ):
            mock_registry_resolve.side_effect = lambda request, fallback_factory: SimpleNamespace(
                inner=fallback_factory()
            )

            monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")

            execution = SimpleNamespace(metadata={})
            node_exec = SimpleNamespace(outputs=None, error=None)

            handled = await svc.execute_tool_calls(
                session_id="s1",
                execution=execution,
                node_id="node_1",
                node_exec=node_exec,
                prompt="Say hello in one sentence",
                system_prompt=None,
                user_prompt="Say hello in one sentence",
                model="deepseek-ai/DeepSeek-V3",
                tool_names=["demo"],
                tool_choice=None,
                max_tool_calls=1,
            )

        assert handled is True
        assert isinstance(node_exec.error, str)
        assert "Tool-calling adapter init failed" in node_exec.error
        assert "ImportError" in node_exec.error
