"""Unit tests for ResearchService."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from houyi_studio.server.research.service import (
    ResearchService,
    RunNotFoundError,
    RunNotTerminalError,
    VersionConflictError,
    _ArchivedRun,
    _normalize_report_sections,
)

from houyi.adapters.llm.base import LLMAdapter, LLMResponse, StreamChunk
from houyi.application.research.types import (
    OrchestrationMode,
    PlanEdit,
    PlanEditOperation,
    PlanStatus,
    ResearchSettings,
    ResearchStatus,
    SearchResult,
    SourceReference,
)
from houyi.application.research.url_validator import (
    URLValidationReport,
    URLValidationResult,
    URLValidator,
)
from houyi.infrastructure.config.env_config import (
    ENV_RESEARCH_MAX_AGENTS,
    ENV_RESEARCH_ORCHESTRATION_MODE,
)
from houyi.skills.web_search.service import WebSearchService
from houyi.skills.web_search.types import (
    WebSearchMetadata,
    WebSearchResponse,
    WebSearchResult,
)

_QUICK = ResearchSettings(depth="quick")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PLAN_JSON = json.dumps(
    {
        "sub_questions": [
            {"question": "Q1", "priority": 5, "search_strategy": "web", "expected_sources": 3},
            {"question": "Q2", "priority": 4, "search_strategy": "web", "expected_sources": 3},
            {"question": "Q3", "priority": 3, "search_strategy": "web", "expected_sources": 3},
            {"question": "Q4", "priority": 2, "search_strategy": "web", "expected_sources": 3},
            {"question": "Q5", "priority": 1, "search_strategy": "web", "expected_sources": 3},
        ],
        "outline": [
            {"title": "Section 1", "objective": "Cover Q1", "related_question_ids": [0]},
            {"title": "Section 2", "objective": "Cover Q2", "related_question_ids": [1]},
            {"title": "Section 3", "objective": "Cover Q3", "related_question_ids": [2]},
            {"title": "Section 4", "objective": "Cover Q4", "related_question_ids": [3]},
            {"title": "Section 5", "objective": "Cover Q5", "related_question_ids": [4]},
        ],
        "estimated_duration_min": 5,
    }
)

_SEARCHER_RESPONSE = json.dumps(
    {
        "sources": [
            {"url": "https://example.com/1", "title": "Source 1", "snippet": "snippet"},
        ],
        "summary": "Found relevant sources",
        "queries_used": ["test query"],
    }
)
_SECTION = json.dumps({"content": "Body text [ref_001].", "citations": []})
_RACE = json.dumps(
    {
        "comprehensiveness": {"score": 80, "reasoning": "ok"},
        "depth": {"score": 75, "reasoning": "ok"},
        "instruction_following": {"score": 85, "reasoning": "ok"},
        "readability": {"score": 90, "reasoning": "ok"},
    }
)
_FACT = json.dumps({"citation_accuracy": 90.0, "effective_citations": 5})

_QUERY_GEN = json.dumps(["test search query", "alternate query"])
_SUFFICIENCY = json.dumps({"sufficient": True, "rationale": "Enough"})


def _plan_with_clarification(plan_json: str, clarification: dict[str, Any]) -> str:
    plan_data = json.loads(plan_json)
    plan_data["clarification"] = clarification
    return json.dumps(plan_data)


def _all_responses() -> list[str]:
    return [_PLAN_JSON, _QUERY_GEN, _SUFFICIENCY, _SECTION, "Summary.", _RACE, _FACT]


class _MockLLM(LLMAdapter):
    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or [_PLAN_JSON]
        self._idx = 0

    async def chat(self, messages: list, **kwargs: Any) -> LLMResponse:
        content = self._responses[self._idx] if self._idx < len(self._responses) else "{}"
        self._idx += 1
        return LLMResponse(content=content, finish_reason="stop", model="mock")

    async def stream_chat(self, messages: list, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        yield StreamChunk()


def _mock_ws() -> WebSearchService:
    svc = AsyncMock(spec=WebSearchService)
    svc.search = AsyncMock(
        return_value=WebSearchResponse(
            query="q",
            provider="mock",
            results=[WebSearchResult(title="R1", url="https://a.com", snippet="Snip")],
            metadata=WebSearchMetadata(
                cached=False, cache_hit=False, latency_ms=5, provider="mock"
            ),
        )
    )
    return svc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fast_url_validation(monkeypatch: pytest.MonkeyPatch):
    """Keep service tests focused on lifecycle and persistence behavior.

    URL reachability has dedicated coverage in URL validator tests. Stubbing it
    here prevents external-network timeouts from dominating runtime.
    """

    async def _validate(_self, urls: list[str]) -> URLValidationReport:
        unique_urls = list(dict.fromkeys(urls))
        return URLValidationReport(
            total=len(unique_urls),
            reachable=len(unique_urls),
            unreachable=0,
            error_rate=0.0,
            results=[
                URLValidationResult(url=url, reachable=True, status_code=200) for url in unique_urls
            ],
        )

    monkeypatch.setattr(URLValidator, "validate", _validate)


class TestCreateRun:
    async def test_creates_plan(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        runtime, plan = await svc.create_run("AI frameworks", settings=_QUICK)
        assert plan is not None
        assert runtime.status == ResearchStatus.PLAN_READY

    async def test_replans_with_refined_query(self, tmp_path):
        responses = [
            _plan_with_clarification(
                _PLAN_JSON,
                {
                    "needs_clarification": True,
                    "confidence": 0.4,
                    "issues": ["Missing time period"],
                    "suggested_questions": ["Which year range matters?"],
                    "refined_query": "AI frameworks in 2026",
                },
            ),
            _PLAN_JSON,
        ]
        svc = ResearchService(_MockLLM(responses), _mock_ws(), data_dir=tmp_path)
        runtime, plan = await svc.create_run(
            "AI frameworks",
            settings=ResearchSettings(depth="standard"),
        )
        assert runtime._clarification is not None
        assert runtime._clarification.refined_query == "AI frameworks in 2026"
        assert plan.query == "AI frameworks in 2026"
        assert runtime.status == ResearchStatus.PLAN_READY

    async def test_idempotency_key(self, tmp_path):
        svc = ResearchService(_MockLLM([_PLAN_JSON, _PLAN_JSON]), _mock_ws(), data_dir=tmp_path)
        r1, _ = await svc.create_run("Q1", idempotency_key="key1", settings=_QUICK)
        r2, _ = await svc.create_run("Q2", idempotency_key="key1", settings=_QUICK)
        assert r1.run_id == r2.run_id

    async def test_create_empty_settings(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        runtime, plan = await svc.create_run("test", settings=_QUICK)
        assert runtime.status == ResearchStatus.PLAN_READY
        assert plan is not None

    async def test_default_mode_delegate(self, tmp_path, monkeypatch):
        monkeypatch.delenv(ENV_RESEARCH_ORCHESTRATION_MODE, raising=False)
        monkeypatch.delenv(ENV_RESEARCH_MAX_AGENTS, raising=False)
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        runtime, _ = await svc.create_run("test", settings=ResearchSettings(depth="quick"))
        assert runtime._settings.orchestration_mode == OrchestrationMode.DELEGATE
        assert runtime._settings.max_agents == 3

    async def test_explicit_mode_kept(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_RESEARCH_ORCHESTRATION_MODE, "delegate")
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        runtime, _ = await svc.create_run(
            "test",
            settings=ResearchSettings(depth="quick", orchestration_mode=OrchestrationMode.DIRECT),
        )
        assert runtime._settings.orchestration_mode == OrchestrationMode.DIRECT


class TestEditPlan:
    async def test_edit_bumps_version(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        runtime, plan = await svc.create_run("test", settings=_QUICK)
        edit = PlanEdit(op=PlanEditOperation.ADD, target_question="Extra?")
        updated = await svc.edit_plan(runtime.run_id, [edit])
        assert updated.version == 2

    async def test_version_conflict(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        runtime, _ = await svc.create_run("test", settings=_QUICK)
        with pytest.raises(VersionConflictError):
            await svc.edit_plan(runtime.run_id, [], client_plan_version=99)


class TestNotFound:
    async def test_require_run(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        with pytest.raises(RunNotFoundError):
            svc.get_progress("nonexistent")


class TestListRuns:
    async def test_list_after_create(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        await svc.create_run("test", settings=_QUICK)
        items = svc.list_runs()
        assert len(items) == 1
        assert items[0]["query"] == "test"


class TestCancelRun:
    async def test_cancel(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        runtime, _ = await svc.create_run("test", settings=_QUICK)
        await svc.cancel_run(runtime.run_id, "user cancelled")
        assert runtime.status == ResearchStatus.CANCELLED


class TestDeleteRun:
    async def test_delete_cancelled(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        runtime, _ = await svc.create_run("test", settings=_QUICK)
        await svc.cancel_run(runtime.run_id)
        await svc.delete_run(runtime.run_id)
        assert svc.get_run(runtime.run_id) is None

    async def test_delete_plan_ready(self, tmp_path):
        """Non-executing runs can be deleted."""
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        runtime, _ = await svc.create_run("test", settings=_QUICK)
        assert runtime.status == ResearchStatus.PLAN_READY
        await svc.delete_run(runtime.run_id)
        assert svc.get_run(runtime.run_id) is None

    async def test_delete_executing_blocked(self, tmp_path):
        """Executing runs cannot be deleted."""
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        runtime, _ = await svc.create_run("test", settings=_QUICK)
        runtime._status = ResearchStatus.EXECUTING
        with pytest.raises(RunNotTerminalError):
            await svc.delete_run(runtime.run_id)


class TestExecuteLifecycle:
    async def test_full_execute_lifecycle(self, tmp_path):
        svc = ResearchService(_MockLLM(_all_responses()), _mock_ws(), data_dir=tmp_path)
        runtime, _ = await svc.create_run("AI frameworks", settings=_QUICK)
        await svc.launch_run(runtime.run_id)
        assert runtime.status == ResearchStatus.COMPLETED

    async def test_execute_nonexistent(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        with pytest.raises(RunNotFoundError):
            await svc.launch_run("nonexistent")


class TestGetters:
    async def test_progress_after_create(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        runtime, _ = await svc.create_run("test", settings=_QUICK)
        progress = svc.get_progress(runtime.run_id)
        assert progress.status == ResearchStatus.PLAN_READY
        assert progress.total_steps == 5

    async def test_emitter_after_create(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        runtime, _ = await svc.create_run("test", settings=_QUICK)
        emitter = svc.get_emitter(runtime.run_id)
        assert emitter is not None

    async def test_emitter_missing_none(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        assert svc.get_emitter("nonexistent") is None

    async def test_execution_clears_buffer(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        runtime, _ = await svc.create_run("test", settings=_QUICK)
        buffer = svc.get_event_buffer(runtime.run_id)
        buffer.extend(["stale-1", "stale-2"])

        prepared = svc.prepare_for_execution(runtime.run_id)

        assert prepared.run_id == runtime.run_id
        assert buffer == []


class TestPersistence:
    async def test_json_written(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        runtime, _ = await svc.create_run("test", settings=_QUICK)
        path = tmp_path / f"{runtime.run_id}.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["run_id"] == runtime.run_id
        assert data["settings"]["orchestration_mode"]

    async def test_runs_loaded_on_startup(self, tmp_path):
        """Runs persisted as JSON are hydrated when a new service starts."""
        svc1 = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        runtime, _ = await svc1.create_run("persisted query", settings=_QUICK)
        run_id = runtime.run_id

        svc2 = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        items = svc2.list_runs()
        assert any(item["run_id"] == run_id for item in items)

    async def test_archived_run_shows_plan(self, tmp_path):
        """Archived run preserves the plan for display."""
        svc1 = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        runtime, plan = await svc1.create_run("test plan", settings=_QUICK)
        run_id = runtime.run_id

        svc2 = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        restored = svc2.get_run(run_id)
        assert restored is not None
        assert restored.plan is not None
        assert restored.plan.query == "test plan"

    async def test_archived_run_preserves_error(self, tmp_path):
        """If a run had an error, it's visible after reload."""
        svc1 = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        runtime, _ = await svc1.create_run("will fail", settings=_QUICK)
        runtime._error = "Research timed out after 420s"
        runtime._status = ResearchStatus.FAILED
        svc1._persist_run(runtime)

        svc2 = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        restored = svc2.get_run(runtime.run_id)
        assert restored is not None
        assert restored.error == "Research timed out after 420s"
        assert restored.status == ResearchStatus.FAILED

    async def test_delete_archived_run(self, tmp_path):
        """Archived terminal runs can be deleted."""
        svc1 = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        runtime, _ = await svc1.create_run("to delete", settings=_QUICK)
        await svc1.cancel_run(runtime.run_id)
        run_id = runtime.run_id

        svc2 = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        await svc2.delete_run(run_id)
        assert svc2.get_run(run_id) is None
        assert not (tmp_path / f"{run_id}.json").exists()


class TestConcurrencyLimit:
    async def test_running_count_zero_initially(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        assert svc.running_run_count() == 0

    async def test_running_count_increments(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        runtime, _ = await svc.create_run("running test", settings=_QUICK)
        assert svc.running_run_count() == 0
        runtime._status = ResearchStatus.EXECUTING
        assert svc.running_run_count() == 1
        runtime._status = ResearchStatus.GENERATING_REPORT
        assert svc.running_run_count() == 1
        runtime._status = ResearchStatus.PLAN_READY
        assert svc.running_run_count() == 0

    async def test_max_concurrent_constant(self, tmp_path):
        svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        assert ResearchService.MAX_CONCURRENT_RUNS == 3
        assert svc.MAX_CONCURRENT_RUNS == 3


class TestMemoryIntegration:
    """Research → Memory extraction pipeline."""

    async def test_memory_push_on_complete(self, tmp_path):
        """Completed runs push memory candidates to MemoryService."""
        from houyi_studio.server.memory.service import MemoryService

        from houyi.adapters.memory.engine import MemoryEngine
        from houyi.adapters.memory.store import MemoryStore

        mem_store = MemoryStore(data_dir=str(tmp_path / "memory"))
        mem_engine = MemoryEngine(mem_store)
        mem_svc = MemoryService(mem_engine)

        svc = ResearchService(
            _MockLLM(_all_responses()),
            _mock_ws(),
            data_dir=tmp_path,
            memory_service=mem_svc,
        )
        runtime, _ = await svc.create_run("AI research", settings=_QUICK)
        await svc.launch_run(runtime.run_id)
        assert runtime.status == ResearchStatus.COMPLETED
        candidates = mem_svc.list_candidates()
        assert len(candidates) > 0

    async def test_without_memory_svc(self, tmp_path):
        """Without memory_service, execution still succeeds."""
        svc = ResearchService(
            _MockLLM(_all_responses()),
            _mock_ws(),
            data_dir=tmp_path,
            memory_service=None,
        )
        runtime, _ = await svc.create_run("AI research", settings=_QUICK)
        await svc.launch_run(runtime.run_id)
        assert runtime.status == ResearchStatus.COMPLETED

    async def test_memory_extraction_failure(self, tmp_path):
        """Memory extraction failure does not crash execution."""
        from unittest.mock import MagicMock

        broken_svc = MagicMock()
        broken_svc.add_candidates.side_effect = RuntimeError("memory broken")

        svc = ResearchService(
            _MockLLM(_all_responses()),
            _mock_ws(),
            data_dir=tmp_path,
            memory_service=broken_svc,
        )
        runtime, _ = await svc.create_run("test", settings=_QUICK)
        await svc.launch_run(runtime.run_id)
        assert runtime.status == ResearchStatus.COMPLETED


class TestRehydrate:
    async def test_rehydrate_resets_plan(self, tmp_path):
        svc1 = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        runtime, _ = await svc1.create_run("rehydrate me", settings=_QUICK)
        run_id = runtime.run_id
        await svc1.cancel_run(run_id, "user cancelled")

        svc2 = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        archived = svc2.get_run(run_id)
        assert archived is not None
        assert isinstance(archived, _ArchivedRun)

        live = svc2._require_live_run(run_id)
        assert live.plan is not None
        assert live.plan.status == PlanStatus.DRAFT

    async def test_restores_search_results(self, tmp_path):
        """Rehydrated run must carry search_results so retry checkpoint works."""
        svc1 = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        runtime, _ = await svc1.create_run("checkpoint test", settings=_QUICK)
        run_id = runtime.run_id

        sr = SearchResult(
            question_id="sq_test1",
            rounds=[],
            sources=[SourceReference(url="https://example.com", title="T", snippet="s")],
            summary="done",
            coverage_score=0.9,
        )
        runtime._search_results.append(sr)
        svc1._persist_run(runtime)

        svc2 = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
        live = svc2._require_live_run(run_id)
        assert len(live._search_results) == 1
        assert live._search_results[0].question_id == "sq_test1"
        assert live._search_results[0].summary == "done"


class TestArchivedReportNormalization:
    """Envelope-leaked sections must be cleaned when loaded from disk."""

    def test_repairs_truncated_envelope(self):
        report = {
            "sections": [
                {
                    "title": "Current status",
                    "content": (
                        '{\n  "content": "### Career status missing\n\n'
                        "### Same-name exclusion\n\n"
                        "Several same-name candidates were ruled out after "
                        "verification [ref_abc]. Beyond that the evidence is thin"
                    ),
                },
            ],
        }
        out = _normalize_report_sections(report)
        assert out is not None
        content = out["sections"][0]["content"]
        assert not content.lstrip().startswith("{")
        assert '"content"' not in content
        assert "Same-name exclusion" in content

    def test_noop_on_clean(self):
        report = {
            "sections": [{"title": "T", "content": "Just prose [ref_1]."}],
        }
        out = _normalize_report_sections(report)
        assert out["sections"][0]["content"] == "Just prose [ref_1]."

    def test_expands_comma_refs(self):
        # Comma-grouped ``[ref_a, ref_b]`` tokens would bypass the
        # single-ref resolver and render as literal bracket noise. The
        # load path must split them into atomic ``[ref_a][ref_b]``.
        report = {
            "sections": [
                {
                    "title": "Evidence",
                    "content": "Evidence [ref_aaaaaaaa, ref_bbbbbbbb] support.",
                },
            ],
        }
        out = _normalize_report_sections(report)
        assert out is not None
        content = out["sections"][0]["content"]
        assert "[ref_aaaaaaaa, ref_bbbbbbbb]" not in content
        assert "[ref_aaaaaaaa][ref_bbbbbbbb]" in content

    def test_strips_citation_trailer(self):
        # An escaped ``","citations":`` trailer embedded in the content
        # string leaks as visible JSON noise; the load path must cut
        # the body at the first such boundary.
        trailer = (
            'Prose body ends here.",\n  "citations": [\n    {\n      '
            '"reference_id": "ref_aaaaaaaa"\n    }\n  ]'
        )
        report = {"sections": [{"title": "Core", "content": trailer}]}
        out = _normalize_report_sections(report)
        assert out is not None
        content = out["sections"][0]["content"]
        assert '"citations"' not in content
        assert '"reference_id"' not in content
        assert "Prose body ends here" in content

    def test_restores_orphan_mermaid_fence(self):
        # An indented diagram body followed by a lone closing ``` must
        # be wrapped in a matching ```mermaid opener rather than silently
        # dropped (otherwise the body renders as an unlabelled indented
        # code block in markdown).
        fence_content = (
            "Narrative paragraph one.\n\n"
            "    Env->>Trigger: change detected\n"
            "    end\n"
            "```\n\n"
            "Narrative paragraph two."
        )
        report = {"sections": [{"title": "Arch", "content": fence_content}]}
        out = _normalize_report_sections(report)
        assert out is not None
        content = out["sections"][0]["content"]
        assert "```mermaid" in content
        assert content.count("```") == 2
        assert "Env->>Trigger: change detected" in content
        assert "Narrative paragraph one" in content
        assert "Narrative paragraph two" in content

    def test_archived_run_applies_normalization(self):
        archived = _ArchivedRun(
            {
                "run_id": "rr_test",
                "status": "completed",
                "report": {
                    "sections": [
                        {
                            "title": "Section one",
                            "content": '{ "content": "### Sub\n\nBody text with evidence [ref_1].", "citations": [] }',
                        }
                    ]
                },
            }
        )
        content = archived.report_data["sections"][0]["content"]
        assert not content.lstrip().startswith("{")
        assert "Body text with evidence" in content
