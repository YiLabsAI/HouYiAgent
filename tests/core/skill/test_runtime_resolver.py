"""Tests for RuntimeResolver: automatic executor binding from runtime contract.

Covers:
- Adapter import and binding for tool/script modes
- Template mode (no executor binding)
- Graceful degradation on import failure
- hooks_root path normalization
- Batch resolution
- Structured audit events and observability counters
- Degradation strategy outcomes
"""

from __future__ import annotations

from houyi.domain.skill.runtime_contract import (
    CapabilityTier,
    RuntimeContract,
    RuntimeMode,
    RuntimeStatus,
)
from houyi.domain.skill.runtime_resolver import (
    ResolutionEvent,
    ResolutionStats,
    RuntimeResolver,
)
from houyi.domain.skill.spec import SkillSpec

# ── Helpers ──────────────────────────────────────────────────────────


def _make_skill(
    name: str = "test",
    runtime: RuntimeContract | None = None,
    executor=None,
    input_schema=None,
    output_schema=None,
) -> SkillSpec:
    from pydantic import BaseModel

    class EmptyModel(BaseModel):
        pass

    return SkillSpec(
        name=name,
        description="test skill",
        input_schema=input_schema or EmptyModel,
        output_schema=output_schema or EmptyModel,
        executor=executor,
        runtime_contract=runtime,
    )


# ── Adapter resolution tests ─────────────────────────────────────────


class TestResolveAdapter:
    """Tests for RuntimeResolver.resolve_adapter()."""

    def test_tool_mode_with_valid_adapter(self, tmp_path):
        """Valid adapter dotted path resolves to a callable."""
        # Create a temporary module with a callable
        mod_dir = tmp_path / "fake_pkg"
        mod_dir.mkdir()
        (mod_dir / "__init__.py").write_text("")
        (mod_dir / "adapter.py").write_text(
            "async def execute(input_data):\n    return {'result': 'ok'}\n"
        )

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            rc = RuntimeContract(mode=RuntimeMode.TOOL, adapter="fake_pkg.adapter:execute")
            skill = _make_skill(runtime=rc)
            resolver = RuntimeResolver()
            resolved = resolver.resolve(skill)
            assert resolved.executor is not None
            assert callable(resolved.executor)
            assert resolved.capability_tier == CapabilityTier.EXECUTABLE
        finally:
            sys.path.pop(0)
            sys.modules.pop("fake_pkg", None)
            sys.modules.pop("fake_pkg.adapter", None)

    def test_tool_mode_missing_adapter_field(self):
        """Tool mode without adapter field leaves executor unbound."""
        rc = RuntimeContract(mode=RuntimeMode.TOOL)
        skill = _make_skill(runtime=rc)
        resolver = RuntimeResolver()
        resolved = resolver.resolve(skill)
        assert resolved.executor is None

    def test_script_mode_with_adapter(self, tmp_path):
        """Script mode with adapter also resolves the callable."""
        mod_dir = tmp_path / "script_pkg"
        mod_dir.mkdir()
        (mod_dir / "__init__.py").write_text("")
        (mod_dir / "run.py").write_text("def run(data):\n    return data\n")

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            rc = RuntimeContract(mode=RuntimeMode.SCRIPT, adapter="script_pkg.run:run")
            skill = _make_skill(runtime=rc)
            resolver = RuntimeResolver()
            resolved = resolver.resolve(skill)
            assert resolved.executor is not None
            assert callable(resolved.executor)
        finally:
            sys.path.pop(0)
            sys.modules.pop("script_pkg", None)
            sys.modules.pop("script_pkg.run", None)

    def test_template_mode_no_executor(self):
        """Template mode does not bind any executor."""
        rc = RuntimeContract(mode=RuntimeMode.TEMPLATE)
        skill = _make_skill(runtime=rc)
        resolver = RuntimeResolver()
        resolved = resolver.resolve(skill)
        assert resolved.executor is None
        assert resolved.capability_tier == CapabilityTier.METADATA

    def test_no_runtime_contract(self):
        """Skill without runtime_contract is returned unchanged."""
        skill = _make_skill()
        resolver = RuntimeResolver()
        resolved = resolver.resolve(skill)
        assert resolved is skill  # identity — no copy needed

    def test_already_has_executor(self):
        """Skill with existing executor is not overwritten."""

        def original_fn(x):
            return x

        rc = RuntimeContract(mode=RuntimeMode.TOOL, adapter="nonexistent.module:fn")
        skill = _make_skill(runtime=rc, executor=original_fn)
        resolver = RuntimeResolver()
        resolved = resolver.resolve(skill)
        assert resolved.executor is original_fn

    def test_adapter_import_failure_graceful(self):
        """Invalid adapter path degrades gracefully, logs warning."""
        rc = RuntimeContract(mode=RuntimeMode.TOOL, adapter="no.such.module:fn")
        skill = _make_skill(runtime=rc)
        resolver = RuntimeResolver()
        resolved = resolver.resolve(skill)
        assert resolved.executor is None
        assert resolved.runtime_status == RuntimeStatus.UNAVAILABLE

    def test_adapter_bad_format_no_colon(self):
        """Adapter string without colon separator degrades gracefully."""
        rc = RuntimeContract(mode=RuntimeMode.TOOL, adapter="just.a.module.path")
        skill = _make_skill(runtime=rc)
        resolver = RuntimeResolver()
        resolved = resolver.resolve(skill)
        assert resolved.executor is None

    def test_adapter_attr_not_found(self, tmp_path):
        """Module exists but attribute not found degrades gracefully."""
        mod_dir = tmp_path / "exists_pkg"
        mod_dir.mkdir()
        (mod_dir / "__init__.py").write_text("")
        (mod_dir / "mod.py").write_text("x = 1\n")

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            rc = RuntimeContract(mode=RuntimeMode.TOOL, adapter="exists_pkg.mod:nonexistent")
            skill = _make_skill(runtime=rc)
            resolver = RuntimeResolver()
            resolved = resolver.resolve(skill)
            assert resolved.executor is None
        finally:
            sys.path.pop(0)
            sys.modules.pop("exists_pkg", None)
            sys.modules.pop("exists_pkg.mod", None)


# ── Path normalization tests ─────────────────────────────────────────


class TestHooksRootNormalization:
    """Tests for hooks_root path normalization."""

    def test_hooks_root_resolved_to_absolute(self, tmp_path):
        """hooks_root is resolved relative to project_root."""
        rc = RuntimeContract(
            mode=RuntimeMode.TEMPLATE,
            hooks_root="houyi/skills/planning",
        )
        skill = _make_skill(runtime=rc)
        resolver = RuntimeResolver(project_root=tmp_path)
        resolved = resolver.resolve(skill)
        expected = tmp_path / "houyi" / "skills" / "planning"
        assert resolved.skill_dir == expected

    def test_hooks_root_none_keeps_original_skill_dir(self, tmp_path):
        """When hooks_root is None, skill_dir is unchanged."""
        rc = RuntimeContract(mode=RuntimeMode.TEMPLATE)
        original_dir = tmp_path / "original"
        skill = _make_skill(runtime=rc)
        skill.skill_dir = original_dir
        resolver = RuntimeResolver(project_root=tmp_path)
        resolved = resolver.resolve(skill)
        assert resolved.skill_dir == original_dir

    def test_hooks_root_already_absolute(self, tmp_path):
        """Absolute hooks_root is used as-is."""
        abs_path = tmp_path / "absolute" / "path"
        rc = RuntimeContract(
            mode=RuntimeMode.TEMPLATE,
            hooks_root=str(abs_path),
        )
        skill = _make_skill(runtime=rc)
        resolver = RuntimeResolver(project_root=tmp_path)
        resolved = resolver.resolve(skill)
        assert resolved.skill_dir == abs_path


# ── Batch resolution tests ───────────────────────────────────────────


class TestResolveBatch:
    """Tests for resolving multiple skills at once."""

    def test_resolve_batch(self):
        """resolve_batch processes a list of skills."""
        skills = [
            _make_skill(name="a"),
            _make_skill(name="b", runtime=RuntimeContract(mode=RuntimeMode.TEMPLATE)),
        ]
        resolver = RuntimeResolver()
        results = resolver.resolve_batch(skills)
        assert len(results) == 2
        assert results[0].name == "a"
        assert results[1].name == "b"

    def test_resolve_batch_empty(self):
        """Empty list returns empty list."""
        resolver = RuntimeResolver()
        assert resolver.resolve_batch([]) == []


# ── Audit event tests ───────────────────────────────────────────────


class TestResolutionEvent:
    """Tests for the ResolutionEvent dataclass."""

    def test_to_dict_minimal(self):
        e = ResolutionEvent(skill_name="s", outcome="bound")
        d = e.to_dict()
        assert d["skill_name"] == "s"
        assert d["outcome"] == "bound"
        assert "adapter" not in d
        assert "error" not in d
        assert "fallback_used" not in d
        assert "timestamp" in d

    def test_to_dict_with_all_fields(self):
        e = ResolutionEvent(
            skill_name="s",
            outcome="degraded",
            adapter="mod:fn",
            error="import failed",
            fallback_used="core",
            duration_ms=1.234,
        )
        d = e.to_dict()
        assert d["adapter"] == "mod:fn"
        assert d["error"] == "import failed"
        assert d["fallback_used"] == "core"
        assert d["duration_ms"] == 1.23


# ── Observability counter tests ─────────────────────────────────────


class TestResolutionStats:
    """Tests for ResolutionStats aggregate counters."""

    def test_empty_stats(self):
        s = ResolutionStats()
        assert s.total == 0
        assert s.success_rate == 0.0
        assert s.degradation_rate == 0.0
        assert s.failure_rate == 0.0

    def test_record_and_rates(self):
        s = ResolutionStats()
        s.record("bound")
        s.record("bound")
        s.record("skipped")
        s.record("template")
        s.record("degraded")
        s.record("failed")
        assert s.total == 6
        assert s.bound == 2
        assert s.skipped == 1
        assert s.template == 1
        assert s.degraded == 1
        assert s.failed == 1
        # success = (bound + skipped + template) / total = 4/6
        assert abs(s.success_rate - 4 / 6) < 1e-6
        assert abs(s.degradation_rate - 1 / 6) < 1e-6
        assert abs(s.failure_rate - 1 / 6) < 1e-6

    def test_to_dict(self):
        s = ResolutionStats()
        s.record("bound")
        s.record("degraded")
        d = s.to_dict()
        assert d["total"] == 2
        assert d["bound"] == 1
        assert d["degraded"] == 1
        assert d["success_rate"] == 0.5
        assert d["degradation_rate"] == 0.5

    def test_unknown_outcome_counted_in_total(self):
        s = ResolutionStats()
        s.record("unknown")
        assert s.total == 1
        assert s.bound == 0
        assert s.failed == 0


# ── Audit trail integration tests ───────────────────────────────────


class TestAuditTrail:
    """Tests that resolve() emits audit events and updates stats."""

    def test_bound_event_on_successful_adapter(self, tmp_path):
        mod_dir = tmp_path / "audit_pkg"
        mod_dir.mkdir()
        (mod_dir / "__init__.py").write_text("")
        (mod_dir / "run.py").write_text("def go(data): return data\n")

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            rc = RuntimeContract(mode=RuntimeMode.TOOL, adapter="audit_pkg.run:go")
            skill = _make_skill(name="audit-test", runtime=rc)
            resolver = RuntimeResolver()
            resolver.resolve(skill)

            assert len(resolver.audit_log) == 1
            evt = resolver.audit_log[0]
            assert evt.skill_name == "audit-test"
            assert evt.outcome == "bound"
            assert evt.adapter == "audit_pkg.run:go"
            assert evt.error is None
            assert evt.duration_ms >= 0
            assert resolver.stats.bound == 1
        finally:
            sys.path.pop(0)
            sys.modules.pop("audit_pkg", None)
            sys.modules.pop("audit_pkg.run", None)

    def test_skipped_event_when_executor_present(self):
        rc = RuntimeContract(mode=RuntimeMode.TOOL, adapter="nope:fn")
        skill = _make_skill(name="skip-test", runtime=rc, executor=lambda x: x)
        resolver = RuntimeResolver()
        resolver.resolve(skill)

        assert len(resolver.audit_log) == 1
        assert resolver.audit_log[0].outcome == "skipped"
        assert resolver.stats.skipped == 1

    def test_template_event(self):
        rc = RuntimeContract(mode=RuntimeMode.TEMPLATE)
        skill = _make_skill(name="tmpl-test", runtime=rc)
        resolver = RuntimeResolver()
        resolver.resolve(skill)

        assert resolver.audit_log[0].outcome == "template"
        assert resolver.stats.template == 1

    def test_degraded_event_on_adapter_failure(self):
        rc = RuntimeContract(mode=RuntimeMode.TOOL, adapter="no.such.mod:fn")
        skill = _make_skill(name="degrade-test", runtime=rc)
        resolver = RuntimeResolver()
        resolver.resolve(skill)

        assert len(resolver.audit_log) == 1
        evt = resolver.audit_log[0]
        assert evt.outcome == "degraded"
        assert evt.adapter == "no.such.mod:fn"
        assert evt.error is not None
        assert "no.such.mod" in evt.error
        assert resolver.stats.degraded == 1

    def test_failed_event_when_no_adapter_no_template(self):
        rc = RuntimeContract(mode=RuntimeMode.TOOL)
        skill = _make_skill(name="fail-test", runtime=rc)
        resolver = RuntimeResolver()
        resolver.resolve(skill)

        assert len(resolver.audit_log) == 1
        assert resolver.audit_log[0].outcome == "failed"
        assert resolver.stats.failed == 1

    def test_no_event_when_no_runtime_contract(self):
        skill = _make_skill(name="no-rc")
        resolver = RuntimeResolver()
        resolver.resolve(skill)

        assert len(resolver.audit_log) == 0
        assert resolver.stats.total == 0

    def test_batch_accumulates_stats(self):
        skills = [
            _make_skill(name="a", runtime=RuntimeContract(mode=RuntimeMode.TEMPLATE)),
            _make_skill(name="b", runtime=RuntimeContract(mode=RuntimeMode.TOOL)),
            _make_skill(
                name="c", runtime=RuntimeContract(mode=RuntimeMode.TOOL, adapter="bad:format")
            ),
        ]
        resolver = RuntimeResolver()
        resolver.resolve_batch(skills)

        assert resolver.stats.total == 3
        assert resolver.stats.template == 1
        assert resolver.stats.failed == 1  # no adapter, tool mode
        assert resolver.stats.degraded == 1  # bad adapter format
        assert len(resolver.audit_log) == 3

    def test_audit_log_duration_is_positive(self):
        """Duration should be a non-negative float for every event."""
        rc = RuntimeContract(mode=RuntimeMode.TOOL, adapter="no.such:fn")
        skill = _make_skill(name="dur-test", runtime=rc)
        resolver = RuntimeResolver()
        resolver.resolve(skill)
        assert resolver.audit_log[0].duration_ms >= 0

    def test_stats_rates_after_mixed_batch(self, tmp_path):
        """Verify success/degradation/failure rates after a realistic batch."""
        mod_dir = tmp_path / "rate_pkg"
        mod_dir.mkdir()
        (mod_dir / "__init__.py").write_text("")
        (mod_dir / "fn.py").write_text("def go(d): return d\n")

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            skills = [
                # bound (adapter success)
                _make_skill(
                    name="s1",
                    runtime=RuntimeContract(mode=RuntimeMode.TOOL, adapter="rate_pkg.fn:go"),
                ),
                # template (success)
                _make_skill(name="s2", runtime=RuntimeContract(mode=RuntimeMode.TEMPLATE)),
                # degraded (bad adapter)
                _make_skill(
                    name="s3", runtime=RuntimeContract(mode=RuntimeMode.TOOL, adapter="no.mod:fn")
                ),
                # failed (tool mode, no adapter)
                _make_skill(name="s4", runtime=RuntimeContract(mode=RuntimeMode.TOOL)),
            ]
            resolver = RuntimeResolver()
            resolver.resolve_batch(skills)

            d = resolver.stats.to_dict()
            assert d["total"] == 4
            assert d["bound"] == 1
            assert d["template"] == 1
            assert d["degraded"] == 1
            assert d["failed"] == 1
            assert d["success_rate"] == 0.5  # 2/4
            assert d["degradation_rate"] == 0.25
            assert d["failure_rate"] == 0.25
        finally:
            sys.path.pop(0)
            sys.modules.pop("rate_pkg", None)
            sys.modules.pop("rate_pkg.fn", None)


# ── Dependency probing tests ────────────────────────────────────────


class TestProbeDependencies:
    """Tests for probe_dependencies pre-flight validation."""

    def test_probe_valid_adapter(self, tmp_path):
        mod_dir = tmp_path / "probe_pkg"
        mod_dir.mkdir()
        (mod_dir / "__init__.py").write_text("")
        (mod_dir / "ok.py").write_text("def run(d): return d\n")

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            skill = _make_skill(
                name="valid",
                runtime=RuntimeContract(mode=RuntimeMode.TOOL, adapter="probe_pkg.ok:run"),
            )
            resolver = RuntimeResolver()
            results = resolver.probe_dependencies([skill])
            assert len(results) == 1
            assert results[0]["importable"] is True
            assert results[0]["callable"] is True
            assert results[0]["error"] is None
        finally:
            sys.path.pop(0)
            sys.modules.pop("probe_pkg", None)
            sys.modules.pop("probe_pkg.ok", None)

    def test_probe_missing_module(self):
        skill = _make_skill(
            name="bad-mod",
            runtime=RuntimeContract(mode=RuntimeMode.TOOL, adapter="no.such.module:fn"),
        )
        resolver = RuntimeResolver()
        results = resolver.probe_dependencies([skill])
        assert len(results) == 1
        assert results[0]["importable"] is False
        assert results[0]["callable"] is False
        assert "import" in results[0]["error"].lower()

    def test_probe_missing_attribute(self, tmp_path):
        mod_dir = tmp_path / "probe_attr"
        mod_dir.mkdir()
        (mod_dir / "__init__.py").write_text("")
        (mod_dir / "mod.py").write_text("x = 1\n")

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            skill = _make_skill(
                name="bad-attr",
                runtime=RuntimeContract(
                    mode=RuntimeMode.TOOL, adapter="probe_attr.mod:nonexistent"
                ),
            )
            resolver = RuntimeResolver()
            results = resolver.probe_dependencies([skill])
            assert len(results) == 1
            assert results[0]["importable"] is True
            assert results[0]["callable"] is False
            assert results[0]["error"] is not None
        finally:
            sys.path.pop(0)
            sys.modules.pop("probe_attr", None)
            sys.modules.pop("probe_attr.mod", None)

    def test_probe_skips_skills_without_adapter(self):
        skills = [
            _make_skill(name="no-rc"),
            _make_skill(name="tmpl", runtime=RuntimeContract(mode=RuntimeMode.TEMPLATE)),
            _make_skill(name="tool-no-adapter", runtime=RuntimeContract(mode=RuntimeMode.TOOL)),
        ]
        resolver = RuntimeResolver()
        results = resolver.probe_dependencies(skills)
        assert len(results) == 0

    def test_probe_bad_format_adapter(self):
        skill = _make_skill(
            name="bad-fmt",
            runtime=RuntimeContract(mode=RuntimeMode.TOOL, adapter="no_colon_here"),
        )
        resolver = RuntimeResolver()
        results = resolver.probe_dependencies([skill])
        assert len(results) == 1
        assert results[0]["importable"] is False
        assert results[0]["error"] is not None

    def test_probe_mixed_batch(self, tmp_path):
        """Probing a mixed batch returns results only for adapter-bearing skills."""
        mod_dir = tmp_path / "mixed_probe"
        mod_dir.mkdir()
        (mod_dir / "__init__.py").write_text("")
        (mod_dir / "ok.py").write_text("def run(d): return d\n")

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            skills = [
                _make_skill(
                    name="good",
                    runtime=RuntimeContract(mode=RuntimeMode.TOOL, adapter="mixed_probe.ok:run"),
                ),
                _make_skill(
                    name="bad",
                    runtime=RuntimeContract(mode=RuntimeMode.TOOL, adapter="nonexistent.mod:fn"),
                ),
                _make_skill(name="no-adapter"),
            ]
            resolver = RuntimeResolver()
            results = resolver.probe_dependencies(skills)
            assert len(results) == 2
            good = next(r for r in results if r["skill_name"] == "good")
            bad = next(r for r in results if r["skill_name"] == "bad")
            assert good["callable"] is True
            assert bad["callable"] is False
        finally:
            sys.path.pop(0)
            sys.modules.pop("mixed_probe", None)
            sys.modules.pop("mixed_probe.ok", None)
