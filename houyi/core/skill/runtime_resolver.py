"""Automatic executor binding from SKILL.md runtime contract.

After a skill is loaded from file/URL/directory, RuntimeResolver inspects
its ``runtime_contract`` and attempts to:

1. Import and bind the ``adapter`` callable as the skill's executor.
2. Normalize ``hooks_root`` to an absolute path for asset resolution.
3. Leave skills with ``mode=template`` or no runtime contract unchanged.

Skills that already have an executor bound are never overwritten.

Runtime resolution records structured audit events and counters so callers can
inspect success, degradation, and failure outcomes.
"""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)


# ── Audit event ───────────────────────────────────────────────────────


@dataclass
class ResolutionEvent:
    """Structured audit record for a single runtime resolution attempt."""

    skill_name: str
    outcome: str  # "bound", "skipped", "template", "degraded", "failed"
    adapter: str | None = None
    error: str | None = None
    duration_ms: float = 0.0
    fallback_used: str | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "skill_name": self.skill_name,
            "outcome": self.outcome,
            "duration_ms": round(self.duration_ms, 2),
            "timestamp": self.timestamp,
        }
        if self.adapter:
            d["adapter"] = self.adapter
        if self.error:
            d["error"] = self.error
        if self.fallback_used:
            d["fallback_used"] = self.fallback_used
        return d


# ── Observability counters ────────────────────────────────────────────


@dataclass
class ResolutionStats:
    """Aggregate counters for runtime resolution outcomes."""

    total: int = 0
    bound: int = 0
    skipped: int = 0
    template: int = 0
    degraded: int = 0
    failed: int = 0

    def record(self, outcome: str) -> None:
        self.total += 1
        if outcome == "bound":
            self.bound += 1
        elif outcome == "skipped":
            self.skipped += 1
        elif outcome == "template":
            self.template += 1
        elif outcome == "degraded":
            self.degraded += 1
        elif outcome == "failed":
            self.failed += 1

    @property
    def success_rate(self) -> float:
        """Fraction of resolutions that resulted in a bound or skipped outcome."""
        if self.total == 0:
            return 0.0
        return (self.bound + self.skipped + self.template) / self.total

    @property
    def degradation_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.degraded / self.total

    @property
    def failure_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.failed / self.total

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "bound": self.bound,
            "skipped": self.skipped,
            "template": self.template,
            "degraded": self.degraded,
            "failed": self.failed,
            "success_rate": round(self.success_rate, 4),
            "degradation_rate": round(self.degradation_rate, 4),
            "failure_rate": round(self.failure_rate, 4),
        }


def _elapsed(t0: float) -> float:
    """Return milliseconds elapsed since *t0* (from ``time.monotonic()``)."""
    return (time.monotonic() - t0) * 1000


class RuntimeResolver:
    """Resolve runtime contracts into bound executors and normalized paths.

    Provides structured audit logging and observability counters for every
    resolution attempt. Degradation strategy:

    1. Adapter import (if declared)
    2. Core fallback (handled by caller, e.g. ``_hydrate_external_runtime``)
    3. Template passthrough (no executor needed)
    4. Unavailable (resolution failed)

    Usage::

        resolver = RuntimeResolver(project_root=Path.cwd())
        resolved_skill = resolver.resolve(skill)
        print(resolver.stats.to_dict())  # observability counters
        print(resolver.audit_log)        # structured audit trail
    """

    def __init__(self, project_root: Path | None = None) -> None:
        self._project_root = project_root or Path.cwd()
        self.stats = ResolutionStats()
        self.audit_log: list[ResolutionEvent] = []

    def resolve(self, skill: object) -> object:
        """Resolve a single skill's runtime contract.

        Returns the original skill if no resolution is needed, or a
        copy with executor and/or skill_dir updated.
        """
        from houyi.core.skill.runtime_contract import RuntimeContract, RuntimeMode

        skill_name = str(getattr(skill, "name", "?"))
        t0 = time.monotonic()

        rc = getattr(skill, "runtime_contract", None)
        if rc is None or not isinstance(rc, RuntimeContract):
            return skill

        # Never overwrite an existing executor
        if callable(getattr(skill, "executor", None)):
            self._emit(skill_name, "skipped", duration_ms=_elapsed(t0))
            return self._maybe_normalize_paths(skill, rc)

        # Template mode: no executor to bind
        if rc.mode == RuntimeMode.TEMPLATE:
            self._emit(skill_name, "template", duration_ms=_elapsed(t0))
            return self._maybe_normalize_paths(skill, rc)

        # Attempt adapter resolution for tool/script modes
        if rc.adapter:
            executor, err = self._import_adapter(rc.adapter, skill_name=skill_name)
            if executor is not None:
                updated = cast(Any, skill).model_copy(update={"executor": executor})
                self._emit(
                    skill_name,
                    "bound",
                    adapter=rc.adapter,
                    duration_ms=_elapsed(t0),
                )
                return self._maybe_normalize_paths(updated, rc)
            # Adapter declared but import failed → degraded
            self._emit(
                skill_name,
                "degraded",
                adapter=rc.adapter,
                error=err,
                duration_ms=_elapsed(t0),
            )
            return self._maybe_normalize_paths(skill, rc)

        # No adapter declared and not template → failed
        self._emit(skill_name, "failed", duration_ms=_elapsed(t0))
        return self._maybe_normalize_paths(skill, rc)

    def resolve_batch(self, skills: list) -> list:
        """Resolve runtime contracts for a batch of skills."""
        return [self.resolve(s) for s in skills]

    def probe_dependencies(self, skills: list) -> list[dict[str, Any]]:
        """Probe adapter dependencies for a batch of skills without binding.

        Returns a list of probe results, one per skill that declares an
        adapter. Each result contains:

        - ``skill_name``: name of the skill
        - ``adapter``: declared adapter string
        - ``importable``: whether the module can be imported
        - ``callable``: whether the attribute is a callable
        - ``error``: error message if any check failed

        This is useful for pre-flight validation before batch import,
        enabling early detection of missing dependencies.
        """
        from houyi.core.skill.runtime_contract import RuntimeContract

        results: list[dict[str, Any]] = []
        for skill in skills:
            rc = getattr(skill, "runtime_contract", None)
            if not isinstance(rc, RuntimeContract) or not rc.adapter:
                continue

            skill_name = str(getattr(skill, "name", "?"))
            adapter = rc.adapter
            probe: dict[str, Any] = {
                "skill_name": skill_name,
                "adapter": adapter,
                "importable": False,
                "callable": False,
                "error": None,
            }

            fn, err = self._import_adapter(adapter, skill_name=skill_name)
            if fn is not None:
                probe["importable"] = True
                probe["callable"] = True
            elif err:
                probe["error"] = err
                # Module was importable if error is about attribute/callable,
                # not about format or import failure
                if "no attribute" in err or "not callable" in err:
                    probe["importable"] = True

            results.append(probe)
        return results

    # ── Private helpers ──────────────────────────────────────────

    def _emit(
        self,
        skill_name: str,
        outcome: str,
        *,
        adapter: str | None = None,
        error: str | None = None,
        duration_ms: float = 0.0,
        fallback_used: str | None = None,
    ) -> None:
        """Record an audit event and update counters."""
        event = ResolutionEvent(
            skill_name=skill_name,
            outcome=outcome,
            adapter=adapter,
            error=error,
            duration_ms=duration_ms,
            fallback_used=fallback_used,
        )
        self.audit_log.append(event)
        self.stats.record(outcome)
        if outcome in ("degraded", "failed"):
            logger.warning(
                "Runtime resolution %s for skill '%s': %s",
                outcome,
                skill_name,
                error or "no adapter",
            )
        else:
            logger.debug(
                "Runtime resolution %s for skill '%s' (%.1fms)",
                outcome,
                skill_name,
                duration_ms,
            )

    def _import_adapter(
        self, adapter: str, skill_name: str = "?"
    ) -> tuple[object | None, str | None]:
        """Import a dotted path like ``'module.path:function_name'``.

        Returns ``(callable, None)`` on success or ``(None, error_msg)`` on failure.
        """
        if ":" not in adapter:
            msg = f"adapter '{adapter}' missing ':' separator (expected 'module.path:attr')"
            logger.warning("Skill '%s': %s", skill_name, msg)
            return None, msg

        module_path, _, attr_name = adapter.partition(":")
        if not module_path or not attr_name:
            msg = f"adapter '{adapter}' has empty module or attribute"
            logger.warning("Skill '%s': %s", skill_name, msg)
            return None, msg

        try:
            module = importlib.import_module(module_path)
        except Exception as exc:
            msg = f"failed to import adapter module '{module_path}': {exc}"
            logger.warning("Skill '%s': %s", skill_name, msg)
            return None, msg

        fn = getattr(module, attr_name, None)
        if fn is None:
            msg = f"module '{module_path}' has no attribute '{attr_name}'"
            logger.warning("Skill '%s': %s", skill_name, msg)
            return None, msg

        if not callable(fn):
            msg = f"adapter '{module_path}:{attr_name}' is not callable ({type(fn).__name__})"
            logger.warning("Skill '%s': %s", skill_name, msg)
            return None, msg

        logger.info(
            "Skill '%s': bound executor from adapter '%s'",
            skill_name,
            adapter,
        )
        return fn, None

    def _maybe_normalize_paths(self, skill: object, rc: object) -> object:
        """Normalize ``hooks_root`` to an absolute ``skill_dir``."""
        hooks_root = getattr(rc, "hooks_root", None)
        if not hooks_root:
            return skill

        hooks_path = Path(hooks_root)
        if not hooks_path.is_absolute():
            hooks_path = self._project_root / hooks_path

        current_dir = getattr(skill, "skill_dir", None)
        if current_dir == hooks_path:
            return skill

        return cast(Any, skill).model_copy(update={"skill_dir": hooks_path})
