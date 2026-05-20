from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

from houyi.application.evolution.artifacts import EvolutionArtifact
from houyi.application.evolution.audit_log import (
    AuditEntry,
    EvolutionAuditLog,
    InMemoryEvolutionAuditLog,
)
from houyi.application.evolution.constraints import BasicConstraintGate, EvolutionConstraintGate
from houyi.application.evolution.dataset import EvolutionDatasetBuilder, SignalDatasetBuilder
from houyi.application.evolution.evaluation import (
    EvolutionEvaluator,
    HeuristicEvolutionEvaluator,
)
from houyi.application.evolution.event_log import EvolutionEventLog, InMemoryEvolutionEventLog
from houyi.application.evolution.events import EvolutionEvent
from houyi.application.evolution.optimizers import (
    DeterministicEvolutionOptimizer,
    EvolutionOptimizer,
)
from houyi.application.evolution.policy_store import (
    EvolutionPolicyStore,
    InMemoryEvolutionPolicyStore,
)
from houyi.application.evolution.promotion import PromotionDecision, PromotionManager
from houyi.application.evolution.providers import (
    EvolutionCursorStore,
    InMemoryEvolutionCursorStore,
)
from houyi.application.evolution.scheduler import EvolutionScheduler
from houyi.application.evolution.shadow import DatasetShadowEvaluator, ShadowEvaluator
from houyi.application.evolution.signals import EvolutionSignalMiner

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvolutionTickReport:
    skipped: bool
    reason: str
    events_read: int = 0
    signals_found: int = 0
    candidates_created: int = 0
    promotion: PromotionDecision | None = None
    cursor: int = 0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class EvolutionRunReport:
    ticks: list[EvolutionTickReport] = field(default_factory=list)
    events_consumed: int = 0
    promotions: int = 0
    errors: int = 0
    last_report: EvolutionTickReport | None = None
    cursor: int = 0


class EvolutionDaemon:
    def __init__(
        self,
        artifact: EvolutionArtifact,
        *,
        event_log: EvolutionEventLog | None = None,
        policy_store: EvolutionPolicyStore | None = None,
        scheduler: EvolutionScheduler | None = None,
        signal_miner: EvolutionSignalMiner | None = None,
        dataset_builder: EvolutionDatasetBuilder | None = None,
        optimizer: EvolutionOptimizer | None = None,
        evaluator: EvolutionEvaluator | None = None,
        constraint_gate: EvolutionConstraintGate | None = None,
        promotion_manager: PromotionManager | None = None,
        cursor_store: EvolutionCursorStore | None = None,
        audit_log: EvolutionAuditLog | None = None,
        consumer_name: str = "evolution_daemon",
        shadow_evaluator: ShadowEvaluator | None = None,
    ) -> None:
        self.artifact = artifact
        self.event_log = event_log or InMemoryEvolutionEventLog()
        self.policy_store = policy_store or InMemoryEvolutionPolicyStore()
        self.policy_store.set_active(artifact)
        self.scheduler = scheduler or EvolutionScheduler()
        self.signal_miner = signal_miner or EvolutionSignalMiner()
        self.dataset_builder = dataset_builder or SignalDatasetBuilder()
        self.optimizer = optimizer or DeterministicEvolutionOptimizer()
        self.evaluator = evaluator or HeuristicEvolutionEvaluator()
        self.constraint_gate = constraint_gate or BasicConstraintGate()
        self.promotion_manager = promotion_manager or PromotionManager()
        self.cursor_store = cursor_store or InMemoryEvolutionCursorStore()
        self.audit_log = audit_log if audit_log is not None else InMemoryEvolutionAuditLog()
        self.consumer_name = consumer_name
        self.shadow_evaluator = shadow_evaluator or DatasetShadowEvaluator(self.evaluator)
        self.cursor = self.cursor_store.get_cursor(self.consumer_name)
        self.running = False

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def emit_event(self, event: EvolutionEvent) -> None:
        self.event_log.append(event)

    def tick(self, *, force: bool = False) -> EvolutionTickReport:
        cursor_before = self.cursor
        try:
            report = self._tick_body(force=force, cursor_before=cursor_before)
        except Exception as exc:
            self._record_audit(
                action="error",
                cursor_before=cursor_before,
                cursor_after=self.cursor,
                events_consumed=0,
                skipped=True,
                reason="exception",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._record_audit(
            action="tick",
            cursor_before=cursor_before,
            cursor_after=report.cursor,
            events_consumed=report.events_read,
            skipped=report.skipped,
            reason=report.reason,
            promotion_level=(None if report.promotion is None else report.promotion.level.value),
        )
        return report

    def _tick_body(self, *, force: bool, cursor_before: int) -> EvolutionTickReport:
        if not self.running:
            return EvolutionTickReport(True, "not_running", cursor=cursor_before)
        if not self.scheduler.should_run(force=force):
            return EvolutionTickReport(True, "scheduler_not_ready", cursor=cursor_before)
        events, next_cursor = self.event_log.read_since(self.cursor)
        if not events:
            self.scheduler.mark_run()
            self.cursor = next_cursor
            self.cursor_store.set_cursor(self.consumer_name, self.cursor)
            return EvolutionTickReport(True, "no_events", cursor=self.cursor)
        signals = self.signal_miner.mine(events)
        if not signals:
            self.scheduler.mark_run()
            self.cursor = next_cursor
            self.cursor_store.set_cursor(self.consumer_name, self.cursor)
            return EvolutionTickReport(
                True,
                "no_signals",
                events_read=len(events),
                cursor=self.cursor,
            )
        dataset = self.dataset_builder.build(signals)
        candidates = self.optimizer.propose(self.artifact, signals)
        evaluations = self.evaluator.evaluate(candidates, dataset)
        passed_candidates = [
            replace(evaluation.candidate, score=evaluation.score)
            for evaluation in evaluations
            if evaluation.passed
        ]
        valid_candidates = [
            candidate
            for candidate in passed_candidates
            if all(result.passed for result in self.constraint_gate.validate(candidate))
        ]
        promotion = self.promotion_manager.stage_shadow(
            valid_candidates,
            self.artifact,
            self.shadow_evaluator,
            dataset,
            policy_store=self.policy_store,
        )
        if promotion.level.value == "active" and promotion.candidate is not None:
            self.artifact = promotion.candidate.artifact
        self.scheduler.mark_run()
        self.cursor = next_cursor
        self.cursor_store.set_cursor(self.consumer_name, self.cursor)
        return EvolutionTickReport(
            False,
            "completed",
            events_read=len(events),
            signals_found=len(signals),
            candidates_created=len(candidates),
            promotion=promotion,
            cursor=self.cursor,
        )

    def run_until_idle(
        self,
        *,
        max_ticks: int = 100,
        force: bool = True,
        swallow_errors: bool = True,
    ) -> EvolutionRunReport:
        ticks: list[EvolutionTickReport] = []
        events_consumed = 0
        promotions = 0
        errors = 0
        last: EvolutionTickReport | None = None
        for _ in range(max_ticks):
            try:
                report = self.tick(force=force)
            except Exception as exc:
                if not swallow_errors:
                    raise
                logger.warning("evolution daemon tick failed: %s", exc)
                errors += 1
                report = EvolutionTickReport(
                    True,
                    "exception",
                    cursor=self.cursor,
                    error=f"{type(exc).__name__}: {exc}",
                )
                ticks.append(report)
                last = report
                continue
            ticks.append(report)
            last = report
            events_consumed += report.events_read
            if report.promotion is not None and report.promotion.candidate is not None:
                promotions += 1
            if report.skipped and report.reason in {
                "no_events",
                "not_running",
                "scheduler_not_ready",
            }:
                break
        return EvolutionRunReport(
            ticks=ticks,
            events_consumed=events_consumed,
            promotions=promotions,
            errors=errors,
            last_report=last,
            cursor=self.cursor,
        )

    def _record_audit(
        self,
        *,
        action: str,
        cursor_before: int,
        cursor_after: int,
        events_consumed: int,
        skipped: bool,
        reason: str,
        promotion_level: str | None = None,
        error: str | None = None,
    ) -> None:
        try:
            self.audit_log.append_audit(
                AuditEntry(
                    consumer=self.consumer_name,
                    action=action,
                    cursor_before=cursor_before,
                    cursor_after=cursor_after,
                    events_consumed=events_consumed,
                    skipped=skipped,
                    reason=reason,
                    promotion_level=promotion_level,
                    error=error,
                )
            )
        except Exception:
            logger.exception("evolution audit_log append failed")
